// Fused tick-to-decision benchmark — the §7 headline path, measured as one
// continuous internal-clock operation instead of per-component micros:
//
//   synthetic order arrival → LOB update (+ matching + telemetry emit)
//     → observation build → strategy eval (LibTorch RlPolicy forward)
//
//   BM_Fused_ArrivalToObs    — arrival → book/publisher → 8-dim obs write.
//     Everything on the path *except* the neural forward. The delta against
//     TickToDecision isolates the LibTorch inference cost.
//
//   BM_Fused_TickToDecision  — the full path, ending in
//     RlPolicy::ComputeNextAction(). The observation is written through
//     `state_ptr()` straight into the preallocated input tensor — the
//     designed zero-copy integration path (task P5.6).
//
// The actor is a TorchScript twin of the exported SAC MlpPolicy actor
// (8 → 256 → ReLU → 256 → ReLU → 2 → tanh, ≈275 KiB of fp32 weights —
// same topology and parameter count as models/sac_policy.pt). It is built
// in-process with seeded weights so the benchmark has no filesystem
// dependency on a trained artifact; set OEP_BENCH_POLICY_PT to a .pt path
// to run the real exported actor instead.
//
// Workload realism follows sim_runner / py_module: a seeded xorshift64
// stream of maker orders 1–5 ticks off the mid on both sides, with 1-in-4
// marketable crossers so the matching path is exercised, a bounded resting
// population enforced by FIFO eviction, and the telemetry ring drained —
// all outside the rdtscp bracket. Only the arrival→decision path is
// sampled; the LatencySampler percentiles are the measurement (Google
// Benchmark's own mean includes housekeeping and is not what the report
// renders).

#include "benchmarks/bench_util.hpp"
#include "core/order_book.hpp"
#include "core/spsc_ring.hpp"
#include "core/tsc_clock.hpp"
#include "exec/rl_policy.hpp"
#include "exec/telemetry_publisher.hpp"
#include "external/kdb_logger.hpp"

#include <ATen/Parallel.h>
#include <torch/script.h>

#include <benchmark/benchmark.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>

namespace {

using oep::bench::LatencySampler;
using oep::bench::ReadCycleCounter;
using oep::core::Fill;
using oep::core::OrderBook;
using oep::core::OrderId;
using oep::core::Price;
using oep::core::Quantity;
using oep::core::Side;
using oep::core::TscNanoClock;
using oep::exec::RlPolicy;

using Event = oep::external::TelemetryEvent;
constexpr std::size_t kRingCapacity = 1u << 16;
using Ring = oep::core::SpscRing<Event, kRingCapacity>;
using Publisher = oep::exec::TelemetryPublisher<Ring, TscNanoClock>;

constexpr std::size_t kSlabCapacity = 8192;
constexpr Price kMid = 100;                    // matches sim_runner / SimEnv
constexpr Price kMinPrewarmPx = 90;
constexpr Price kMaxPrewarmPx = 110;
constexpr Quantity kParentQty = 1'000'000;
// RlPolicy's per-step schedule caches are sized at construction; rebuild the
// (cheap, module-sharing) policy shell every kPolicySteps decisions so
// current_step never runs off the cache. 64 Ki steps ⇒ a handful of rebuilds
// per benchmark repetition, all outside the timed bracket.
constexpr std::size_t kPolicySteps = 1u << 16;
// FIFO of optimistically-tracked resting ids for the eviction guard.
// Cancel of an already-consumed id is a safe no-op (same trick as SimEnv).
constexpr std::size_t kIdFifoCapacity = 1u << 15;

// Deterministic xorshift64 — bit-identical to sim_runner.cpp.
struct Xorshift64 {
  explicit Xorshift64(std::uint64_t seed) noexcept : s(seed ? seed : 1) {}
  std::uint64_t Next() noexcept {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  std::uint64_t s;
};

// TorchScript twin of the exported SAC actor: same topology and parameter
// count as models/sac_policy.pt (SB3 MlpPolicy net_arch=[256, 256]).
// matmul/relu/tanh only — all TorchScript builtins, no dtype ceremony.
torch::jit::script::Module MakeSacActorTwin() {
  torch::manual_seed(0xC0FFEEBABEULL);
  const auto opt = torch::TensorOptions().dtype(torch::kFloat32);
  torch::jit::script::Module m("sac_actor_twin");
  m.register_parameter("w1", torch::randn({8, 256}, opt) * 0.05f, false);
  m.register_parameter("b1", torch::zeros({256}, opt), false);
  m.register_parameter("w2", torch::randn({256, 256}, opt) * 0.05f, false);
  m.register_parameter("b2", torch::zeros({256}, opt), false);
  m.register_parameter("w3", torch::randn({256, 2}, opt) * 0.05f, false);
  m.register_parameter("b3", torch::zeros({2}, opt), false);
  m.define(
      "def forward(self, x: Tensor) -> Tensor:\n"
      "    h1 = torch.relu(torch.matmul(x, self.w1) + self.b1)\n"
      "    h2 = torch.relu(torch.matmul(h1, self.w2) + self.b2)\n"
      "    return torch.tanh(torch.matmul(h2, self.w3) + self.b3)\n");
  return m;
}

torch::jit::script::Module LoadActor() {
  if (const char* pt = std::getenv("OEP_BENCH_POLICY_PT");
      pt != nullptr && pt[0] != '\0') {
    return torch::jit::load(pt);
  }
  return MakeSacActorTwin();
}

// One synthetic arrival, decoded from a single PRNG draw. Buys rest 1–5
// ticks below mid / sells above (no self-cross), except 1-in-4 marketable
// orders that cross the touch so the matching + fill path stays hot.
struct Arrival {
  Side side;
  Price px;
  Quantity qty;
};

inline Arrival NextArrival(Xorshift64& rng) noexcept {
  const std::uint64_t r = rng.Next();
  const bool buy = (r & 1ULL) != 0ULL;
  const bool marketable = ((r >> 1) & 3ULL) == 0ULL;
  const Price depth = static_cast<Price>((r >> 3) % 5ULL);  // 0..4 ticks
  Price px;
  if (buy) {
    px = marketable ? (kMid + 1 + depth) : (kMid - 1 - depth);
  } else {
    px = marketable ? (kMid - 1 - depth) : (kMid + 1 + depth);
  }
  const Quantity qty = 1 + ((r >> 8) % 8ULL);
  return {buy ? Side::kBuy : Side::kSell, px, qty};
}

// Shared fixture: the same wired stack SimEnv / sim_runner use, minus the
// relay thread (external clock — its absence only means we drain the ring
// ourselves, outside the timed bracket).
struct FusedFixture {
  FusedFixture()
      : ring(std::make_unique<Ring>()), book(kSlabCapacity),
        publisher(book, *ring, clock), rng(0xC0FFEEBABEULL) {
    for (Price p = kMinPrewarmPx; p <= kMaxPrewarmPx; ++p) {
      book.PrewarmLevel(Side::kBuy, p);
      book.PrewarmLevel(Side::kSell, p);
    }
    id_fifo.fill(0);
  }

  // Post-iteration housekeeping, outside the rdtscp bracket: drain the
  // telemetry ring (keeps TryPush on the success path, as it is when the
  // relay is keeping up) and evict the oldest resting orders when the slab
  // nears exhaustion (mirrors sim_runner's guard).
  void Housekeep() noexcept {
    Event e;
    while (ring->TryPop(e)) {
    }
    while (book.open_order_count() >= kSlabCapacity - 64 &&
           fifo_count > 0) {
      (void)publisher.Cancel(id_fifo[fifo_head]);
      fifo_head = (fifo_head + 1) & (kIdFifoCapacity - 1);
      --fifo_count;
    }
  }

  void TrackId(OrderId id) noexcept {
    if (fifo_count == kIdFifoCapacity) {  // overwrite-oldest, like the ring
      fifo_head = (fifo_head + 1) & (kIdFifoCapacity - 1);
      --fifo_count;
    }
    id_fifo[(fifo_head + fifo_count) & (kIdFifoCapacity - 1)] = id;
    ++fifo_count;
  }

  std::unique_ptr<Ring> ring;
  OrderBook book;
  TscNanoClock clock{};
  Publisher publisher;
  Xorshift64 rng;
  std::array<OrderId, kIdFifoCapacity> id_fifo{};
  std::size_t fifo_head = 0;
  std::size_t fifo_count = 0;
  OrderId next_id = 1;
};

// Build the 8-dim observation exactly as SimEnv::WriteState lays it out.
// `remaining_frac` / `time_frac` come from the strategy schedule; the four
// price features are live book queries — the same best-of scans the RL
// bridge performs every step.
inline void WriteObs(const OrderBook& book, float remaining_frac,
                     float time_frac, float* obs) noexcept {
  const Price bb = book.has_bid() ? book.best_bid() : 0;
  const Price ba = book.has_ask() ? book.best_ask() : 0;
  const float bb_f = static_cast<float>(bb);
  const float ba_f = static_cast<float>(ba);
  obs[0] = remaining_frac;
  obs[1] = time_frac;
  obs[2] = bb_f;
  obs[3] = ba_f;
  obs[4] = (bb == 0 || ba == 0) ? 0.0f : (ba_f - bb_f);
  obs[5] = (bb == 0 || ba == 0) ? static_cast<float>(kMid)
                                : 0.5f * (bb_f + ba_f);
  obs[6] = 1.0f - remaining_frac;
  obs[7] = 1.0f;
}

void BM_Fused_ArrivalToObs(benchmark::State& state) {
  FusedFixture fx;
  alignas(64) float obs[RlPolicy::kStateDim] = {0.0f};
  std::size_t step = 0;

  LatencySampler sampler(static_cast<std::size_t>(state.max_iterations));
  for (auto _ : state) {
    const Arrival a = NextArrival(fx.rng);
    const OrderId id = fx.next_id++;
    const float remaining_frac =
        1.0f - static_cast<float>(step % kPolicySteps) /
                   static_cast<float>(kPolicySteps);

    const std::uint64_t t0 = ReadCycleCounter();
    bool ok = fx.publisher.AddLimit(id, a.side, a.px, a.qty);
    WriteObs(fx.book, remaining_frac, remaining_frac, obs);
    const std::uint64_t t1 = ReadCycleCounter();

    benchmark::DoNotOptimize(ok);
    benchmark::DoNotOptimize(obs[5]);
    sampler.Record(t1 - t0);

    fx.TrackId(id);
    fx.Housekeep();
    ++step;
  }
  sampler.Publish(state);
}
BENCHMARK(BM_Fused_ArrivalToObs);

void BM_Fused_TickToDecision(benchmark::State& state) {
  FusedFixture fx;
  const torch::jit::script::Module actor = LoadActor();
  const RlPolicy::Params params{kParentQty, Side::kSell, kPolicySteps};

  std::optional<RlPolicy> policy;
  policy.emplace(actor, params);

  // Warm-up: the JIT graph executor profiles the first invocations before
  // settling on an optimized plan; absorb that outside the sample window
  // (same recipe as rl_policy_test).
  {
    const std::array<float, RlPolicy::kStateDim> warm{
        0.5f, 1.0f, 99.0f, 101.0f, 2.0f, 100.0f, 0.0f, 1.0f};
    for (int i = 0; i < 200; ++i) {
      policy->WriteState(warm);
      policy->ComputeNextAction();
    }
  }

  std::size_t fills_observed = 0;
  torch::NoGradGuard no_grad;

  LatencySampler sampler(static_cast<std::size_t>(state.max_iterations));
  for (auto _ : state) {
    // Recycle the policy shell before its schedule cache runs out. Shares
    // the same Module (and its optimized graph), so this is a cheap
    // re-init, and it happens outside the timed bracket.
    if (policy->current_step() + 1 >= kPolicySteps) {
      policy.emplace(actor, params);
    }
    const Arrival a = NextArrival(fx.rng);
    const OrderId id = fx.next_id++;
    const std::size_t step = policy->current_step();
    const float remaining_frac =
        static_cast<float>(policy->RemainingAfterStep(step)) /
        static_cast<float>(kParentQty);
    const float time_frac = 1.0f - static_cast<float>(step) /
                                       static_cast<float>(kPolicySteps);

    const std::uint64_t t0 = ReadCycleCounter();
    bool ok = fx.publisher.AddLimit(id, a.side, a.px, a.qty);
    WriteObs(fx.book, remaining_frac, time_frac, policy->state_ptr());
    policy->ComputeNextAction();
    const std::uint64_t t1 = ReadCycleCounter();

    benchmark::DoNotOptimize(ok);
    benchmark::DoNotOptimize(policy->last_size_fraction());
    sampler.Record(t1 - t0);
    if (ok) {
      ++fills_observed;  // proxy: accepted arrivals (fills counted in ring)
    }

    fx.TrackId(id);
    fx.Housekeep();
  }
  sampler.Publish(state);
  state.counters["accepted_per_iter"] =
      benchmark::Counter(static_cast<double>(fills_observed) /
                         static_cast<double>(state.iterations()));
  state.counters["ring_dropped"] = benchmark::Counter(
      static_cast<double>(fx.ring->dropped_messages()));
}
BENCHMARK(BM_Fused_TickToDecision);

}  // namespace

int main(int argc, char** argv) {
  // The decision path is single-threaded on the internal clock by design;
  // pin Torch's intraop pool to one thread so the (1×8) MLP never fans out.
  at::set_num_threads(1);
  const int pinned = oep::bench::MaybePinToEnvCpu("OEP_BENCH_PIN_CPU");
  if (pinned >= 0) {
    benchmark::AddCustomContext("pinned_cpu", std::to_string(pinned));
  }
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
