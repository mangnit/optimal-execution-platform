// Microbenchmark: OrderBook.
//
// Three scenarios cover the three latency-critical paths of the LOB:
//
//   BM_OB_AddLimit_Resting — a passive order that does not cross. The
//     level is prewarmed so `std::map::try_emplace` does not allocate. The
//     path exercised is: slab allocate → intrusive push-back → id-index
//     insert. This is the "quote" case in a typical maker workload.
//
//   BM_OB_Cancel_Mid — cancel-by-id of an order sitting in the middle of a
//     deep FIFO. Exercises the splitmix64 hash lookup, the back-shift
//     erase, and O(1) intrusive unlink from a non-terminal position — the
//     property the intrusive design exists for.
//
//   BM_OB_Cross_Sweep — an aggressive taker that sweeps through several
//     prewarmed ask levels, generating a fill stream per level. Measures
//     the matching path, which is the reason the book exists.
//
// Each iteration does its own setup/teardown outside the timed region using
// `state.PauseTiming` / `state.ResumeTiming`; the rdtsc sample brackets
// only the operation under test.

#include "benchmarks/bench_util.hpp"
#include "core/order_book.hpp"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

namespace {

using oep::bench::LatencySampler;
using oep::bench::ReadCycleCounter;
using oep::core::Fill;
using oep::core::OrderBook;
using oep::core::OrderId;
using oep::core::Price;
using oep::core::Quantity;
using oep::core::Side;

constexpr std::size_t kSlabCapacity = 1u << 14;   // 16 384 resting orders
constexpr Price kRestPrice = 10'000;              // arbitrary tick price
constexpr Price kCrossBase = 10'000;              // ask ladder base
constexpr std::size_t kCrossLevels = 8;           // ask levels to sweep
constexpr Quantity kQtyPerOrder = 10;
constexpr std::uint32_t kMidLevelDepth = 64;      // orders per level for mid

// Sink lambda captured as a plain function pointer: we don't want the
// benchmark to conflate fill-handling with book work.
struct NullSink {
  void operator()(const Fill&) const noexcept {}
};

void BM_OB_AddLimit_Resting(benchmark::State& state) {
  OrderBook book(kSlabCapacity);
  book.PrewarmLevel(Side::kBuy, kRestPrice);

  LatencySampler sampler(static_cast<std::size_t>(state.max_iterations));
  OrderId next_id = 1;
  for (auto _ : state) {
    const OrderId id = next_id++;
    const std::uint64_t t0 = ReadCycleCounter();
    bool ok = book.AddLimit(id, Side::kBuy, kRestPrice, kQtyPerOrder,
                            NullSink{});
    const std::uint64_t t1 = ReadCycleCounter();
    benchmark::DoNotOptimize(ok);
    sampler.Record(t1 - t0);

    // Keep the pool from filling up over the benchmark. The cancel is
    // deliberately outside the timed region — it is the previous
    // iteration's teardown, not this iteration's work.
    state.PauseTiming();
    (void)book.Cancel(id);
    state.ResumeTiming();
  }
  sampler.Publish(state);
}
BENCHMARK(BM_OB_AddLimit_Resting);

void BM_OB_Cancel_Mid(benchmark::State& state) {
  OrderBook book(kSlabCapacity);
  book.PrewarmLevel(Side::kBuy, kRestPrice);
  NullSink sink;

  // Pre-fill the level with a deep FIFO. We'll rotate through the middle
  // third of the queue, cancelling and re-adding so the level stays roughly
  // the same depth throughout the run.
  std::vector<OrderId> ids;
  ids.reserve(kMidLevelDepth);
  OrderId next_id = 1;
  for (std::uint32_t i = 0; i < kMidLevelDepth; ++i) {
    (void)book.AddLimit(next_id, Side::kBuy, kRestPrice, kQtyPerOrder, sink);
    ids.push_back(next_id);
    ++next_id;
  }
  const std::uint32_t middle_slot = kMidLevelDepth / 2;

  LatencySampler sampler(static_cast<std::size_t>(state.max_iterations));
  for (auto _ : state) {
    const OrderId victim = ids[middle_slot];
    const std::uint64_t t0 = ReadCycleCounter();
    bool cancelled = book.Cancel(victim);
    const std::uint64_t t1 = ReadCycleCounter();
    benchmark::DoNotOptimize(cancelled);
    sampler.Record(t1 - t0);

    // Re-add outside the timed region so the level depth stays stable.
    // The new order lands at the tail; we rotate `ids` so the cancel next
    // iteration hits a still-mid-of-queue victim (the head neighbour of
    // the previous victim shifts one slot toward the front).
    state.PauseTiming();
    const OrderId reborn = next_id++;
    (void)book.AddLimit(reborn, Side::kBuy, kRestPrice, kQtyPerOrder, sink);
    ids.erase(ids.begin() + middle_slot);
    ids.push_back(reborn);
    state.ResumeTiming();
  }
  sampler.Publish(state);
}
BENCHMARK(BM_OB_Cancel_Mid);

void BM_OB_Cross_Sweep(benchmark::State& state) {
  // The taker sweep needs a fresh ask ladder every iteration; PauseTiming
  // covers the rebuild so it doesn't pollute the sample. Slab capacity is
  // sized to hold one full ladder plus headroom for slop.
  const Quantity qty_per_level = kQtyPerOrder;
  const Quantity sweep_qty = qty_per_level * kCrossLevels;
  const Price top_price =
      kCrossBase + static_cast<Price>(kCrossLevels) - 1;

  OrderBook book(kSlabCapacity);
  for (std::size_t i = 0; i < kCrossLevels; ++i) {
    book.PrewarmLevel(Side::kSell,
                      kCrossBase + static_cast<Price>(i));
  }

  NullSink sink;
  std::size_t fills_observed = 0;
  auto counting_sink = [&](const Fill&) noexcept { ++fills_observed; };

  LatencySampler sampler(static_cast<std::size_t>(state.max_iterations));
  OrderId next_id = 1;
  for (auto _ : state) {
    // (Re-)stock every ask level with one maker.
    state.PauseTiming();
    for (std::size_t i = 0; i < kCrossLevels; ++i) {
      (void)book.AddLimit(next_id++, Side::kSell,
                          kCrossBase + static_cast<Price>(i),
                          qty_per_level, sink);
    }
    state.ResumeTiming();

    const OrderId taker_id = next_id++;
    const std::uint64_t t0 = ReadCycleCounter();
    bool ok = book.AddLimit(taker_id, Side::kBuy, top_price,
                            sweep_qty, counting_sink);
    const std::uint64_t t1 = ReadCycleCounter();
    benchmark::DoNotOptimize(ok);
    sampler.Record(t1 - t0);
  }
  sampler.Publish(state);
  state.counters["fills_per_iter"] =
      benchmark::Counter(static_cast<double>(fills_observed) /
                        static_cast<double>(state.iterations()));
}
BENCHMARK(BM_OB_Cross_Sweep);

}  // namespace

int main(int argc, char** argv) {
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
