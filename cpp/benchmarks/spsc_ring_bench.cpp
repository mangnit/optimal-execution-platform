// Microbenchmark: SpscRing<T, Capacity>.
//
// Two things need measuring here, and they answer different questions:
//
//   BM_Spsc_PushPopRoundtrip — single-threaded round-trip (`TryPush` +
//     `TryPop` back-to-back). Isolates the raw instruction cost of the
//     atomic bookkeeping. No cross-core traffic; the ring's head/tail lines
//     stay owner-exclusive in L1D. This is the "no producer/consumer
//     contention" lower bound.
//
//   BM_Spsc_TryPushLoaded — producer measured against a real consumer thread
//     draining in a tight loop on a second logical CPU. Now the producer's
//     `acquire` load of read_idx_ can miss L1D and the store to write_idx_
//     is snooped across the MESI mesh. This is the number that matters for
//     the internal→relay clock bridge.
//
// The consumer thread inherits pinning from the `OEP_BENCH_PIN_CPU_CONSUMER`
// env var (independent of the producer's `OEP_BENCH_PIN_CPU`), so you can
// exercise same-core-sibling vs cross-socket topologies without recompiling.

#include "benchmarks/bench_util.hpp"
#include "core/spsc_ring.hpp"

#include <benchmark/benchmark.h>

#include <atomic>
#include <cstdint>
#include <thread>

namespace {

using oep::bench::LatencySampler;
using oep::bench::ReadCycleCounter;
using oep::core::SpscRing;

// The payload is deliberately a plain uint64_t — SPSC ring users on the
// hot path shovel small POD records (fill events, order acks), and the
// benchmark's job is to isolate the ring's cost, not model any particular
// downstream schema.
using Payload = std::uint64_t;
constexpr std::size_t kRingCapacity = 1024;

void BM_Spsc_PushPopRoundtrip(benchmark::State& state) {
  SpscRing<Payload, kRingCapacity> ring;
  LatencySampler sampler(static_cast<std::size_t>(state.max_iterations));
  Payload out{};
  for (auto _ : state) {
    const Payload v = static_cast<Payload>(state.iterations());
    const std::uint64_t t0 = ReadCycleCounter();
    bool pushed = ring.TryPush(v);
    bool popped = ring.TryPop(out);
    const std::uint64_t t1 = ReadCycleCounter();
    benchmark::DoNotOptimize(pushed);
    benchmark::DoNotOptimize(popped);
    benchmark::DoNotOptimize(out);
    sampler.Record(t1 - t0);
  }
  sampler.Publish(state);
}
BENCHMARK(BM_Spsc_PushPopRoundtrip);

void BM_Spsc_TryPushLoaded(benchmark::State& state) {
  SpscRing<Payload, kRingCapacity> ring;
  std::atomic<bool> stop{false};

  std::thread consumer([&] {
    oep::bench::MaybePinToEnvCpu("OEP_BENCH_PIN_CPU_CONSUMER");
    Payload out{};
    while (!stop.load(std::memory_order_relaxed)) {
      (void)ring.TryPop(out);
      benchmark::DoNotOptimize(out);
    }
    // Drain the tail so the ring's destructor observes an empty state.
    while (ring.TryPop(out)) {
      benchmark::DoNotOptimize(out);
    }
  });

  LatencySampler sampler(static_cast<std::size_t>(state.max_iterations));
  Payload v = 0;
  for (auto _ : state) {
    ++v;
    const std::uint64_t t0 = ReadCycleCounter();
    bool pushed = ring.TryPush(v);
    const std::uint64_t t1 = ReadCycleCounter();
    benchmark::DoNotOptimize(pushed);
    sampler.Record(t1 - t0);
  }
  stop.store(true, std::memory_order_relaxed);
  consumer.join();

  sampler.Publish(state);
  state.counters["dropped"] =
      benchmark::Counter(static_cast<double>(ring.dropped_messages()));
}
BENCHMARK(BM_Spsc_TryPushLoaded);

}  // namespace

int main(int argc, char** argv) {
  const int pinned = oep::bench::MaybePinToEnvCpu("OEP_BENCH_PIN_CPU");
  if (pinned >= 0) {
    benchmark::AddCustomContext("pinned_cpu_producer", std::to_string(pinned));
  }
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
