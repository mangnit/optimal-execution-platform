// Microbenchmark: SlabAllocator<Order>.
//
// What's measured (per-iteration, rdtsc-bracketed): the two hot-path calls
// that stand between the matching engine and the "no `new` on the hot path"
// invariant. `Allocate` pops the intrusive free-list head; `Deallocate`
// pushes it back. Both should be a handful of cycles each — this
// benchmark's job is to prove that and to catch a regression the day it
// creeps in.
//
// Two scenarios:
//   BM_Slab_AllocDealloc  — hot round-trip (alloc then immediately dealloc).
//                            Every iteration touches the same slot, so this
//                            is the L1-hot lower bound.
//   BM_Slab_DrainRefill   — alloc/dealloc walks the whole pool. Not a
//                            hot-path pattern per se, but exercises the
//                            free-list threading and stresses the D-cache.

#include "benchmarks/bench_util.hpp"
#include "core/order_book.hpp"        // brings in the POD `Order` type
#include "core/slab_allocator.hpp"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

namespace {

using oep::bench::LatencySampler;
using oep::bench::ReadCycleCounter;
using oep::core::Order;
using oep::core::SlabAllocator;

// Fixed capacity: large enough to hold the drain-refill scenario in one go,
// small enough to fit comfortably in L1 (Order is a cache line, so 1024
// slots = 64 KiB — L1D-sized on typical x86 cores; the drain benchmark
// deliberately spills, and that's a signal we want to see).
constexpr std::size_t kCapacity = 1024;

void BM_Slab_AllocDealloc(benchmark::State& state) {
  SlabAllocator<Order> slab(kCapacity);
  LatencySampler sampler(static_cast<std::size_t>(state.max_iterations));
  for (auto _ : state) {
    const std::uint64_t t0 = ReadCycleCounter();
    Order* p = slab.Allocate();
    benchmark::DoNotOptimize(p);
    slab.Deallocate(p);
    const std::uint64_t t1 = ReadCycleCounter();
    sampler.Record(t1 - t0);
  }
  sampler.Publish(state);
}
BENCHMARK(BM_Slab_AllocDealloc);

void BM_Slab_DrainRefill(benchmark::State& state) {
  SlabAllocator<Order> slab(kCapacity);
  std::vector<Order*> live;
  live.reserve(kCapacity);
  LatencySampler sampler(static_cast<std::size_t>(state.max_iterations));
  for (auto _ : state) {
    // Drain the pool.
    const std::uint64_t t0 = ReadCycleCounter();
    for (std::size_t i = 0; i < kCapacity; ++i) {
      Order* p = slab.Allocate();
      benchmark::DoNotOptimize(p);
      live.push_back(p);
    }
    // Refill.
    for (Order* p : live) {
      slab.Deallocate(p);
    }
    const std::uint64_t t1 = ReadCycleCounter();
    // One "iteration" here is one full drain-refill cycle; normalise the
    // sample so p50/p99 are still per-op numbers.
    sampler.Record((t1 - t0) / (2 * kCapacity));
    live.clear();
  }
  sampler.Publish(state);
}
BENCHMARK(BM_Slab_DrainRefill);

}  // namespace

// Custom main so we can honour OEP_BENCH_PIN_CPU before the harness starts.
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
