// Production TSC → nanosecond clock for the internal hot path
// (docs/architecture.md §0, task P3_sim_runner).
//
// The latency benchmarks already used an rdtsc-based sampler
// (`cpp/benchmarks/bench_util.hpp`) — this header extracts the two primitives
// that the sim runner / live loop need to plumb into
// `TelemetryPublisher::ClockFn`, so nothing on the internal clock has to pull
// in `<benchmark/benchmark.h>`:
//
//   * `ReadCycleCounter()` — one `__rdtscp` on x86-64 (a lightly serialising
//     instruction that prevents the surrounding ops from being reordered
//     across the sample), a `steady_clock` fallback everywhere else.
//   * `NanosPerCycle()`   — one-shot busy-spin calibration against
//     `steady_clock` at first use, cached for the process lifetime.
//
// The bench harness keeps its `oep::bench::` aliases (see `bench_util.hpp`) so
// `-DOEP_BUILD_BENCHMARKS=ON` still builds unchanged.

#ifndef OEP_CORE_TSC_CLOCK_HPP_
#define OEP_CORE_TSC_CLOCK_HPP_

#include <chrono>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#define OEP_CORE_HAVE_RDTSC 1
#else
#define OEP_CORE_HAVE_RDTSC 0
#endif

namespace oep::core {

// Read the invariant TSC. `rdtscp` waits for prior instructions to retire, so
// the sample brackets what we intended to measure without a reordered tail.
inline std::uint64_t ReadCycleCounter() noexcept {
#if OEP_CORE_HAVE_RDTSC
  unsigned aux;
  return __rdtscp(&aux);
#else
  return static_cast<std::uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

// Busy-spin the TSC against `steady_clock` for ~50 ms and cache the ratio.
// A busy loop (rather than a sleep) keeps the calibrating thread on-core so
// the scheduler can't migrate us mid-window and skew the sample.
inline double NanosPerCycle() {
  static const double cached = [] {
#if OEP_CORE_HAVE_RDTSC
    using clock = std::chrono::steady_clock;
    const auto wall_start = clock::now();
    const std::uint64_t tsc_start = ReadCycleCounter();
    while (clock::now() - wall_start < std::chrono::milliseconds(50)) {
      // Compiler barrier: force the call to happen every iteration without
      // depending on <benchmark>'s DoNotOptimize.
      const std::uint64_t sample = ReadCycleCounter();
      asm volatile("" : : "r"(sample) : "memory");
    }
    const std::uint64_t tsc_end = ReadCycleCounter();
    const auto wall_end = clock::now();
    const double ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            wall_end - wall_start).count());
    const double cycles = static_cast<double>(tsc_end - tsc_start);
    return ns / cycles;
#else
    // Fallback path: ReadCycleCounter already returns nanoseconds.
    return 1.0;
#endif
  }();
  return cached;
}

// Callable adapter that satisfies `TelemetryPublisher`'s `ClockFn` contract.
// Captures a base TSC reading at construction so the emitted `ts_ns` values
// start near zero — that keeps `static_cast<std::uint64_t>(cycles *
// ns_per_cycle)` in double's 53-bit exact-integer range for the lifetime of
// any realistic sim / trading session (2^53 ns ≈ 104 days).
class TscNanoClock {
 public:
  TscNanoClock() noexcept
      : ns_per_cycle_(NanosPerCycle()),
        base_cycles_(ReadCycleCounter()) {}

  std::uint64_t operator()() const noexcept {
    const std::uint64_t now = ReadCycleCounter();
    const std::uint64_t delta = now - base_cycles_;
    return static_cast<std::uint64_t>(
        static_cast<double>(delta) * ns_per_cycle_);
  }

 private:
  double ns_per_cycle_;
  std::uint64_t base_cycles_;
};

}  // namespace oep::core

#endif  // OEP_CORE_TSC_CLOCK_HPP_
