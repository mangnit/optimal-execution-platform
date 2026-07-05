// Shared benchmark utilities for the internal-clock hot-path suite.
//
// Google Benchmark by default reports mean iteration time. What
// `latency_report.md` actually needs is the tail: p50, p99, p99.9. This
// header provides a `LatencySampler` that collects one rdtsc-based cycle
// delta per benchmark iteration, then — at the end of the benchmark — sorts
// and publishes the percentiles as custom Google Benchmark counters so they
// flow out through `--benchmark_format=json` and into the report generator.
//
// Timing method: `__rdtscp()` reads the invariant TSC and acts as a lightly
// serialising instruction, so the surrounding ops don't reorder across it.
// Cycles are converted to nanoseconds via a one-off TSC-frequency calibration
// performed at first use (see `NanosPerCycle`).
//
// Thread pinning scaffold: `MaybePinToConfiguredCpu()` reads the
// `OEP_BENCH_PIN_CPU` environment variable and, if set to a non-negative
// integer, pins the calling thread to that logical CPU via
// `pthread_setaffinity_np`. Pinning typically requires the process to have
// the appropriate CAP_SYS_NICE / core-isolation setup on Linux; leaving the
// env var unset makes the benchmarks portable to CI runners that can't pin.

#ifndef OEP_BENCHMARKS_BENCH_UTIL_HPP_
#define OEP_BENCHMARKS_BENCH_UTIL_HPP_

#include "core/tsc_clock.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <thread>
#include <vector>

namespace oep::bench {

// The rdtsc primitive and its cycle→ns calibration live in core/tsc_clock.hpp
// so the sim_runner / live_loop entry points can reach for them without
// pulling in Google Benchmark. Re-exported into `oep::bench::` so existing
// benchmark TUs (and any external callers) compile unchanged.
using oep::core::NanosPerCycle;
using oep::core::ReadCycleCounter;

// Collects per-iteration cycle deltas and, on `Publish`, computes tail
// percentiles into the benchmark's counters map. The vector is reserved up
// front so `Record` is allocation-free — otherwise its own `push_back` would
// bleed into every sample we take.
class LatencySampler {
 public:
  explicit LatencySampler(std::size_t reserve = 1u << 20) {
    samples_.reserve(reserve);
  }

  void Record(std::uint64_t cycles) noexcept {
    samples_.push_back(cycles);
  }

  // Publish p50/p99/p99.9 in nanoseconds as Google Benchmark counters. Also
  // records `min_ns`, `max_ns`, and `samples` so the report can quote them.
  void Publish(benchmark::State& state) {
    if (samples_.empty()) {
      return;
    }
    std::sort(samples_.begin(), samples_.end());
    const double ns_per_cycle = NanosPerCycle();
    auto quantile = [&](double q) {
      const std::size_t n = samples_.size();
      // Nearest-rank on a zero-indexed sorted vector.
      std::size_t rank = static_cast<std::size_t>(q * (n - 1) + 0.5);
      if (rank >= n) rank = n - 1;
      return samples_[rank] * ns_per_cycle;
    };
    state.counters["p50_ns"] = benchmark::Counter(quantile(0.50));
    state.counters["p99_ns"] = benchmark::Counter(quantile(0.99));
    state.counters["p999_ns"] = benchmark::Counter(quantile(0.999));
    state.counters["min_ns"] =
        benchmark::Counter(samples_.front() * ns_per_cycle);
    state.counters["max_ns"] =
        benchmark::Counter(samples_.back() * ns_per_cycle);
    state.counters["samples"] =
        benchmark::Counter(static_cast<double>(samples_.size()));
  }

  std::size_t size() const noexcept { return samples_.size(); }
  void clear() noexcept { samples_.clear(); }

 private:
  std::vector<std::uint64_t> samples_;
};

// Pin the calling thread to `cpu`. Non-fatal on failure (CI runners often
// deny setaffinity); returns true on success.
inline bool PinThreadToCpu(int cpu) {
  if (cpu < 0) return false;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

// Optional pinning scaffold. When `OEP_BENCH_PIN_CPU` is set, pin the caller
// to that CPU. Returns the CPU actually pinned to (or -1 if not attempted).
// The benchmark's `main()` calls this before the first iteration so every
// benchmark inherits the affinity; per-benchmark threads (e.g. the SPSC
// consumer) can re-invoke with `OEP_BENCH_PIN_CPU_CONSUMER` if set.
inline int MaybePinToEnvCpu(const char* env_name) {
  const char* v = std::getenv(env_name);
  if (v == nullptr || v[0] == '\0') return -1;
  const int cpu = std::atoi(v);
  if (cpu < 0) return -1;
  const bool ok = PinThreadToCpu(cpu);
  return ok ? cpu : -1;
}

}  // namespace oep::bench

#endif  // OEP_BENCHMARKS_BENCH_UTIL_HPP_
