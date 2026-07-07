# Latency Report — Internal Clock

Generated 2026-07-07 13:38:09 UTC from `scripts/run_benchmarks.sh`.

Every benchmark below runs on the **internal clock** (docs/architecture.md §7):
the matching path, its allocator, and the SPSC ring that bridges it to the
relay. FIX, kdb writes, and dashboard IO live on the external clock and are
deliberately not measured here.

Percentiles are computed from per-iteration `rdtscp` samples, converted to
nanoseconds via a boot-time TSC-frequency calibration. `samples` is the
number of per-op observations backing each row; p99.9 is only trustworthy
when that count is well above a thousand.

**Run environment (this report):** producer thread pinned to CPU 3
(`OEP_BENCH_PIN_CPU=3`), SPSC consumer pinned to CPU 5
(`OEP_BENCH_PIN_CPU_CONSUMER=5`), on a WSL2 guest. WSL2 exposes no cpufreq
sysfs, so the CPU governor cannot be forced to `performance` from the
guest and the host may still migrate/throttle the underlying core — the
millisecond-class `max` values are host-scheduling artifacts, and p99.9
should be read with that caveat. Pinning is real (verified via
`pthread_setaffinity_np` return codes); frequency control is not. A
core-isolated bare-metal Linux run remains the final word for the tail.
Treat the numbers as the engine's regression baseline, not a marketing
claim (per §7's "no pre-committed numbers" rule).

Hardware PMU counters (§ "Cache & branch behaviour" below) were collected
with a minimal `perf_event_open(2)` wrapper (`perf` is not installable on
this box; the WSL2 kernel does expose the hardware PMU): user-space-only
counts (`exclude_kernel`), `inherit` across benchmark worker threads,
attached at exec — the same counting mode as `perf stat -e ... --all-user`.

### `slab_allocator_bench`

host=`LAPTOP-AEPE3M3P` · cpu=`1382 MHz × 8` · cache-scaled=`False` · pinned_cpu=`3`

| Benchmark | p50 | p99 | p99.9 | min | max | samples | extra |
|---|---:|---:|---:|---:|---:|---:|---|
| `BM_Slab_AllocDealloc` | 10.1 ns | 15.9 ns | 23.1 ns | 7.2 ns | 2.10 ms | 99,136,123 |  |
| `BM_Slab_DrainRefill` | 0.7 ns | 2.2 ns | 8.0 ns | 0.7 ns | 33.21 µs | 968,757 |  |

### `spsc_ring_bench`

host=`LAPTOP-AEPE3M3P` · cpu=`1382 MHz × 8` · cache-scaled=`False` · pinned_cpu_producer=`3`

| Benchmark | p50 | p99 | p99.9 | min | max | samples | extra |
|---|---:|---:|---:|---:|---:|---:|---|
| `BM_Spsc_PushPopRoundtrip` | 11.6 ns | 18.8 ns | 27.5 ns | 8.7 ns | 1.49 ms | 83,650,402 |  |
| `BM_Spsc_TryPushLoaded` | 18.8 ns | 86.8 ns | 123.0 ns | 8.7 ns | 805.46 µs | 49,544,177 | dropped=7.49e+05 |

### `order_book_bench`

host=`LAPTOP-AEPE3M3P` · cpu=`1382 MHz × 8` · cache-scaled=`False` · pinned_cpu=`3`

| Benchmark | p50 | p99 | p99.9 | min | max | samples | extra |
|---|---:|---:|---:|---:|---:|---:|---|
| `BM_OB_AddLimit_Resting` | 23.1 ns | 149.0 ns | 512.2 ns | 11.6 ns | 411.11 µs | 9,695,525 |  |
| `BM_OB_Cancel_Mid` | 21.7 ns | 47.7 ns | 157.7 ns | 13.0 ns | 171.95 µs | 10,361,326 |  |
| `BM_OB_Cross_Sweep` | 81.0 ns | 173.6 ns | 358.8 ns | 56.4 ns | 4.02 ms | 8,262,913 | fills_per_iter=8 |

### `fused_hotpath_bench`

host=`LAPTOP-AEPE3M3P` · cpu=`1382 MHz × 8` · cache-scaled=`False` · pinned_cpu=`3`

| Benchmark | p50 | p99 | p99.9 | min | max | samples | extra |
|---|---:|---:|---:|---:|---:|---:|---|
| `BM_Fused_ArrivalToObs` | 52.1 ns | 198.2 ns | 327.0 ns | 27.5 ns | 364.09 µs | 19,079,072 |  |
| `BM_Fused_TickToDecision` | 16.57 µs | 40.51 µs | 112.59 µs | 13.37 µs | 22.08 ms | 160,268 | accepted_per_iter=1, ring_dropped=0 |

---

## Before / after: the `std::map` → direct-indexed array price ladder

The 2026-07-04 report was taken with the original `std::map<Price,
LevelFIFO>` ladder (and, unknown at the time, a latent back-shift-deletion
bug in the cancel path's id index — found by the fused benchmark, fixed in
`59ed42d`). This report is the direct-indexed array ladder (`13ce3d0`):
one contiguous `vector<LevelFIFO>` per side, subtract-and-index level
lookup, incrementally-maintained best bid/ask (O(1) reads).

| Benchmark | 07-04 map (p50 / p99 / p99.9) | 07-07 array (p50 / p99 / p99.9) | p50 delta |
|---|---:|---:|---:|
| `BM_OB_AddLimit_Resting` | 26.0 / 167.8 / 528.1 ns | 23.1 / 149.0 / 512.2 ns | −11% |
| `BM_OB_Cancel_Mid` | 26.0 / 49.2 / 169.3 ns | 21.7 / 47.7 / 157.7 ns | −17% |
| `BM_OB_Cross_Sweep` (8 fills) | 186.6 / 425.3 / 813.1 ns | **81.0 / 173.6 / 358.8 ns** | **−57%** |

Controls for comparability: the slab benchmark (untouched code) reproduces
its 07-04 p50 exactly (10.1 ns), so the two sessions are comparable; the
07-04 run was unpinned, this one is pinned, which mainly helps the
cross-thread SPSC row (p50 31.8 → 18.8 ns). Honest reading of the deltas:

- **The matching path is where the map actually cost:** `Cross_Sweep`
  drops 2.3× at p50 and 2.4× at p99 — the sweep now walks a contiguous
  array with an incrementally-repaired touch instead of hopping map nodes
  (~10 ns/fill including the telemetry-free fill callback).
- **Single-op insert/cancel were never map-bound** on a prewarmed hot
  level (−11%/−17% at p50); their residual p99.9 (~0.5 µs) tracks WSL2
  host jitter, not data-structure cost.
- **The loaded path gains far more than the micros show.** Under the
  fused benchmark's sustained churn workload (live book, matching,
  telemetry, eviction), per-op cost fell 129 → 52 ns for AddLimit and
  165 → 11 ns for the best-bid/ask observation reads — the O(1) touch is
  what the RL bridge hits every step.

## The fused tick-to-decision path: the book is not the bottleneck

`BM_Fused_TickToDecision` = synthetic arrival → LOB update (+ matching +
telemetry emit) → 8-dim observation build → LibTorch SAC-actor forward.
Everything before the neural forward (`BM_Fused_ArrivalToObs`) costs
**52 ns at p50**; the full decision costs **16.6 µs** — i.e. **~99.7% of
tick-to-decision is the LibTorch JIT forward pass** (a 8→256→256→2 MLP on
one CPU core at ~1.4 GHz, dominated by interpreter/dispatch overhead on
tiny tensors, not FLOPs). Two-clock consequence (§0): the C++ matching
engine sustains its nanosecond budget; the RL decision runs at
microseconds and belongs on the parent-order cadence, exactly as the
architecture assumes. If inference latency ever mattered, the levers are
graph freezing / `torch::jit::optimize_for_inference`, a hand-rolled
GEMM path, or quantization — not book work.

## Cache & branch behaviour (hardware PMU, user-space counts)

Whole-process counts over each filtered 1 s benchmark run (includes the
Google-Benchmark harness, so read these as workload-level *rates*, not
per-op attributions):

| Benchmark | IPC | cache-misses / 1k instr | branch-miss rate |
|---|---:|---:|---:|
| `BM_OB_AddLimit_Resting` | 1.16 | 0.64 | 1.00% |
| `BM_OB_Cancel_Mid` | 1.18 | 0.46 | 1.15% |
| `BM_OB_Cross_Sweep` | 1.64 | 0.22 | 0.57% |
| `BM_Fused_ArrivalToObs` | 1.29 | 1.38 | 4.91% |
| `BM_Fused_TickToDecision` | 1.65 | 0.52 | 0.72% |
| `BM_Slab_AllocDealloc` | 1.38 | 2.88 | 1.40% |
| `BM_Spsc_TryPushLoaded` | **0.60** | 1.18 | 3.28% |

Notes: the matching sweep runs at IPC 1.64 with 0.22 cache-misses per
thousand instructions — the contiguous ladder keeps the sweep in-cache
and branch-predictable (0.57% miss rate). The loaded SPSC bench is the
outlier by design: IPC 0.60 with 292 M cache references is the
producer/consumer cache-line ping-pong made visible — the cost the
`alignas(64)` layout bounds but cannot eliminate while two cores share a
line stream. The fused arrival→obs loop's 4.9% branch-miss rate is the
PRNG-driven synthetic flow (side/price/qty decoded from random bits every
iteration), not the book.
