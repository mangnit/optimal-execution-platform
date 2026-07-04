# Latency Report — Internal Clock

Generated 2026-07-04 15:23:00 UTC from `scripts/run_benchmarks.sh`.

Every benchmark below runs on the **internal clock** (docs/architecture.md §7):
the matching path, its allocator, and the SPSC ring that bridges it to the
relay. FIX, kdb writes, and dashboard IO live on the external clock and are
deliberately not measured here.

Percentiles are computed from per-iteration `rdtscp` samples, converted to
nanoseconds via a boot-time TSC-frequency calibration. `samples` is the
number of per-op observations backing each row; p99.9 is only trustworthy
when that count is well above a thousand.

The samples are wall-clock timings on whatever CPU the harness landed on;
without core isolation and CPU-frequency pinning, the tail includes
scheduling jitter. Treat the numbers as a baseline for regression tracking,
not as a marketing claim (per §7's "no pre-committed numbers" rule).

### `slab_allocator_bench`

host=`LAPTOP-AEPE3M3P` · cpu=`1382 MHz × 8` · cache-scaled=`False`

| Benchmark | p50 | p99 | p99.9 | min | max | samples | extra |
|---|---:|---:|---:|---:|---:|---:|---|
| `BM_Slab_AllocDealloc` | 10.1 ns | 17.4 ns | 23.1 ns | 7.2 ns | 762.03 µs | 106,213,249 |  |
| `BM_Slab_DrainRefill` | 1.4 ns | 2.2 ns | 8.0 ns | 0.7 ns | 543.3 ns | 886,091 |  |

### `spsc_ring_bench`

host=`LAPTOP-AEPE3M3P` · cpu=`1382 MHz × 8` · cache-scaled=`False`

| Benchmark | p50 | p99 | p99.9 | min | max | samples | extra |
|---|---:|---:|---:|---:|---:|---:|---|
| `BM_Spsc_PushPopRoundtrip` | 11.6 ns | 17.4 ns | 26.0 ns | 8.7 ns | 472.67 µs | 94,146,058 |  |
| `BM_Spsc_TryPushLoaded` | 31.8 ns | 85.4 ns | 118.6 ns | 8.7 ns | 663.50 µs | 48,450,108 | dropped=5.94e+05 |

### `order_book_bench`

host=`LAPTOP-AEPE3M3P` · cpu=`1382 MHz × 8` · cache-scaled=`False`

| Benchmark | p50 | p99 | p99.9 | min | max | samples | extra |
|---|---:|---:|---:|---:|---:|---:|---|
| `BM_OB_AddLimit_Resting` | 26.0 ns | 167.8 ns | 528.1 ns | 17.4 ns | 283.06 µs | 7,917,803 |  |
| `BM_OB_Cancel_Mid` | 26.0 ns | 49.2 ns | 169.3 ns | 15.9 ns | 345.57 µs | 7,273,719 |  |
| `BM_OB_Cross_Sweep` | 186.6 ns | 425.3 ns | 813.1 ns | 150.5 ns | 551.64 µs | 5,124,646 | fills_per_iter=8 |
