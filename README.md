# Optimal-Execution Platform

A low-latency C++20 matching/simulation engine with an RL optimal-execution
research stack on top. The engine is the product: nanosecond-budget
order-book operations, lock-free telemetry, seeded determinism, and a
benchmark-gated CI. The RL head is the research story — including the
honest negative result that rigorous accounting produced.

**Design rule (the two clocks):** the *internal clock* (matching, fills,
strategy eval) is deterministic, allocation-free, and benchmarked in
nanoseconds; the *external clock* (FIX edge, kdb+ tape, dashboards) is
minutes-to-hours and deliberately off the latency budget. The hot path
writes to exactly one lock-free SPSC ring and never blocks — if downstream
IO stalls, telemetry is dropped and counted, never the matching path.

```
 INTERNAL CLOCK — pinned, deterministic, benchmarked
 ┌────────────────────────────────────────────────────────┐
 │ synthetic/UDP order arrival → intrusive LevelFIFO book │
 │   → matching + queue-aware fills                       │
 │   → strategy eval (CRTP: TWAP / AC / VWAP / RL·LibTorch)│
 └──────────────────────┬─────────────────────────────────┘
                        │ one SPSC ring · drop-on-full · never blocks
                        ▼
 EXTERNAL CLOCK — relay thread: kdb+ writer · FIX gateway · Streamlit cockpit
```

## Latency (pinned run, WSL2 guest, 1.4 GHz laptop core — see caveats)

| Path | p50 | p99 | p99.9 |
|---|---:|---:|---:|
| Slab alloc/dealloc | 10.1 ns | 15.9 ns | 23.1 ns |
| SPSC push under load (cross-core) | 18.8 ns | 86.8 ns | 123.0 ns |
| Book: add resting limit | 23.1 ns | 149.0 ns | 512.2 ns |
| Book: cancel mid-FIFO | 21.7 ns | 47.7 ns | 157.7 ns |
| Book: 8-level aggressive sweep | 81.0 ns | 173.6 ns | 358.8 ns |
| **Fused: arrival → LOB → observation** | **52.1 ns** | 198.2 ns | 327.0 ns |
| **Fused: tick → RL decision (LibTorch)** | **16.6 µs** | 40.5 µs | 112.6 µs |

Hardware PMU counters (collected with a minimal `perf_event_open(2)`
wrapper — no `perf` binary, no root): the matching sweep runs at
**IPC 1.64 with 0.22 cache-misses per 1k instructions** and a 0.57%
branch-miss rate; the loaded SPSC ring's IPC 0.60 is the producer/consumer
cache-line ping-pong made visible. Full tables, method, and honest
environment caveats: [docs/latency_report.md](docs/latency_report.md).

The fused rows are the point: **the C++ book costs ~52 ns; the neural
forward costs ~16.6 µs — 99.7% of tick-to-decision is inference**, which
runs on the parent-order clock where microseconds are irrelevant. The
engine is not the bottleneck.

## Engineering highlights

- **`std::map` → direct-indexed array price ladder.** One contiguous
  `vector<LevelFIFO>` per side over a fixed price band; level lookup is a
  subtract-and-index and the touch is maintained incrementally, so
  `best_bid()/best_ask()` are O(1) reads. Cut the matching sweep 57% at
  p50 / 59% at p99, and the loaded-workload AddLimit from 129 → 52 ns.
- **A latent hash-table bug, found by a benchmark.** The order book's
  O(1) cancel index used Knuth back-shift deletion with a wrong
  movability test — displaced entries got stranded as unreachable
  orphans, and under sustained churn the table saturated until insert
  spun forever. The fused benchmark's sustained workload exposed it (a
  21-minute hang, bisected with per-component rdtsc accumulators); it is
  fixed and locked in with a 200k-round churn regression test that fails
  in seconds on the old code.
- **Fused tick-to-decision benchmark.** The headline path measured as one
  continuous operation — arrival → LOB update (+ matching + telemetry
  emit) → observation build → TorchScript SAC-actor forward — not a sum
  of micros.
- **PMU counters without root.** A ~90-line `perf_event_open` wrapper
  (fork, attach with `enable_on_exec` + `inherit`, exec, read at exit)
  provides `perf stat --all-user`-equivalent counters on a box where
  `perf` cannot be installed.
- **Drop-tolerant telemetry.** The hot path is allowed to outrun its
  observers: a bounded lock-free SPSC ring with an atomic drop counter —
  telemetry loss is acceptable, hot-path jitter is not.
- **Seeded determinism as a test, not a hope.** Identical seed ⇒
  byte-identical fill sequence, enforced by unit test — and verified
  end-to-end when the ladder swap reproduced the RL evaluation numbers
  to the decimal on all three seeds.
- **CI regression gate.** Every push rebuilds, runs the 70-test suite,
  re-runs the latency harness, and gates p99 against a committed baseline
  (`scripts/check_latency_regression.py`; advisory across host changes,
  strict on same-class hardware).

## The RL story (honest by construction)

Trained a SAC execution agent (SELL Q=1000 over 32 steps against a
churned synthetic book) that, after a reward-geometry fix — replacing a
linear terminal penalty cliff with an Almgren–Chriss-style quadratic —
**appeared to beat TWAP on 2 of 3 evaluation seeds.** Implementing
rigorous **mark-to-market accounting** (residual inventory priced at the
terminal mid instead of arrival) showed the win was an artifact: the
agent was resting an unfilled tail whose true opportunity cost the reward
never charged. Executing the pre-registered contingency — a dense
advantage-vs-running-TWAP reward with the benchmark tracked in a 9-D
state — produced a sharper failure: the agent learned to **bang its own
benchmark**, crossing early to crater the mid its ghost TWAP executed at,
then harvesting the spread between the legs. Both reward-hacks are
documented as negative results (`experiments.md`, `research_state.md`),
the RL head is frozen at the κ=1000 milestone, and the engineering effort
went where the measurements pointed: the C++ engine. The pipeline —
Gymnasium env over a zero-copy pybind11 bridge with the GIL released in
the C++ step, SB3 SAC training, TorchScript export, in-process LibTorch
inference — is complete and benchmarked above.

## Build & run

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build            # 70/70
scripts/run_benchmarks.sh         # regenerates docs/latency_report.md
```

Optional components: `-DOEP_USE_TORCH=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch`
(RL inference + fused benchmark), `-DOEP_BUILD_PYBIND=ON` (the `oep_env`
training bridge), `-DOEP_USE_KDB=ON` (live kdb+ ticker-plant logger).
`python/cockpit/app.py` is a Streamlit desk-style cockpit that runs
SAC-vs-TWAP head-to-head with arrival-mid and mark-to-market IS tiles.

## Layout

```
cpp/core/       slab allocator · SPSC ring · order book · TSC clock
cpp/exec/       CRTP strategies: TWAP, Almgren-Chriss, VWAP, RL (LibTorch)
cpp/bindings/   pybind11 SimEnv (flat float obs, GIL-released step)
cpp/benchmarks/ Google Benchmark suite incl. fused tick-to-decision
cpp/tests/      70 GoogleTests: invariants, determinism, zero-alloc, churn
python/         Gymnasium env · SB3 SAC training · TorchScript export · cockpit
q/              kdb+ schema + TCA replay (sequence gaps, landing-lag drift)
scripts/        benchmark runner · latency-report renderer · p99 CI gate
```

## Measurement honesty

Numbers above were taken pinned but on a WSL2 guest without CPU-governor
control; millisecond-class maxima are host-scheduling artifacts and p99.9
carries that caveat — a core-isolated bare-metal run is the final word.
Simulation results use disjoint train/eval seeds; the sim's known
unrealisms (no adversarial latency arbitrage, simplified cancel dynamics,
single venue) make all RL results upper bounds. No dollar-P&L claims
anywhere: the deliverable is measured friction against named benchmarks,
and the measurements are committed alongside the code.
