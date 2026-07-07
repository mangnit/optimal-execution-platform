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

## Project structure

Where to look first as a reviewer, and why each piece exists:

```
cpp/
├── core/                    ← THE hot path. Read this first.
│   ├── slab_allocator.hpp     O(1) cache-line-aligned pool; zero hot-path new/delete
│   ├── order_book.hpp         intrusive LevelFIFO book on a direct-indexed ARRAY
│   │                          price ladder (O(1) level lookup, O(1) best bid/ask,
│   │                          O(1) cancel via back-shift-deletion hash index —
│   │                          the churn bug fix + regression test live here)
│   ├── spsc_ring.hpp          lock-free bounded SPSC, drop-on-full + drop counter:
│   │                          the hot path may outrun observers, never block on them
│   └── tsc_clock.hpp          rdtscp invariant-TSC clock, one-shot calibration
├── exec/                    ← strategy layer, CRTP (no virtual dispatch on hot path)
│   ├── strategy.hpp           the CRTP vocabulary all strategies compile through
│   ├── almgren_chriss.*       sinh urgency schedule, precomputed at construction
│   ├── twap.* / vwap.*        analytic baselines
│   ├── rl_policy.*            LibTorch TorchScript inference: preallocated input
│   │                          tensor + reused interpreter stack, alloc-free steps
│   └── telemetry_publisher.hpp book ops → TelemetryEvents → SPSC ring, inlined
├── bindings/py_module.cpp   ← pybind11 SimEnv: flat float32 obs (zero-copy view),
│                              GIL RELEASED across the entire C++ engine step —
│                              proven by 4 SubprocVecEnv workers scaling in parallel
├── main/sim_runner.cpp      ← deterministic seeded synthetic-flow driver (the
│                              same xorshift flow recipe the RL env uses); wires
│                              book → publisher → ring → relay → kdb+ end-to-end
├── external/                ← EXTERNAL clock: EventRelay consumer thread,
│                              in-memory + live kdb+ IPC loggers
├── benchmarks/              ← Google Benchmark + rdtscp percentile sampler;
│                              fused_hotpath_bench.cpp is the headline
│                              arrival → LOB → obs → LibTorch-decision path
│                              (this benchmark found the hash-table bug)
└── tests/                   ← 70 GoogleTests: matching invariants, seeded
                               determinism (byte-identical fills), zero-alloc
                               guards via intercepted operator new, id-index
                               sustained-churn regression, relay/integration

python/
├── env/execution_env.py     ← Gymnasium wrapper over the C++ SimEnv (no per-step
│                              Python-side allocation)
├── train/                   ← SB3 SAC training (SubprocVecEnv + VecNormalize),
│                              TorchScript actor export for C++ inference
└── cockpit/app.py           ← Streamlit desk cockpit: SAC vs TWAP head-to-head,
                               arrival-mid AND mark-to-market IS tiles

scripts/
├── run_benchmarks.sh          runs the suite, renders docs/latency_report.md
├── check_latency_regression.py  CI p99 gate vs committed baseline (15% + 15ns)
└── eval_sweep.py              3-seed RL eval incl. the MTM IS accounting that
                               exposed the reward-hacks

benchmarks/                  ← committed benchmark JSONs; ci_baseline/ is the
                               regression-gate reference
q/                           ← kdb+ ticker-plant schema + TCA replay (sequence
                               gaps, landing-lag drift between the two clocks)
docs/                        ← architecture.md (the blueprint) ·
                               latency_report.md (pinned numbers + PMU counters) ·
                               current_state.md (running engineering log)
.github/workflows/ci.yml     ← build → ctest → benchmarks → p99 gate → artifacts
```

Research trail: `experiments.md` (per-run log) and `research_state.md`
(durable state) document the full RL trajectory, including both
reward-hacks and the decision record. Synthetic background flow is the
seeded deterministic recipe in `sim_runner.cpp` / `py_module.cpp`; the
calibrated marked-Hawkes generator from the blueprint (§5.3) is future
work, not shipped code.

## Measurement honesty

Numbers above were taken pinned but on a WSL2 guest without CPU-governor
control; millisecond-class maxima are host-scheduling artifacts and p99.9
carries that caveat — a core-isolated bare-metal run is the final word.
Simulation results use disjoint train/eval seeds; the sim's known
unrealisms (no adversarial latency arbitrage, simplified cancel dynamics,
single venue) make all RL results upper bounds. No dollar-P&L claims
anywhere: the deliverable is measured friction against named benchmarks,
and the measurements are committed alongside the code.
