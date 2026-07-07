# Current State

**C++ systems milestones — hash-table bug fix, array price ladder, fused
benchmark (2026-07-07):** the project's primary metric is now **C++ engine
latency** (the RL workstream is frozen — see below). Three coupled changes
landed on `feature/sac-quadratic-terminal-k1000`:

- **Correctness: id-index back-shift deletion bug** (`59ed42d`,
  [cpp/core/order_book.hpp](cpp/core/order_book.hpp)). The Knuth 6.4-R
  erase in `LookupAndEraseId` stopped at the first entry sitting on its
  home slot and moved entries whose home lay inside `(hole, next]` —
  both defects strand displaced entries behind empty slots: orphaned,
  unreachable by lookup, never erased. Under sustained insert/erase churn
  orphans accumulate until the table saturates and `InsertId` (which
  needs an empty slot to terminate) spins forever. Exposed by the fused
  hot-path benchmark's sustained workload (a 21-minute 100%-CPU hang,
  bisected via per-component rdtsc accumulators); invisible to all prior
  tests/workloads. Fixed with the canonical movability test
  (`(next−home) mod cap ≥ (next−hole) mod cap`, skip-and-continue) and
  regression-guarded by `OrderBookTest.IdIndexSurvivesSustainedChurn`
  (200k churn rounds — fails in seconds on the old code).
- **Performance: direct-indexed array price ladder** (`13ce3d0`). The
  `std::map<Price, LevelFIFO>` ladder is replaced by one contiguous
  `vector<LevelFIFO>` per side over a fixed band ([0, 16383] default,
  narrowable via ctor), preallocated at construction. Level lookup is a
  subtract-and-index; the touch is maintained incrementally so
  `has_bid()/best_bid()` are O(1) reads (previously map-node scans).
  Churn-workload deltas: AddLimit 129→52 ns, best-bid/ask obs write
  165→11 ns, `Cross_Sweep` p50 187→74 ns. Out-of-band residuals drop
  (same policy as slab exhaustion). **Seeded determinism verified
  byte-identical**: the rebuilt `oep_env` reproduces the committed
  κ=1000 TWAP numbers to the decimal on all three eval seeds.
- **Benchmark: fused tick-to-decision** (`baf78b9`,
  [cpp/benchmarks/fused_hotpath_bench.cpp](cpp/benchmarks/fused_hotpath_bench.cpp)).
  The §7 headline path measured as one continuous internal-clock op:
  synthetic arrival → LOB update (+ matching + telemetry emit) → 8-dim
  obs build → LibTorch `RlPolicy` forward. `BM_Fused_ArrivalToObs`
  isolates everything but the neural forward. Unpinned WSL2 numbers:
  arrival→obs p50 **52 ns**; full tick-to-decision p50 **16.8 µs** —
  ~99.7% of the fused path is the LibTorch forward, i.e. **the book is
  not the bottleneck, inference is** (and inference runs on the
  parent-order cadence, per the two-clock model). Gated on
  `OEP_USE_TORCH`, wired into `scripts/run_benchmarks.sh`.

**Test suite: ctest 70/70** (all prior invariants — price/time priority,
determinism replay, zero-alloc guards, integration — plus the new churn
test) under `-Werror`.

**RL workstream: FROZEN at the κ=1000 8-D milestone** (`75902d2` +
MTM eval `e57746c`). The 9-D dense TWAP-advantage contingency was executed
and rejected on evidence — the agent learned to bang its own benchmark
(commit `9c1e266` records the negative result; details in
`research_state.md` §1f / `experiments.md`). The RL narrative ships as
"caught two reward-hacks through rigorous mark-to-market accounting."

**RL execution agent — environment realism & reward research (2026-07-06):**
the SAC optimal-execution agent (P5.4) was filling only 0–3/1000 units. Root
cause was the **environment, not the hyperparameters**: no resting bid ever
formed and the child order poisoned the book, capping the fill ceiling at ~1%
for any policy. Fixed across 5 diverse trials (liquidity, reward structure,
kPhi, learning rate, obs/reward normalization) plus a background-liquidity
**churn** pass — the simulation is now a realistic Almgren–Chriss landscape
(hoard-and-dump is dead; TWAP is the optimal *simple* baseline). Open problem:
the trained agent executes actively but is **over-aggressive** and does **not
yet beat TWAP** after convergence — the terminal stress penalty dominates the
spread/impact signal, so the reward-optimal policy over-crosses. Next step is a
principled reward redesign (not more compute). **The live source of truth for
this workstream is `research_state.md` (durable state) and `experiments.md`
(per-run log) at the repo root — this file is NOT tracking the reward-design
iteration.** Commits: `35a317c` (5-trial fix), `b47e74e` (liquidity churn).

**Latest tuning pass — training stability (2026-07-05):** three changes
to stop the supervisor's early-stopping trigger from killing SAC runs
during transient dips.

- **SAC hyperparameters** (`python/train/train_sac.py`): pinned
  `learning_rate=1e-4` (down from SB3's default 3e-4) on the SAC
  constructor and kept `ent_coef="auto"` explicit so the entropy
  temperature is auto-tuned rather than left implicit. Smaller step
  size trades convergence speed for a smoother `ep_rew_mean` curve,
  which is what the plateau detector actually looks at.
- **Early-stopping patience** (`python/train/supervisor.py`):
  `_PLATEAU_PATIENCE` raised from 10 → 50 consecutive eval intervals
  without improvement. Gives the agent 5× the runway to climb out of
  the negative-reward spirals we were seeing on short smoke runs.
  Memory-leak detector and status heartbeat unchanged.
- **Inventory penalty** (`cpp/bindings/py_module.cpp`): `kPhi`
  cut 0.5f → 0.25f in `StepImpl` — the φ·(q/Q)² running cost
  is now half as steep, so the shortfall term dominates by a
  wider margin during exploration and the agent is less
  "stressed" about carrying inventory mid-episode. Requires a
  pybind rebuild (`cmake --build build --target oep_env`) before
  the change is picked up by the trainer; no other C++ TU
  depends on this constant.

**Phase:** P6.1 – Execution Cockpit (`python/cockpit/app.py`) — **COMPLETE**.
Streamlit-based dashboard drives the C++ engine through the Pybind11
bridge, runs SAC vs TWAP head-to-head from the sidebar, and renders
per-step fill-price / cum-avg traces against arrival mid with live
IS-bps KPI tiles.

- **Execution Cockpit** (`python/cockpit/app.py`,
  `python/cockpit/requirements.txt`): built the Streamlit-based
  cockpit that closes the Phase 6 §6.1 loop — sidebar drives
  `parent_qty` / `horizon_steps` / `arrival_mid` / `seed` and a policy
  selector (SAC when `models/sac_oep_baseline.zip` is present, else
  TWAP / Random), the "Run Execution" button plays two episodes
  (chosen policy + TWAP baseline) through `ExecutionEnv` at the same
  seed, and results are surfaced as (i) four KPI tiles — policy IS
  (bps), TWAP IS (bps), Δ vs TWAP, and filled/parent with the
  realised average fill — and (ii) a Plotly overlay of per-step fill
  price and cumulative-average price for both policies against a
  dashed arrival-mid reference. Sign convention pinned to docs/architecture.md:
  parent SELL of Q, IS as `(S₀ − avg_fill) / S₀` in bps with unfilled
  inventory marked at S₀ so the metric is comparable across episodes
  even when the child sweep doesn't empty the book. TWAP baseline
  uses `size_frac = 1 / (T − t)` at aggression 1 so the C++ engine
  (which interprets `size_frac` as a fraction of `remaining_`) sweeps
  a uniform absolute size across the horizon; end-to-end smoke run
  through the app confirmed the engine loads, both episodes complete,
  and the Plotly analytics render live. `stable_baselines3` import is
  behind a try/except so the app degrades to TWAP/Random when SB3 (or
  the `.zip` artifact) is absent — cockpit stays runnable from a
  fresh checkout that hasn't trained yet. `models/` stays
  `.gitignore`d — the cockpit reads the artifact if present, but the
  binary itself is not tracked.

**Prior phase snapshot (P5.5, complete):** Full `ctest` suite passes
**62/62**, `scripts/test_bridge.py` and `scripts/test_gym.py` smoke
tests are green, and `python3 python/train/export_torchscript.py`
emits a native `models/sac_policy.pt` ready for the C++ LibTorch
loader.

- **TorchScript export** (`python/train/export_torchscript.py`):
  extracted the SAC actor from `models/sac_oep_baseline.zip`, wrapped
  it in a thin `_DeterministicActor` `nn.Module` that always calls
  `actor(obs, deterministic=True)` — trace only sees the mean-action
  branch, so the SB3 stochastic sampling path (which is not
  script-safe) never enters the graph. Traced with a `(1, 8)` float32
  dummy obs matching `ExecutionEnv.observation_space`, then saved via
  `torch.jit.trace(...).save()` to `models/sac_policy.pt` (287.4 KiB,
  actor MLP only — critic / replay buffer / optimizer state from the
  `.zip` are deliberately dropped). Post-save reload with
  `torch.jit.load` and a forward pass confirmed a `(1, 2)` output
  shape, matching `ExecutionEnv.action_space = Box([0,1]^2, float32)`,
  so a broken export fails inside Python before the C++ side ever
  touches the file. CLI knobs (`--model-path`, `--out-path`) let the
  same script serve later sweeps without an edit. `models/` stays
  `.gitignore`d — the artifact is a build output, not source.

- **SAC training loop** (`python/train/train_sac.py`,
  `python/train/__init__.py`): wired Stable-Baselines3 SAC to
  `ExecutionEnv` behind a `SubprocVecEnv` of N=4 workers, closing out
  the P5 stack (§5.4). Each rank gets a distinct seed
  (`_DEFAULT_TRAIN_SEED + rank`) so the four subprocesses explore
  decorrelated trajectories rather than N copies of the same episode;
  a single-process `DummyVecEnv` on `_DEFAULT_EVAL_SEED` backs the
  `EvalCallback` so eval numbers stay reproducible across runs.
  `SubprocVecEnv` correctly spun up four parallel C++ environments
  without hanging — proving the `py::gil_scoped_release` in the P5.2
  bridge is flawless (a single missed release would have serialised
  all four workers behind the interpreter lock). `learn()` returned
  cleanly; the run dropped a `.zip` model artifact to
  `models/sac_oep_baseline.zip` and TensorBoard traces to `logs/` (both
  paths configurable via `--model-path` / `--log-dir`). CLI knobs
  (`--n-envs`, `--total-timesteps`, `--eval-freq`, `--n-eval-episodes`,
  `--seed`, `--eval-seed`) let the same script serve as both the P5.4
  smoke test and the entry point for a fuller sweep. `models/` and
  `logs/` are `.gitignore`d — training artifacts do not belong in the
  tree.

**Prior phase snapshot (P5.3, complete):** `scripts/test_gym.py` drives
`gymnasium.utils.env_checker.check_env` cleanly against `ExecutionEnv`.

- **Gymnasium wrapper** (`python/env/execution_env.py`,
  `python/env/__init__.py`, `scripts/test_gym.py`): wrapped the C++
  `oep_env.SimEnv` in a standard `gymnasium.Env` so Stable-Baselines3
  can drive the engine unchanged. `ExecutionEnv` declares
  `action_space = Box([0,1]^2, float32)` and
  `observation_space = Box([-inf,inf]^8, float32)` per §5.4; `reset`
  forwards the caller's seed (default `0xC0FFEEBABE`) into the C++
  PRNG and returns `(obs, {})`; `step` calls into the bridge and
  returns `(obs, reward, terminated=done(), truncated=False, {})`.
  Both methods return the zero-copy view exposed by
  `SimEnv.state()` — no per-step Python-side allocation, matching the
  docs/architecture.md "host-to-device" boundary rule. `close()` drops the
  SimEnv so its dtor stops the relay thread. `scripts/test_gym.py`
  runs `check_env(skip_render_check=True)` with warnings escalated to
  errors (only the informational "Box bound is ±infinity" note is
  filtered, since those bounds are required by the task) and reports
  **`test_gym: OK — ExecutionEnv passes gymnasium.check_env`**.

- **Pybind11 bridge** (`cpp/bindings/py_module.cpp`,
  `CMakeLists.txt` gated on `-DOEP_BUILD_PYBIND=ON`,
  `scripts/test_bridge.py`): built the `oep_env` shared module and
  implemented the zero-allocation `SimEnv` bridge — same
  `OrderBook → TelemetryPublisher → SpscRing → EventRelay →
  InMemoryKdbLogger` stack `sim_runner` uses, wrapped behind a flat
  float32 `step(action)` / `state()` interface. State buffer
  (`kStateDim=8`) is `alignas(64)` and wrapped once as a zero-copy
  `py::array_t<float>` view; the length-2 action is read into locals
  before the C++ body runs. **GIL discipline (docs/architecture.md P5+ rule):**
  `py::gil_scoped_release` guards the entire C++ internal-clock step
  — background flow, child order via `book_.AddLimit` with a fused
  on-fill lambda (telemetry push + reward accumulation, one book call,
  zero virtual dispatch) — reacquired by the guard's dtor before
  `WriteState()` touches the numpy buffer. Ring is heap-allocated once
  in the ctor (2.5 MiB `SpscRing<TelemetryEvent, 1<<16>`) so `step()`
  never allocates. `scripts/test_bridge.py` drives two policies
  (TWAP-passive and all-in-aggressive) through a full 32-step horizon
  and asserts `dropped_messages() == 0`, finite reward, and bounded
  episode length — **all smoke tests pass**.

**Prior phase snapshot (P5.1, complete):** Implemented the
Almgren–Chriss and VWAP analytical baselines as
  zero-allocation CRTP strategies
  ([cpp/exec/strategy.hpp](cpp/exec/strategy.hpp),
  [cpp/exec/almgren_chriss.{hpp,cpp}](cpp/exec/almgren_chriss.hpp),
  [cpp/exec/vwap.{hpp,cpp}](cpp/exec/vwap.hpp)). No virtual dispatch,
  no exceptions, no RTTI on the hot path; the sinh urgency curve and
  the cumulative-rounded VWAP schedule are pre-computed at construction
  (external clock) so `ChildQuantity` / `RemainingAfterStep` collapse
  to a single indexed vector read. κ = √(λσ²/η) is calculated once and
  stored; λ→0 collapses cleanly to the TWAP linear limit. Both
  schedules anchor Σ n_j ≡ Q by construction so integer rounding does
  not drift close-out. `Strategy::SubmitStep` forwards through a
  templated `Publisher::AddLimit` (matches `TelemetryPublisher`) so
  child orders reach the SPSC ring with zero heap traffic — proven
  by allocator-counter guards in the two new gtests
  ([cpp/tests/almgren_chriss_test.cpp](cpp/tests/almgren_chriss_test.cpp),
  [cpp/tests/vwap_test.cpp](cpp/tests/vwap_test.cpp), 14 cases total
  covering the κ formula, boundary anchors, monotonicity, λ→0 = TWAP,
  large-λ front-loading, a hand-computed N=4 schedule, cumulative-round
  VWAP, sub-unit profile close-out, hot-path zero-alloc, and the
  end-to-end `SubmitStep → TelemetryPublisher → SpscRing` Ack
  round-trip).
- **Fixed [cpp/tests/relay_test.cpp](cpp/tests/relay_test.cpp)** to
  match the Phase 4 threading contract: `EventRelay`'s consumer thread
  no longer calls `logger_.Flush()` (kdb+ 5.0 C-API is not thread-safe
  from a thread other than the one that opened the handle — see the
  P4 fix in [event_relay.hpp:112](cpp/external/event_relay.hpp#L112)),
  so the `DrainsBurstOfEventsInFifoOrder` test now calls
  `logger.Flush()` explicitly from the main thread after
  `relay.Stop()` returns before asserting `flush_count() >= 1`.
  Aligns the test with the new ownership rule without changing the
  KDB+ segfault-safe production path.

**Prior phase snapshot (P4, complete):** Live end-to-end run against
a stood-up `q schema.q -p 5010` plant delivered **94 907 events with
zero dropped frames** through the full `OrderBook →
TelemetryPublisher → SpscRing → EventRelay → IpcKdbLogger → kdb+ TP`
pipeline; `q/tca_analysis.q` confirmed strict `seq` monotonicity and
a landing-lag distribution with `max drift ≈ 56 ms` from the median
(dominated by WSL2 scheduling tail — the constant TSC↔wall epoch
offset drops out of the median subtraction by construction).

**KDB+ 5.0 compatibility fixes required to close the loop:**
- **Main-thread `Flush()` refactor** ([cpp/external/event_relay.hpp:112](cpp/external/event_relay.hpp#L112)):
  the relay's consumer thread no longer calls `logger_.Flush()` in
  its residual-drain pass. The vendored `c.o` client (KXVER=3) does
  lazy arena / mixed-list init on first `k()` touch and segfaults
  when that init happens on a `std::thread` other than the one that
  called `Connect()` under kdb+ 5.0. Contract is now: whoever owns
  the relay/logger pair calls `logger.Flush()` on the same thread
  that ran `Connect()` after `Stop()` returns. `InMemoryKdbLogger`
  `Flush()` is a no-op, so the in-memory integration tests are
  unaffected.
- **`type` / `med` reserved-keyword workarounds** ([q/schema.q](q/schema.q)):
  the original `([] type:...; ...)` table constructor and a naive
  `med lag` in the analysis script both trip kdb+ 5.0's stricter
  parser (`type` now shadows the built-in, `med` collides in some
  scopes). `schema.q` was rewritten as
  `trade:flip \`type\`seq\`...\`wall_ns!(\`symbol$(); ... )` and
  `.u.upd` was collapsed to a one-liner that broadcasts `.z.p` to a
  per-row `long` vector before insert — same on-tape shape as before,
  just no reserved-keyword collision on load.

**Prior phase snapshot (P3, complete):** Production TSC clock plumbed
into `TelemetryPublisher::ClockFn`; `sim_runner` drives a
deterministic seeded workload end-to-end through
`OrderBook → TelemetryPublisher → SpscRing → EventRelay → KdbLogger`;
first live run closed **100 000 operations in 17 ms with zero SPSC
drops** against the in-memory sink.

**Completed:**
- Workspace directory initialization.
- Top-level CMake + Ninja build (`-Wall -Wextra -Wpedantic -Werror`, C++20,
  release defaults `-O3 -march=native`).
- GoogleTest integrated via `FetchContent` (system libstdc++ compatible).
- **Slab allocator** (`cpp/core/slab_allocator.hpp`):
  - Fixed-size, cache-aligned (`kCacheLine=64`) block allocator with intrusive
    free-list. `Allocate`/`Deallocate` are O(1), `noexcept`, and touch no
    global allocator after construction. Slot size rounded up to a full cache
    line so distinct slots never share one.
  - 11 GoogleTests (`cpp/tests/slab_allocator_test.cpp`) pass under `ctest`.
    Coverage includes: per-slot 64-byte alignment across a fully drained pool,
    no-hot-path-heap-traffic (via intercepted aligned `new[]`/`delete[]`),
    construct/destruct balance across 128 cycles (leak prevention), O(1)
    scaling assertion across a 64× capacity range, exhaustion → `nullptr`,
    LIFO reuse, cross-pool `Owns` rejection, zero-capacity edge case.
- **SPSC ring** (`cpp/core/spsc_ring.hpp`):
  - Lock-free, bounded, drop-on-full ring bridging the internal (hot) clock
    to the external (relay) clock per `docs/architecture.md` §2. Power-of-two
    capacity, bitmask modulo, `std::atomic<size_t>` head/tail with strict
    acquire/release pairing; each hot atomic (`write_idx_`, `read_idx_`,
    `dropped_messages_`) is `alignas(64)` so the producer/consumer never
    MESI-thrash a shared line.
  - `TryPush` never blocks: on full, the write is discarded and
    `dropped_messages_` ticks up — telemetry loss is acceptable, hot-path
    jitter is not. `write_idx_` is not advanced on drop, so surviving
    elements retain strict FIFO order (drops punch holes in the observer's
    view, never in the ordering of what is delivered).
  - Compile-time guards: `T` must be trivially copyable & destructible;
    `std::atomic<size_t>` / `std::atomic<uint64_t>` must be always lock-free.
  - 14 GoogleTests (`cpp/tests/spsc_ring_test.cpp`) pass under `ctest`.
    Coverage: lock-free traits, per-atomic 64-byte line separation,
    whole-struct cache alignment, empty pop, single push/pop, order
    preservation across wraparound (32 rounds × capacity-8), exact capacity
    occupancy, drop counter exact under 10 000-attempt overflow, producer
    never stalls (100 000 back-to-back drops < 1 s), drops do not corrupt
    the FIFO of survivors, interleaved push/pop counter monotonicity, and
    two concurrent producer/consumer tests — a slow consumer that forces
    saturation (`produced == received + dropped`, strictly monotone
    delivery) and a retrying producer that must deliver every value in
    issue order.
- **Latency benchmark harness** (`cpp/benchmarks/`):
  - Google Benchmark integrated via `FetchContent` (tag v1.8.3), gated on
    `-DOEP_BUILD_BENCHMARKS=ON` (default on). One executable per core
    component keeps CI triage cheap and lets the eventual regression job
    parallelise.
  - `bench_util.hpp` supplies a shared per-iteration `LatencySampler`:
    each iteration is bracketed by `__rdtscp()`, deltas are recorded
    allocation-free into a pre-reserved vector, and on completion the p50,
    p99, and p99.9 (plus min/max/sample-count) are published as custom
    Google Benchmark counters. Cycles→ns comes from a one-off TSC
    calibration against `steady_clock` at first use, cached for the run.
  - Thread-pinning scaffold: `OEP_BENCH_PIN_CPU` (and
    `OEP_BENCH_PIN_CPU_CONSUMER` for the SPSC consumer thread) drive
    `pthread_setaffinity_np` before the harness starts. Non-fatal on
    failure so CI runners without `CAP_SYS_NICE` still produce a report.
  - Coverage matches the P1 task spec:
    - Slab: `AllocDealloc` (L1-hot round-trip) and `DrainRefill` (whole
      pool alloc/dealloc walk).
    - SPSC: `PushPopRoundtrip` (single-thread lower bound) and
      `TryPushLoaded` (real consumer thread draining on a second core;
      `dropped_messages` reported as a counter).
    - Order book: `AddLimit_Resting` on a prewarmed level,
      `Cancel_Mid` on a 64-order FIFO (rotating victim keeps depth
      stable), `Cross_Sweep` on an eight-level prewarmed ask ladder with
      one maker per level — a level rebuild between iterations is
      pushed out of the timed region via `PauseTiming`.
  - `scripts/run_benchmarks.sh` runs all three binaries with
    `--benchmark_format=json` and hands the JSON to
    `scripts/render_latency_report.py`, which emits `docs/latency_report.md`
    (per-binary tables of p50/p99/p99.9 in ns, plus min/max/sample count and
    any auxiliary counters).
  - First baseline captured in `docs/latency_report.md` (unpinned WSL2
    host — the tail includes scheduling jitter and should be re-taken on a
    core-isolated Linux box before it is treated as a regression baseline).

- **Order book** (`cpp/core/order_book.hpp`):
  - Intrusive `LevelFIFO` limit order book on the internal clock. Orders
    are POD nodes carved from the `SlabAllocator<Order>` — no `new`/
    `delete` on the hot path. Each price level is a doubly-linked FIFO
    (`prev`/`next` threaded through the `Order` itself), so any resting
    order can be unlinked in O(1) regardless of queue position.
  - **O(1) cancel-by-id** via an open-addressing hash table
    (`id_index_`) keyed by splitmix64-hashed `OrderId`. Sized to the
    next power of two ≥ 2×capacity for a ≤50% load factor; erase uses
    **Knuth 6.4-R back-shift deletion** to keep the table tombstone-free
    so lookup probe counts stay bounded across arbitrary insert/cancel
    churn.
  - Price ladder is `std::map<Price, LevelFIFO>` (`greater<>` for bids,
    `less<>` for asks) with a `PrewarmLevel()` hook that pre-inserts the
    levels the hot path will use, so resting inserts never trigger a
    `std::map` node allocation after warmup. Empty prewarmed levels are
    skipped during matching.
  - Matching is standard price/time priority, drop-on-full when the slab
    is exhausted (mirrors `SpscRing::TryPush` — never block). Fills are
    delivered synchronously through a templated `FillHandler` so a
    lambda or SPSC-push functor inlines without virtual dispatch.
  - 11 GoogleTests (`cpp/tests/order_book_test.cpp`) pass under `ctest`.
    Coverage: time priority within a level, price priority across
    levels, residual rests as passive, non-crossing does not match,
    cancel from the middle of a FIFO preserves head/tail/order-count
    topology, cancel of head and tail, cancel of an unknown id returns
    false, cancelled orders never fill, cancel releases the slab slot,
    no hot-path heap traffic during a 4 096-op mixed workload
    (add/cancel/match) via intercepted aligned `new[]`/`delete[]`, and
    seeded-determinism replay — the same op sequence produces a
    byte-identical fill stream on two independent books.

- **CI workflow** (`.github/workflows/ci.yml`):
  - GitHub Actions job on `push`/`pull_request` to `main`. Installs
    Ninja + CMake + GCC, configures Release with `-O3 -march=native` and
    `-DOEP_BUILD_BENCHMARKS=ON`, builds, runs the full `ctest` suite
    (`--output-on-failure`), then executes `scripts/run_benchmarks.sh`.
  - Uploads `docs/latency_report.md` and `benchmarks/*.json` as
    artifacts on every run (including failures) so future PRs can be
    diffed against a committed baseline for a p99 regression check.
  - Committed as `6e3d69a` — closes out P1.

- **KDB+ logger interface** (`cpp/external/kdb_logger.hpp`):
  - `TelemetryEvent` POD (fill / cancel / ack discriminator, sequence,
    internal-clock ns timestamp, order/counter ids, price, qty, side).
    `static_assert`s pin trivially copyable / destructible so it can
    ride the `SpscRing<TelemetryEvent, N>` unchanged. Factory
    constructors (`Fill`, `Cancel`, `Ack`) keep the producer call
    sites terse and keep the discriminator/field pair consistent.
  - `kCsvHeader` constant (`type,seq,ts_ns,order_id,counter_id,price,
    qty,side`) and a shared `FormatCsvRow()` free function so the
    in-memory test sink and the eventual `k.h` IPC sink emit the same
    tape schema — `q/schema.q` can pin against the same column order.
  - Abstract `KdbLogger` base with `Log(const TelemetryEvent&)` and
    `Flush()`. `InMemoryKdbLogger` supplies the CI/test implementation
    (mutex-guarded `std::vector<std::string>` of CSV rows, snapshot
    copy on read to avoid handing the relay thread's buffer out under
    a live producer). A real `IpcKdbLogger` slots in without touching
    the relay.

- **Order-book → SPSC wiring / TelemetryPublisher**
  (`cpp/exec/telemetry_publisher.hpp`):
  - Templated on `Ring` and `ClockFn` so nothing about the ring capacity
    or the timestamp source (TSC in production, monotone counter in
    tests) leaks into the header. Wraps `OrderBook::AddLimit` and
    `OrderBook::Cancel`; every hot-path outcome — one `Fill` event per
    maker/taker pair, one `Ack` iff a residual actually rests, one
    `Cancel` on successful cancel — is converted to a `TelemetryEvent`
    and pushed via `SpscRing::TryPush`. Producer-side monotone
    `sequence` is assigned at emit; `timestamp_ns` comes from the
    injected clock.
  - The `FillHandler` (already templated in P1) and a new templated
    `OrderBook::Cancel(id, on_cancel)` overload let the publisher
    observe `(id, side, price, residual qty)` *before* the slab
    reclaims the node — captured by-reference lambdas inline through
    the templated call sites, so the wired hot path has zero virtual
    dispatch and zero allocation. The legacy `Cancel(id)→bool` is kept
    as a thin default-callback wrapper so existing tests are unchanged.
  - Drop-tolerant: `TryPush` failures are counted by the ring but not
    retried on the hot path (§2 architecture rule — telemetry loss is
    acceptable, hot-path jitter is not).
  - 5 GoogleTests (`cpp/tests/system_integration_test.cpp`) pass under
    `ctest` (48/48 total suite green). Coverage:
    - End-to-end CSV tape verification through the full
      `OrderBook → TelemetryPublisher → SpscRing → EventRelay →
      InMemoryKdbLogger` pipeline (Ack + Fill + Ack-of-residual +
      Cancel land in the logger in exactly the expected schema and
      sequence order, zero drops).
    - Pure aggressor emits only `Fill`, no `Ack` (no residual to rest).
    - `Cancel` of an unknown id emits nothing and does not burn a
      sequence number.
    - **Zero-allocation guard for the wired hot path**: a 100 000-op
      mixed workload through the publisher + ring produces zero global
      `new`/`delete` deltas after warmup (relay deliberately not
      started — the logger's mutex + string formatting live on the
      external clock and are allowed to allocate).
    - 2 000-order concurrent producer / relay-consumer run: strict
      monotone sequence in the logged tape, zero SPSC drops.
  - The existing `OrderBook.NoHotPathHeapTraffic` test still passes
    unchanged, confirming the new `Cancel` overload did not regress
    the order-book's zero-alloc guarantee.

- **External-clock event relay** (`cpp/external/event_relay.hpp`):
  - Templated on the ring type; owns one dedicated consumer thread
    that polls `TryPop` and hands each event to the injected
    `KdbLogger&`. All I/O / allocation / dispatch stays on the
    external clock; the producer side of the ring never blocks on
    anything the relay does.
  - Idempotent `Start()` / `Stop()` gated by `std::atomic<bool>`
    flags with acquire/release ordering. `Stop()` sets
    `stop_requested_`, and the consumer runs one final drain pass
    *after* observing the flag so anything the producer landed
    between the last drain and its stop signal is still logged
    before `Flush()` and join. Destructor calls `Stop()`, so a
    stack-scoped relay is always joined.
  - Idle policy is a configurable `std::this_thread::sleep_for` on
    empty (default 50 µs) — keeps CI CPU sane without introducing a
    condition variable on the producer side. Tests drive the sleep
    to zero for spin-mode or to 100 ms to prove the residual-drain
    path.
  - Racy `processed()` counter (relaxed atomic) for test assertions
    and later §P4 TCA tape sanity checks.
  - 6 GoogleTests (`cpp/tests/relay_test.cpp`) pass under `ctest`
    (43/43 total suite green). Coverage: CSV row matches the pinned
    schema for fill / cancel / ack; burst of 500 events drains in
    strict FIFO order with `flush_count ≥ 1` and zero SPSC drops;
    double-`Start()` / double-`Stop()` idempotent with no hang;
    destructor stops cleanly without an explicit `Stop()` and still
    drains queued events; `Stop()` drains residual elements when the
    consumer is parked in its idle sleep; 20 000-event concurrent
    producer/consumer (retrying producer, spin-mode relay) delivers
    every event in strict monotone sequence order.

- **KDB+ ticker-plant schema** (`q/schema.q`):
  - `trade` table pinned column-for-column to `kCsvHeader` and to the
    mixed-list order that `IpcKdbLogger::SendBatchLocked` publishes:
    `type` (symbol), `seq` / `ts_ns` / `order_id` / `counter_id` /
    `price` / `qty` (long), `side` (symbol). `ts_ns` carries the
    internal (TSC) clock; the wall-clock stamp is added on landing
    via `.z.p` per §P4 TCA replay, so both clocks live on the tape.
  - Standard kdb+ tick `.u.upd:{[t;x] t insert x}` handler makes the
    script runnable standalone — a `q schema.q -p 5010` on the ticker
    plant host is enough to smoke-test the C++ IPC frame end-to-end
    without pulling in a full TP/RDB. The IpcKdbLogger already targets
    `.u.upd[`trade; ...]` so wiring is symmetric.

- **Production TSC clock** (`cpp/core/tsc_clock.hpp`):
  - Lifts the invariant-TSC primitives out of the bench harness so the
    internal clock can be reached without pulling in
    `<benchmark/benchmark.h>`. `ReadCycleCounter()` is a single
    `__rdtscp` on x86-64 (lightly serialising — retires prior ops
    before sampling), with a `steady_clock` fallback for non-x86 CI.
  - `NanosPerCycle()` runs a one-shot ~50 ms busy-spin against
    `steady_clock` and caches the ratio in a function-local static;
    subsequent calls are free. Busy-spin (not `sleep_for`) keeps the
    calibrating thread on-core so the scheduler can't migrate mid-window
    and skew the ratio.
  - `TscNanoClock` is a callable that satisfies
    `TelemetryPublisher::ClockFn`: captures a base TSC reading at
    construction so the emitted `ts_ns` values start near zero, keeping
    `cycles * ns_per_cycle` inside `double`'s 53-bit exact-integer
    window for the lifetime of any realistic session (~104 days).
    `operator()` is one `rdtscp` plus a scalar float multiply — zero
    allocation, zero syscalls on the hot path.
  - `bench_util.hpp` now re-exports `oep::bench::ReadCycleCounter` /
    `oep::bench::NanosPerCycle` as aliases of the core primitives, so
    every existing benchmark TU compiles unchanged.

- **Sim runner** (`cpp/main/sim_runner.cpp`, new `CMakeLists.txt`
  target):
  - Standalone executable that wires the full P1/P2 pipeline together
    behind the production TSC clock: `OrderBook + TelemetryPublisher`
    on the internal clock, `SpscRing<TelemetryEvent, 1<<16>` as the
    drop-tolerant bridge, `EventRelay` + `KdbLogger` on the external
    clock. Ring is heap-allocated once at startup (2.5 MiB frame is
    too large for stack), never on the hot path.
  - Workload is a seeded `xorshift64` mix of aggressive / passive
    limit orders and 1-in-4 cancels — the same recipe the P2 zero-
    allocation integration test uses, so the sim exercises exactly the
    code paths already verified allocate-free. A narrow price band
    (90..110) is prewarmed so resting inserts never trigger a
    `std::map` node allocation after startup. Bit-identical PRNG
    across libstdc++/libc++ satisfies the docs/architecture.md "identical seed
    ⇒ identical fill sequence" determinism rule.
  - Defensive slab-eviction guard cancels a resting order when
    `open_order_count()` approaches `kBookCapacity`, so a normal run
    never trips the `AddLimit(→false)` slab-exhaustion path.
  - CLI: `--host / --port / --ops / --seed / --help`. Build picks the
    sink at compile time — `-DOEP_USE_KDB=ON` links the real
    `IpcKdbLogger` (and a failed `Connect()` is warned-not-fatal so
    the relay still drains into the logger's per-column batches until
    an operator brings the ticker plant up); otherwise falls back to
    `InMemoryKdbLogger` so a `docker compose`-less checkout can still
    `sim_runner` end-to-end.
  - **First live run: 100 000 ops end-to-end in ~17 ms with zero SPSC
    drops** through the in-memory sink — the wired hot path meets the
    §2 architecture rule (producer never blocks on the relay) at
    real workload volume, on top of the deterministic seed.
  - Fixed a latent segfault in `IpcKdbLogger::Connect()` uncovered by
    the sim run: the vendored `k.h` client calls `strlen()` on the
    credentials pointer unconditionally, so a `nullptr` for the
    "no auth" case dereferenced inside `khpu`. Collapsed to a static
    empty-string sentinel — behavioural contract for KDB+ is
    "empty string == no auth", not "null pointer".

- **KDB+ build enabled** (`build/CMakeCache.txt`):
  - Reconfigured with `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DOEP_USE_KDB=ON`
    and rebuilt. `nm build/sim_runner` now lists the full
    `oep::external::IpcKdbLogger` symbol set (`Log`, `Flush`,
    `Connect`, `Disconnect`, `SendBatchLocked`, `ClearBatchLocked`),
    so a `--host / --port` invocation now hits the vendored `k.h`
    IPC client instead of the in-memory fallback. Warned-not-fatal
    connect path (P3 fix) keeps the relay draining into the
    per-column batches until the ticker plant is brought up.

- **Ticker-plant wall-clock stamp** (`q/schema.q`):
  - `trade` gained a `wall_ns` column (kdb+ `long`), stamped in
    `.u.upd` on landing from `.z.p` (local UTC nanoseconds since
    2000-01-01). `.u.upd` broadcasts the scalar wall stamp to a
    per-row vector matching the batched-list frame the
    `IpcKdbLogger::SendBatchLocked` emits, so a full flush lands as
    one insert with both clocks aligned. `ts_ns` still carries the
    TSC clock rebased near zero at `TscNanoClock` construction; the
    two are epoch-different by design, and the drift about the
    median (not the constant offset) is what the TCA script
    interprets.

- **TCA replay + clock-delta script** (`q/tca_analysis.q`):
  - Three load modes: `-host / -port` opens an IPC handle to a
    running TP and pulls `0!trade`; `-tape PATH` reads a
    `save`-serialised snapshot from disk; no args falls back to a
    top-level `trade` variable in the current session so the
    script can be `\l`-ed into a populated TP.
  - Sequence check: `1_ deltas seq` yields per-row diffs;
    `where diffs>1` counts SPSC-drop gaps, `where diffs<=0` counts
    reorders / duplicates. Reports first-seq / last-seq / delivered
    rows / expected span (`1 + last - first`) so an operator can
    read off both drop rate and any TP re-order at a glance.
    Non-zero exit code on failure so a CI step can gate on it.
  - Landing-lag distribution: `lag = wall_ns - ts_ns`, then
    `drift = |lag - median lag|` — the constant epoch offset drops
    out, leaving landing jitter. Quantiles come from an in-memory
    sort (`asc drift`) indexed at
    `floor (n-1)*[0.5, 0.9, 0.99, 0.999]`; also reports max drift
    and both clocks' elapsed span across the run so their rate
    ratio can be sanity-checked against the TSC calibration in
    `NanosPerCycle()`.

**Pending (deferred to later phases):**
- `perf` cache/branch-miss numbers to sit alongside the p50/p99/p99.9 table
  (harness ready, needs a core-isolated Linux run — CI runner is shared
  hardware and not suitable for the tail).
- HdrHistogram integration to replace the sort-in-memory percentile path
  when sample counts push past ~10^8.
- Numerical p99 regression threshold on top of the CI artifacts (JSON is
  already uploaded; a follow-up job can diff current vs baseline).
- Re-take the latency baseline on a core-isolated Linux box (with
  `OEP_BENCH_PIN_CPU` + `OEP_BENCH_PIN_CPU_CONSUMER`) and commit the
  resulting `docs/latency_report.md` alongside the pinning notes.

**Bugs/Issues:** None known. Live P4 run passed on the first
end-to-end drive after the `Flush()` and reserved-keyword fixes
above. WSL2 host tail (max 56 ms drift about the median) is
scheduling jitter, not pipeline jitter — the P3 in-memory sink
closed 100 k ops in 17 ms with zero drops on the same host, so
the tail is entirely `.z.p`-side scheduling and kernel scheduling
of the relay thread, not producer-side. Re-take on a
core-isolated Linux box (per the pending item below) is expected
to collapse the tail by >100×.

**Next steps — closing out P5 and moving into P6.**
1. C++ LibTorch loader that mmaps `models/sac_policy.pt` and drives
   `SimEnv` inference on the external clock — the traced graph's
   `(1, 8) → (1, 2)` contract is the interface to build against.
2. Extend `q/tca_analysis.q` to reconstruct fills against arrival
   mid and compute IS in bps with the docs/architecture.md sign convention
   (parent SELL of Q over [0, T]), so the trained SAC policy can be
   benchmarked against the P5.1 TWAP/VWAP/AC baselines head-to-head.
3. Wire the live loop's FIX gateway onto the external clock the
   relay + KDB+ logger already occupy — no gateway I/O may leak
   onto the hot path.
4. Re-take the latency baseline on a core-isolated Linux host with
   `OEP_BENCH_PIN_CPU` + `OEP_BENCH_PIN_CPU_CONSUMER` set and commit
   the resulting `docs/latency_report.md` alongside the pinning notes.
