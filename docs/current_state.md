# Current State

**Phase:** P3 – Sim runner / live loop — **COMPLETE (100%)**.
Production TSC clock plumbed into `TelemetryPublisher::ClockFn`, the
`sim_runner` binary drives a deterministic seeded workload end-to-end
through `OrderBook → TelemetryPublisher → SpscRing → EventRelay →
KdbLogger`, and a first live run just closed **100 000 operations in
17 ms with zero SPSC drops** against the in-memory sink. Ready to
open P4 (TCA replay + core-isolated tail measurements).

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

**Bugs/Issues:** None.

**Next Phase — P4 (TCA replay + core-isolated tail measurements).**
1. Re-take the latency baseline on a core-isolated Linux host with
   `OEP_BENCH_PIN_CPU` + `OEP_BENCH_PIN_CPU_CONSUMER` set and commit
   the resulting `docs/latency_report.md` alongside the pinning notes.
2. Stand up the TCA replay path: read the `trade` table off the
   `q schema.q` ticker plant, reconstruct fills against arrival mid,
   and compute IS with the docs/architecture.md sign convention (parent SELL of
   Q over [0, T]).
3. Wire the live loop's FIX gateway onto the external clock the relay
   + KDB+ logger already occupy — no gateway I/O may leak onto the
   hot path.
