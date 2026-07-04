# Current State

**Phase:** P1 – C++ core (in progress)
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

**Pending:**
- Benchmark harness (Google Benchmark, HdrHistogram) + `latency_report.md`.
- CI (`.github/workflows/ci.yml`) with the benchmark-regression gate.
- Wire order-book state changes through the SPSC ring to the relay side.

**Bugs/Issues:** None.

**Next Tasks:**
1. Stand up the Google Benchmark harness so P1's gate — p50/p99/p99.9
   histogram + `perf` counters — has somewhere to publish.
2. Route order-book fills/acks through the SPSC ring to the external
   (relay) clock.
3. Add the CI workflow with the benchmark-regression gate.
