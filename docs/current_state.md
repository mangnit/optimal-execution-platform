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

**Pending:**
- `cpp/core/spsc_ring.hpp` — bounded, overflow-tolerant, drop-counter.
- `cpp/core/order_book.{hpp,cpp}` — intrusive `LevelFIFO`, O(1) cancel.
- Benchmark harness (Google Benchmark, HdrHistogram) + `latency_report.md`.
- CI (`.github/workflows/ci.yml`) with the benchmark-regression gate.

**Bugs/Issues:** None.

**Next Tasks:**
1. Implement the SPSC ring (single hot-path writer → relay), with the
   overflow-tolerant drop counter called out in `docs/architecture.md` §2.
2. Wire the slab into the eventual `LevelFIFO` order book.
3. Stand up the Google Benchmark harness so P1's gate — p50/p99/p99.9
   histogram + `perf` counters — has somewhere to publish.
