// Lock-free Single-Producer Single-Consumer ring for the internal→external
// clock bridge (docs/architecture.md §2).
//
// Invariants the hot path relies on:
//   * TryPush never blocks. If the consumer is behind and the ring is full,
//     the new write is dropped and `dropped_messages_` ticks up. Telemetry
//     loss is acceptable; hot-path jitter is not.
//   * No mutexes. Two monotonically increasing 64-bit indices (write, read)
//     synchronise via std::memory_order_acquire / release. The head atomic
//     of each side is padded to its own cache line so the producer's write
//     of write_idx_ and the consumer's write of read_idx_ never contend
//     over a shared line (false-sharing guard, alignas(64)).
//   * Capacity is a power of two so the modulo becomes a bitmask, and the
//     unsigned index difference (w - r) correctly reports occupancy even
//     after the counters wrap at 2^64.
//
// Sequence integrity under drops: the producer never advances write_idx_
// on a drop, so the slot at (write_idx_ & mask) is untouched. The next
// successful push overwrites that slot and the consumer's read sequence is
// unaffected — drops punch holes in what the *observer* sees, never in the
// FIFO order of what it does see.

#ifndef OEP_CORE_SPSC_RING_HPP_
#define OEP_CORE_SPSC_RING_HPP_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace oep::core {

inline constexpr std::size_t kSpscCacheLine = 64;

template <typename T, std::size_t Capacity>
class SpscRing {
 public:
  static_assert(Capacity > 0, "Capacity must be positive");
  static_assert((Capacity & (Capacity - 1)) == 0,
                "Capacity must be a power of two");
  static_assert(std::is_trivially_copyable_v<T>,
                "T must be trivially copyable for hot-path use");
  static_assert(std::is_trivially_destructible_v<T>,
                "T must be trivially destructible");
  static_assert(std::atomic<std::size_t>::is_always_lock_free,
                "std::atomic<size_t> must be lock-free on this platform");
  static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
                "std::atomic<uint64_t> must be lock-free on this platform");

  using value_type = T;
  static constexpr std::size_t kCapacity = Capacity;
  static constexpr std::size_t kMask = Capacity - 1;

  SpscRing() noexcept
      : write_idx_{0}, read_idx_{0}, dropped_messages_{0} {}

  SpscRing(const SpscRing&) = delete;
  SpscRing& operator=(const SpscRing&) = delete;
  SpscRing(SpscRing&&) = delete;
  SpscRing& operator=(SpscRing&&) = delete;

  // Producer side. Never blocks. Returns true on success; on full, increments
  // dropped_messages_ and returns false. Only one thread may call this.
  [[nodiscard]] bool TryPush(const T& value) noexcept {
    // relaxed: only the producer writes write_idx_, so no cross-thread sync
    // is needed to read our own last position.
    const std::size_t w = write_idx_.load(std::memory_order_relaxed);
    // acquire: pair with the consumer's release-store of read_idx_. This
    // ensures the slot at (r-1) & mask is fully drained before we consider
    // reusing it below.
    const std::size_t r = read_idx_.load(std::memory_order_acquire);
    if (w - r >= Capacity) {
      dropped_messages_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    storage_[w & kMask] = value;
    // release: publish both the slot write above and the new head to the
    // consumer's acquire-load of write_idx_.
    write_idx_.store(w + 1, std::memory_order_release);
    return true;
  }

  // Consumer side. Returns true and drains one element into `out`, or false
  // if empty. Only one thread may call this.
  [[nodiscard]] bool TryPop(T& out) noexcept {
    // relaxed: only the consumer writes read_idx_.
    const std::size_t r = read_idx_.load(std::memory_order_relaxed);
    // acquire: pair with the producer's release-store of write_idx_. This
    // ensures the slot write is visible before we read it below.
    const std::size_t w = write_idx_.load(std::memory_order_acquire);
    if (r == w) {
      return false;
    }
    out = storage_[r & kMask];
    // release: signal to the producer's acquire-load of read_idx_ that the
    // slot is free for reuse.
    read_idx_.store(r + 1, std::memory_order_release);
    return true;
  }

  // Observer helpers. Racy under live producer/consumer; safe to call from
  // either side or from a third thread that only reads.
  [[nodiscard]] std::uint64_t dropped_messages() const noexcept {
    return dropped_messages_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::size_t size_approx() const noexcept {
    const std::size_t w = write_idx_.load(std::memory_order_acquire);
    const std::size_t r = read_idx_.load(std::memory_order_acquire);
    return w - r;
  }

  [[nodiscard]] bool empty_approx() const noexcept {
    return size_approx() == 0;
  }

  [[nodiscard]] static constexpr std::size_t capacity() noexcept {
    return Capacity;
  }

  // Layout accessors — for tests only. offsetof inside the class works even
  // when the members are private.
  static constexpr std::size_t write_index_offset_for_testing() noexcept {
    return offsetof(SpscRing, write_idx_);
  }
  static constexpr std::size_t read_index_offset_for_testing() noexcept {
    return offsetof(SpscRing, read_idx_);
  }
  static constexpr std::size_t dropped_offset_for_testing() noexcept {
    return offsetof(SpscRing, dropped_messages_);
  }

 private:
  // Each hot atomic lives on its own cache line. The producer touches
  // write_idx_ every push; the consumer touches read_idx_ every pop. Sharing
  // a line would MESI-thrash the pair on every operation.
  alignas(kSpscCacheLine) std::atomic<std::size_t> write_idx_;
  alignas(kSpscCacheLine) std::atomic<std::size_t> read_idx_;
  alignas(kSpscCacheLine) std::atomic<std::uint64_t> dropped_messages_;
  // Payload storage. Individual slots may share a line among themselves; that
  // is unavoidable for small T and is not a false-sharing hazard here because
  // the producer and consumer only ever contend at the same slot when the
  // ring is empty, which is precisely when there is no throughput to protect.
  // Zero-initialized on construction (external clock, one-time cost). Two
  // reasons: (a) pre-faults every page of the ring so the producer never
  // takes a first-touch page fault on the hot path, and (b) newer GCC's
  // -Wmaybe-uninitialized cannot prove TryPop's empty-check guards the
  // storage read when fully inlined, and -Werror turns that false positive
  // into a build break.
  alignas(kSpscCacheLine) T storage_[Capacity] = {};
};

}  // namespace oep::core

#endif  // OEP_CORE_SPSC_RING_HPP_
