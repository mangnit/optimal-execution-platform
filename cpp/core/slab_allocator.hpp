// Fixed-size, cache-aligned block allocator for the internal-clock hot path.
//
// Design: on construction, one aligned buffer is reserved for `capacity` slots.
// Every slot is `kCacheLine`-aligned and sized up to a cache line, so no two
// slots ever share a line (false-sharing guard). Free slots are threaded onto
// an intrusive singly linked list rooted at `free_head_`; the link uses the
// slot's own storage, so the free-list carries zero heap overhead.
//
// Allocate = pop head. Deallocate = push head. Both are O(1) and touch no
// global allocator after construction — the invariant the hot path depends on.

#ifndef OEP_CORE_SLAB_ALLOCATOR_HPP_
#define OEP_CORE_SLAB_ALLOCATOR_HPP_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

namespace oep::core {

inline constexpr std::size_t kCacheLine = 64;

template <typename T>
class SlabAllocator {
 public:
  using value_type = T;

  static_assert(alignof(T) <= kCacheLine,
                "T over-aligned beyond a cache line");

  // Round the slot up to a full cache line so distinct slots never share one.
  static constexpr std::size_t kSlotSize =
      (sizeof(T) + kCacheLine - 1) & ~(kCacheLine - 1);
  static_assert(kSlotSize >= sizeof(void*),
                "slot cannot hold intrusive free-list link");

  explicit SlabAllocator(std::size_t capacity)
      : capacity_(capacity),
        storage_(capacity == 0
                     ? nullptr
                     : static_cast<std::byte*>(::operator new[](
                           capacity * kSlotSize,
                           std::align_val_t{kCacheLine}))),
        free_head_(nullptr),
        in_use_(0) {
    // Thread every slot onto the free list. Highest-index slot becomes head so
    // Allocate() hands them out in ascending address order — nice for tests
    // and for prefetchers, and costs nothing at construction time.
    for (std::size_t i = 0; i < capacity_; ++i) {
      Node* node = SlotAt(i);
      node->next = free_head_;
      free_head_ = node;
    }
  }

  ~SlabAllocator() {
    if (storage_ != nullptr) {
      ::operator delete[](storage_, std::align_val_t{kCacheLine});
    }
  }

  SlabAllocator(const SlabAllocator&) = delete;
  SlabAllocator& operator=(const SlabAllocator&) = delete;
  SlabAllocator(SlabAllocator&&) = delete;
  SlabAllocator& operator=(SlabAllocator&&) = delete;

  // O(1). Returns nullptr on exhaustion — the hot path decides how to react;
  // dropping is preferable to blocking or calling new.
  [[nodiscard]] T* Allocate() noexcept {
    if (free_head_ == nullptr) {
      return nullptr;
    }
    Node* node = free_head_;
    free_head_ = node->next;
    ++in_use_;
    // Slot memory is now live as a T. std::launder respects the object-model
    // transition from "storage" to "T".
    return std::launder(reinterpret_cast<T*>(node));
  }

  // O(1). Caller must have already destroyed *p (this is a raw allocator, not
  // a container). Pointer must have come from Allocate() on this instance.
  void Deallocate(T* p) noexcept {
    assert(p != nullptr);
    assert(Owns(p));
    assert(in_use_ > 0);
    Node* node = reinterpret_cast<Node*>(p);
    node->next = free_head_;
    free_head_ = node;
    --in_use_;
  }

  // Convenience: allocate + construct in place. Off-hot-path helper.
  template <typename... Args>
  [[nodiscard]] T* New(Args&&... args) {
    T* raw = Allocate();
    if (raw == nullptr) {
      return nullptr;
    }
    return ::new (static_cast<void*>(raw)) T(std::forward<Args>(args)...);
  }

  void Delete(T* p) noexcept(std::is_nothrow_destructible_v<T>) {
    if (p == nullptr) {
      return;
    }
    p->~T();
    Deallocate(p);
  }

  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::size_t in_use() const noexcept { return in_use_; }
  [[nodiscard]] std::size_t available() const noexcept {
    return capacity_ - in_use_;
  }

  [[nodiscard]] bool Owns(const T* p) const noexcept {
    if (p == nullptr || storage_ == nullptr) {
      return false;
    }
    const auto* raw = reinterpret_cast<const std::byte*>(p);
    const auto* base = storage_;
    const auto* end = storage_ + capacity_ * kSlotSize;
    if (raw < base || raw >= end) {
      return false;
    }
    return (static_cast<std::size_t>(raw - base) % kSlotSize) == 0;
  }

 private:
  struct Node {
    Node* next;
  };

  Node* SlotAt(std::size_t i) noexcept {
    return reinterpret_cast<Node*>(storage_ + i * kSlotSize);
  }

  std::size_t capacity_;
  std::byte* storage_;
  Node* free_head_;
  std::size_t in_use_;
};

}  // namespace oep::core

#endif  // OEP_CORE_SLAB_ALLOCATOR_HPP_
