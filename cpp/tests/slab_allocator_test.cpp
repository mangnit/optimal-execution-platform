// Tests for oep::core::SlabAllocator.
//
// Focus areas (per task P1_slab_allocator):
//   1. Every slot handed out is 64-byte cache-aligned — false-sharing guard.
//   2. Allocate() / Deallocate() are O(1) and touch no global allocator after
//      construction — verified via a scaling assertion and via intercepting
//      the aligned new/delete variants the slab uses.
//   3. No leaks: aligned new/delete counts match on shutdown.

#include "core/slab_allocator.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <random>
#include <unordered_set>
#include <vector>

// Intercept only the aligned new/delete variants. The slab allocator is the
// sole user of these in this test binary, so counting them gives a direct read
// of the slab's heap traffic — gtest / STL noise stays on the plain overloads.
namespace {
std::atomic<std::size_t> g_aligned_new_calls{0};
std::atomic<std::size_t> g_aligned_del_calls{0};

std::size_t AlignedNews() {
  return g_aligned_new_calls.load(std::memory_order_relaxed);
}
std::size_t AlignedDels() {
  return g_aligned_del_calls.load(std::memory_order_relaxed);
}
}  // namespace

void* operator new[](std::size_t sz, std::align_val_t al) {
  g_aligned_new_calls.fetch_add(1, std::memory_order_relaxed);
  void* p = nullptr;
  if (::posix_memalign(&p, static_cast<std::size_t>(al), sz) != 0) {
    throw std::bad_alloc();
  }
  return p;
}

void operator delete[](void* p, std::align_val_t) noexcept {
  if (p != nullptr) {
    g_aligned_del_calls.fetch_add(1, std::memory_order_relaxed);
  }
  std::free(p);
}

void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
  if (p != nullptr) {
    g_aligned_del_calls.fetch_add(1, std::memory_order_relaxed);
  }
  std::free(p);
}

namespace {

// A payload big enough to exercise the sizeof-rounding branch of kSlotSize but
// smaller than a cache line so the slot is padded up.
struct SmallOrder {
  std::uint64_t id;
  std::int64_t price;
  std::uint32_t qty;
};

// A payload that already fills a cache line exactly.
struct alignas(64) LineOrder {
  std::uint64_t words[8];
};

TEST(SlabAllocatorTest, SlotSizeIsCacheLineMultiple) {
  EXPECT_EQ(oep::core::SlabAllocator<SmallOrder>::kSlotSize % 64u, 0u);
  EXPECT_GE(oep::core::SlabAllocator<SmallOrder>::kSlotSize, 64u);
  EXPECT_EQ(oep::core::SlabAllocator<LineOrder>::kSlotSize, 64u);
}

TEST(SlabAllocatorTest, EverySlotIs64ByteAligned) {
  constexpr std::size_t kCapacity = 4096;
  oep::core::SlabAllocator<SmallOrder> pool(kCapacity);

  std::vector<SmallOrder*> handed_out;
  handed_out.reserve(kCapacity);
  for (std::size_t i = 0; i < kCapacity; ++i) {
    SmallOrder* p = pool.Allocate();
    ASSERT_NE(p, nullptr) << "unexpected exhaustion at i=" << i;
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(p) % 64u, 0u)
        << "slot #" << i << " not 64-byte aligned";
    handed_out.push_back(p);
  }
  EXPECT_EQ(pool.Allocate(), nullptr);  // pool drained

  // No two slots overlap or share a cache line.
  std::unordered_set<std::uintptr_t> lines;
  lines.reserve(kCapacity);
  for (auto* p : handed_out) {
    const auto line = reinterpret_cast<std::uintptr_t>(p) / 64u;
    ASSERT_TRUE(lines.insert(line).second)
        << "two slots landed on the same cache line";
  }

  for (auto* p : handed_out) pool.Deallocate(p);
  EXPECT_EQ(pool.in_use(), 0u);
}

TEST(SlabAllocatorTest, AlignedForLineSizedPayload) {
  constexpr std::size_t kCapacity = 256;
  oep::core::SlabAllocator<LineOrder> pool(kCapacity);
  for (std::size_t i = 0; i < kCapacity; ++i) {
    LineOrder* p = pool.Allocate();
    ASSERT_NE(p, nullptr);
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(p) % 64u, 0u);
  }
}

TEST(SlabAllocatorTest, ExhaustionReturnsNullptr) {
  oep::core::SlabAllocator<SmallOrder> pool(3);
  auto* a = pool.Allocate();
  auto* b = pool.Allocate();
  auto* c = pool.Allocate();
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(pool.Allocate(), nullptr);
  EXPECT_EQ(pool.in_use(), 3u);
  EXPECT_EQ(pool.available(), 0u);
  pool.Deallocate(b);
  auto* d = pool.Allocate();
  EXPECT_EQ(d, b) << "freed slot must be reused (LIFO)";
  pool.Deallocate(a);
  pool.Deallocate(c);
  pool.Deallocate(d);
  EXPECT_EQ(pool.in_use(), 0u);
}

TEST(SlabAllocatorTest, OwnsRejectsForeignPointers) {
  oep::core::SlabAllocator<SmallOrder> pool_a(8);
  oep::core::SlabAllocator<SmallOrder> pool_b(8);
  auto* p = pool_a.Allocate();
  ASSERT_NE(p, nullptr);
  EXPECT_TRUE(pool_a.Owns(p));
  EXPECT_FALSE(pool_b.Owns(p));
  EXPECT_FALSE(pool_a.Owns(nullptr));
  SmallOrder stack_object{};
  EXPECT_FALSE(pool_a.Owns(&stack_object));
  pool_a.Deallocate(p);
}

TEST(SlabAllocatorTest, NoHotPathHeapTraffic) {
  constexpr std::size_t kCapacity = 1024;
  constexpr std::size_t kOps = 200'000;

  // Isolate the hot-path snapshot from setup/teardown.
  const std::size_t news_before = AlignedNews();
  const std::size_t dels_before = AlignedDels();

  oep::core::SlabAllocator<SmallOrder> pool(kCapacity);

  // Slab construction should account for exactly one aligned new.
  EXPECT_EQ(AlignedNews() - news_before, 1u);
  EXPECT_EQ(AlignedDels() - dels_before, 0u);

  const std::size_t news_after_ctor = AlignedNews();
  const std::size_t dels_after_ctor = AlignedDels();

  // Hot loop: alternating alloc/dealloc plus a rolling live set. Zero global
  // allocator traffic must occur here.
  std::vector<SmallOrder*> live;
  live.reserve(kCapacity);
  std::mt19937_64 rng(0xC0FFEEULL);
  for (std::size_t i = 0; i < kOps; ++i) {
    const bool do_alloc =
        live.size() < kCapacity && (live.empty() || (rng() & 1ULL));
    if (do_alloc) {
      auto* p = pool.Allocate();
      ASSERT_NE(p, nullptr);
      live.push_back(p);
    } else {
      pool.Deallocate(live.back());
      live.pop_back();
    }
  }

  EXPECT_EQ(AlignedNews(), news_after_ctor)
      << "Allocate/Deallocate must not call operator new";
  EXPECT_EQ(AlignedDels(), dels_after_ctor)
      << "Allocate/Deallocate must not call operator delete";

  // Drain and let the pool destruct — matched delete, no leak.
  for (auto* p : live) pool.Deallocate(p);
}

TEST(SlabAllocatorTest, ConstructionAndDestructionAreBalanced) {
  const std::size_t news_before = AlignedNews();
  const std::size_t dels_before = AlignedDels();
  constexpr int kCycles = 128;
  for (int i = 0; i < kCycles; ++i) {
    oep::core::SlabAllocator<SmallOrder> pool(64 + i);
    auto* p = pool.Allocate();
    ASSERT_NE(p, nullptr);
    pool.Deallocate(p);
  }
  const std::size_t news_delta = AlignedNews() - news_before;
  const std::size_t dels_delta = AlignedDels() - dels_before;
  EXPECT_EQ(news_delta, static_cast<std::size_t>(kCycles));
  EXPECT_EQ(dels_delta, static_cast<std::size_t>(kCycles))
      << "destructor leaked storage";
}

TEST(SlabAllocatorTest, RoundTripInUseReturnsToZero) {
  constexpr std::size_t kCapacity = 512;
  oep::core::SlabAllocator<SmallOrder> pool(kCapacity);
  std::vector<SmallOrder*> live;
  live.reserve(kCapacity);
  std::mt19937_64 rng(42);
  for (int i = 0; i < 50'000; ++i) {
    const bool go_up = live.size() < kCapacity &&
                       (live.empty() || (rng() % 3ULL) != 0ULL);
    if (go_up) {
      auto* p = pool.Allocate();
      ASSERT_NE(p, nullptr);
      live.push_back(p);
    } else {
      // Pop a random live pointer, not just the tail — stresses free-list
      // rewiring under LIFO reuse.
      const std::size_t idx = static_cast<std::size_t>(rng() % live.size());
      std::swap(live[idx], live.back());
      pool.Deallocate(live.back());
      live.pop_back();
    }
    ASSERT_EQ(pool.in_use(), live.size());
    ASSERT_EQ(pool.available(), kCapacity - live.size());
  }
  for (auto* p : live) pool.Deallocate(p);
  EXPECT_EQ(pool.in_use(), 0u);
  EXPECT_EQ(pool.available(), kCapacity);
}

// Per-op cost of Allocate+Deallocate must not scale with pool capacity — the
// defining property of an O(1) allocator. We time drain-and-refill loops at
// two capacities that differ by 64x and assert the per-op ratio stays inside a
// generous envelope (accounts for cache effects, CI noise, TLB pressure).
TEST(SlabAllocatorTest, PerOpCostIsIndependentOfPoolSize) {
  auto measure_ns_per_op = [](std::size_t capacity, std::size_t ops) {
    oep::core::SlabAllocator<SmallOrder> pool(capacity);
    std::vector<SmallOrder*> live;
    live.reserve(capacity);
    // Prime: half-fill so we're not fault-hitting during timing.
    for (std::size_t i = 0; i < capacity / 2; ++i) {
      live.push_back(pool.Allocate());
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < ops; ++i) {
      auto* p = pool.Allocate();
      // Use the return value so the compiler can't elide the call.
      asm volatile("" : : "r"(p) : "memory");
      pool.Deallocate(p);
    }
    const auto t1 = std::chrono::steady_clock::now();
    for (auto* p : live) pool.Deallocate(p);
    const auto ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return static_cast<double>(ns) / static_cast<double>(ops);
  };

  constexpr std::size_t kOps = 2'000'000;
  const double small_ns = measure_ns_per_op(1024, kOps);
  const double large_ns = measure_ns_per_op(65'536, kOps);

  // If Allocate/Deallocate were O(N), large_ns would explode. A generous 5x
  // envelope tolerates CI jitter while still failing a linear implementation.
  const double ratio = large_ns / small_ns;
  EXPECT_LT(ratio, 5.0)
      << "per-op cost scaled with capacity (small=" << small_ns
      << "ns, large=" << large_ns << "ns)";
  // Sanity: both measurements should be in the low-nanosecond regime for an
  // intrusive free-list allocator; 500ns/op indicates something is very wrong.
  EXPECT_LT(small_ns, 500.0);
  EXPECT_LT(large_ns, 500.0);
}

TEST(SlabAllocatorTest, NewDeleteConstructsAndDestroys) {
  struct Counted {
    static int& live() {
      static int v = 0;
      return v;
    }
    int payload;
    explicit Counted(int x) : payload(x) { ++live(); }
    ~Counted() { --live(); }
  };
  oep::core::SlabAllocator<Counted> pool(8);
  auto* a = pool.New(7);
  auto* b = pool.New(9);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(a->payload, 7);
  EXPECT_EQ(b->payload, 9);
  EXPECT_EQ(Counted::live(), 2);
  pool.Delete(a);
  pool.Delete(b);
  EXPECT_EQ(Counted::live(), 0);
  EXPECT_EQ(pool.in_use(), 0u);
}

TEST(SlabAllocatorTest, ZeroCapacityIsWellFormed) {
  oep::core::SlabAllocator<SmallOrder> pool(0);
  EXPECT_EQ(pool.capacity(), 0u);
  EXPECT_EQ(pool.Allocate(), nullptr);
}

}  // namespace
