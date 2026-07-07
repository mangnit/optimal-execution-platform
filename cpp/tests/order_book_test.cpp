// Tests for oep::core::OrderBook.
//
// Focus areas (per task P1_order_book):
//   1. Price/time priority — a marketable order takes the best price first,
//      then within a level consumes head-of-queue first.
//   2. Partial fills, level-crossing sweeps, and residual resting.
//   3. O(1) cancel from head / middle / tail of a FIFO — verified
//      structurally (list topology after unlink) and via the fact that
//      cancel never touches the slab's global allocator.
//   4. Zero hot-path heap traffic. Every global new/delete overload is
//      intercepted and counted; after prewarming, the hot loop must produce
//      zero deltas on both counters (see NoHotPathHeapTraffic).
//   5. Determinism — replaying the same op sequence yields byte-identical
//      fills (this is the property the sim engine depends on, per
//      docs/architecture.md §12).

#include "core/order_book.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <random>
#include <vector>

// Intercept every standard operator new/delete variant. Unlike the slab test,
// which only intercepts the aligned overloads, the order book uses std::map
// (plain new) internally — so we need to see all traffic to prove zero
// allocation during the hot phase.
namespace {
std::atomic<std::size_t> g_new_calls{0};
std::atomic<std::size_t> g_del_calls{0};

std::size_t TotalNews() { return g_new_calls.load(std::memory_order_relaxed); }
std::size_t TotalDels() { return g_del_calls.load(std::memory_order_relaxed); }

void* DoAlloc(std::size_t sz, std::size_t align) {
  g_new_calls.fetch_add(1, std::memory_order_relaxed);
  void* p = nullptr;
  if (align < sizeof(void*)) align = sizeof(void*);
  if (::posix_memalign(&p, align, sz == 0 ? 1 : sz) != 0) {
    throw std::bad_alloc();
  }
  return p;
}

void DoFree(void* p) {
  if (p == nullptr) return;
  g_del_calls.fetch_add(1, std::memory_order_relaxed);
  std::free(p);
}
}  // namespace

void* operator new(std::size_t sz) { return DoAlloc(sz, alignof(std::max_align_t)); }
void* operator new[](std::size_t sz) { return DoAlloc(sz, alignof(std::max_align_t)); }
void* operator new(std::size_t sz, std::align_val_t al) {
  return DoAlloc(sz, static_cast<std::size_t>(al));
}
void* operator new[](std::size_t sz, std::align_val_t al) {
  return DoAlloc(sz, static_cast<std::size_t>(al));
}
void operator delete(void* p) noexcept { DoFree(p); }
void operator delete[](void* p) noexcept { DoFree(p); }
void operator delete(void* p, std::size_t) noexcept { DoFree(p); }
void operator delete[](void* p, std::size_t) noexcept { DoFree(p); }
void operator delete(void* p, std::align_val_t) noexcept { DoFree(p); }
void operator delete[](void* p, std::align_val_t) noexcept { DoFree(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { DoFree(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { DoFree(p); }

namespace {

using oep::core::Fill;
using oep::core::OrderBook;
using oep::core::OrderId;
using oep::core::Price;
using oep::core::Quantity;
using oep::core::Side;

// Convenience: a lambda-friendly fill sink that appends to a caller vector.
struct FillSink {
  std::vector<Fill>* out;
  void operator()(const Fill& f) const { out->push_back(f); }
};

// -------------------------------------------------------------------------
// 1. Price/time priority.
// -------------------------------------------------------------------------

TEST(OrderBook, TimePriorityWithinLevel) {
  OrderBook book(64);
  std::vector<Fill> fills;
  FillSink sink{&fills};

  // Three bids at the same price. The first to arrive should fill first.
  ASSERT_TRUE(book.AddLimit(/*id=*/1, Side::kBuy, /*px=*/100, /*qty=*/5, sink));
  ASSERT_TRUE(book.AddLimit(/*id=*/2, Side::kBuy, /*px=*/100, /*qty=*/5, sink));
  ASSERT_TRUE(book.AddLimit(/*id=*/3, Side::kBuy, /*px=*/100, /*qty=*/5, sink));
  EXPECT_TRUE(fills.empty());  // no crossing yet

  // An 8-lot sell at price 100 crosses. It should consume order 1 (5) and
  // partially fill order 2 (3), leaving order 2 with 2 and order 3 intact.
  ASSERT_TRUE(book.AddLimit(/*id=*/99, Side::kSell, /*px=*/100, /*qty=*/8, sink));

  ASSERT_EQ(fills.size(), 2u);
  EXPECT_EQ(fills[0].maker_id, 1u);
  EXPECT_EQ(fills[0].taker_id, 99u);
  EXPECT_EQ(fills[0].price, 100);
  EXPECT_EQ(fills[0].qty, 5u);
  EXPECT_EQ(fills[0].taker_side, Side::kSell);
  EXPECT_EQ(fills[1].maker_id, 2u);
  EXPECT_EQ(fills[1].qty, 3u);

  // Level state after the sweep.
  EXPECT_EQ(book.level_qty(Side::kBuy, 100), 2u + 5u);
  EXPECT_EQ(book.level_order_count(Side::kBuy, 100), 2u);
  EXPECT_EQ(book.best_bid(), 100);
}

TEST(OrderBook, PricePriorityAcrossLevels) {
  OrderBook book(64);
  std::vector<Fill> fills;
  FillSink sink{&fills};

  // Three bid levels. A crossing sell should sweep the best (highest) first.
  ASSERT_TRUE(book.AddLimit(1, Side::kBuy, 98, 4, sink));
  ASSERT_TRUE(book.AddLimit(2, Side::kBuy, 99, 3, sink));
  ASSERT_TRUE(book.AddLimit(3, Side::kBuy, 100, 2, sink));
  EXPECT_EQ(book.best_bid(), 100);

  // Sell 6 at 98 crosses all three levels: 2@100, 3@99, 1@98.
  ASSERT_TRUE(book.AddLimit(99, Side::kSell, 98, 6, sink));

  ASSERT_EQ(fills.size(), 3u);
  EXPECT_EQ(fills[0].price, 100);
  EXPECT_EQ(fills[0].qty, 2u);
  EXPECT_EQ(fills[1].price, 99);
  EXPECT_EQ(fills[1].qty, 3u);
  EXPECT_EQ(fills[2].price, 98);
  EXPECT_EQ(fills[2].qty, 1u);

  EXPECT_EQ(book.best_bid(), 98);
  EXPECT_EQ(book.level_qty(Side::kBuy, 98), 3u);
  EXPECT_FALSE(book.has_ask());
}

TEST(OrderBook, ResidualRestsAsPassive) {
  OrderBook book(64);
  std::vector<Fill> fills;
  FillSink sink{&fills};

  ASSERT_TRUE(book.AddLimit(1, Side::kSell, 101, 5, sink));
  // Buy 10 at 102: fills 5 vs the ask, then rests 5 at 102.
  ASSERT_TRUE(book.AddLimit(2, Side::kBuy, 102, 10, sink));

  ASSERT_EQ(fills.size(), 1u);
  EXPECT_EQ(fills[0].price, 101);
  EXPECT_EQ(fills[0].qty, 5u);
  EXPECT_FALSE(book.has_ask());
  EXPECT_EQ(book.best_bid(), 102);
  EXPECT_EQ(book.level_qty(Side::kBuy, 102), 5u);
}

TEST(OrderBook, NonCrossingDoesNotMatch) {
  OrderBook book(64);
  std::vector<Fill> fills;
  FillSink sink{&fills};

  ASSERT_TRUE(book.AddLimit(1, Side::kBuy, 99, 3, sink));
  ASSERT_TRUE(book.AddLimit(2, Side::kSell, 101, 3, sink));
  EXPECT_TRUE(fills.empty());
  EXPECT_EQ(book.best_bid(), 99);
  EXPECT_EQ(book.best_ask(), 101);
}

// -------------------------------------------------------------------------
// 2. Cancellation — O(1) from any position in the queue.
// -------------------------------------------------------------------------

TEST(OrderBook, CancelFromMiddlePreservesFifoTopology) {
  OrderBook book(64);
  std::vector<Fill> fills;
  FillSink sink{&fills};

  // Five bids at the same price. FIFO order: 1 -> 2 -> 3 -> 4 -> 5.
  for (OrderId i = 1; i <= 5; ++i) {
    ASSERT_TRUE(book.AddLimit(i, Side::kBuy, 100, 1, sink));
  }
  ASSERT_EQ(book.level_order_count(Side::kBuy, 100), 5u);

  // Cancel the middle order (id=3). Intrusive unlink should give O(1) removal
  // and leave the level as 1 -> 2 -> 4 -> 5.
  EXPECT_TRUE(book.Cancel(3));
  EXPECT_EQ(book.level_order_count(Side::kBuy, 100), 4u);
  EXPECT_EQ(book.level_qty(Side::kBuy, 100), 4u);

  const auto* lvl = book.level(Side::kBuy, 100);
  ASSERT_NE(lvl, nullptr);
  std::vector<OrderId> order;
  for (auto* n = lvl->head; n != nullptr; n = n->next) order.push_back(n->id);
  ASSERT_EQ(order.size(), 4u);
  EXPECT_EQ(order[0], 1u);
  EXPECT_EQ(order[1], 2u);
  EXPECT_EQ(order[2], 4u);
  EXPECT_EQ(order[3], 5u);

  // Reverse-walk must be consistent (doubly-linked).
  std::vector<OrderId> rev;
  for (auto* n = lvl->tail; n != nullptr; n = n->prev) rev.push_back(n->id);
  ASSERT_EQ(rev.size(), 4u);
  EXPECT_EQ(rev[0], 5u);
  EXPECT_EQ(rev[3], 1u);

  // A crossing sell now consumes in the surviving FIFO order.
  ASSERT_TRUE(book.AddLimit(99, Side::kSell, 100, 4, sink));
  ASSERT_EQ(fills.size(), 4u);
  EXPECT_EQ(fills[0].maker_id, 1u);
  EXPECT_EQ(fills[1].maker_id, 2u);
  EXPECT_EQ(fills[2].maker_id, 4u);
  EXPECT_EQ(fills[3].maker_id, 5u);
}

TEST(OrderBook, CancelHeadAndTail) {
  OrderBook book(64);
  std::vector<Fill> fills;
  FillSink sink{&fills};
  for (OrderId i = 1; i <= 3; ++i) {
    ASSERT_TRUE(book.AddLimit(i, Side::kBuy, 100, 1, sink));
  }
  EXPECT_TRUE(book.Cancel(1));    // head
  EXPECT_TRUE(book.Cancel(3));    // tail (of the survivors)
  const auto* lvl = book.level(Side::kBuy, 100);
  ASSERT_NE(lvl, nullptr);
  ASSERT_EQ(lvl->head, lvl->tail);
  ASSERT_NE(lvl->head, nullptr);
  EXPECT_EQ(lvl->head->id, 2u);
  EXPECT_EQ(lvl->head->prev, nullptr);
  EXPECT_EQ(lvl->head->next, nullptr);
}

TEST(OrderBook, CancelUnknownReturnsFalse) {
  OrderBook book(8);
  EXPECT_FALSE(book.Cancel(42));
}

TEST(OrderBook, CancelledOrderCannotBeFilled) {
  OrderBook book(16);
  std::vector<Fill> fills;
  FillSink sink{&fills};
  ASSERT_TRUE(book.AddLimit(1, Side::kBuy, 100, 5, sink));
  ASSERT_TRUE(book.AddLimit(2, Side::kBuy, 100, 5, sink));
  EXPECT_TRUE(book.Cancel(1));
  ASSERT_TRUE(book.AddLimit(99, Side::kSell, 100, 5, sink));
  ASSERT_EQ(fills.size(), 1u);
  EXPECT_EQ(fills[0].maker_id, 2u);
}

TEST(OrderBook, CancelReleasesSlabSlot) {
  OrderBook book(2);
  std::vector<Fill> fills;
  FillSink sink{&fills};
  ASSERT_TRUE(book.AddLimit(1, Side::kBuy, 100, 1, sink));
  ASSERT_TRUE(book.AddLimit(2, Side::kBuy, 100, 1, sink));
  EXPECT_EQ(book.open_order_count(), 2u);
  // Slab full — a third order can't rest.
  EXPECT_FALSE(book.AddLimit(3, Side::kBuy, 99, 1, sink));
  EXPECT_TRUE(book.Cancel(1));
  EXPECT_EQ(book.open_order_count(), 1u);
  // Now there is room again.
  EXPECT_TRUE(book.AddLimit(4, Side::kBuy, 99, 1, sink));
}

// -------------------------------------------------------------------------
// 3. Zero hot-path heap traffic.
// -------------------------------------------------------------------------

TEST(OrderBook, NoHotPathHeapTraffic) {
  constexpr std::size_t kCapacity = 4096;
  constexpr Price kMinPx = 90;
  constexpr Price kMaxPx = 110;  // inclusive
  constexpr std::size_t kOps = 200'000;

  OrderBook book(kCapacity);
  for (Price p = kMinPx; p <= kMaxPx; ++p) {
    book.PrewarmLevel(Side::kBuy, p);
    book.PrewarmLevel(Side::kSell, p);
  }

  // Pre-size the fills buffer so emplace_back never grows the vector.
  std::vector<Fill> fills;
  fills.reserve(kCapacity * 4);
  FillSink sink{&fills};

  // Track order ids we've placed so we can cancel real ones (some entries in
  // this list may already have been consumed by a marketable order — Cancel
  // will just return false in that case). We push once per AddLimit and pop
  // once per cancel, so the vector can grow up to ~kOps if cancels are rare;
  // reserve for that worst case so push_back never reallocates in the hot
  // loop.
  std::vector<OrderId> resting;
  resting.reserve(kOps + 16);

  // Snapshot AFTER all setup — this is the demarcation between warmup and
  // hot phase. Anything from here on out must be slab / stack only.
  const std::size_t news_before = TotalNews();
  const std::size_t dels_before = TotalDels();

  std::mt19937_64 rng(0xC0FFEEBABEULL);
  OrderId next_id = 1;

  for (std::size_t i = 0; i < kOps; ++i) {
    fills.clear();  // reuse the reserved buffer

    const bool cancel =
        !resting.empty() && ((rng() % 4ULL) == 0ULL);
    if (cancel) {
      const std::size_t idx =
          static_cast<std::size_t>(rng() % resting.size());
      std::swap(resting[idx], resting.back());
      const OrderId victim = resting.back();
      resting.pop_back();
      // Some victims may have already been consumed by a marketable order;
      // Cancel returns false in that case. Either outcome is fine here.
      (void)book.Cancel(victim);
      continue;
    }

    // Room in the slab? If not, force a cancel next iter by skipping.
    if (book.open_order_count() >= kCapacity - 2) {
      if (!resting.empty()) {
        const OrderId victim = resting.back();
        resting.pop_back();
        (void)book.Cancel(victim);
      }
      continue;
    }

    const bool buy = (rng() & 1ULL) != 0ULL;
    const Side side = buy ? Side::kBuy : Side::kSell;
    // Skew prices so buys mostly rest below asks and vice versa, keeping the
    // mix healthy (some rests, some crosses).
    const Price offset = static_cast<Price>(rng() % 11) - 5;
    const Price px = buy ? (99 + offset) : (101 + offset);
    const Quantity qty = 1 + static_cast<Quantity>(rng() % 4);
    const OrderId id = next_id++;
    const bool ok = book.AddLimit(id, side, px, qty, sink);
    ASSERT_TRUE(ok) << "unexpected slab exhaustion at op " << i;
    // If the order rested (not fully filled), track it as cancellable.
    // We can detect resting via slab in_use change, but simpler: if fills
    // didn't fully consume qty, the residual rested. In this test we don't
    // strictly need to track exact residuals — we track the id and let
    // Cancel return false for consumed orders.
    resting.push_back(id);
  }

  EXPECT_EQ(TotalNews(), news_before)
      << "hot loop caused " << (TotalNews() - news_before)
      << " allocations — the order book is not zero-alloc";
  EXPECT_EQ(TotalDels(), dels_before)
      << "hot loop caused " << (TotalDels() - dels_before)
      << " deallocations";
}

// -------------------------------------------------------------------------
// 4. Determinism (docs/architecture.md §12).
// -------------------------------------------------------------------------

TEST(OrderBook, ReplayingOpsGivesIdenticalFills) {
  auto run = []() {
    OrderBook book(1024);
    std::vector<Fill> fills;
    FillSink sink{&fills};
    std::mt19937_64 rng(0xDEADBEEFULL);
    OrderId next_id = 1;
    for (int i = 0; i < 5000; ++i) {
      const bool buy = (rng() & 1ULL) != 0ULL;
      const Side side = buy ? Side::kBuy : Side::kSell;
      const Price offset = static_cast<Price>(rng() % 7) - 3;
      const Price px = buy ? (100 + offset) : (102 + offset);
      const Quantity qty = 1 + static_cast<Quantity>(rng() % 3);
      (void)book.AddLimit(next_id++, side, px, qty, sink);
    }
    return fills;
  };
  const auto a = run();
  const auto b = run();
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].maker_id, b[i].maker_id) << " at i=" << i;
    EXPECT_EQ(a[i].taker_id, b[i].taker_id);
    EXPECT_EQ(a[i].price, b[i].price);
    EXPECT_EQ(a[i].qty, b[i].qty);
    EXPECT_EQ(a[i].taker_side, b[i].taker_side);
  }
}

// -------------------------------------------------------------------------
// 6. Id-index integrity under sustained insert/erase churn.
// -------------------------------------------------------------------------

// Regression guard for the back-shift deletion bug found by the fused
// hot-path benchmark: the erase loop stopped at the first entry sitting on
// its home slot (instead of skipping unmovable entries and continuing to
// the cluster end), stranding displaced entries behind the new hole. Those
// orphans were unreachable by lookup — so Cancel returned false for live
// orders — and accumulated until the table saturated, at which point
// InsertId (which needs an empty slot to terminate) spun forever.
//
// A small book makes clusters dense so the pattern fires fast: keep a
// rolling window of live resting orders and churn add+cancel for many
// rounds. With the bug, a cancel of a live id starts failing within a few
// hundred rounds; with the fix, every cancel succeeds and the book drains
// to empty at the end.
TEST(OrderBook, IdIndexSurvivesSustainedChurn) {
  constexpr std::size_t kCapacity = 64;      // id table = 128 slots
  constexpr OrderId kWindow = 32;            // live population (25% load)
  constexpr OrderId kRounds = 200'000;
  OrderBook book(kCapacity);
  std::vector<Fill> fills;
  FillSink sink{&fills};

  for (OrderId n = 1; n <= kRounds; ++n) {
    ASSERT_TRUE(book.AddLimit(n, Side::kBuy, /*px=*/100, /*qty=*/1, sink))
        << "rest failed at round " << n;
    if (n > kWindow) {
      ASSERT_TRUE(book.Cancel(n - kWindow))
          << "live order lost from id index at round " << n;
    }
  }
  EXPECT_TRUE(fills.empty());  // same-side resting flow never crosses

  // Drain the window; every remaining live id must still be reachable.
  for (OrderId n = kRounds - kWindow + 1; n <= kRounds; ++n) {
    ASSERT_TRUE(book.Cancel(n)) << "drain failed for id " << n;
  }
  EXPECT_EQ(book.open_order_count(), 0u);
  EXPECT_FALSE(book.has_bid());
}

}  // namespace
