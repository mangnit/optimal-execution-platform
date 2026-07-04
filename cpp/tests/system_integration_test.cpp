// End-to-end wiring test for the P2 pipeline (task P2_wiring).
//
// Pipeline under test:
//     OrderBook ── TelemetryPublisher ── SpscRing<TelemetryEvent> ── EventRelay ── InMemoryKdbLogger
//     └───────── internal (hot) clock ─────────┘ ┴ └────────── external clock ──────────┘
//
// What we prove:
//   1. Every hot-path outcome — Ack (residual rests), Fill (per maker/taker
//      pair), Cancel — reaches the logger, in producer sequence order, in the
//      exact CSV schema `q/schema.q` will pin against.
//   2. Sequence numbers are strictly monotone starting at 0; there are no
//      duplicates and no gaps under normal (no-drop) operation.
//   3. Zero allocation on the hot path. Between an explicit warmup snapshot
//      and the end of the busy phase, `operator new` / `operator delete`
//      counters must not tick. This is the pipeline-level version of the
//      OrderBook.NoHotPathHeapTraffic guard — if the publisher, ring or
//      relay quietly introduced an allocation, the counters fire and this
//      test fails.

#include "core/order_book.hpp"
#include "core/spsc_ring.hpp"
#include "exec/telemetry_publisher.hpp"
#include "external/event_relay.hpp"
#include "external/kdb_logger.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <string>
#include <thread>
#include <vector>

// Intercept every operator new/delete variant so we can prove the hot phase
// touches no global allocator. Same technique as order_book_test.cpp; kept
// self-contained so this TU can be built independently.
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

using oep::core::OrderBook;
using oep::core::OrderId;
using oep::core::Price;
using oep::core::Quantity;
using oep::core::Side;
using oep::core::SpscRing;
using oep::exec::TelemetryPublisher;
using oep::external::EventRelay;
using oep::external::InMemoryKdbLogger;
using oep::external::TelemetryEvent;

template <typename Pred>
bool WaitUntil(Pred pred,
               std::chrono::milliseconds timeout = std::chrono::seconds(2)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred()) return true;
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return pred();
}

// Deterministic clock: monotone counter so the ts_ns column in the CSV tape
// is predictable and comparable. In production this is replaced with the
// TSC-derived nanosecond counter (see bench_util.hpp).
struct CountingClock {
  std::uint64_t* tick;
  std::uint64_t operator()() noexcept { return (*tick)++; }
};

// -------------------------------------------------------------------------
// 1. End-to-end tape verification: Ack / Fill / Cancel land in the logger
//    with the exact expected CSV rows, in producer sequence order.
// -------------------------------------------------------------------------

TEST(SystemIntegration, AckFillCancelPipelineProducesExpectedTape) {
  SpscRing<TelemetryEvent, 1024> ring;
  InMemoryKdbLogger logger;
  EventRelay relay(ring, logger, std::chrono::microseconds(0));

  OrderBook book(64);
  std::uint64_t tick = 1000;
  TelemetryPublisher publisher(book, ring, CountingClock{&tick});

  relay.Start();

  // Passive sell 5@101 → Ack (no cross).
  ASSERT_TRUE(publisher.AddLimit(/*id=*/1, Side::kSell, /*px=*/101, /*qty=*/5));
  // Passive buy 3@99 → Ack (no cross).
  ASSERT_TRUE(publisher.AddLimit(/*id=*/2, Side::kBuy, /*px=*/99, /*qty=*/3));
  // Aggressive buy 7@102: crosses the ask fully (5@101), rests 2@102.
  //   → one Fill event (maker=1, taker=3, price=101, qty=5, side=BUY)
  //   → one Ack event for the residual 2@102.
  ASSERT_TRUE(publisher.AddLimit(/*id=*/3, Side::kBuy, /*px=*/102, /*qty=*/7));
  // Cancel the resting bid at 99 (id=2). → Cancel event (residual 3@99, BUY).
  ASSERT_TRUE(publisher.Cancel(/*id=*/2));

  constexpr std::uint64_t kExpectedEvents = 5;
  ASSERT_TRUE(WaitUntil(
      [&]() { return relay.processed() >= kExpectedEvents; }));
  relay.Stop();

  const auto rows = logger.Snapshot();
  ASSERT_EQ(rows.size(), kExpectedEvents);

  // seq 0 (ts 1000): Ack for the passive sell id=1.
  EXPECT_EQ(rows[0], "ACK,0,1000,1,0,101,5,SELL");
  // seq 1 (ts 1001): Ack for the passive buy id=2.
  EXPECT_EQ(rows[1], "ACK,1,1001,2,0,99,3,BUY");
  // seq 2 (ts 1002): Fill from the aggressive buy id=3 lifting maker id=1.
  EXPECT_EQ(rows[2], "FILL,2,1002,3,1,101,5,BUY");
  // seq 3 (ts 1003): Ack for id=3's residual 2@102.
  EXPECT_EQ(rows[3], "ACK,3,1003,3,0,102,2,BUY");
  // seq 4 (ts 1004): Cancel of id=2 (still 3@99, BUY).
  EXPECT_EQ(rows[4], "CANCEL,4,1004,2,0,99,3,BUY");

  EXPECT_EQ(ring.dropped_messages(), 0u);
  EXPECT_EQ(publisher.next_sequence(), kExpectedEvents);
}

// -------------------------------------------------------------------------
// 2. No ack when the taker fully aggresses away (no residual to rest).
// -------------------------------------------------------------------------

TEST(SystemIntegration, PureAggressorEmitsOnlyFillNoAck) {
  SpscRing<TelemetryEvent, 256> ring;
  InMemoryKdbLogger logger;
  EventRelay relay(ring, logger, std::chrono::microseconds(0));

  OrderBook book(16);
  std::uint64_t tick = 500;
  TelemetryPublisher publisher(book, ring, CountingClock{&tick});

  relay.Start();

  // Passive ask 10@100 → Ack.
  ASSERT_TRUE(publisher.AddLimit(1, Side::kSell, 100, 10));
  // Aggressive buy 4@100 fully lifts 4@100 → Fill only (no residual, no ack).
  ASSERT_TRUE(publisher.AddLimit(2, Side::kBuy, 100, 4));

  ASSERT_TRUE(WaitUntil([&]() { return relay.processed() >= 2; }));
  relay.Stop();

  const auto rows = logger.Snapshot();
  ASSERT_EQ(rows.size(), 2u);
  EXPECT_EQ(rows[0], "ACK,0,500,1,0,100,10,SELL");
  EXPECT_EQ(rows[1], "FILL,1,501,2,1,100,4,BUY");
}

// -------------------------------------------------------------------------
// 3. Cancel of an unknown id emits nothing (and returns false).
// -------------------------------------------------------------------------

TEST(SystemIntegration, CancelUnknownIdEmitsNothing) {
  SpscRing<TelemetryEvent, 64> ring;
  InMemoryKdbLogger logger;
  EventRelay relay(ring, logger, std::chrono::microseconds(0));

  OrderBook book(4);
  std::uint64_t tick = 1;
  TelemetryPublisher publisher(book, ring, CountingClock{&tick});

  relay.Start();
  EXPECT_FALSE(publisher.Cancel(/*id=*/999));
  relay.Stop();

  EXPECT_EQ(logger.size(), 0u);
  EXPECT_EQ(publisher.next_sequence(), 0u);
}

// -------------------------------------------------------------------------
// 4. Zero-allocation guarantee for the wired hot path.
// -------------------------------------------------------------------------

TEST(SystemIntegration, WiredHotPathIsZeroAllocation) {
  // Same recipe as OrderBook.NoHotPathHeapTraffic, but exercising the whole
  // publisher → ring path. The relay is deliberately NOT started here — the
  // logger's mutex + std::string formatting live on the external clock and
  // are permitted to allocate; that would confuse this alloc counter. What
  // we care about is the internal-clock producer side.
  constexpr std::size_t kCapacity = 4096;
  constexpr Price kMinPx = 90;
  constexpr Price kMaxPx = 110;
  constexpr std::size_t kOps = 100'000;

  SpscRing<TelemetryEvent, 8192> ring;
  OrderBook book(kCapacity);
  for (Price p = kMinPx; p <= kMaxPx; ++p) {
    book.PrewarmLevel(Side::kBuy, p);
    book.PrewarmLevel(Side::kSell, p);
  }
  std::uint64_t tick = 0;
  TelemetryPublisher publisher(book, ring, CountingClock{&tick});

  // Buffers used by the hot loop; grow them once, before the snapshot.
  std::vector<OrderId> resting;
  resting.reserve(kOps + 16);

  // Snapshot demarcates the warmup / hot boundary. Anything after must be
  // slab/stack/ring only.
  const std::size_t news_before = TotalNews();
  const std::size_t dels_before = TotalDels();

  // Cheap deterministic PRNG (xorshift64) — no <random> engine so we don't
  // inadvertently allocate inside the loop.
  std::uint64_t rng = 0xC0FFEEBABEULL;
  auto next = [&rng]() noexcept {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return rng;
  };

  OrderId next_id = 1;
  for (std::size_t i = 0; i < kOps; ++i) {
    const bool cancel = !resting.empty() && ((next() % 4ULL) == 0ULL);
    if (cancel) {
      const std::size_t idx =
          static_cast<std::size_t>(next() % resting.size());
      std::swap(resting[idx], resting.back());
      const OrderId victim = resting.back();
      resting.pop_back();
      (void)publisher.Cancel(victim);
      continue;
    }

    if (book.open_order_count() >= kCapacity - 2) {
      if (!resting.empty()) {
        const OrderId victim = resting.back();
        resting.pop_back();
        (void)publisher.Cancel(victim);
      }
      continue;
    }

    const bool buy = (next() & 1ULL) != 0ULL;
    const Side side = buy ? Side::kBuy : Side::kSell;
    const Price offset = static_cast<Price>(next() % 11) - 5;
    const Price px = buy ? (99 + offset) : (101 + offset);
    const Quantity qty = 1 + static_cast<Quantity>(next() % 4);
    const OrderId id = next_id++;
    const bool ok = publisher.AddLimit(id, side, px, qty);
    ASSERT_TRUE(ok) << "slab exhaustion at op " << i;
    resting.push_back(id);
  }

  EXPECT_EQ(TotalNews(), news_before)
      << "wired hot loop caused " << (TotalNews() - news_before)
      << " allocations — publisher + ring is not zero-alloc";
  EXPECT_EQ(TotalDels(), dels_before)
      << "wired hot loop caused " << (TotalDels() - dels_before)
      << " deallocations";

  // Publisher must have emitted at least one event per op minus cancels of
  // unknown ids (of which there are none here — every cancel targets a real
  // id or is skipped). A loose sanity floor: at least kOps/4 events.
  EXPECT_GT(publisher.next_sequence(), kOps / 4);
}

// -------------------------------------------------------------------------
// 5. Sequence monotonicity holds under a concurrent producer/consumer.
// -------------------------------------------------------------------------

TEST(SystemIntegration, SequenceIsStrictlyMonotoneUnderConcurrentRelay) {
  SpscRing<TelemetryEvent, 4096> ring;
  InMemoryKdbLogger logger;
  EventRelay relay(ring, logger, std::chrono::microseconds(0));

  OrderBook book(2048);
  // Prewarm a narrow band so resting never triggers std::map allocs on the
  // hot loop (defence in depth — this test doesn't count allocations, but
  // we want the same steady-state as production).
  for (Price p = 90; p <= 110; ++p) {
    book.PrewarmLevel(Side::kBuy, p);
    book.PrewarmLevel(Side::kSell, p);
  }
  std::uint64_t tick = 0;
  TelemetryPublisher publisher(book, ring, CountingClock{&tick});

  relay.Start();

  constexpr std::size_t kOrders = 2'000;
  for (std::size_t i = 0; i < kOrders; ++i) {
    const Side side = (i & 1U) ? Side::kBuy : Side::kSell;
    // Non-crossing prices so every submission emits exactly one Ack — makes
    // the expected event count deterministic (== kOrders).
    const Price px = side == Side::kBuy ? 95 : 105;
    ASSERT_TRUE(publisher.AddLimit(static_cast<OrderId>(i + 1), side, px, 1));
  }

  ASSERT_TRUE(WaitUntil([&]() { return relay.processed() >= kOrders; }));
  relay.Stop();

  const auto rows = logger.Snapshot();
  ASSERT_EQ(rows.size(), kOrders);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const std::string& row = rows[i];
    const auto c1 = row.find(',');
    const auto c2 = row.find(',', c1 + 1);
    ASSERT_NE(c1, std::string::npos);
    ASSERT_NE(c2, std::string::npos);
    const std::uint64_t seq =
        std::stoull(row.substr(c1 + 1, c2 - c1 - 1));
    EXPECT_EQ(seq, static_cast<std::uint64_t>(i)) << "row " << i;
  }
  EXPECT_EQ(ring.dropped_messages(), 0u);
}

}  // namespace
