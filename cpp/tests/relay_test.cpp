// P2 event-relay / KDB logger unit tests.
//
// What these tests must prove (task P2_relay_and_logger):
//   1. The relay drains a burst of events pushed onto the SPSC ring, in FIFO
//      order, without loss when the ring never overflows.
//   2. The CSV formatter emits the exact `type,seq,ts_ns,order_id,counter_id,
//      price,qty,side` schema the tape consumers will pin against.
//   3. Start/Stop is clean: no hang, no double-join, idempotent, safe to
//      destroy without an explicit Stop, and any in-flight producer push
//      landing during Stop is still logged.

#include "core/order_book.hpp"
#include "core/spsc_ring.hpp"
#include "external/event_relay.hpp"
#include "external/kdb_logger.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using oep::core::OrderId;
using oep::core::Price;
using oep::core::Quantity;
using oep::core::Side;
using oep::core::SpscRing;
using oep::external::EventRelay;
using oep::external::FormatCsvRow;
using oep::external::InMemoryKdbLogger;
using oep::external::kCsvHeader;
using oep::external::TelemetryEvent;
using oep::external::TelemetryEventType;

// Small helper: spin-wait (with timeout) until `pred` holds. Keeps tests from
// flaking on slow CI schedulers without introducing a real sleep loop.
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

TEST(KdbLoggerFormat, CsvRowMatchesSchema) {
  // The schema string documents exactly the columns FormatCsvRow emits.
  EXPECT_EQ(kCsvHeader, "type,seq,ts_ns,order_id,counter_id,price,qty,side");

  const oep::core::Fill f{/*maker_id=*/11, /*taker_id=*/22, /*price=*/10025,
                          /*qty=*/300, /*taker_side=*/Side::kSell};
  const TelemetryEvent ev = TelemetryEvent::Fill(/*sequence=*/7,
                                                  /*timestamp_ns=*/123456, f);
  const std::string row = FormatCsvRow(ev);
  EXPECT_EQ(row, "FILL,7,123456,22,11,10025,300,SELL");
}

TEST(KdbLoggerFormat, CancelAndAckRowsMatchSchema) {
  const TelemetryEvent cancel = TelemetryEvent::Cancel(
      /*sequence=*/3, /*timestamp_ns=*/999, /*id=*/42, Side::kBuy,
      /*price=*/9900, /*residual=*/50);
  EXPECT_EQ(FormatCsvRow(cancel), "CANCEL,3,999,42,0,9900,50,BUY");

  const TelemetryEvent ack = TelemetryEvent::Ack(
      /*sequence=*/4, /*timestamp_ns=*/1000, /*id=*/43, Side::kSell,
      /*price=*/9910, /*qty=*/25);
  EXPECT_EQ(FormatCsvRow(ack), "ACK,4,1000,43,0,9910,25,SELL");
}

TEST(EventRelay, DrainsBurstOfEventsInFifoOrder) {
  SpscRing<TelemetryEvent, 1024> ring;
  InMemoryKdbLogger logger;
  EventRelay relay(ring, logger, std::chrono::microseconds(0));

  relay.Start();

  constexpr std::uint64_t kBurst = 500;
  for (std::uint64_t i = 0; i < kBurst; ++i) {
    const TelemetryEvent e = TelemetryEvent::Ack(
        /*sequence=*/i, /*timestamp_ns=*/1'000'000 + i, /*id=*/i + 1,
        Side::kSell, /*price=*/10000 + static_cast<Price>(i),
        /*qty=*/1 + i);
    ASSERT_TRUE(ring.TryPush(e));
  }

  ASSERT_TRUE(WaitUntil([&]() { return relay.processed() >= kBurst; }));

  relay.Stop();
  EXPECT_FALSE(relay.running());
  EXPECT_EQ(relay.processed(), kBurst);

  const auto rows = logger.Snapshot();
  ASSERT_EQ(rows.size(), kBurst);
  for (std::uint64_t i = 0; i < kBurst; ++i) {
    std::ostringstream oss;
    oss << "ACK," << i << ',' << (1'000'000 + i) << ',' << (i + 1) << ",0,"
        << (10000 + i) << ',' << (1 + i) << ",SELL";
    EXPECT_EQ(rows[i], oss.str()) << "row " << i;
  }

  EXPECT_GE(logger.flush_count(), 1u);
  EXPECT_EQ(ring.dropped_messages(), 0u);
}

TEST(EventRelay, StartStopIsIdempotentAndDoesNotHang) {
  SpscRing<TelemetryEvent, 64> ring;
  InMemoryKdbLogger logger;
  EventRelay relay(ring, logger, std::chrono::microseconds(50));

  // Double-start should not spawn a second thread nor deadlock.
  relay.Start();
  EXPECT_TRUE(relay.running());
  relay.Start();
  EXPECT_TRUE(relay.running());

  // Double-stop is a no-op after the first.
  relay.Stop();
  EXPECT_FALSE(relay.running());
  relay.Stop();
  EXPECT_FALSE(relay.running());

  EXPECT_EQ(relay.processed(), 0u);
  EXPECT_EQ(logger.size(), 0u);
}

TEST(EventRelay, DestructorStopsCleanlyWithoutExplicitStop) {
  SpscRing<TelemetryEvent, 32> ring;
  InMemoryKdbLogger logger;
  {
    EventRelay relay(ring, logger, std::chrono::microseconds(50));
    relay.Start();
    for (std::uint64_t i = 0; i < 8; ++i) {
      const TelemetryEvent e = TelemetryEvent::Cancel(
          /*sequence=*/i, /*timestamp_ns=*/i, /*id=*/i, Side::kBuy,
          /*price=*/100, /*residual=*/1);
      ASSERT_TRUE(ring.TryPush(e));
    }
    // Falling off the scope must invoke Stop() through the destructor.
  }
  EXPECT_EQ(logger.size(), 8u);
}

TEST(EventRelay, StopDrainsResidualElements) {
  // Force the relay to have work still queued at the moment Stop() is called.
  // We stall the consumer with a large idle sleep, land a burst, then Stop —
  // it must drain everything before joining.
  SpscRing<TelemetryEvent, 128> ring;
  InMemoryKdbLogger logger;
  EventRelay relay(ring, logger, std::chrono::milliseconds(100));
  relay.Start();

  // Small pause so the relay is parked in its idle sleep, not busy-looping.
  std::this_thread::sleep_for(std::chrono::milliseconds(5));

  constexpr std::uint64_t kBurst = 100;
  for (std::uint64_t i = 0; i < kBurst; ++i) {
    const TelemetryEvent e = TelemetryEvent::Ack(
        /*sequence=*/i, /*timestamp_ns=*/i, /*id=*/i, Side::kBuy,
        /*price=*/1, /*qty=*/1);
    ASSERT_TRUE(ring.TryPush(e));
  }
  relay.Stop();

  EXPECT_EQ(relay.processed(), kBurst);
  EXPECT_EQ(logger.size(), kBurst);
}

TEST(EventRelay, HandlesConcurrentProducerUnderNormalLoad) {
  // Producer pushes on a background thread while the relay drains. With ample
  // ring headroom and a modest event count, nothing should be dropped and
  // every event should end up in the logger in strict sequence order.
  SpscRing<TelemetryEvent, 4096> ring;
  InMemoryKdbLogger logger;
  EventRelay relay(ring, logger, std::chrono::microseconds(0));
  relay.Start();

  constexpr std::uint64_t kEvents = 20'000;
  std::thread producer([&ring] {
    for (std::uint64_t i = 0; i < kEvents; ++i) {
      const TelemetryEvent e = TelemetryEvent::Ack(
          /*sequence=*/i, /*timestamp_ns=*/i, /*id=*/i, Side::kSell,
          /*price=*/1, /*qty=*/1);
      // Retry loop: this test asserts no-drop, so if the consumer is briefly
      // behind we spin rather than dropping.
      while (!ring.TryPush(e)) {
        std::this_thread::yield();
      }
    }
  });

  producer.join();
  ASSERT_TRUE(WaitUntil([&]() { return relay.processed() >= kEvents; }));
  relay.Stop();

  EXPECT_EQ(relay.processed(), kEvents);
  // Note: `dropped_messages` counts *push attempts* that hit a full ring, not
  // events lost from the tape. A retrying producer legitimately increments
  // it; the no-loss guarantee we care about is `logger.size() == kEvents`
  // plus strict monotone sequence in the rows below.

  const auto rows = logger.Snapshot();
  ASSERT_EQ(rows.size(), kEvents);
  // Verify strict monotone sequence in the logged tape.
  for (std::uint64_t i = 0; i < kEvents; ++i) {
    // Column 1 (0-indexed) is the sequence. Cheap parse: find the first two
    // commas.
    const std::string& row = rows[i];
    const auto c1 = row.find(',');
    const auto c2 = row.find(',', c1 + 1);
    ASSERT_NE(c1, std::string::npos);
    ASSERT_NE(c2, std::string::npos);
    const std::uint64_t seq =
        std::stoull(row.substr(c1 + 1, c2 - c1 - 1));
    EXPECT_EQ(seq, i);
  }
}

}  // namespace
