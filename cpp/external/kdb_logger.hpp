// KDB+ tick-logger interface for the external clock (docs/architecture.md §2,
// task P2_relay_and_logger). This header lives strictly on the *external*
// clock — the same relay thread that drains the SPSC ring calls into it, so
// std::string / std::mutex / new / delete are all in-bounds. The one hard rule
// is the mirror of the ring: the logger MUST NOT reach back into anything on
// the internal clock. It only ever consumes trivially copyable telemetry
// structs the producer has already deposited in the ring.
//
// Interface shape: a small abstract base (`KdbLogger`) so we can drop a real
// `k.h` IPC implementation in later without touching the relay. For CI and
// unit tests we ship an in-memory CSV sink (`InMemoryKdbLogger`) that records
// exactly what a real KDB writer would receive, in the same row schema
// (`schema.q` under `q/` will consume the same columns). This keeps the relay
// testable without a live KDB tick plant and gives the "structured to easily
// drop in the k.h calls later" property the task asks for.

#ifndef OEP_EXTERNAL_KDB_LOGGER_HPP_
#define OEP_EXTERNAL_KDB_LOGGER_HPP_

#include "core/order_book.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace oep::external {

// Discriminator for the small set of relay-observable events. Kept minimal so
// the telemetry payload stays trivially copyable (SPSC ring requirement).
enum class TelemetryEventType : std::uint8_t {
  kFill = 0,
  kCancel = 1,
  kAck = 2,
};

// Single POD event the internal clock deposits into the SPSC ring. All members
// are fixed-width so the layout is stable across TUs; the struct is trivially
// copyable so `SpscRing<TelemetryEvent, N>` compiles.
//
// Field convention (mirrors `core::Fill` for the fill case):
//   * order_id      — the subject order (taker on fills, cancelled order on
//                     cancel, acknowledged order on ack)
//   * counter_id    — the maker on a fill; 0 for cancel/ack
//   * price / qty   — trade price and quantity for fill; resting price and
//                     residual for cancel; resting price and initial qty for ack
//   * side          — taker side for fill; order side for cancel/ack
//   * timestamp_ns  — internal clock at emit time (TSC-derived, see
//                     `bench_util.hpp`). The relay may add a wall-clock stamp
//                     when it lands; on-tape both are useful (§P4 TCA).
//   * sequence      — producer-side monotone counter; lets the tape detect a
//                     drop even when the SPSC drop counter is not sampled.
struct TelemetryEvent {
  TelemetryEventType type;
  oep::core::Side side;
  std::uint64_t sequence;
  std::uint64_t timestamp_ns;
  oep::core::OrderId order_id;
  oep::core::OrderId counter_id;
  oep::core::Price price;
  oep::core::Quantity qty;

  // Convenience factories keep the producer's call sites terse and make it
  // impossible to leave a discriminator/field pair inconsistent.
  static constexpr TelemetryEvent Fill(std::uint64_t sequence,
                                       std::uint64_t timestamp_ns,
                                       const oep::core::Fill& f) noexcept {
    return TelemetryEvent{TelemetryEventType::kFill, f.taker_side, sequence,
                          timestamp_ns, f.taker_id, f.maker_id, f.price, f.qty};
  }
  static constexpr TelemetryEvent Cancel(std::uint64_t sequence,
                                         std::uint64_t timestamp_ns,
                                         oep::core::OrderId id,
                                         oep::core::Side side,
                                         oep::core::Price price,
                                         oep::core::Quantity residual) noexcept {
    return TelemetryEvent{TelemetryEventType::kCancel, side, sequence,
                          timestamp_ns, id, 0, price, residual};
  }
  static constexpr TelemetryEvent Ack(std::uint64_t sequence,
                                      std::uint64_t timestamp_ns,
                                      oep::core::OrderId id,
                                      oep::core::Side side,
                                      oep::core::Price price,
                                      oep::core::Quantity qty) noexcept {
    return TelemetryEvent{TelemetryEventType::kAck, side, sequence,
                          timestamp_ns, id, 0, price, qty};
  }
};

// Trivially copyable / destructible for SpscRing<TelemetryEvent, N>.
static_assert(std::is_trivially_copyable_v<TelemetryEvent>,
              "TelemetryEvent must stay trivially copyable for SPSC transport");
static_assert(std::is_trivially_destructible_v<TelemetryEvent>,
              "TelemetryEvent must stay trivially destructible");

// CSV schema exposed as a constant so tests (and later `q/schema.q`) can
// pin to the same column order the logger emits.
inline constexpr std::string_view kCsvHeader =
    "type,seq,ts_ns,order_id,counter_id,price,qty,side";

inline constexpr std::string_view TelemetryEventTypeName(
    TelemetryEventType t) noexcept {
  switch (t) {
    case TelemetryEventType::kFill:
      return "FILL";
    case TelemetryEventType::kCancel:
      return "CANCEL";
    case TelemetryEventType::kAck:
      return "ACK";
  }
  return "?";
}

inline constexpr std::string_view SideName(oep::core::Side s) noexcept {
  return s == oep::core::Side::kBuy ? "BUY" : "SELL";
}

// One row of the CSV tape. Kept as a free function so both the in-memory sink
// and the eventual `k.h` sink can share the exact same formatter.
inline std::string FormatCsvRow(const TelemetryEvent& e) {
  std::ostringstream oss;
  oss << TelemetryEventTypeName(e.type) << ',' << e.sequence << ','
      << e.timestamp_ns << ',' << e.order_id << ',' << e.counter_id << ','
      << e.price << ',' << e.qty << ',' << SideName(e.side);
  return oss.str();
}

// Abstract sink. The relay depends only on this interface. A real
// `IpcKdbLogger` (via `k.h`) would implement the same three methods and slot
// straight into `EventRelay` with no producer-side change.
class KdbLogger {
 public:
  virtual ~KdbLogger() = default;

  // Called once per telemetry event by the relay thread. Serialization/IO is
  // the logger's problem; the relay does not block for it beyond the call.
  virtual void Log(const TelemetryEvent& event) = 0;

  // Called by the relay on shutdown, and (optionally) periodically. A real
  // KDB logger flushes its batched IPC frame here.
  virtual void Flush() = 0;
};

// CI/test implementation. Appends one CSV row per event to an internal vector,
// guarded by a mutex so tests can peek from any thread. Not the fast path;
// it exists so the relay is testable without a live KDB instance.
class InMemoryKdbLogger : public KdbLogger {
 public:
  InMemoryKdbLogger() = default;

  InMemoryKdbLogger(const InMemoryKdbLogger&) = delete;
  InMemoryKdbLogger& operator=(const InMemoryKdbLogger&) = delete;

  void Log(const TelemetryEvent& event) override {
    std::string row = FormatCsvRow(event);
    std::lock_guard<std::mutex> lock(mu_);
    rows_.push_back(std::move(row));
  }

  void Flush() override {
    std::lock_guard<std::mutex> lock(mu_);
    ++flush_count_;
  }

  // Snapshot of rows in insertion order. Copy on purpose — callers race the
  // relay thread and must not hold a reference into the live buffer.
  std::vector<std::string> Snapshot() const {
    std::lock_guard<std::mutex> lock(mu_);
    return rows_;
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return rows_.size();
  }

  std::size_t flush_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return flush_count_;
  }

 private:
  mutable std::mutex mu_;
  std::vector<std::string> rows_;
  std::size_t flush_count_ = 0;
};

}  // namespace oep::external

#endif  // OEP_EXTERNAL_KDB_LOGGER_HPP_
