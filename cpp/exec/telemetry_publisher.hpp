// Hot-path adapter that drives an OrderBook and publishes every observable
// outcome — ack (residual rests), fill (each maker/taker pair), cancel — as
// a `TelemetryEvent` into an `SpscRing<TelemetryEvent, N>`. This is the last
// wire on the P2 pipeline: the internal clock finishes here at TryPush; the
// external-clock `EventRelay` picks up on the other side of the ring.
//
// Design notes:
//   * Templated on `Ring` and `ClockFn` so nothing about the concrete ring
//     capacity or the timestamp source (TSC in production, monotone counter
//     in tests) leaks into this header. Both the FillHandler passed into
//     `OrderBook::AddLimit` and the OnCancel passed into `OrderBook::Cancel`
//     are stateless-ish lambdas that inline through the templated call site
//     — zero virtual dispatch on the hot path.
//   * Zero allocation. The lambdas capture by reference; no std::function,
//     no heap. `TelemetryEvent` is trivially copyable so the SpscRing slot
//     assignment is a plain memcpy.
//   * Drop-tolerant: `TryPush` is not retried. Telemetry loss is acceptable
//     per §2 of docs/architecture.md; hot-path jitter is not. The SpscRing's
//     own `dropped_messages` counter accounts for any losses.
//   * Sequence numbers are producer-monotone and assigned at emit time. The
//     tape can detect a drop even when the ring's drop counter is not
//     sampled by comparing consecutive `seq` values (see §P4 TCA).
//   * Ack semantics: emitted iff `AddLimit` returned true *and* some residual
//     actually rested (i.e. the taker did not fully cross away). A pure
//     aggressive fill (no residual) emits fills only, no ack. A slab-exhausted
//     rest (AddLimit → false) emits neither ack nor extra fills beyond what
//     already crossed. This mirrors the observability of a real exchange.

#ifndef OEP_EXEC_TELEMETRY_PUBLISHER_HPP_
#define OEP_EXEC_TELEMETRY_PUBLISHER_HPP_

#include "core/order_book.hpp"
#include "external/kdb_logger.hpp"

#include <cstdint>
#include <type_traits>

namespace oep::exec {

template <typename Ring, typename ClockFn>
class TelemetryPublisher {
 public:
  using Event = oep::external::TelemetryEvent;

  static_assert(std::is_same_v<typename Ring::value_type, Event>,
                "Ring must carry TelemetryEvent");

  TelemetryPublisher(oep::core::OrderBook& book, Ring& ring,
                     ClockFn clock) noexcept
      : book_(book), ring_(ring), clock_(clock) {}

  TelemetryPublisher(const TelemetryPublisher&) = delete;
  TelemetryPublisher& operator=(const TelemetryPublisher&) = delete;
  TelemetryPublisher(TelemetryPublisher&&) = delete;
  TelemetryPublisher& operator=(TelemetryPublisher&&) = delete;

  // Submit a limit order. Fills are published as they occur; if any residual
  // rests, a single Ack is published carrying (id, side, price, residual).
  // Returns the underlying book's success flag (false only on slab
  // exhaustion of the residual — drop-on-full mirrors SpscRing::TryPush).
  bool AddLimit(oep::core::OrderId id, oep::core::Side side,
                oep::core::Price price, oep::core::Quantity qty) {
    oep::core::Quantity filled = 0;
    auto on_fill = [this, &filled](const oep::core::Fill& f) {
      filled += f.qty;
      const std::uint64_t seq = seq_++;
      const std::uint64_t ts = clock_();
      (void)ring_.TryPush(Event::Fill(seq, ts, f));
    };
    const bool ok = book_.AddLimit(id, side, price, qty, on_fill);
    if (ok && filled < qty) {
      const oep::core::Quantity residual = qty - filled;
      const std::uint64_t seq = seq_++;
      const std::uint64_t ts = clock_();
      (void)ring_.TryPush(Event::Ack(seq, ts, id, side, price, residual));
    }
    return ok;
  }

  // Cancel a resting order. On success, publishes a Cancel event carrying
  // the residual (side, price, qty) captured before the slab reclaims the
  // node. Returns false (and publishes nothing) if the id is unknown.
  bool Cancel(oep::core::OrderId id) {
    auto on_cancel = [this](oep::core::OrderId cid, oep::core::Side cside,
                            oep::core::Price cprice,
                            oep::core::Quantity residual) {
      const std::uint64_t seq = seq_++;
      const std::uint64_t ts = clock_();
      (void)ring_.TryPush(
          Event::Cancel(seq, ts, cid, cside, cprice, residual));
    };
    return book_.Cancel(id, on_cancel);
  }

  // Next sequence number that will be assigned. Test-only observer.
  std::uint64_t next_sequence() const noexcept { return seq_; }

 private:
  oep::core::OrderBook& book_;
  Ring& ring_;
  ClockFn clock_;
  std::uint64_t seq_ = 0;
};

}  // namespace oep::exec

#endif  // OEP_EXEC_TELEMETRY_PUBLISHER_HPP_
