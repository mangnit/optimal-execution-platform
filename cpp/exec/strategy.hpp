// CRTP execution-strategy interface (docs/architecture.md §5, task P5).
//
// The two-clock discipline puts strategy evaluation on the *internal* clock
// (§2). Nothing on that clock is allowed to allocate, throw, or dispatch
// virtually. The base type here defines a small fixed vocabulary
// (ChildQuantity / RemainingAfterStep / side / …) and forwards each call to
// `Derived::…Impl(...)`. The compiler inlines through the CRTP boundary and
// no vtable is generated. Concrete derived strategies (AC, VWAP, later RL)
// pre-compute their schedule at construction time (external clock) so the
// per-step call collapses to a single indexed lookup.
//
// Sign convention (docs/architecture.md §0): quantities are non-negative and
// the parent direction is carried by `side()`. A SELL of Q sends N children
// whose `Quantity` values sum to Q.

#ifndef OEP_EXEC_STRATEGY_HPP_
#define OEP_EXEC_STRATEGY_HPP_

#include "core/order_book.hpp"

#include <cstddef>

namespace oep::exec {

template <typename Derived>
class Strategy {
 public:
  using OrderId = oep::core::OrderId;
  using Price = oep::core::Price;
  using Quantity = oep::core::Quantity;
  using Side = oep::core::Side;

  // Child order size for step j in [0, num_steps()). Zero-allocation.
  Quantity ChildQuantity(std::size_t j) const noexcept {
    return derived().ChildQuantityImpl(j);
  }

  // Parent inventory remaining *after* step j has been sent. Convention:
  // RemainingAfterStep(0) is the full parent qty (no child yet sent),
  // RemainingAfterStep(num_steps()) is 0 (full close-out).
  Quantity RemainingAfterStep(std::size_t j) const noexcept {
    return derived().RemainingAfterStepImpl(j);
  }

  Side side() const noexcept { return derived().side_impl(); }
  Quantity parent_qty() const noexcept { return derived().parent_qty_impl(); }
  std::size_t num_steps() const noexcept { return derived().num_steps_impl(); }

  // Submit step j's child through a templated Publisher. The publisher's
  // AddLimit signature — (OrderId, Side, Price, Quantity) — matches
  // TelemetryPublisher::AddLimit; the CRTP forwarding here means the whole
  // chain inlines and no heap allocation happens on the hot path. Returns
  // whatever AddLimit returns (false only on slab exhaustion). A zero-quantity
  // step is a no-op that returns true without touching the publisher.
  template <typename Publisher>
  bool SubmitStep(std::size_t j, OrderId id, Price px,
                  Publisher& publisher) noexcept {
    const Quantity q = ChildQuantity(j);
    if (q == 0) return true;
    return publisher.AddLimit(id, side(), px, q);
  }

 protected:
  Strategy() = default;
  Strategy(const Strategy&) = default;
  Strategy(Strategy&&) = default;
  Strategy& operator=(const Strategy&) = default;
  Strategy& operator=(Strategy&&) = default;
  ~Strategy() = default;

 private:
  Derived& derived() noexcept { return static_cast<Derived&>(*this); }
  const Derived& derived() const noexcept {
    return static_cast<const Derived&>(*this);
  }
};

}  // namespace oep::exec

#endif  // OEP_EXEC_STRATEGY_HPP_
