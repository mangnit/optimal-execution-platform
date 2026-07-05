#include "exec/vwap.hpp"

#include <cassert>
#include <cmath>

namespace oep::exec {

VwapStrategy::VwapStrategy(const Params& p)
    : side_(p.parent_side),
      parent_qty_(p.parent_qty),
      child_qty_(p.num_steps, 0),
      remaining_(p.num_steps + 1, 0) {
  assert(p.num_steps >= 1);
  assert(p.volume_profile != nullptr);
  const std::size_t N = p.num_steps;
  const double Qf = static_cast<double>(p.parent_qty);

  remaining_[0] = p.parent_qty;

  double cum = 0.0;
  Quantity sent = 0;
  for (std::size_t j = 0; j < N; ++j) {
    const double w = p.volume_profile[j];
    assert(w >= 0.0);
    cum += w;
    Quantity target;
    if (j + 1 == N) {
      // Anchor the final cumulative target so close-out is exact even if the
      // profile does not quite sum to 1.0.
      target = p.parent_qty;
    } else {
      double c = cum;
      if (c < 0.0) c = 0.0;
      if (c > 1.0) c = 1.0;
      target = static_cast<Quantity>(std::llround(Qf * c));
      if (target > p.parent_qty) target = p.parent_qty;
      if (target < sent) target = sent;
    }
    child_qty_[j] = target - sent;
    sent = target;
    remaining_[j + 1] = p.parent_qty - sent;
  }
}

}  // namespace oep::exec
