// VWAP execution baseline (docs/architecture.md §5.2, task P5).
//
// Given a user-supplied volume profile u_j with Σ u_j ≈ 1, the child sizes
// track n_j = round(Q · Σ_{k<=j} u_k) - Σ_{k<j} n_k using cumulative
// rounding, so Σ child_qty == parent_qty exactly regardless of floating-point
// residuals in the profile. The final step is anchored to the parent qty for
// belt-and-braces close-out.
//
// A uniform profile (u_j = 1/N) collapses to TWAP. Same CRTP contract as
// Almgren-Chriss: schedule is pre-computed at construction (external clock),
// the hot path is a single indexed lookup.

#ifndef OEP_EXEC_VWAP_HPP_
#define OEP_EXEC_VWAP_HPP_

#include "exec/strategy.hpp"

#include <cstddef>
#include <vector>

namespace oep::exec {

class VwapStrategy : public Strategy<VwapStrategy> {
 public:
  struct Params {
    Quantity parent_qty;
    Side parent_side;
    // Non-owning pointer to a length-`num_steps` array of non-negative weights.
    // Copied into the internal schedule at construction; the caller may free
    // the array immediately after the ctor returns.
    const double* volume_profile;
    std::size_t num_steps;
  };

  explicit VwapStrategy(const Params& p);

  Quantity ChildQuantityImpl(std::size_t j) const noexcept {
    return child_qty_[j];
  }
  Quantity RemainingAfterStepImpl(std::size_t j) const noexcept {
    return remaining_[j];
  }
  Side side_impl() const noexcept { return side_; }
  std::size_t num_steps_impl() const noexcept { return child_qty_.size(); }
  Quantity parent_qty_impl() const noexcept { return parent_qty_; }

 private:
  Side side_;
  Quantity parent_qty_;
  std::vector<Quantity> child_qty_;
  std::vector<Quantity> remaining_;
};

}  // namespace oep::exec

#endif  // OEP_EXEC_VWAP_HPP_
