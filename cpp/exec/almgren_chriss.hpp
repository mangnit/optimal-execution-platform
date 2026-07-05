// Almgren-Chriss execution baseline (docs/architecture.md §5.1, task P5).
//
// Continuous-limit urgency schedule for liquidating Q over [0, T]:
//
//     x_j = Q · sinh( κ (T - t_j) ) / sinh( κ T ),   κ = sqrt(λ σ² / η)
//     n_j = x_{j-1} - x_j
//
// κ is pre-calculated once at construction; the resulting real-valued x_j are
// discretised into an integer child-size table (`child_qty_`) that also sums
// exactly to Q even after rounding — the constructor forces close-out on the
// final step. The hot path is a single indexed lookup into `child_qty_` (or
// `remaining_`), so ChildQuantity / RemainingAfterStep are zero-allocation and
// zero-branch beyond the vector bounds check the compiler elides in release.
//
// λ → 0 collapses κ → 0 and the schedule to linear (TWAP). The constructor
// handles this limit explicitly to avoid dividing 0/0 through the sinh ratio.

#ifndef OEP_EXEC_ALMGREN_CHRISS_HPP_
#define OEP_EXEC_ALMGREN_CHRISS_HPP_

#include "exec/strategy.hpp"

#include <cstddef>
#include <vector>

namespace oep::exec {

class AlmgrenChriss : public Strategy<AlmgrenChriss> {
 public:
  struct Params {
    Quantity parent_qty;    // Q
    Side parent_side;
    std::size_t num_steps;  // N (>= 1)
    double horizon;         // T (> 0, arbitrary time units)
    double lambda;          // risk aversion λ (>= 0)
    double sigma;           // volatility σ (>= 0)
    double eta;             // linear temporary impact η (> 0)
  };

  // κ = sqrt(λ σ² / η); returns 0 when λσ² = 0 so the caller can special-case
  // the TWAP limit. External-clock helper (not on the hot path).
  static double ComputeKappa(double lambda, double sigma, double eta) noexcept;

  explicit AlmgrenChriss(const Params& p);

  // CRTP forwards — all noexcept, zero-alloc, single indexed lookup.
  Quantity ChildQuantityImpl(std::size_t j) const noexcept {
    return child_qty_[j];
  }
  Quantity RemainingAfterStepImpl(std::size_t j) const noexcept {
    return remaining_[j];
  }
  Side side_impl() const noexcept { return params_.parent_side; }
  std::size_t num_steps_impl() const noexcept { return params_.num_steps; }
  Quantity parent_qty_impl() const noexcept { return params_.parent_qty; }

  double kappa() const noexcept { return kappa_; }

 private:
  Params params_;
  double kappa_;
  std::vector<Quantity> child_qty_;  // size = N; Σ = Q
  std::vector<Quantity> remaining_;  // size = N+1; [0]=Q, [N]=0
};

}  // namespace oep::exec

#endif  // OEP_EXEC_ALMGREN_CHRISS_HPP_
