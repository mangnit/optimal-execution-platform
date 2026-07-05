#include "exec/almgren_chriss.hpp"

#include <cassert>
#include <cmath>

namespace oep::exec {

double AlmgrenChriss::ComputeKappa(double lambda, double sigma,
                                   double eta) noexcept {
  assert(eta > 0.0);
  const double num = lambda * sigma * sigma;
  if (num <= 0.0) return 0.0;
  return std::sqrt(num / eta);
}

AlmgrenChriss::AlmgrenChriss(const Params& p)
    : params_(p),
      kappa_(ComputeKappa(p.lambda, p.sigma, p.eta)),
      child_qty_(p.num_steps, 0),
      remaining_(p.num_steps + 1, 0) {
  assert(p.num_steps >= 1);
  assert(p.horizon > 0.0);
  const std::size_t N = p.num_steps;
  const double T = p.horizon;
  const double Qf = static_cast<double>(p.parent_qty);

  // Boundary anchors: full inventory at start, forced close-out at horizon.
  // Anchoring these instead of trusting the rounded sinh values guarantees
  // Σ child_qty == parent_qty and the monotone-non-increasing invariant.
  remaining_[0] = p.parent_qty;
  remaining_[N] = 0;

  const double kT = kappa_ * T;
  // κT ≈ 0 collapses sinh(κ(T-t)) / sinh(κT) to (T-t)/T (the TWAP limit).
  // The threshold keeps us away from dividing by ~0 in the ratio.
  const bool linear = !(kT > 1e-9);
  const double denom = linear ? 0.0 : std::sinh(kT);

  Quantity prev = p.parent_qty;
  for (std::size_t j = 1; j < N; ++j) {
    const double t = (static_cast<double>(j) * T) / static_cast<double>(N);
    double frac;
    if (linear) {
      frac = (T - t) / T;
    } else {
      frac = std::sinh(kappa_ * (T - t)) / denom;
    }
    if (frac < 0.0) frac = 0.0;
    if (frac > 1.0) frac = 1.0;
    Quantity x = static_cast<Quantity>(std::llround(Qf * frac));
    // Rounding can flatten the ratio at very small κ; clamp to keep the
    // remaining-inventory curve monotone non-increasing.
    if (x > prev) x = prev;
    remaining_[j] = x;
    child_qty_[j - 1] = prev - x;
    prev = x;
  }
  // Force close-out at step N so Σ child_qty == Q exactly regardless of the
  // rounding path taken above.
  child_qty_[N - 1] = prev;
}

}  // namespace oep::exec
