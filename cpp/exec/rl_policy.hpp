// RL execution policy — TorchScript inference on the internal clock
// (docs/architecture.md §5.4, task P5.6).
//
// Unlike AC/VWAP, which precompute a static schedule at construction, the
// RL policy is state-dependent: each call sees the current book, feeds the
// 8-dim observation vector to the SAC actor, and reads a 2-dim action
// (size_fraction, aggression). The CRTP surface still hands the strategy
// engine a `ChildQuantity(j) / RemainingAfterStep(j)` view — we back it
// with a per-step cache so the interface is bit-identical to the static
// baselines.
//
// Two-clock discipline (docs/architecture.md §0):
//   - Constructor: external clock. Loads TorchScript, calls torch::empty
//     for the input tensor, initialises schedule caches. Anything here is
//     allowed to allocate.
//   - Hot path (`WriteState`, `ComputeNextAction`): internal clock. Writes
//     the 8 floats directly into the preallocated tensor's storage via
//     `data_ptr<float>()`; feeds it through `Function::run(stack&)` using
//     a preallocated `Stack` (no vector-by-value copy). No `torch::empty`,
//     no `std::vector` growth, no heap traffic from our code — proven by
//     the atomic new/delete counters in rl_policy_test.cpp.
//
// Sign convention: parent SELL of Q over [0,T]; quantities non-negative;
// the sign is carried by `parent_side` alone (§0).

#ifndef OEP_EXEC_RL_POLICY_HPP_
#define OEP_EXEC_RL_POLICY_HPP_

#include "exec/strategy.hpp"

#include <torch/script.h>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace oep::exec {

class RlPolicy : public Strategy<RlPolicy> {
 public:
  static constexpr std::size_t kStateDim = 8;   // matches ExecutionEnv
  static constexpr std::size_t kActionDim = 2;  // (size_frac, aggression)
  // Reserved interpreter-stack capacity. Torch's JIT interpreter pushes
  // intermediates onto the caller-provided Stack; giving it headroom up
  // front means it never has to realloc during a step. Set generously —
  // the SAC actor's MLP peaks at ~10 IValues.
  static constexpr std::size_t kMaxStackDepth = 64;

  struct Params {
    Quantity parent_qty;
    Side parent_side;
    std::size_t num_steps;  // >= 1
  };

  // Load the TorchScript module from disk (external clock). Throws on I/O
  // or schema failure so the caller finds out on startup, not step 1.
  RlPolicy(const std::string& model_path, const Params& p);

  // Test-only overload: adopt a pre-built module (used by rl_policy_test
  // to inject a trivial dummy policy without going through the filesystem).
  RlPolicy(torch::jit::script::Module module, const Params& p);

  RlPolicy(const RlPolicy&) = delete;
  RlPolicy& operator=(const RlPolicy&) = delete;

  // Write the 8-dim observation directly into the preallocated tensor
  // storage. Zero-alloc — no new tensor, no vector, no reshape.
  void WriteState(const std::array<float, kStateDim>& state) noexcept;

  // Escape hatch for callers that already have a pointer to fill (e.g.
  // the pybind bridge's own state buffer). The returned pointer is
  // writable and points at the input tensor's underlying float storage.
  float* state_ptr() noexcept { return input_tensor_.data_ptr<float>(); }

  // Run inference for the *current* step:
  //   1. Feed the preallocated input tensor to the actor.
  //   2. Decode (size_fraction, aggression) into an integer child qty
  //      clamped to the remaining parent inventory.
  //   3. Update the per-step cache so ChildQuantity / RemainingAfterStep
  //      answer correctly for this step.
  //   4. Force close-out on the final step (Σ child == parent_qty).
  // Non-noexcept because Torch's forward may throw on a broken model; on
  // the hot path with a validated model this is a nothrow single call.
  void ComputeNextAction();

  // CRTP forwards — single indexed lookup, no allocation, no branching
  // beyond bounds the compiler elides in release.
  Quantity ChildQuantityImpl(std::size_t j) const noexcept {
    return child_qty_[j];
  }
  Quantity RemainingAfterStepImpl(std::size_t j) const noexcept {
    return remaining_[j];
  }
  Side side_impl() const noexcept { return params_.parent_side; }
  std::size_t num_steps_impl() const noexcept { return params_.num_steps; }
  Quantity parent_qty_impl() const noexcept { return params_.parent_qty; }

  // Introspection (external clock; used by tests + telemetry).
  std::size_t current_step() const noexcept { return current_step_; }
  float last_size_fraction() const noexcept { return last_size_fraction_; }
  float last_aggression() const noexcept { return last_aggression_; }

 private:
  void Init();

  Params params_;
  torch::jit::script::Module module_;
  torch::Tensor input_tensor_;          // shape (1, 8), float32, contiguous
  torch::jit::Function* forward_fn_;    // borrowed from module_
  std::vector<c10::IValue> stack_;      // preallocated interpreter stack
  std::vector<Quantity> child_qty_;     // size = N; Σ = parent_qty
  std::vector<Quantity> remaining_;     // size = N+1; [0]=Q, [N]=0
  std::size_t current_step_ = 0;
  float last_size_fraction_ = 0.0f;
  float last_aggression_ = 0.0f;
};

}  // namespace oep::exec

#endif  // OEP_EXEC_RL_POLICY_HPP_
