#include "exec/rl_policy.hpp"

#include <ATen/core/function.h>

#include <cassert>
#include <cmath>
#include <utility>

namespace oep::exec {

namespace {

// Clamp a raw actor output into [0, 1] without touching std::clamp (which
// depending on the STL version can bring in <algorithm> template ceremony
// that shows up in profiles). Zero-branch after the compiler folds the
// predicates.
inline float ClampUnit(float x) noexcept {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

}  // namespace

RlPolicy::RlPolicy(const std::string& model_path, const Params& p)
    : params_(p),
      module_(torch::jit::load(model_path)),
      input_tensor_(torch::empty(
          {1, static_cast<long>(kStateDim)},
          torch::TensorOptions().dtype(torch::kFloat32))),
      forward_fn_(nullptr) {
  Init();
}

RlPolicy::RlPolicy(torch::jit::script::Module module, const Params& p)
    : params_(p),
      module_(std::move(module)),
      input_tensor_(torch::empty(
          {1, static_cast<long>(kStateDim)},
          torch::TensorOptions().dtype(torch::kFloat32))),
      forward_fn_(nullptr) {
  Init();
}

void RlPolicy::Init() {
  assert(params_.num_steps >= 1);
  // eval() flips dropout / batchnorm off. External clock; safe to allocate.
  module_.eval();

  // Cache the forward Function* — bypasses the by-value stack copy that
  // Module::forward and Method::operator() perform, which would allocate
  // one vector slot per call. Function::run(Stack&) takes the stack by
  // reference and lets us reuse the same preallocated buffer forever.
  forward_fn_ = &module_.get_method("forward").function();

  child_qty_.assign(params_.num_steps, 0);
  remaining_.assign(params_.num_steps + 1, 0);
  remaining_[0] = params_.parent_qty;

  // Preallocate the interpreter stack. Once reserved, subsequent
  // push_back / emplace_back calls from Function::run stay within capacity
  // and do not touch the heap.
  stack_.reserve(kMaxStackDepth);

  // Zero the input tensor storage so an uninitialised WriteState doesn't
  // hand the actor NaNs.
  float* buf = input_tensor_.data_ptr<float>();
  for (std::size_t i = 0; i < kStateDim; ++i) {
    buf[i] = 0.0f;
  }
}

void RlPolicy::WriteState(
    const std::array<float, kStateDim>& state) noexcept {
  // Direct memcpy-equivalent into the preallocated tensor storage. No new
  // tensor, no from_blob, no reshape. This is the P5.6 hot-path contract.
  float* dst = input_tensor_.data_ptr<float>();
  for (std::size_t i = 0; i < kStateDim; ++i) {
    dst[i] = state[i];
  }
}

void RlPolicy::ComputeNextAction() {
  assert(current_step_ < params_.num_steps);
  assert(forward_fn_ != nullptr);

  // Inference: no autograd, no gradient tape, no graph edge bookkeeping.
  torch::NoGradGuard no_grad;

  // Reuse the preallocated stack: clear() keeps capacity, emplace_back
  // stays within it, Function::run mutates in place. Zero heap traffic
  // from our side; Torch's tensor allocations flow through c10's caching
  // allocator, not ::operator new, so the allocation-counter tests stay
  // clean too.
  //
  // Stack layout for a module method: [self, arg1, ...]. Skipping `self`
  // triggers "expected 2 inputs, but got only 1" from GraphExecutor —
  // Method::operator()'s by-value stack copy hides this by prepending
  // owner_ for us; using Function::run(Stack&) directly means we have to
  // push it ourselves.
  stack_.clear();
  stack_.emplace_back(module_._ivalue());
  stack_.emplace_back(input_tensor_);
  forward_fn_->run(stack_);

  // After run(), the stack top holds the actor output — shape (1, 2),
  // float32, contiguous. Extract without copying the storage.
  const auto& output = stack_.back().toTensor();
  const float* action = output.data_ptr<float>();
  const float size_fraction = ClampUnit(action[0]);
  const float aggression = ClampUnit(action[1]);

  const Quantity remaining_before = remaining_[current_step_];
  Quantity q;
  if (current_step_ + 1 == params_.num_steps) {
    // Force close-out on the final step so Σ child_qty == parent_qty
    // regardless of what the actor asked for.
    q = remaining_before;
  } else {
    q = static_cast<Quantity>(std::llround(
        static_cast<double>(remaining_before) * size_fraction));
    if (q > remaining_before) q = remaining_before;
  }

  child_qty_[current_step_] = q;
  remaining_[current_step_ + 1] = remaining_before - q;
  last_size_fraction_ = size_fraction;
  last_aggression_ = aggression;
  ++current_step_;
}

}  // namespace oep::exec
