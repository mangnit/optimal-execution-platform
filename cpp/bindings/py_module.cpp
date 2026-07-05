// Pybind11 bridge for the C++ matching engine (docs/architecture.md §3, §6,
// task P5.2). The Python side treats this like a host-to-device transfer:
// flat float buffers cross the boundary, the C++ step function runs with the
// GIL released, and nothing on the internal clock allocates during a step.
//
// Layout mirrors `cpp/main/sim_runner.cpp` so the RL environment exercises
// the exact same OrderBook → TelemetryPublisher → SpscRing → EventRelay →
// KdbLogger pipeline the sim runner does. The InMemoryKdbLogger sink is used
// here — the ML training loop does not need a live ticker plant, and the
// in-memory sink's `Flush()` is a no-op so its cost stays on the external
// clock (the relay thread) and never reaches back into the producer.
//
// Zero-allocation contract on the hot path:
//   * `state_buf_` is a fixed-size float array pre-allocated at construction.
//     `state_np_` wraps it in a `py::array_t` once, so `state()` returns a
//     zero-copy view.
//   * The action array is passed in from Python; we read two floats out of
//     it into locals before releasing the GIL. No numpy allocation in step().
//   * The C++ step drives the publisher exactly like sim_runner does — the
//     P2 zero-alloc integration tests already prove that path allocation-free.
//   * The child order goes through `book_.AddLimit` directly with a fused
//     lambda that (a) pushes telemetry into the ring (same shape as
//     TelemetryPublisher) and (b) accumulates fill qty / cash for reward.
//     Two observers, one book call — same zero-alloc guarantee.
//
// GIL discipline: `py::gil_scoped_release release;` is the last statement
// before the C++ step body and the guard's dtor reacquires when the scope
// exits (before we touch `state_np_` again). That is the docs/architecture.md rule for
// the Python/C++ boundary.

#include "core/order_book.hpp"
#include "core/spsc_ring.hpp"
#include "core/tsc_clock.hpp"
#include "exec/telemetry_publisher.hpp"
#include "external/event_relay.hpp"
#include "external/kdb_logger.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace py = pybind11;

namespace {

// Ring capacity matches sim_runner.cpp so training-time allocations look the
// same as production. 64 Ki slots × ~48 B ≈ 3 MiB — heap-allocated once at
// construction, never on the hot path.
constexpr std::size_t kRingCapacity = 1u << 16;
constexpr std::size_t kBookCapacity = 8192;
constexpr oep::core::Price kMinPrewarmPx = 90;
constexpr oep::core::Price kMaxPrewarmPx = 110;

// State vector layout (kept small and flat — docs/architecture.md "host-to-device"):
//   [0] q_remaining / parent_qty         — normalised inventory left
//   [1] (T - t) / T                      — normalised time-to-go
//   [2] best_bid                         — raw tick price (float32)
//   [3] best_ask                         — raw tick price
//   [4] spread                           — best_ask - best_bid
//   [5] mid                              — (bid + ask) / 2
//   [6] qty_filled / parent_qty          — cumulative fill fraction
//   [7] avg_fill_price / arrival_mid     — 1.0 when nothing has filled yet
constexpr std::size_t kStateDim = 8;
constexpr std::size_t kActionDim = 2;

// Deterministic PRNG (xorshift64) — bit-identical to sim_runner.cpp so a
// shared seed produces the same exogenous flow in both drivers.
class Xorshift64 {
 public:
  explicit Xorshift64(std::uint64_t seed) noexcept : state_(seed ? seed : 1) {}
  std::uint64_t Next() noexcept {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 7;
    state_ ^= state_ << 17;
    return state_;
  }
  void Reseed(std::uint64_t seed) noexcept { state_ = seed ? seed : 1; }

 private:
  std::uint64_t state_;
};

using Event = oep::external::TelemetryEvent;
using Ring = oep::core::SpscRing<Event, kRingCapacity>;
using Publisher = oep::exec::TelemetryPublisher<Ring, oep::core::TscNanoClock>;
using Relay = oep::external::EventRelay<Ring>;

class SimEnv {
 public:
  SimEnv(std::uint64_t parent_qty, std::size_t horizon_steps,
         std::uint64_t seed, oep::core::Price arrival_mid)
      : parent_qty_(parent_qty ? parent_qty : 1),
        horizon_steps_(horizon_steps ? horizon_steps : 1),
        arrival_mid_(arrival_mid),
        ring_(std::make_unique<Ring>()),
        logger_(std::make_unique<oep::external::InMemoryKdbLogger>()),
        book_(kBookCapacity),
        publisher_(book_, *ring_, clock_),
        relay_(std::make_unique<Relay>(*ring_, *logger_)),
        rng_(seed) {
    for (oep::core::Price p = kMinPrewarmPx; p <= kMaxPrewarmPx; ++p) {
      book_.PrewarmLevel(oep::core::Side::kBuy, p);
      book_.PrewarmLevel(oep::core::Side::kSell, p);
    }
    // Wrap the state buffer once. `py::none()` as the owner leaves lifetime
    // management to the SimEnv — callers must not keep the returned array
    // alive past this SimEnv. That contract matches numpy-view semantics.
    state_np_ = py::array_t<float>(
        {static_cast<py::ssize_t>(kStateDim)},
        {static_cast<py::ssize_t>(sizeof(float))}, state_buf_, py::none());

    ResetInternal();
    relay_->Start();
    WriteState();
  }

  ~SimEnv() {
    // Stop the relay before we tear down anything it consumes from.
    if (relay_) {
      relay_->Stop();
    }
    if (logger_) {
      logger_->Flush();
    }
  }

  SimEnv(const SimEnv&) = delete;
  SimEnv& operator=(const SimEnv&) = delete;

  py::array_t<float> Reset(std::uint64_t seed) {
    rng_.Reseed(seed);
    ResetInternal();
    WriteState();
    return state_np_;
  }

  // step(action) — action is a flat float32 array of length >= kActionDim:
  //   action[0] = size_fraction  ∈ [0, 1]  fraction of remaining to send
  //   action[1] = aggression     ∈ [0, 1]  0 = post passively, 1 = cross
  // Returns the scalar reward for this step. Reward for a SELL is the
  // per-step realised bps versus the arrival mid, minus an inventory
  // penalty at the terminal step (§5.4). The state array is updated
  // in-place — read it back through `state()` (zero-copy).
  float Step(py::array_t<float, py::array::c_style | py::array::forcecast> action) {
    auto buf = action.request();
    if (static_cast<std::size_t>(buf.size) < kActionDim) {
      throw std::runtime_error("action must have at least 2 elements");
    }
    const float* in = static_cast<const float*>(buf.ptr);
    const float size_frac = std::clamp(in[0], 0.0f, 1.0f);
    const float aggression = std::clamp(in[1], 0.0f, 1.0f);

    float reward = 0.0f;
    {
      // docs/architecture.md P5+ rule: GIL released around the C++ internal clock step.
      py::gil_scoped_release release;
      reward = StepImpl(size_frac, aggression);
    }
    WriteState();
    return reward;
  }

  py::array_t<float> State() const { return state_np_; }
  bool Done() const { return done_; }
  std::uint64_t Remaining() const { return remaining_; }
  std::uint64_t StepIndex() const { return step_idx_; }
  std::uint64_t QtyFilled() const { return qty_filled_; }
  double AvgFillPrice() const {
    return qty_filled_ == 0
               ? 0.0
               : static_cast<double>(cash_from_fills_) /
                     static_cast<double>(qty_filled_);
  }
  std::uint64_t DroppedMessages() const { return ring_->dropped_messages(); }
  std::uint64_t Processed() const { return relay_->processed(); }

  static constexpr std::size_t state_dim() { return kStateDim; }
  static constexpr std::size_t action_dim() { return kActionDim; }

 private:
  void ResetInternal() {
    // Fresh book: we can't safely wipe an OrderBook in-place (slab tracks
    // live nodes), so on Reset we drain via cancels for anything resting.
    // Simpler: rebuild by moving to a fresh publisher/book pair would break
    // the pinned reference held by publisher_. Instead we cancel every
    // resting order tracked in `resting_ids_` so the slab returns to empty.
    for (auto id : resting_ids_) {
      (void)publisher_.Cancel(id);
    }
    resting_ids_.clear();

    remaining_ = parent_qty_;
    step_idx_ = 0;
    qty_filled_ = 0;
    cash_from_fills_ = 0;
    done_ = false;
    next_id_ = 1;
  }

  // Actual C++ step body — runs with the GIL released. No allocations, no
  // Python object touches. All the pybind11-facing state is updated by the
  // caller (`WriteState`) after we return.
  float StepImpl(float size_frac, float aggression) {
    if (done_) {
      return 0.0f;
    }

    // 1) Background flow: a small burst of synthetic maker/taker orders to
    //    keep the book alive. Same recipe as sim_runner.cpp so the seeded
    //    determinism rule is preserved end-to-end.
    for (int k = 0; k < 4; ++k) {
      const bool buy = (rng_.Next() & 1ULL) != 0ULL;
      const oep::core::Side side =
          buy ? oep::core::Side::kBuy : oep::core::Side::kSell;
      const oep::core::Price offset =
          static_cast<oep::core::Price>(rng_.Next() % 11ULL) - 5;
      const oep::core::Price px =
          buy ? (arrival_mid_ - 1 + offset) : (arrival_mid_ + 1 + offset);
      const oep::core::Quantity qty = 1 + (rng_.Next() % 4ULL);
      const oep::core::OrderId id = next_id_++;

      // Slab-eviction guard mirrors sim_runner.cpp — never trip AddLimit→false.
      if (book_.open_order_count() >= kBookCapacity - 8) {
        if (!resting_ids_.empty()) {
          const auto victim = resting_ids_.back();
          resting_ids_.pop_back();
          (void)publisher_.Cancel(victim);
        }
      }
      if (publisher_.AddLimit(id, side, px, qty)) {
        // Only track if a residual could rest (matches publisher's Ack rule
        // implicitly — a fully-aggressive order rested nothing, but tracking
        // its id here is still safe: Cancel on an unknown id is a no-op).
        resting_ids_.push_back(id);
      }
    }

    // 2) Child order sized by the action.
    // Sign convention (docs/architecture.md): parent is a SELL of Q units. size_fraction
    // ∈ [0, 1] scales the remaining inventory; aggression picks the price
    // relative to the current touch.
    const std::uint64_t max_step_qty = remaining_;
    std::uint64_t child_qty =
        static_cast<std::uint64_t>(size_frac * static_cast<float>(max_step_qty));
    // Terminal step: sweep whatever is left ("forced liquidation at T", §5.4).
    if (step_idx_ + 1 >= horizon_steps_) {
      child_qty = remaining_;
    }
    if (child_qty == 0) {
      ++step_idx_;
      MaybeCloseOut();
      return 0.0f;
    }

    const oep::core::Price best_bid_now =
        book_.has_bid() ? book_.best_bid() : arrival_mid_ - 1;
    const oep::core::Price best_ask_now =
        book_.has_ask() ? book_.best_ask() : arrival_mid_ + 1;
    // aggression=0 posts at the far side of the touch (passive rest);
    // aggression=1 crosses through the best bid (marketable sell).
    const oep::core::Price passive_px = best_ask_now;    // rest at ask
    const oep::core::Price aggressive_px = best_bid_now; // sweep bid
    const oep::core::Price child_px =
        (aggression >= 0.5f) ? aggressive_px : passive_px;

    // Fused fill observer: emits telemetry (matches TelemetryPublisher's
    // AddLimit exactly — one Fill event per maker/taker pair, one Ack iff
    // residual > 0) and accumulates fill stats for the reward.
    oep::core::Quantity filled = 0;
    std::int64_t child_cash = 0;
    const oep::core::OrderId child_id = next_id_++;

    auto on_fill = [&](const oep::core::Fill& f) {
      filled += f.qty;
      child_cash += static_cast<std::int64_t>(f.price) *
                    static_cast<std::int64_t>(f.qty);
      const std::uint64_t seq = fused_seq_++;
      const std::uint64_t ts = clock_();
      (void)ring_->TryPush(Event::Fill(seq, ts, f));
    };
    const bool ok = book_.AddLimit(child_id, oep::core::Side::kSell, child_px,
                                    child_qty, on_fill);
    if (ok && filled < child_qty) {
      const oep::core::Quantity residual = child_qty - filled;
      const std::uint64_t seq = fused_seq_++;
      const std::uint64_t ts = clock_();
      (void)ring_->TryPush(Event::Ack(seq, ts, child_id,
                                       oep::core::Side::kSell, child_px,
                                       residual));
      resting_ids_.push_back(child_id);
    }

    // Reward: per-step realised bps against the arrival mid for the SELL.
    //   r_t = Σ f_qty · (f_price - S_0) / S_0 · 1e4  /  parent_qty (normaliser)
    // We normalise by parent_qty so the total episode reward stays O(bps).
    const std::int64_t s0 =
        arrival_mid_ > 0 ? arrival_mid_ : static_cast<oep::core::Price>(1);
    float reward = 0.0f;
    if (filled > 0) {
      const double numerator = static_cast<double>(child_cash) -
                               static_cast<double>(filled) *
                                   static_cast<double>(s0);
      reward = static_cast<float>(numerator / static_cast<double>(s0) * 1e4 /
                                   static_cast<double>(parent_qty_));
    }
    qty_filled_ += filled;
    cash_from_fills_ += child_cash;
    remaining_ = (filled >= remaining_) ? 0 : (remaining_ - filled);

    // Inventory holding penalty φ · (q/Q)^2 — a modest running cost that
    // matches §5.4. φ chosen small so the shortfall term dominates in the
    // smoke test.
    constexpr float kPhi = 0.25f;
    const float inv_frac = static_cast<float>(remaining_) /
                           static_cast<float>(parent_qty_);
    reward -= kPhi * inv_frac * inv_frac;

    ++step_idx_;
    MaybeCloseOut();
    return reward;
  }

  void MaybeCloseOut() noexcept {
    if (remaining_ == 0 || step_idx_ >= horizon_steps_) {
      done_ = true;
    }
  }

  // Called with the GIL held — safe to touch the numpy buffer.
  void WriteState() {
    const oep::core::Price bb = book_.has_bid() ? book_.best_bid() : 0;
    const oep::core::Price ba = book_.has_ask() ? book_.best_ask() : 0;
    const float bb_f = static_cast<float>(bb);
    const float ba_f = static_cast<float>(ba);
    state_buf_[0] = static_cast<float>(remaining_) /
                    static_cast<float>(parent_qty_);
    state_buf_[1] = (horizon_steps_ == 0)
                        ? 0.0f
                        : (1.0f - static_cast<float>(step_idx_) /
                                       static_cast<float>(horizon_steps_));
    state_buf_[2] = bb_f;
    state_buf_[3] = ba_f;
    state_buf_[4] = (bb == 0 || ba == 0) ? 0.0f : (ba_f - bb_f);
    state_buf_[5] = (bb == 0 || ba == 0) ? static_cast<float>(arrival_mid_)
                                          : 0.5f * (bb_f + ba_f);
    state_buf_[6] = static_cast<float>(qty_filled_) /
                    static_cast<float>(parent_qty_);
    state_buf_[7] = (qty_filled_ == 0 || arrival_mid_ == 0)
                        ? 1.0f
                        : static_cast<float>(
                              static_cast<double>(cash_from_fills_) /
                              (static_cast<double>(qty_filled_) *
                               static_cast<double>(arrival_mid_)));
  }

  // Order matters: ring / logger / relay must outlive the publisher that
  // pushes into them, so declare them ahead of the members that reference
  // them (C++ initialises members in declaration order).
  const std::uint64_t parent_qty_;
  const std::size_t horizon_steps_;
  const oep::core::Price arrival_mid_;

  std::unique_ptr<Ring> ring_;
  std::unique_ptr<oep::external::InMemoryKdbLogger> logger_;

  oep::core::OrderBook book_;
  oep::core::TscNanoClock clock_{};
  Publisher publisher_;

  std::unique_ptr<Relay> relay_;

  Xorshift64 rng_;

  // Fused seq for the child-order path — kept distinct from the publisher's
  // internal counter so both paths remain monotone within themselves. The
  // relay does not depend on cross-path seq ordering.
  std::uint64_t fused_seq_ = 1'000'000'000ULL;

  std::uint64_t remaining_ = 0;
  std::uint64_t qty_filled_ = 0;
  std::int64_t cash_from_fills_ = 0;
  std::size_t step_idx_ = 0;
  bool done_ = false;
  oep::core::OrderId next_id_ = 1;

  // Simple resting-id tracker so Reset can drain resting orders back into
  // the slab. Reserved once at construction — no push_back allocation on
  // the hot path in the common case (workloads under this cap).
  std::vector<oep::core::OrderId> resting_ids_ = [] {
    std::vector<oep::core::OrderId> v;
    v.reserve(kBookCapacity);
    return v;
  }();

  alignas(64) float state_buf_[kStateDim] = {0.0f};
  py::array_t<float> state_np_;
};

}  // namespace

PYBIND11_MODULE(oep_env, m) {
  m.doc() =
      "Pybind11 bridge for the C++ matching engine (Phase 5.2). "
      "Wraps the sim_runner stack — OrderBook, SpscRing, TelemetryPublisher, "
      "EventRelay, InMemoryKdbLogger — behind a flat-float `step()` interface "
      "with the GIL released across the C++ internal-clock body.";

  py::class_<SimEnv>(m, "SimEnv")
      .def(py::init<std::uint64_t, std::size_t, std::uint64_t,
                    oep::core::Price>(),
           py::arg("parent_qty") = 1000,
           py::arg("horizon_steps") = 32,
           py::arg("seed") = 0xC0FFEEBABEULL,
           py::arg("arrival_mid") = 100)
      .def("reset", &SimEnv::Reset, py::arg("seed") = 0xC0FFEEBABEULL,
           "Reset internal state and re-seed the exogenous flow PRNG. "
           "Returns the initial observation as a flat float32 array.")
      .def("step", &SimEnv::Step, py::arg("action"),
           "Advance one bar. `action` is a length-2 float32 array "
           "[size_fraction, aggression]. Returns the scalar reward. "
           "The GIL is released for the entirety of the C++ engine step.")
      .def("state", &SimEnv::State,
           "Return the current observation as a zero-copy view of the "
           "internal state buffer. Do not hold the reference past the "
           "SimEnv's lifetime.")
      .def("done", &SimEnv::Done)
      .def("remaining", &SimEnv::Remaining)
      .def("step_index", &SimEnv::StepIndex)
      .def("qty_filled", &SimEnv::QtyFilled)
      .def("avg_fill_price", &SimEnv::AvgFillPrice)
      .def("dropped_messages", &SimEnv::DroppedMessages,
           "SPSC ring drop counter — should stay at 0 for well-sized rings.")
      .def("processed", &SimEnv::Processed,
           "Number of events the relay has drained from the ring.")
      .def_property_readonly_static(
          "state_dim", [](py::object) { return SimEnv::state_dim(); })
      .def_property_readonly_static(
          "action_dim", [](py::object) { return SimEnv::action_dim(); });
}
