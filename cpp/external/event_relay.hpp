// External-clock event relay (docs/architecture.md §2, task P2_relay_and_logger).
//
// The internal clock (matching engine, order book) writes telemetry to exactly
// one SPSC ring and never blocks. This class owns the other end: a single
// dedicated consumer thread that polls `TryPop` and hands each event to a
// `KdbLogger`. It lives entirely on the external clock — std::thread, mutexes
// inside the logger, and dynamic dispatch through `KdbLogger&` are all fine
// here. The one hard constraint is the same as the ring's: nothing this class
// does may reach back and block the producer.
//
// Shutdown contract:
//   * Start()   — idempotent; spawns the consumer thread once.
//   * Stop()    — idempotent; sets an atomic flag, drains everything currently
//                 in the ring, calls Flush(), and joins. Callers should stop
//                 producing before invoking Stop; anything pushed strictly
//                 after Stop() returns is not guaranteed to be observed.
//   * dtor      — calls Stop(), so an EventRelay on the stack is always joined.
//
// The idle policy is a short std::this_thread::sleep_for when the ring drains
// empty. That keeps CPU usage sane in CI without introducing a condition
// variable on the producer side (which would violate the "hot path never
// blocks" rule). The sleep duration is configurable so tests can drive it to
// zero when they want the relay to spin.

#ifndef OEP_EXTERNAL_EVENT_RELAY_HPP_
#define OEP_EXTERNAL_EVENT_RELAY_HPP_

#include "external/kdb_logger.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <utility>

namespace oep::external {

// `Ring` is templated so we can hook up any `SpscRing<TelemetryEvent, N>`
// without pulling a specific capacity into this header. It must expose
// `value_type` and `bool TryPop(value_type&)` (matches core::SpscRing).
template <typename Ring>
class EventRelay {
 public:
  using Event = typename Ring::value_type;

  EventRelay(Ring& ring, KdbLogger& logger,
             std::chrono::microseconds idle_sleep =
                 std::chrono::microseconds(50)) noexcept
      : ring_(ring), logger_(logger), idle_sleep_(idle_sleep) {}

  EventRelay(const EventRelay&) = delete;
  EventRelay& operator=(const EventRelay&) = delete;
  EventRelay(EventRelay&&) = delete;
  EventRelay& operator=(EventRelay&&) = delete;

  ~EventRelay() { Stop(); }

  // Idempotent. Spawns the consumer thread on the first call; subsequent
  // calls while running are no-ops.
  void Start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true,
                                          std::memory_order_acq_rel)) {
      return;
    }
    stop_requested_.store(false, std::memory_order_release);
    thread_ = std::thread(&EventRelay::Run, this);
  }

  // Idempotent. Signals the consumer, waits for it to drain and join.
  void Stop() {
    if (!running_.load(std::memory_order_acquire)) {
      return;
    }
    stop_requested_.store(true, std::memory_order_release);
    if (thread_.joinable()) {
      thread_.join();
    }
    running_.store(false, std::memory_order_release);
  }

  [[nodiscard]] bool running() const noexcept {
    return running_.load(std::memory_order_acquire);
  }

  // Monotone counter of events handed to the logger. Racy against a live
  // consumer but useful for tests and for the eventual §P4 TCA tape check.
  [[nodiscard]] std::uint64_t processed() const noexcept {
    return processed_.load(std::memory_order_relaxed);
  }

 private:
  void Run() {
    Event event{};
    while (true) {
      bool got_any = false;
      while (ring_.TryPop(event)) {
        logger_.Log(event);
        processed_.fetch_add(1, std::memory_order_relaxed);
        got_any = true;
      }
      // Ordering matters: check the stop flag *after* a drain pass. If the
      // producer pushed and then set stop_requested_, we will observe both
      // states and drain the pushed event on this iteration or the next.
      if (stop_requested_.load(std::memory_order_acquire)) {
        // Final race-close pass: catch anything the producer landed between
        // the drain loop above and its own last push before signalling stop.
        while (ring_.TryPop(event)) {
          logger_.Log(event);
          processed_.fetch_add(1, std::memory_order_relaxed);
        }
        logger_.Flush();
        return;
      }
      if (!got_any && idle_sleep_.count() > 0) {
        std::this_thread::sleep_for(idle_sleep_);
      }
    }
  }

  Ring& ring_;
  KdbLogger& logger_;
  std::chrono::microseconds idle_sleep_;
  std::thread thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<std::uint64_t> processed_{0};
};

}  // namespace oep::external

#endif  // OEP_EXTERNAL_EVENT_RELAY_HPP_
