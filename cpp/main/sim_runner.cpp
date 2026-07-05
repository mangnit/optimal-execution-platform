// Simulation runner (docs/architecture.md §2, task P3_sim_runner).
//
// Wires the full P1/P2 pipeline into a single executable and pumps a
// deterministic synthetic workload through it:
//
//     OrderBook + TelemetryPublisher   ── internal clock ──▶
//        │                                                  │
//        └── TSC-derived ClockFn (core::TscNanoClock) ─────┘
//                       │
//                       ▼
//        SpscRing<TelemetryEvent, N> ── drop-tolerant bridge ──▶
//                       │
//                       ▼   external clock (relay thread)
//        EventRelay ──▶ KdbLogger  ── IPC to `q schema.q` on 5010
//                            └─── (or InMemoryKdbLogger when built without
//                                  -DOEP_USE_KDB=ON so `docker compose up`
//                                  and CI still run standalone.)
//
// The internal clock uses `oep::core::TscNanoClock`, which calibrates
// nanoseconds-per-cycle against `steady_clock` at first use so each
// `TelemetryEvent::timestamp_ns` value is a true hardware-clock nanosecond
// stamp. The KDB+ ticker plant receives both this internal stamp and (via
// `q/schema.q`'s `.z.p` landing) the external wall clock, satisfying the §P4
// "both clocks on tape" requirement.
//
// The workload is a seeded xorshift64 mix of aggressive / passive limit
// orders and cancels — the same recipe the P2 zero-allocation integration
// test uses, so the sim runner exercises exactly the code paths we already
// verified allocate-free. Prewarming a narrow band of price levels means the
// hot path never triggers a `std::map` node allocation after startup.

#include "core/order_book.hpp"
#include "core/spsc_ring.hpp"
#include "core/tsc_clock.hpp"
#include "exec/telemetry_publisher.hpp"
#include "external/event_relay.hpp"
#include "external/kdb_logger.hpp"

#if OEP_USE_KDB
#include "external/ipc_kdb_logger.hpp"
#endif

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

// Ring capacity: 64 Ki slots keeps ~2.5 MiB of TelemetryEvents resident, which
// is a comfortable margin even if the relay thread is briefly descheduled.
constexpr std::size_t kRingCapacity = 1u << 16;
constexpr std::size_t kBookCapacity = 8192;
constexpr std::size_t kDefaultOps = 100'000;
constexpr oep::core::Price kMinPrewarmPx = 90;
constexpr oep::core::Price kMaxPrewarmPx = 110;

// Deterministic PRNG (xorshift64) — no `<random>` engine, no allocation, and
// bit-identical across libstdc++/libc++ so the same seed yields the same
// event sequence on every host.
class Xorshift64 {
 public:
  explicit Xorshift64(std::uint64_t seed) noexcept : state_(seed) {}
  std::uint64_t Next() noexcept {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 7;
    state_ ^= state_ << 17;
    return state_;
  }

 private:
  std::uint64_t state_;
};

struct Config {
  std::string host = "localhost";
  int port = 5010;
  std::size_t ops = kDefaultOps;
  std::uint64_t seed = 0xC0FFEEBABEULL;
};

Config ParseArgs(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "sim_runner: --" << name << " requires an argument\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--host") {
      cfg.host = next("host");
    } else if (a == "--port") {
      cfg.port = std::atoi(next("port"));
    } else if (a == "--ops") {
      cfg.ops = static_cast<std::size_t>(std::strtoull(next("ops"), nullptr, 10));
    } else if (a == "--seed") {
      cfg.seed = std::strtoull(next("seed"), nullptr, 0);
    } else if (a == "--help" || a == "-h") {
      std::cout
          << "usage: sim_runner [--host H] [--port P] [--ops N] [--seed S]\n"
          << "  --host   KDB+ ticker plant host (default: localhost)\n"
          << "  --port   KDB+ ticker plant port (default: 5010)\n"
          << "  --ops    number of book operations to drive (default: "
          << kDefaultOps << ")\n"
          << "  --seed   xorshift64 seed for the workload (default: 0xC0FFEEBABE)\n";
      std::exit(0);
    } else {
      std::cerr << "sim_runner: unknown argument '" << a << "'\n";
      std::exit(2);
    }
  }
  return cfg;
}

using Event = oep::external::TelemetryEvent;
using Ring = oep::core::SpscRing<Event, kRingCapacity>;

// Factory returns the concrete logger by base pointer so main() can hand the
// same reference to the relay in both build modes. When OEP_USE_KDB=ON we try
// to connect; a failed connection is non-fatal because the relay still drains
// the ring (batching into the logger's per-column buffers) and a follow-up
// operator can bring the ticker plant up.
std::unique_ptr<oep::external::KdbLogger> BuildLogger(const Config& cfg) {
#if OEP_USE_KDB
  auto logger = std::make_unique<oep::external::IpcKdbLogger>(cfg.host, cfg.port);
  if (!logger->Connect()) {
    std::cerr << "sim_runner: warning: could not connect to " << cfg.host
              << ':' << cfg.port
              << " — telemetry will batch in memory until reconnect\n";
  } else {
    std::cerr << "sim_runner: connected to KDB+ ticker plant at "
              << cfg.host << ':' << cfg.port << '\n';
  }
  return logger;
#else
  (void)cfg;
  std::cerr << "sim_runner: built without OEP_USE_KDB — using in-memory sink\n";
  return std::make_unique<oep::external::InMemoryKdbLogger>();
#endif
}

}  // namespace

int main(int argc, char** argv) {
  const Config cfg = ParseArgs(argc, argv);

  // Ring is heap-allocated once at startup (never on the hot path) because it
  // is far too large for a stack frame.
  auto ring = std::make_unique<Ring>();

  auto logger = BuildLogger(cfg);
  oep::external::EventRelay<Ring> relay(*ring, *logger);
  relay.Start();

  oep::core::OrderBook book(kBookCapacity);
  for (oep::core::Price p = kMinPrewarmPx; p <= kMaxPrewarmPx; ++p) {
    book.PrewarmLevel(oep::core::Side::kBuy, p);
    book.PrewarmLevel(oep::core::Side::kSell, p);
  }

  // Production TSC clock: calibrates in-ctor, then `operator()` is one rdtscp
  // plus a scalar float multiply. Zero allocation, zero syscalls on the hot
  // path — exactly what `TelemetryPublisher::ClockFn` promised.
  oep::core::TscNanoClock clock;
  oep::exec::TelemetryPublisher publisher(book, *ring, clock);

  Xorshift64 rng(cfg.seed);
  std::vector<oep::core::OrderId> resting;
  resting.reserve(cfg.ops + 16);
  oep::core::OrderId next_id = 1;
  std::size_t adds = 0;
  std::size_t cancels = 0;

  const auto wall_start = std::chrono::steady_clock::now();

  for (std::size_t i = 0; i < cfg.ops; ++i) {
    // 1-in-4 cancel of a random resting order (rotating victim keeps depth
    // stable and exercises the mid-FIFO cancel path).
    if (!resting.empty() && ((rng.Next() % 4ULL) == 0ULL)) {
      const std::size_t idx =
          static_cast<std::size_t>(rng.Next() % resting.size());
      std::swap(resting[idx], resting.back());
      const oep::core::OrderId victim = resting.back();
      resting.pop_back();
      (void)publisher.Cancel(victim);
      ++cancels;
      continue;
    }

    // Defensive: if the slab is getting close to full, evict before adding so
    // we never hit the AddLimit(→false) slab-exhaustion path in normal runs.
    if (book.open_order_count() >= kBookCapacity - 4) {
      if (!resting.empty()) {
        const oep::core::OrderId victim = resting.back();
        resting.pop_back();
        (void)publisher.Cancel(victim);
        ++cancels;
      }
      continue;
    }

    const bool buy = (rng.Next() & 1ULL) != 0ULL;
    const oep::core::Side side =
        buy ? oep::core::Side::kBuy : oep::core::Side::kSell;
    // Offset ∈ [-5, +5]; buy quotes centre on 99 and sell on 101 so the
    // upper half of the buy distribution crosses the ask ladder.
    const oep::core::Price offset =
        static_cast<oep::core::Price>(rng.Next() % 11ULL) - 5;
    const oep::core::Price px = buy ? (99 + offset) : (101 + offset);
    const oep::core::Quantity qty = 1 + (rng.Next() % 4ULL);
    const oep::core::OrderId id = next_id++;
    if (publisher.AddLimit(id, side, px, qty)) {
      resting.push_back(id);
      ++adds;
    }
  }

  const auto wall_end = std::chrono::steady_clock::now();
  const auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(wall_end - wall_start)
          .count();

  // Graceful teardown: Stop() runs one final drain pass after observing the
  // stop flag, so anything the producer wrote between the last relay sweep
  // and this call is still logged and Flush()'d before the thread joins.
  // On the IPC logger, Flush() ships the last accumulated batch to KDB+.
  relay.Stop();

  // Final Flush is a belt-and-braces safety net for a hypothetical relay
  // implementation that only flushes on shutdown; harmless with the current
  // in-memory + IPC loggers.
  logger->Flush();

  std::cout << "sim_runner: ops=" << cfg.ops
            << " adds=" << adds
            << " cancels=" << cancels
            << " events=" << publisher.next_sequence()
            << " processed=" << relay.processed()
            << " dropped=" << ring->dropped_messages()
            << " elapsed_ms=" << elapsed_ms << '\n';

  return 0;
}
