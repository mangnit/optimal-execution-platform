// Concrete KDB+ IPC logger (docs/architecture.md §2, task P2_kdb_ipc).
//
// Sits on the external clock and drains telemetry events into a live KDB+
// ticker plant via KX Systems' C client (`k.h` + `c.o`). Behaviour:
//
//   * `Log()` — buffers the event into per-column vectors. Cheap, mutex-guarded.
//   * `Flush()` — if a socket is open and the batch is non-empty, builds a
//     single `.u.upd[`<table>; <cols>]` async IPC frame and pushes it. On send
//     failure the socket is closed and the batch is preserved so a later
//     `Connect()` + `Flush()` can retry without dropping tape.
//   * `Connect()` / `Disconnect()` — idempotent connection management via
//     `khpu()`; failures return false rather than throwing so the relay thread
//     never sees an exception cross the KdbLogger boundary.
//
// Schema on the wire matches `kCsvHeader` from `kdb_logger.hpp` exactly:
//
//     type    (`FILL/`CANCEL/`ACK)          — symbol
//     seq                                   — long
//     ts_ns    (internal-clock nanoseconds) — long (nanos since epoch source)
//     order_id                              — long
//     counter_id                            — long
//     price                                 — long
//     qty                                   — long
//     side    (`BUY/`SELL)                  — symbol
//
// `q/schema.q` pins the receiving ticker plant to the same column order.
//
// The header intentionally does NOT `#include "k.h"` — that file leaks a slew
// of single-letter preprocessor macros (`R` for `return`, `Z` for `static`,
// `P(x,y)`, etc.) which we absolutely do not want polluting every translation
// unit that consumes an oep::external interface. All K-side manipulation is
// isolated to `ipc_kdb_logger.cpp`.

#ifndef OEP_EXTERNAL_IPC_KDB_LOGGER_HPP_
#define OEP_EXTERNAL_IPC_KDB_LOGGER_HPP_

#include "external/kdb_logger.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace oep::external {

class IpcKdbLogger : public KdbLogger {
 public:
  // `table_name` is the q table `.u.upd` publishes to (default matches
  // `q/schema.q`). `credentials` is the `user:password` string handed to
  // `khpu()`; empty means "no auth". No connection is opened here — call
  // `Connect()` explicitly so the relay start-up path can decide what to do
  // if the ticker plant is offline.
  IpcKdbLogger(std::string host, int port,
               std::string table_name = "trade",
               std::string credentials = "");

  ~IpcKdbLogger() override;

  IpcKdbLogger(const IpcKdbLogger&) = delete;
  IpcKdbLogger& operator=(const IpcKdbLogger&) = delete;
  IpcKdbLogger(IpcKdbLogger&&) = delete;
  IpcKdbLogger& operator=(IpcKdbLogger&&) = delete;

  // Opens the socket via `khpu()`. Idempotent — a successful call while
  // already connected returns true without touching the handle. Returns false
  // if the ticker plant is unreachable or rejects the credentials.
  bool Connect();

  // Closes the socket. Idempotent. Any batched-but-unflushed events remain
  // queued in memory so a subsequent `Connect()` + `Flush()` can still send
  // them.
  void Disconnect();

  [[nodiscard]] bool connected() const noexcept;

  // KdbLogger interface. Both are safe to call from the relay thread only
  // (single-consumer), but the internal mutex means an out-of-thread stats
  // read is safe too.
  void Log(const TelemetryEvent& event) override;
  void Flush() override;

  // Test / observability accessors.
  [[nodiscard]] std::size_t pending_count() const;
  [[nodiscard]] std::uint64_t sent_frames() const noexcept {
    return sent_frames_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::uint64_t send_failures() const noexcept {
    return send_failures_.load(std::memory_order_relaxed);
  }

 private:
  // Both require `mu_` held. `SendBatchLocked` returns true on success; on
  // failure the caller is expected to close the handle and preserve the
  // batch. `ClearBatchLocked` resets all per-column vectors.
  bool SendBatchLocked();
  void ClearBatchLocked() noexcept;

  std::string host_;
  int port_;
  std::string table_name_;
  std::string credentials_;

  mutable std::mutex mu_;
  int handle_ = 0;  // KDB socket; `> 0` open, `<= 0` closed

  // Per-column batches. Kept as native C++ types (not raw K objects) so this
  // header is `k.h`-free and Log() stays cheap in the common case.
  std::vector<std::string> type_batch_;
  std::vector<std::int64_t> seq_batch_;
  std::vector<std::int64_t> ts_ns_batch_;
  std::vector<std::int64_t> order_id_batch_;
  std::vector<std::int64_t> counter_id_batch_;
  std::vector<std::int64_t> price_batch_;
  std::vector<std::int64_t> qty_batch_;
  std::vector<std::string> side_batch_;

  std::atomic<std::uint64_t> sent_frames_{0};
  std::atomic<std::uint64_t> send_failures_{0};
};

}  // namespace oep::external

#endif  // OEP_EXTERNAL_IPC_KDB_LOGGER_HPP_
