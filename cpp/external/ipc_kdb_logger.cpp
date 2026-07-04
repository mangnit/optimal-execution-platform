// See ipc_kdb_logger.hpp for the interface contract. This TU is the only place
// in the codebase that touches `k.h` — everything below the `#include "k.h"`
// line is written knowing that k.h leaks single-letter preprocessor macros
// (`R`, `Z`, `P(x,y)`, `O`, ...). We deliberately keep this file small so
// nothing we depend on collides with them.

#include "external/ipc_kdb_logger.hpp"

// k.h already provides `extern "C"` when __cplusplus is defined, so a bare
// #include is correct here. The vendored header uses two C99isms that trip
// `-Wpedantic -Werror` under GCC (a flexible array member and an anonymous
// struct inside a union), so we disable pedantic warnings *only* across the
// include — every other TU in the project still compiles under full pedantic.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include "k.h"
#pragma GCC diagnostic pop

#include <utility>

namespace oep::external {

namespace {

// `ss` (intern-symbol) is declared as `S ss(const S)` where `S` is `char*` —
// i.e. `char* const`, not `const char*`. That's incompatible with a string
// literal / `.c_str()` under -Werror, so we const_cast at the boundary.
// `ss` copies into KDB's symbol table; it does not mutate the input.
inline char* AsMutable(const char* s) noexcept {
  return const_cast<char*>(s);
}
inline char* AsMutable(const std::string& s) noexcept {
  return const_cast<char*>(s.c_str());
}

}  // namespace

IpcKdbLogger::IpcKdbLogger(std::string host, int port,
                           std::string table_name, std::string credentials)
    : host_(std::move(host)),
      port_(port),
      table_name_(std::move(table_name)),
      credentials_(std::move(credentials)) {}

IpcKdbLogger::~IpcKdbLogger() {
  // Best-effort drain on teardown. Flush() is a no-op if nothing is queued or
  // the socket is closed; Disconnect() is idempotent.
  Flush();
  Disconnect();
}

bool IpcKdbLogger::Connect() {
  std::lock_guard<std::mutex> lock(mu_);
  if (handle_ > 0) return true;

  // `khpu(host, port, credentials)`: >0 on success; 0 = bad auth; -1 = socket
  // failure; -2 = timeout. We collapse all failures to "not connected" so the
  // relay just sees a boolean.
  char* creds = credentials_.empty() ? nullptr : AsMutable(credentials_);
  const int h = khpu(AsMutable(host_), port_, creds);
  if (h <= 0) {
    handle_ = 0;
    return false;
  }
  handle_ = h;
  return true;
}

void IpcKdbLogger::Disconnect() {
  std::lock_guard<std::mutex> lock(mu_);
  if (handle_ > 0) {
    kclose(handle_);
    handle_ = 0;
  }
}

bool IpcKdbLogger::connected() const noexcept {
  std::lock_guard<std::mutex> lock(mu_);
  return handle_ > 0;
}

void IpcKdbLogger::Log(const TelemetryEvent& event) {
  std::lock_guard<std::mutex> lock(mu_);
  // Symbol strings are the short fixed set from kdb_logger.hpp; ss() will
  // intern the same underlying pointer on every call, so the per-string
  // allocation cost is only paid here in the batch, not later on send.
  type_batch_.emplace_back(TelemetryEventTypeName(event.type));
  seq_batch_.push_back(static_cast<std::int64_t>(event.sequence));
  ts_ns_batch_.push_back(static_cast<std::int64_t>(event.timestamp_ns));
  order_id_batch_.push_back(static_cast<std::int64_t>(event.order_id));
  counter_id_batch_.push_back(static_cast<std::int64_t>(event.counter_id));
  price_batch_.push_back(static_cast<std::int64_t>(event.price));
  qty_batch_.push_back(static_cast<std::int64_t>(event.qty));
  side_batch_.emplace_back(SideName(event.side));
}

std::size_t IpcKdbLogger::pending_count() const {
  std::lock_guard<std::mutex> lock(mu_);
  return seq_batch_.size();
}

void IpcKdbLogger::Flush() {
  std::lock_guard<std::mutex> lock(mu_);
  if (seq_batch_.empty()) return;
  if (handle_ <= 0) {
    // No socket — leave the batch queued for the next Connect() + Flush().
    return;
  }
  if (SendBatchLocked()) {
    ClearBatchLocked();
    sent_frames_.fetch_add(1, std::memory_order_relaxed);
  } else {
    // Assume a network-level failure; drop the socket so a future Connect()
    // starts clean. Preserve the batch — we haven't handed it over.
    kclose(handle_);
    handle_ = 0;
    send_failures_.fetch_add(1, std::memory_order_relaxed);
  }
}

void IpcKdbLogger::ClearBatchLocked() noexcept {
  type_batch_.clear();
  seq_batch_.clear();
  ts_ns_batch_.clear();
  order_id_batch_.clear();
  counter_id_batch_.clear();
  price_batch_.clear();
  qty_batch_.clear();
  side_batch_.clear();
}

bool IpcKdbLogger::SendBatchLocked() {
  const J n = static_cast<J>(seq_batch_.size());

  // Build one KDB vector per column. `ktn(type, n)` allocates an n-element
  // typed vector; the accessor macros (`kS`, `kJ`) give us the raw pointer
  // for a tight fill loop.
  K col_type    = ktn(KS, n);
  K col_seq     = ktn(KJ, n);
  K col_ts      = ktn(KJ, n);  // internal-clock ns; a real ticker plant maps
                               // this to `.z.p` on landing, giving both clocks
                               // on the tape (§P4 TCA replay).
  K col_order   = ktn(KJ, n);
  K col_counter = ktn(KJ, n);
  K col_price   = ktn(KJ, n);
  K col_qty     = ktn(KJ, n);
  K col_side    = ktn(KS, n);

  for (J i = 0; i < n; ++i) {
    kS(col_type)[i]    = ss(AsMutable(type_batch_[i]));
    kJ(col_seq)[i]     = seq_batch_[i];
    kJ(col_ts)[i]      = ts_ns_batch_[i];
    kJ(col_order)[i]   = order_id_batch_[i];
    kJ(col_counter)[i] = counter_id_batch_[i];
    kJ(col_price)[i]   = price_batch_[i];
    kJ(col_qty)[i]     = qty_batch_[i];
    kS(col_side)[i]    = ss(AsMutable(side_batch_[i]));
  }

  // Wrap the columns in a mixed list, matching `kCsvHeader` column order.
  K cols = knk(8, col_type, col_seq, col_ts, col_order,
               col_counter, col_price, col_qty, col_side);

  // Async publish: negative handle, `.u.upd[`<table>; <cols>]`. The
  // ticker-plant convention is one-way — we get no reply, so the relay is
  // not stalled waiting on q-side processing. k() consumes its K arguments,
  // so we do NOT r0() `cols` or the individual column Ks.
  K r = k(-handle_, AsMutable(".u.upd"),
          ks(AsMutable(table_name_)), cols, static_cast<K>(0));

  // For an async call, k() returns `(K)0` on success. A non-null return with
  // type -128 signals a q-side error; anything else non-null is unexpected
  // but we still r0() it to be safe. A `(K)0` return on a positive-handle
  // (sync) call would mean a network failure, but we're always async here,
  // so we treat `!r` as the success path.
  if (r) {
    const bool err = (r->t == -128);
    r0(r);
    if (err) return false;
  }
  return true;
}

}  // namespace oep::external
