/ OEP ticker-plant schema.
/
/ Pinned to `oep::external::kCsvHeader` and the column order emitted by
/ `IpcKdbLogger::SendBatchLocked` (cpp/external/ipc_kdb_logger.cpp). The
/ C++ side sends one async `.u.upd[`trade; (type;seq;ts_ns;order_id;
/ counter_id;price;qty;side)]` per Flush(); the columns are, in order:
/
/   type       KS  symbol   fill / cancel / ack
/   seq        KJ  long     producer-side monotone sequence
/   ts_ns      KJ  long     internal (TSC) clock, nanoseconds
/   order_id   KJ  long     taker / cancelled / acked order id
/   counter_id KJ  long     resting maker id on fills, else 0
/   price      KJ  long     integer ticks (scaled fixed-point)
/   qty        KJ  long     fill / residual / cancelled qty
/   side       KS  symbol   buy / sell
/
/ The wall-clock stamp on landing is added by the ticker plant / RDB
/ (`.z.p`) rather than the C++ side, giving both clocks on the tape for
/ the P4 TCA replay pipeline.

trade:([] type:`symbol$(); seq:`long$(); ts_ns:`long$();
        order_id:`long$(); counter_id:`long$();
        price:`long$(); qty:`long$(); side:`symbol$())

/ Minimal `.u.upd` so this script is runnable standalone (no full
/ Tick / TP required for a smoke test of the IpcKdbLogger IPC frame).
/ Signature matches the standard kdb+ tick convention: table name (as
/ symbol) + a list of column vectors, insert as-is.
.u.upd:{[t;x] t insert x}
