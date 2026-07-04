#!/usr/bin/env python3
"""Render docs/latency_report.md from Google Benchmark JSON outputs.

Consumes the per-binary JSON files produced by `scripts/run_benchmarks.sh`
and emits a markdown report with one table per binary showing the p50, p99,
and p99.9 tail latencies (in nanoseconds) that our benchmarks stash into
`counters.p50_ns` / `p99_ns` / `p999_ns`.

Also captures min/max, sample count, and any auxiliary counters
(e.g. `dropped`, `fills_per_iter`) so the report is self-describing enough
to survive a "which of these is the real measurement" review.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
from typing import Any


TAIL_COUNTERS = ("p50_ns", "p99_ns", "p999_ns", "min_ns", "max_ns", "samples")


def fmt_ns(v: float) -> str:
    if v < 1_000:
        return f"{v:.1f} ns"
    if v < 1_000_000:
        return f"{v / 1_000:.2f} µs"
    return f"{v / 1_000_000:.2f} ms"


def render_binary_section(path: pathlib.Path) -> str:
    data = json.loads(path.read_text())
    ctx = data.get("context", {})
    benches = data.get("benchmarks", [])

    lines: list[str] = []
    lines.append(f"### `{path.stem}`")
    lines.append("")

    # Environment / context row.
    ctx_bits = [
        f"host=`{ctx.get('host_name', '?')}`",
        f"cpu=`{ctx.get('mhz_per_cpu', '?')} MHz × {ctx.get('num_cpus', '?')}`",
        f"cache-scaled=`{ctx.get('cpu_scaling_enabled', '?')}`",
    ]
    for extra in ("pinned_cpu", "pinned_cpu_producer", "pinned_cpu_consumer"):
        if extra in ctx:
            ctx_bits.append(f"{extra}=`{ctx[extra]}`")
    lines.append(" · ".join(ctx_bits))
    lines.append("")

    # Latency table.
    lines.append(
        "| Benchmark | p50 | p99 | p99.9 | min | max | samples | extra |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---|"
    )
    # Google Benchmark flattens user-defined counters at the top level of
    # each benchmark object rather than nesting them under `counters`, so we
    # sift the benchmark dict directly and treat every non-metadata numeric
    # field as a counter.
    METADATA_KEYS = {
        "name", "family_index", "per_family_instance_index", "run_name",
        "run_type", "repetitions", "repetition_index", "threads",
        "iterations", "real_time", "cpu_time", "time_unit", "label",
        "error_occurred", "error_message", "aggregate_name",
        "aggregate_unit", "big_o", "rms",
    }
    for b in benches:
        def g(k: str) -> float:
            v = b.get(k)
            if v is None:
                return float("nan")
            if isinstance(v, dict):
                return float(v.get("value", 0.0))
            return float(v)

        extras = []
        for k, v in b.items():
            if k in METADATA_KEYS or k in TAIL_COUNTERS:
                continue
            if not isinstance(v, (int, float, dict)):
                continue
            val = v if not isinstance(v, dict) else v.get("value", 0.0)
            try:
                extras.append(f"{k}={float(val):.3g}")
            except (TypeError, ValueError):
                continue
        extras_str = ", ".join(extras) if extras else ""

        samples = g("samples")
        samples_str = f"{int(samples):,}" if samples == samples else "—"

        lines.append(
            "| `{name}` | {p50} | {p99} | {p999} | {mn} | {mx} | {n} | {ex} |".format(
                name=b.get("name", "?"),
                p50=fmt_ns(g("p50_ns")),
                p99=fmt_ns(g("p99_ns")),
                p999=fmt_ns(g("p999_ns")),
                mn=fmt_ns(g("min_ns")),
                mx=fmt_ns(g("max_ns")),
                n=samples_str,
                ex=extras_str,
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", nargs="+", required=True,
                    help="Google Benchmark JSON output files")
    ap.add_argument("--output", required=True,
                    help="Path to write the rendered markdown report")
    args = ap.parse_args()

    inputs = [pathlib.Path(p) for p in args.input]
    for p in inputs:
        if not p.exists():
            print(f"missing input: {p}", file=sys.stderr)
            return 1

    generated_at = dt.datetime.now(dt.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    header = f"""# Latency Report — Internal Clock

Generated {generated_at} from `scripts/run_benchmarks.sh`.

Every benchmark below runs on the **internal clock** (docs/architecture.md §7):
the matching path, its allocator, and the SPSC ring that bridges it to the
relay. FIX, kdb writes, and dashboard IO live on the external clock and are
deliberately not measured here.

Percentiles are computed from per-iteration `rdtscp` samples, converted to
nanoseconds via a boot-time TSC-frequency calibration. `samples` is the
number of per-op observations backing each row; p99.9 is only trustworthy
when that count is well above a thousand.

The samples are wall-clock timings on whatever CPU the harness landed on;
without core isolation and CPU-frequency pinning, the tail includes
scheduling jitter. Treat the numbers as a baseline for regression tracking,
not as a marketing claim (per §7's "no pre-committed numbers" rule).

"""

    body = "\n".join(render_binary_section(p) for p in inputs)
    pathlib.Path(args.output).write_text(header + body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
