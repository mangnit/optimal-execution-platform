#!/usr/bin/env python3
"""CI p99 regression gate for the internal-clock hot path.

Compares the p99_ns counter of each benchmark in a *current* Google
Benchmark JSON against a committed *baseline* JSON and fails (exit 1) on
regression. A benchmark regresses when

    current_p99 > baseline_p99 * (1 + tol_pct) + tol_abs_ns

The relative term catches real slowdowns; the absolute slack keeps
single-digit-nanosecond benchmarks (where one cache miss is >30% of the
budget) from tripping on scheduler noise.

Which benchmarks: by default every benchmark name present in BOTH files is
gated — on a torch-enabled build that includes BM_Fused_ArrivalToObs (the
headline arrival -> LOB -> observation path); on a CI build without
LibTorch the gate covers the core component paths (order book add/cancel/
sweep, SPSC, slab). Pass --benchmarks to pin an explicit list; a requested
name missing from either file is itself a failure, so the gate cannot rot
silently.

Cross-host honesty: absolute nanoseconds are only comparable on the same
hardware. With --advisory-on-host-mismatch, if context.host_name differs
between baseline and current the script reports the comparison but always
exits 0 — CI stays green on shared runners until a baseline captured on
that runner class is committed (download the benchmark-json artifact from
a trusted run and commit it under benchmarks/ci_baseline/ to arm the gate
strictly).

Usage:
  check_latency_regression.py BASELINE.json CURRENT.json
      [--benchmarks NAME[,NAME...]] [--tol-pct 0.15] [--tol-abs-ns 15]
      [--advisory-on-host-mismatch]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> tuple[str, dict[str, float]]:
    """Return (host_name, {benchmark_name: p99_ns}) from a GB JSON file."""
    with open(path) as f:
        doc = json.load(f)
    host = doc.get("context", {}).get("host_name", "<unknown>")
    p99s: dict[str, float] = {}
    for bench in doc.get("benchmarks", []):
        if bench.get("run_type") == "aggregate":
            continue  # gate on raw runs, not mean/median/stddev aggregates
        name = bench.get("name")
        p99 = bench.get("p99_ns")
        if name is not None and isinstance(p99, (int, float)):
            p99s[name] = float(p99)
    return host, p99s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("baseline", type=Path)
    ap.add_argument("current", type=Path)
    ap.add_argument(
        "--benchmarks",
        help="comma-separated benchmark names to gate "
        "(default: all names present in both files)",
    )
    ap.add_argument("--tol-pct", type=float, default=0.15,
                    help="relative degradation tolerance (default 0.15 = 15%%)")
    ap.add_argument("--tol-abs-ns", type=float, default=15.0,
                    help="absolute slack in ns added on top (default 15)")
    ap.add_argument(
        "--advisory-on-host-mismatch",
        action="store_true",
        help="report but never fail when baseline/current host_name differ",
    )
    args = ap.parse_args()

    base_host, base = load(args.baseline)
    cur_host, cur = load(args.current)

    if args.benchmarks:
        names = [n.strip() for n in args.benchmarks.split(",") if n.strip()]
        missing = [n for n in names if n not in base or n not in cur]
        if missing:
            print(f"FAIL: requested benchmark(s) missing from input: {missing}")
            print(f"  baseline has: {sorted(base)}")
            print(f"  current  has: {sorted(cur)}")
            return 1
    else:
        names = sorted(set(base) & set(cur))
        if not names:
            print("FAIL: no common benchmarks between baseline and current")
            return 1

    advisory = args.advisory_on_host_mismatch and base_host != cur_host
    if advisory:
        print(f"NOTE: host mismatch (baseline={base_host!r}, "
              f"current={cur_host!r}) — advisory mode, gate will not fail.")
        print("      To arm strictly, commit a baseline captured on this "
              "runner class (see benchmark-json CI artifact).")

    failures = 0
    print(f"{'benchmark':40s} {'baseline':>10s} {'current':>10s} "
          f"{'limit':>10s}  verdict")
    for name in names:
        b, c = base[name], cur[name]
        limit = b * (1.0 + args.tol_pct) + args.tol_abs_ns
        regressed = c > limit
        verdict = "REGRESSED" if regressed else "ok"
        print(f"{name:40s} {b:9.1f}ns {c:9.1f}ns {limit:9.1f}ns  {verdict}")
        if regressed:
            failures += 1

    if failures and not advisory:
        print(f"\nFAIL: {failures}/{len(names)} benchmark(s) exceeded "
              f"p99 limit (tol {args.tol_pct:.0%} + {args.tol_abs_ns:.0f}ns)")
        return 1
    if failures and advisory:
        print(f"\nADVISORY: {failures}/{len(names)} would have failed on "
              f"same-host hardware — investigate before trusting this run.")
        return 0
    print(f"\nOK: {len(names)} benchmark(s) within p99 limit "
          f"(tol {args.tol_pct:.0%} + {args.tol_abs_ns:.0f}ns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
