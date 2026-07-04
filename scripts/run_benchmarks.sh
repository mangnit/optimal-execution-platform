#!/usr/bin/env bash
# Runs the three internal-clock latency benchmarks and renders
# docs/latency_report.md from their JSON output.
#
# Usage:
#   scripts/run_benchmarks.sh                      # unpinned
#   OEP_BENCH_PIN_CPU=3 OEP_BENCH_PIN_CPU_CONSUMER=5 \
#     scripts/run_benchmarks.sh                    # pin producer + consumer
#
# The script does *not* rebuild automatically; call `cmake --build build` first.
# We keep the split so a benchmark iteration doesn't accidentally trigger a
# full rebuild in the middle of a measurement session.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
build_dir="${OEP_BENCH_BUILD_DIR:-$repo_root/build}"
report_dir="$repo_root/benchmarks"
mkdir -p "$report_dir"

# Minimum wall-clock per benchmark. 2s is enough for millions of iterations
# on the O(10ns) hot-path ops and gives a stable tail estimate.
min_time="${OEP_BENCH_MIN_TIME:-2.0s}"

run_one() {
  local name="$1"
  local out="$report_dir/${name}.json"
  echo "==> $name (min_time=$min_time)"
  "$build_dir/cpp/benchmarks/$name" \
    --benchmark_format=json \
    --benchmark_min_time="$min_time" \
    --benchmark_out="$out" \
    --benchmark_out_format=json
}

run_one slab_allocator_bench
run_one spsc_ring_bench
run_one order_book_bench

python3 "$repo_root/scripts/render_latency_report.py" \
  --input "$report_dir/slab_allocator_bench.json" \
          "$report_dir/spsc_ring_bench.json" \
          "$report_dir/order_book_bench.json" \
  --output "$repo_root/docs/latency_report.md"

echo "==> wrote $repo_root/docs/latency_report.md"
