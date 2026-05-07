#!/usr/bin/env bash
# Compare NCCL_ALGO=TREE vs RING vs default (NCCL chooses) on the
# random_sample_001 topology, both with NCCL_BLINK=0 and =1.
# This isolates "Blink trees vs. NCCL trees" from "Blink trees vs. NCCL rings".
set -euo pipefail
ROOT=/home/bq/projects/alex-bill-micah/nccl-blink
LIB="$ROOT/build/lib"
ARP="$ROOT/benchmarks/nccl-tests/build/all_reduce_perf"
TOPO="$ROOT/benchmarks/results/topology_search/20260412T221537.513412Z/topologies/random_sample_001.xml"

OUT="$ROOT/benchmarks/results/algo_compare"
mkdir -p "$OUT"

run_one() {
  local algo="$1"
  local blink="$2"
  local rep="$3"
  local outlog="$OUT/algo${algo}_blink${blink}_rep${rep}.log"
  echo "--- ALGO=$algo BLINK=$blink rep=$rep ---"
  local algo_env=()
  if [ "$algo" != "default" ]; then
    algo_env=( "NCCL_ALGO=$algo" )
  fi
  env LD_LIBRARY_PATH="$LIB" \
      NCCL_BLINK="$blink" \
      NCCL_TOPO_FILE="$TOPO" \
      NCCL_IGNORE_DISABLED_P2P=2 \
      NCCL_IGNORE_CPU_AFFINITY=1 \
      "${algo_env[@]}" \
      "$ARP" -b 1M -e 1G -f 2 -w 5 -n 20 -g 8 \
      > "$outlog" 2>&1 || echo "  (exit $?)"
  grep -E 'Avg bus bandwidth' "$outlog" | head -1
}

for algo in default Tree Ring; do
  for blink in 0 1; do
    # Skip Blink with non-default algo (Blink only affects tree graph)
    if [ "$algo" = "Ring" ] && [ "$blink" = "1" ]; then continue; fi
    for rep in 0 1; do
      run_one "$algo" "$blink" "$rep"
    done
  done
done
echo "Done"
