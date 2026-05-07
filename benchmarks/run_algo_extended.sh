#!/usr/bin/env bash
# Test NCCL_ALGO=NVLS, NCCL_ALGO=NVLSTree, etc. to find which algo
# NCCL is choosing when Blink is on
set -euo pipefail
ROOT=/home/bq/projects/alex-bill-micah/nccl-blink
LIB="$ROOT/build/lib"
ARP="$ROOT/benchmarks/nccl-tests/build/all_reduce_perf"
TOPO="$ROOT/benchmarks/results/topology_search/20260412T221537.513412Z/topologies/random_sample_001.xml"

OUT="$ROOT/benchmarks/results/algo_extended"
mkdir -p "$OUT"

run_one() {
  local algo="$1"
  local blink="$2"
  local rep="$3"
  local outlog="$OUT/algo${algo}_blink${blink}_rep${rep}.log"
  local algo_env=()
  if [ "$algo" != "default" ]; then
    algo_env=( "NCCL_ALGO=$algo" )
  fi
  echo "--- ALGO=$algo BLINK=$blink rep=$rep ---"
  env LD_LIBRARY_PATH="$LIB" \
      NCCL_BLINK="$blink" \
      NCCL_DEBUG=INFO \
      NCCL_DEBUG_SUBSYS=GRAPH \
      NCCL_TOPO_FILE="$TOPO" \
      NCCL_IGNORE_DISABLED_P2P=2 \
      NCCL_IGNORE_CPU_AFFINITY=1 \
      "${algo_env[@]}" \
      "$ARP" -b 512M -e 1G -f 2 -w 3 -n 10 -g 8 \
      > "$outlog" 2>&1 || echo "  (exit $?)"
  grep -E 'algo selected|^[ ]*1073741824|Avg bus' "$outlog" 2>/dev/null | tail -3
}

for algo in default Tree Ring NVLS NVLSTree CollnetDirect CollnetChain; do
  for blink in 0 1; do
    if [ "$algo" = "Ring" ] && [ "$blink" = "1" ]; then continue; fi
    run_one "$algo" "$blink" 0
  done
done

echo "Done"
