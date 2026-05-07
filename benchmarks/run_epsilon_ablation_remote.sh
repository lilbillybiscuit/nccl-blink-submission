#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/bq/projects/alex-bill-micah/nccl-blink
LIB="$ROOT/build/lib"
ARP="$ROOT/benchmarks/nccl-tests/build/all_reduce_perf"
TOPO="$ROOT/benchmarks/results/topology_search/20260412T221537.513412Z/topologies/random_sample_001.xml"
OUT="$ROOT/benchmarks/results/epsilon_ablation"
mkdir -p "$OUT"

run_one() {
  local eps="$1"
  local rep="$2"
  local outlog="$OUT/eps${eps}_rep${rep}.log"
  echo "--- BLINK_EPSILON=$eps rep=$rep ---"
  LD_LIBRARY_PATH="$LIB" \
    NCCL_BLINK=1 \
    BLINK_EPSILON="$eps" \
    NCCL_DEBUG=INFO \
    NCCL_DEBUG_SUBSYS=GRAPH \
    NCCL_TOPO_FILE="$TOPO" \
    NCCL_IGNORE_DISABLED_P2P=2 \
    NCCL_IGNORE_CPU_AFFINITY=1 \
    "$ARP" -b 1G -e 1G -f 2 -w 5 -n 30 -g 8 \
    > "$outlog" 2>&1 || echo "  (exit $?)"
  grep -E 'Avg bus bandwidth|selected.*trees|computed|timing' "$outlog" | head -3
}

# Sweep ε at 1 GiB (the size where the 2.61x win shows up)
for eps in 0.03 0.05 0.075 0.1 0.15 0.2 0.3 0.5; do
  for rep in 0 1 2; do
    run_one "$eps" "$rep"
  done
done

# Also baseline: BLINK=0
echo "--- BLINK=0 baseline ---"
for rep in 0 1 2; do
  outlog="$OUT/blink0_rep${rep}.log"
  LD_LIBRARY_PATH="$LIB" NCCL_BLINK=0 NCCL_TOPO_FILE="$TOPO" NCCL_IGNORE_DISABLED_P2P=2 NCCL_IGNORE_CPU_AFFINITY=1 \
    "$ARP" -b 1G -e 1G -f 2 -w 5 -n 30 -g 8 > "$outlog" 2>&1 || echo "  (exit $?)"
  grep 'Avg bus bandwidth' "$outlog"
done

echo "Done"
