#!/usr/bin/env bash
# Sweep BLINK_EPSILON over the original random_sample_001 topology
# (the one that delivers a 2.61x win at 1 GiB), measuring at each ε:
#   - bus bandwidth at 1 GiB
#   - selected_trees count from the Blink debug log
#   - total init wall time
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIB="$ROOT/build/lib"
ARP="$SCRIPT_DIR/nccl-tests/build/all_reduce_perf"
TOPO="$SCRIPT_DIR/results/topology_search/20260412T221537.513412Z/topologies/random_sample_001.xml"

OUT="$SCRIPT_DIR/results/epsilon_ablation"
mkdir -p "$OUT"

run_one() {
  local eps="$1"
  local rep="$2"
  local outlog="$OUT/eps${eps}_rep${rep}.log"
  echo "--- BLINK_EPSILON=$eps rep=$rep ---"
  LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}" \
    NCCL_BLINK=1 \
    BLINK_EPSILON="$eps" \
    NCCL_DEBUG=INFO \
    NCCL_DEBUG_SUBSYS=GRAPH \
    NCCL_TOPO_FILE="$TOPO" \
    NCCL_IGNORE_DISABLED_P2P=2 \
    NCCL_IGNORE_CPU_AFFINITY=1 \
    "$ARP" -b 512M -e 1G -f 2 -w 5 -n 30 -g 8 \
    > "$outlog" 2>&1 || echo "  (exit $?)"
  grep -E 'Avg bus bandwidth|Blink (timing|MWU|refine|computed)|BLINK_EPSILON override' "$outlog" | head -10
}

# Sweep ε
for eps in 0.05 0.075 0.1 0.15 0.2 0.3; do
  for rep in 0 1 2; do
    run_one "$eps" "$rep"
  done
done

echo "Done. Results in $OUT/"
