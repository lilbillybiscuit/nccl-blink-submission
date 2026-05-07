#!/usr/bin/env bash
# Reproducibility check: re-run the random_sample_001 win on gpu006
# (same hardware, same shared mount, different host) to confirm the
# 2.61x speedup is a property of the topology, not of gpu003.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIB="$ROOT/build/lib"
ARP="$SCRIPT_DIR/nccl-tests/build/all_reduce_perf"
TOPO="$SCRIPT_DIR/results/topology_search/20260412T221537.513412Z/topologies/random_sample_001.xml"

OUT="$SCRIPT_DIR/results/gpu006_validation"
mkdir -p "$OUT"

run_one() {
  local blink="$1"
  local rep="$2"
  local outlog="$OUT/blink${blink}_rep${rep}.log"
  echo "--- BLINK=$blink rep=$rep ---"
  LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}" \
    NCCL_BLINK="$blink" \
    NCCL_TOPO_FILE="$TOPO" \
    NCCL_IGNORE_DISABLED_P2P=2 \
    NCCL_IGNORE_CPU_AFFINITY=1 \
    "$ARP" -b 256M -e 1G -f 2 -w 5 -n 30 -g 8 \
    > "$outlog" 2>&1 || echo "  (exit $?)"
  grep -E 'Avg bus bandwidth' "$outlog" || tail -2 "$outlog"
}

for rep in 0 1 2 3 4; do
  run_one 0 "$rep"
  run_one 1 "$rep"
done
