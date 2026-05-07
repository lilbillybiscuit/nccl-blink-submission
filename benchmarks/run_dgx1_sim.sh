#!/usr/bin/env bash
# Run all_reduce_perf with NCCL_TOPO_FILE override pointed at the simulated
# DGX-1 V100 topologies. This emulates Blink's original target hardware on
# H200 silicon. We compare NCCL_BLINK=0 vs 1 across 1MB-1GB.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIB="$ROOT/build/lib"
ARP="$SCRIPT_DIR/nccl-tests/build/all_reduce_perf"
TOPO_DIR="$SCRIPT_DIR/topologies"

OUT="$SCRIPT_DIR/results/dgx1_sim"
mkdir -p "$OUT"

run_one() {
  local label="$1"
  local topo="$2"
  local nGpus="$3"
  local cvd="$4"
  local blink="$5"
  local outlog="$OUT/${label}_blink${blink}.log"
  echo "--- ${label} BLINK=${blink} (g=${nGpus}, cvd=${cvd}) ---"
  CUDA_VISIBLE_DEVICES="$cvd" \
    LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}" \
    NCCL_BLINK="$blink" \
    NCCL_DEBUG=INFO \
    NCCL_DEBUG_SUBSYS=GRAPH \
    NCCL_TOPO_FILE="$topo" \
    NCCL_IGNORE_DISABLED_P2P=2 \
    NCCL_IGNORE_CPU_AFFINITY=1 \
    "$ARP" -b 1M -e 1G -f 2 -w 5 -n 20 -g "$nGpus" \
    > "$outlog" 2>&1 || echo "  (exit $?)"
  grep -E 'Avg bus bandwidth' "$outlog" || echo "  (no avg bw line)"
}

# DGX-1 simulated allocations:
#   3 GPUs: ranks 0,1,4
#   5 GPUs: ranks 0,1,2,3,5
#   6 GPUs: ranks 0,1,2,3,5,6
#   8 GPUs: ranks 0..7
for blink in 0 1; do
  run_one "dgx1_3gpu" "$TOPO_DIR/dgx1_3gpu.xml" 3 "0,1,4" "$blink"
  run_one "dgx1_5gpu" "$TOPO_DIR/dgx1_5gpu.xml" 5 "0,1,2,3,5" "$blink"
  run_one "dgx1_6gpu" "$TOPO_DIR/dgx1_6gpu.xml" 6 "0,1,2,3,5,6" "$blink"
  run_one "dgx1_8gpu" "$TOPO_DIR/dgx1_8gpu.xml" 8 "0,1,2,3,4,5,6,7" "$blink"
done
