#!/usr/bin/env bash
# Run a tiny AllReduce on each of a representative set of topologies
# with NCCL_DEBUG=GRAPH enabled, so we capture the
# `Blink timing: extract=Xus MWU=Xus refine=Xus chain=Xus total=Xus`
# line that ncclBlinkCompute emits per init.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIB="$ROOT/build/lib"
ARP="$SCRIPT_DIR/nccl-tests/build/all_reduce_perf"
SEARCH_TOPOS="$SCRIPT_DIR/results/topology_search/20260412T221537.513412Z/topologies"

OUT="$SCRIPT_DIR/results/solver_overhead"
mkdir -p "$OUT"

run_one() {
  local label="$1"
  local topo="$2"
  local outlog="$OUT/${label}.log"
  echo "--- ${label} ---"
  LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}" \
    NCCL_BLINK=1 \
    NCCL_DEBUG=INFO \
    NCCL_DEBUG_SUBSYS=GRAPH \
    NCCL_TOPO_FILE="$topo" \
    NCCL_IGNORE_DISABLED_P2P=2 \
    NCCL_IGNORE_CPU_AFFINITY=1 \
    "$ARP" -b 1M -e 1M -f 2 -w 1 -n 1 -g 8 \
    > "$outlog" 2>&1 || echo "  (exit $?)"
  grep -E 'Blink (timing|MWU|refine|extracted|computed)' "$outlog" || echo "  no Blink lines"
}

# Run base + every structured family + every random sample present
for f in "$SEARCH_TOPOS"/*.xml; do
  label=$(basename "$f" .xml)
  run_one "$label" "$f"
done
