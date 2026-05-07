#!/usr/bin/env bash
# Full size sweep (1MB-1GB) for NVLS algo on random_sample_001
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
  env LD_LIBRARY_PATH="$LIB" \
      NCCL_BLINK="$blink" \
      NCCL_TOPO_FILE="$TOPO" \
      NCCL_IGNORE_DISABLED_P2P=2 \
      NCCL_IGNORE_CPU_AFFINITY=1 \
      NCCL_ALGO="$algo" \
      "$ARP" -b 1M -e 1G -f 2 -w 5 -n 20 -g 8 \
      > "$outlog" 2>&1 || echo "  (exit $?)"
  grep -E 'Avg bus bandwidth' "$outlog" || true
}

# Add NVLS to algo_compare (already have default, Tree, Ring)
for blink in 0 1; do
  for rep in 0 1; do
    run_one NVLS "$blink" "$rep"
  done
done
echo "Done"
