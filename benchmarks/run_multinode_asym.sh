#!/usr/bin/env bash
# 16-rank multinode using the random_sample_001 asymmetric NVLink topology
# applied per-host. Tests whether STARE's intra-host packing wins multinode
# when the intra-host fabric is asymmetric.
set -euo pipefail
ROOT=/home/bq/projects/alex-bill-micah/nccl-blink
LIB="$ROOT/build/lib"
ARP="$ROOT/benchmarks/nccl-tests/build/all_reduce_perf_mpich"
TOPO="$ROOT/benchmarks/results/topology_search/20260412T221537.513412Z/topologies/random_sample_001.xml"

OUT="$ROOT/benchmarks/results/multinode_asym"
mkdir -p "$OUT"

run_one() {
  local blink="$1"
  local rep="$2"
  local outlog="$OUT/blink${blink}_rep${rep}.log"
  echo "--- BLINK=$blink rep=$rep ---"
  mpirun -hosts 172.16.8.33,172.16.8.36 -n 16 -ppn 8 \
    -genv LD_LIBRARY_PATH "$LIB" \
    -genv NCCL_BLINK "$blink" \
    -genv NCCL_TOPO_FILE "$TOPO" \
    -genv NCCL_IGNORE_DISABLED_P2P 2 \
    -genv NCCL_IGNORE_CPU_AFFINITY 1 \
    -genv NCCL_SOCKET_IFNAME ens99f0 \
    -genv NCCL_DEBUG WARN \
    "$ARP" -b 1M -e 4G -f 2 -g 1 -w 5 -n 30 \
    > "$outlog" 2>&1 || echo "  (exit $?)"
  grep -E '^# Avg bus|^[ ]*1073741824' "$outlog" | head -3
}
for rep in 0 1 2; do
  for blink in 0 1; do
    run_one "$blink" "$rep"
  done
done
echo "Done"
