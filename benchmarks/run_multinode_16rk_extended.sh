#!/usr/bin/env bash
# Targeted sweep: 16-rank (8 GPU/host) multinode with more repeats and
# extended message range out to 4 GiB to see whether STARE wins persist
# beyond the 1 GiB link-saturation point.
set -euo pipefail
ROOT=/home/bq/projects/alex-bill-micah/nccl-blink
LIB="$ROOT/build/lib"
ARP="$ROOT/benchmarks/nccl-tests/build/all_reduce_perf_mpich"

OUT="$ROOT/benchmarks/results/multinode_16rk_extended"
mkdir -p "$OUT"
NET_IF=ens99f0

run_one() {
  local blink="$1"
  local rep="$2"
  local outlog="$OUT/blink${blink}_rep${rep}.log"
  echo "--- BLINK=$blink rep=$rep ---"
  mpirun -hosts 172.16.8.33,172.16.8.36 -n 16 -ppn 8 \
    -genv LD_LIBRARY_PATH "$LIB" \
    -genv NCCL_BLINK "$blink" \
    -genv NCCL_SOCKET_IFNAME "$NET_IF" \
    -genv NCCL_DEBUG WARN \
    "$ARP" -b 1M -e 4G -f 2 -g 1 -w 5 -n 30 \
    > "$outlog" 2>&1 || echo "  (exit $?)"
  grep -E '^# Avg bus|^[ ]*4294967296' "$outlog" | head -3
}

for rep in 0 1 2 3 4; do
  for blink in 0 1; do
    run_one "$blink" "$rep"
  done
done
echo "Done"
