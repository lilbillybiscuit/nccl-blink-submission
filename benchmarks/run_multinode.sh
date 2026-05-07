#!/usr/bin/env bash
# 2-host multinode AllReduce sweep across gpu003 + gpu006 (8 GPUs each = 16
# H200 ranks total). Uses MPICH + the _mpich nccl-tests binary.
# Compares NCCL_BLINK=0 vs 1 across full message sweep.
set -euo pipefail
ROOT=/home/bq/projects/alex-bill-micah/nccl-blink
LIB="$ROOT/build/lib"
ARP="$ROOT/benchmarks/nccl-tests/build/all_reduce_perf_mpich"

OUT="$ROOT/benchmarks/results/multinode"
mkdir -p "$OUT"

# Inter-host network interface (172.16.8.x subnet, /22)
NET_IF=ens99f0

run_one() {
  local nproc="$1"      # total ranks
  local ppn="$2"        # ranks per node
  local blink="$3"
  local rep="$4"
  local label="${nproc}rk_${ppn}ppn"
  local outlog="$OUT/${label}_blink${blink}_rep${rep}.log"
  echo "--- $label BLINK=$blink rep=$rep ---"
  mpirun -hosts 172.16.8.33,172.16.8.36 -n "$nproc" -ppn "$ppn" \
    -genv LD_LIBRARY_PATH "$LIB" \
    -genv NCCL_BLINK "$blink" \
    -genv NCCL_SOCKET_IFNAME "$NET_IF" \
    -genv NCCL_DEBUG WARN \
    "$ARP" -b 1M -e 1G -f 2 -g 1 -w 5 -n 20 \
    > "$outlog" 2>&1 || echo "  (exit $?)"
  grep -E 'Avg bus bandwidth' "$outlog" || tail -3 "$outlog"
}

# Vary scale: 2 ranks (1 GPU/host), 4 ranks (2/host), 8 (4/host), 16 (8/host)
for ppn in 1 2 4 8; do
  nproc=$((ppn * 2))
  for blink in 0 1; do
    for rep in 0 1; do
      run_one "$nproc" "$ppn" "$blink" "$rep"
    done
  done
done
echo "Done"
