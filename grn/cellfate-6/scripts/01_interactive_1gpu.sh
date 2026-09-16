#!/usr/bin/env bash
# Step 1 of the ladder: one GPU, interactive.  Run this by hand, not with sbatch.
#
#   srun --gpus-per-node=1 --nodes=1 --time=1:00:00 --pty bash
#   bash scripts/01_interactive_1gpu.sh
#
# Run it with `bash`, not `source`.  Sourcing a script that sets -e means any
# non-zero exit kills your interactive shell and with it the allocation.
#
# Purpose: prove correctness on the real device, then measure the single-GPU
# throughput that every subsequent projection depends on.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "Run this with 'bash scripts/01_interactive_1gpu.sh', not 'source'." >&2
  return 1
fi
set -uo pipefail

SCRATCH=/lus/lfs1aip2/scratch/u6pl/mat611.u6pl/grn/2-grn-gh200
RESULTS="${SCRATCH}/results/step1_1gpu"
mkdir -p "${RESULTS}"

module load cuda/12.6
# shellcheck disable=SC1091
source "${SCRATCH}/venv/bin/activate"

echo "=== correctness ==="
# Deliberately not fatal: a selftest failure is worth seeing alongside the
# throughput numbers rather than instead of them.  The gate is applied at the
# end, once everything has been measured.
cellfate selftest --device cuda:0 --walkers 200000 | tee "${RESULTS}/selftest.log"
SELFTEST_RC=${PIPESTATUS[0]}

echo
echo "=== throughput sweep ==="
# Sweep the walker count across three orders of magnitude.  The interesting
# question on a GH200 is how far past HBM the coherent NVLink-C2C host memory
# lets the walker array grow before throughput falls off; the walker population
# is the only large array in the code.
for W in 10000 100000 1000000 4000000 10000000; do
  echo "--- walkers=${W} ---"
  cellfate bench --device cuda:0 --model th17_treg --scale 3.0 \
      --walkers "${W}" --steps 200 --dtype float64 \
      --json "${RESULTS}/bench_fp64_w${W}.json"
done

echo
echo "=== float32 comparison at the saturating size ==="
cellfate bench --device cuda:0 --model th17_treg --scale 3.0 \
    --walkers 1000000 --steps 200 --dtype float32 \
    --json "${RESULTS}/bench_fp32_w1000000.json"

echo
echo "=== end-to-end validation run ==="
cellfate run configs/toggle_validation.yaml \
    --device cuda:0 --out "${RESULTS}/toggle_validation.json"

cat <<MSG

Step 1 complete.  selftest exit code: ${SELFTEST_RC}
Before moving to step 2, confirm:
  * selftest reported 0 failures
  * every generation log shows clamped=0 (if not, reduce we.tau)
  * you have a walker-steps/s figure and know where the GPU saturates

Results in ${RESULTS}
MSG
exit "${SELFTEST_RC}"
