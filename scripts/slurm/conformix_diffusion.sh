#!/bin/bash
#SBATCH --job-name=conformix_diffusion
#SBATCH --output=logs/conformix_diffusion_%A_%a.out
#SBATCH --error=logs/conformix_diffusion_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-101

# Default Boltz-2 diffusion on the ConforMix benchmark: the calibration baseline.
# The ConforMix paper reports coverage for default Boltz; comparing our walk-jump
# numbers against their published figures is only meaningful if our evaluation
# reproduces their baseline, so we generate it ourselves and score it with the
# same code.
#   DIFFUSION_SAMPLES=100 scripts/slurm/submit.sh gpu scripts/slurm/conformix_diffusion.sh

mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f .env ] && { set -a; source .env; set +a; }
source .venv/bin/activate

CONFIG_FILES=($(find configs/conformix_benchmark -name "*.yaml" -type f | sort))
TOTAL=${#CONFIG_FILES[@]}
if [ ${SLURM_ARRAY_TASK_ID} -ge ${TOTAL} ]; then
    echo "Array index ${SLURM_ARRAY_TASK_ID} exceeds number of configs (${TOTAL})"
    exit 0
fi
CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}
NAME=$(basename "${CONFIG_FILE}" .yaml)
SUBSET=$(basename "$(dirname "${CONFIG_FILE}")")
OUT_DIR="${RESULTS_DIR}/conformix_diffusion_${DIFFUSION_SAMPLES:-100}/${SUBSET}"
mkdir -p "${OUT_DIR}"

CMD="boltz predict ${CONFIG_FILE} --no_kernels --override --out_dir ${OUT_DIR} \
    --diffusion_samples ${DIFFUSION_SAMPLES:-100} --max_parallel_samples ${MAX_PARALLEL:-5} \
    --output_format mmcif --seed ${SEED:-0}"
echo "Running command: ${CMD}"
LOG=$(mktemp)
set -o pipefail   # otherwise the status below is tee's, not the command's
if ! ${CMD} 2>&1 | tee "${LOG}"; then
    echo "Failed to process ${SUBSET}/${NAME}"
    exit 1
fi
# boltz exits 0 even when an example fails (e.g. GPU OOM), so check its report
if grep -qE "Number of failed examples: [1-9]" "${LOG}"; then
    echo "Failed to process ${SUBSET}/${NAME}: boltz reported failed examples"
    exit 1
fi
if [ -z "$(ls -d ${OUT_DIR}/boltz_results_*${NAME}/predictions/${NAME} 2>/dev/null)" ]; then
    echo "Failed to process ${SUBSET}/${NAME}: no predictions written"
    exit 1
fi
echo "Successfully processed ${SUBSET}/${NAME}"
