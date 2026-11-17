#!/bin/bash
#SBATCH --job-name=docking_cofold
#SBATCH --output=logs/docking_cofold_%A_%a.out
#SBATCH --error=logs/docking_cofold_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-427

# Boltz-2 co-folding of the PoseBusters entries (scripts/docking/prepare_posebusters.py).
# ACCELERATOR=cpu runs on the CPU partition (e.g. for entries that run out of GPU memory).
# CONFIG_LIST=<file of config paths> restricts the array (e.g. to the screened
# failures); DIFFUSION_SAMPLES (default 5) and RUN_TAG set the sample count and
# output subdirectory, e.g. for the matched-budget baseline:
#   CONFIG_LIST=$DOCKING_DIR/posebusters/failure_configs.txt DIFFUSION_SAMPLES=600 RUN_TAG=_600 \
#       scripts/slurm/submit.sh gpu scripts/slurm/docking_cofold.sh --array=0-<n-1>

mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f .env ] && { set -a; source .env; set +a; }
source .venv/bin/activate

if [ -n "${CONFIG_LIST:-}" ]; then
    mapfile -t CONFIG_FILES < "${CONFIG_LIST}"
else
    CONFIG_FILES=($(find "${DOCKING_DIR}/posebusters/cofold_configs" -name "*.yaml" -type f | sort))
fi
TOTAL_CONFIGS=${#CONFIG_FILES[@]}
if [ ${SLURM_ARRAY_TASK_ID} -ge ${TOTAL_CONFIGS} ]; then
    echo "Array index ${SLURM_ARRAY_TASK_ID} exceeds number of configs (${TOTAL_CONFIGS})"
    exit 0
fi
CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="${DOCKING_DIR}/posebusters/cofold${RUN_TAG:-}"
mkdir -p ${OUT_DIR}

CMD="boltz predict ${CONFIG_FILE} --no_kernels --override --out_dir ${OUT_DIR} \
    --diffusion_samples ${DIFFUSION_SAMPLES:-5} --max_parallel_samples ${MAX_PARALLEL:-2} --recycling_steps 3 \
    --sampling_steps 200 --output_format mmcif --seed ${SEED:-0} --accelerator ${ACCELERATOR:-gpu}"
NAME=$(basename "${CONFIG_FILE}" .yaml)
echo "Running command: ${CMD}"
LOG=$(mktemp)
set -o pipefail   # otherwise the status below is tee's, not the command's
if ! ${CMD} 2>&1 | tee "${LOG}"; then
    echo "Failed to process ${NAME}"
    exit 1
fi
# boltz exits 0 even when an example fails (e.g. GPU OOM), so check its report
if grep -qE "Number of failed examples: [1-9]" "${LOG}"; then
    echo "Failed to process ${NAME}: boltz reported failed examples"
    exit 1
fi
if [ -z "$(ls -d ${OUT_DIR}/boltz_results_*${NAME}/predictions/${NAME} 2>/dev/null)" ]; then
    echo "Failed to process ${NAME}: no predictions written"
    exit 1
fi
echo "Successfully processed ${NAME}"
