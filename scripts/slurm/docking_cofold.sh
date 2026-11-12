#!/bin/bash
#SBATCH --job-name=docking_cofold
#SBATCH --output=logs/docking_cofold_%A_%a.out
#SBATCH --error=logs/docking_cofold_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-427

# Boltz-2 co-folding of the PoseBusters entries (scripts/docking/prepare_posebusters.py).
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
    --diffusion_samples ${DIFFUSION_SAMPLES:-5} --max_parallel_samples 5 --recycling_steps 3 \
    --sampling_steps 200 --output_format mmcif --seed ${SEED:-0}"
echo "Running command: ${CMD}"
if ${CMD}; then
    echo "Successfully processed $(basename ${CONFIG_FILE} .yaml)"
else
    echo "Failed to process $(basename ${CONFIG_FILE} .yaml)"
    exit 1
fi
