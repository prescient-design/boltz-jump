#!/bin/bash
#SBATCH --job-name=boltz_jump_conformix
#SBATCH --output=boltz_jump_conformix.log
#SBATCH --error=logs/boltz_jump_conformix_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=10
#SBATCH --array=0-103            # 38 + 15 + 34 + 15 = 102 (0-indexed)

# ============================================================
# EDIT THESE PATHS
# ============================================================
BOLTZ_JUMP_DIR="${BOLTZ_JUMP_DIR}"
BENCHMARK_DIR="${BOLTZ_JUMP_DIR}/configs/conformix_benchmark"
OUT_DIR="${BOLTZ_JUMP_DIR}/results/conformix_benchmark_1000"
# ============================================================

mkdir -p logs
source "${BOLTZ_JUMP_DIR}/.venv/bin/activate"

# Gather all config yamls across all datasets
CONFIG_FILES=($(find "${BENCHMARK_DIR}" -name "*.yaml" -type f | sort))
TOTAL_CONFIGS=${#CONFIG_FILES[@]}

echo "Found ${TOTAL_CONFIGS} config files"

if [ ${SLURM_ARRAY_TASK_ID} -ge ${TOTAL_CONFIGS} ]; then
    echo "Array index ${SLURM_ARRAY_TASK_ID} exceeds number of configs (${TOTAL_CONFIGS})"
    exit 0
fi

CONFIG_FILE=${CONFIG_FILES[${SLURM_ARRAY_TASK_ID}]}
CONFIG_BASENAME=$(basename "${CONFIG_FILE}" .yaml)

# Determine which dataset this config belongs to
DATASET=$(basename $(dirname "${CONFIG_FILE}"))

echo "Processing [${SLURM_ARRAY_TASK_ID}/${TOTAL_CONFIGS}]: ${DATASET}/${CONFIG_BASENAME}"

CMD="boltz-jump predict ${CONFIG_FILE} \
    --no_kernels \
    --override \
    --walk_jump_steps 1000 \
    --use_msa_server \
    --out_dir ${OUT_DIR}/${DATASET}"

echo "Running: ${CMD}"
${CMD}

echo "Done: ${CONFIG_BASENAME}"
