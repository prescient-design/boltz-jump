#!/bin/bash
#SBATCH --job-name=noised_seed_one_step_mdcath
#SBATCH --output=logs/noised_seed_one_step_mdcath_%A_%a.out
#SBATCH --error=logs/noised_seed_one_step_mdcath_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --array=0-39

mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f .env ] && { set -a; source .env; set +a; }
source .venv/bin/activate

CONFIGS_DIR="configs/mdcath_msa/configs"
OUT_DIR="${RESULTS_DIR}/noised_seed_one_step_mdcath${RUN_TAG:-}"

mkdir -p ${OUT_DIR}

CONFIG_FILES=($(find "${CONFIGS_DIR}" -name "*.yaml" -type f | sort))
TOTAL_CONFIGS=${#CONFIG_FILES[@]}

echo "Found ${TOTAL_CONFIGS} config files"

if [ ${SLURM_ARRAY_TASK_ID} -ge ${TOTAL_CONFIGS} ]; then
    echo "Array index ${SLURM_ARRAY_TASK_ID} exceeds number of configs (${TOTAL_CONFIGS})"
    exit 0
fi

CONFIG_FILE=${CONFIG_FILES[${SLURM_ARRAY_TASK_ID}]}
CONFIG_BASENAME=$(basename "${CONFIG_FILE}" .yaml)

echo "Processing [${SLURM_ARRAY_TASK_ID}/${TOTAL_CONFIGS}]: ${CONFIG_BASENAME}"

CMD="boltz-jump predict ${CONFIG_FILE} \
    --no_kernels \
    --override \
    --use_noised_seed \
    --noised_seed_one_step \
    --walk_jump_sigma 2.0 \
    --num_noised_seed_samples 100 \
    
    --out_dir ${OUT_DIR}"

echo "Running: ${CMD}"
# Optional: SEED=<int> makes the run reproducible; RUN_TAG=<suffix> writes to a separate results directory.
[ -n "${SEED:-}" ] && CMD="${CMD} --seed ${SEED}"
${CMD}

echo "Done: ${CONFIG_BASENAME}"
