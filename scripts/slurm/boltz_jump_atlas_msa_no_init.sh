#!/bin/bash
#SBATCH --job-name=boltz_jump_no_init
#SBATCH --output=logs/boltz_jump_%A_%a.out
#SBATCH --error=logs/boltz_jump_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --array=0-39

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up paths
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f .env ] && { set -a; source .env; set +a; }
source .venv/bin/activate

CONFIGS_DIR="configs/atlas_test_msa_no_init/configs"
OUT_DIR="${RESULTS_DIR}/atlas_test_msa_no_init${RUN_TAG:-}"

mkdir -p ${OUT_DIR}

# Get list of all config files
CONFIG_FILES=($(find "${CONFIGS_DIR}" -name "*.yaml" -type f | sort))

# Calculate total number of configs
TOTAL_CONFIGS=${#CONFIG_FILES[@]}

echo "Found ${TOTAL_CONFIGS} config files"

# Check if array index is within bounds
if [ ${SLURM_ARRAY_TASK_ID} -ge ${TOTAL_CONFIGS} ]; then
    echo "Array index ${SLURM_ARRAY_TASK_ID} exceeds number of configs (${TOTAL_CONFIGS})"
    exit 0
fi

# Get the config file for this array task (0-based)
CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}
CONFIG_BASENAME=$(basename "${CONFIG_FILE}" .yaml)

echo "Processing config ${SLURM_ARRAY_TASK_ID}/${TOTAL_CONFIGS}: ${CONFIG_FILE}"

# Run boltz-jump predict for this specific config
CMD="boltz-jump predict ${CONFIG_FILE} --no_kernels --override --walk_jump_steps 100 --out_dir ${OUT_DIR}"
echo "Running command: ${CMD}"

# Run the command
# Optional: SEED=<int> makes the run reproducible; RUN_TAG=<suffix> writes to a separate results directory.
[ -n "${SEED:-}" ] && CMD="${CMD} --seed ${SEED}"
if ${CMD}; then
    echo "Successfully processed ${CONFIG_BASENAME}"
else
    echo "Failed to process ${CONFIG_BASENAME}"
    exit 1
fi

echo "Completed processing ${CONFIG_BASENAME}"