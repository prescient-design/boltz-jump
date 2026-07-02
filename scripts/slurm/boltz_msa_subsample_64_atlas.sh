#!/bin/bash
#SBATCH --job-name=boltz_msa_subsample_64
#SBATCH --output=logs/boltz_msa_subsample_64_%A_%a.out
#SBATCH --error=logs/boltz_msa_subsample_64_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=g6e
#SBATCH --gres=gpu:1
#SBATCH --array=0-39

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up paths
cd ${BOLTZ_JUMP_DIR}
source "${BOLTZ_JUMP_DIR}/.venv/bin/activate"

CONFIGS_DIR="${BOLTZ_JUMP_DIR}/configs/atlas_test_msa/configs"
OUT_DIR="${BOLTZ_JUMP_DIR}/results/atlas_test_msa"

mkdir -p ${OUT_DIR}

# Get list of all config files
CONFIG_FILES=($(find "${CONFIGS_DIR}" -name "*.yaml" -type f | sort))

# Calculate total number of configs
TOTAL_CONFIGS=${#CONFIG_FILES[@]}

echo "Found ${TOTAL_CONFIGS} config files"

# Check if array index is within bounds
if [ ${SLURM_ARRAY_TASK_ID} -gt ${TOTAL_CONFIGS} ]; then
    echo "Array index ${SLURM_ARRAY_TASK_ID} exceeds number of configs (${TOTAL_CONFIGS})"
    exit 0
fi

# Get the config file for this array task (subtract 1 for 0-based indexing)
CONFIG_FILE=${CONFIG_FILES[$((SLURM_ARRAY_TASK_ID - 1))]}
CONFIG_BASENAME=$(basename "${CONFIG_FILE}" .yaml)

echo "Processing config ${SLURM_ARRAY_TASK_ID}/${TOTAL_CONFIGS}: ${CONFIG_FILE}"

# Run boltz predict for this specific config
CMD="boltz predict ${CONFIG_FILE} --no_kernels --override --out_dir ${OUT_DIR} --subsample_msa --num_subsampled_msa 64 --diffusion_samples 100 --max_parallel_samples 1"
echo "Running command: ${CMD}"

# Run the command
if ${CMD}; then
    echo "Successfully processed ${CONFIG_BASENAME}"
else
    echo "Failed to process ${CONFIG_BASENAME}"
    exit 1
fi

echo "Completed processing ${CONFIG_BASENAME}"

