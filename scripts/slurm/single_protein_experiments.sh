#!/bin/bash
#SBATCH --job-name=boltz_hypersweep
#SBATCH --output=logs/boltz_sweep_%A_%a.out
#SBATCH --error=logs/boltz_sweep_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=g6e
#SBATCH --gres=gpu:1
#SBATCH --array=0-80

# Create logs and output directories
NAME="1a62_A"
mkdir -p logs
mkdir -p ${NAME}_experiments

# Set up paths
cd ${BOLTZ_JUMP_DIR}
source "${BOLTZ_JUMP_DIR}/.venv/bin/activate"

# Define hyperparameter grid
SIGMA_VALUES=(1.0 2.0)
DELTA_VALUES=(1.0 2.0)
FRICTION_VALUES=(0.02 0.05 0.1)

# Build array of valid combinations (delta <= sigma)
VALID_COMBINATIONS=()
for SIGMA in "${SIGMA_VALUES[@]}"; do
    for DELTA in "${DELTA_VALUES[@]}"; do
        # Only include if delta <= sigma
        if (( $(echo "$DELTA <= $SIGMA" | bc -l) )); then
            for FRICTION in "${FRICTION_VALUES[@]}"; do
                VALID_COMBINATIONS+=("${SIGMA},${DELTA},${FRICTION}")
            done
        fi
    done
done

TOTAL_COMBINATIONS=${#VALID_COMBINATIONS[@]}

echo "Total valid hyperparameter combinations (delta <= sigma): ${TOTAL_COMBINATIONS}"
echo "Current array task ID: ${SLURM_ARRAY_TASK_ID}"

# Check if array index is within bounds
if [ ${SLURM_ARRAY_TASK_ID} -ge ${TOTAL_COMBINATIONS} ]; then
    echo "Array index ${SLURM_ARRAY_TASK_ID} exceeds number of valid combinations (${TOTAL_COMBINATIONS})"
    exit 0
fi

# Get the combination for this array task
COMBINATION=${VALID_COMBINATIONS[$SLURM_ARRAY_TASK_ID]}
IFS=',' read -r SIGMA DELTA FRICTION <<< "$COMBINATION"

# Create output directory for this combination
OUT_DIR="${NAME}_experiments/sigma_${SIGMA}_delta_${DELTA}_friction_${FRICTION}"
mkdir -p "${OUT_DIR}"

echo "=========================================="
echo "Processing combination $((SLURM_ARRAY_TASK_ID + 1))/${TOTAL_COMBINATIONS}"
echo "Hyperparameters:"
echo "  walk_jump_sigma: ${SIGMA}"
echo "  walk_jump_delta: ${DELTA}"
echo "  walk_jump_friction: ${FRICTION}"
echo "  Output directory: ${OUT_DIR}"
echo "=========================================="

# Run boltz-jump predict
CMD="boltz-jump predict atlas_msa/configs/${NAME}.yaml \
    --walk_jump_sigma ${SIGMA} \
    --walk_jump_delta ${DELTA} \
    --walk_jump_friction ${FRICTION} \
    --no_kernels \
    --override \
    --out_dir ${OUT_DIR}"

echo "Running command:"
echo "${CMD}"
echo ""

# Execute the command
if ${CMD}; then
    echo "Successfully completed combination: sigma=${SIGMA}, delta=${DELTA}, friction=${FRICTION}"
else
    echo "Failed to process combination: sigma=${SIGMA}, delta=${DELTA}, friction=${FRICTION}"
    exit 1
fi

echo "Completed processing"