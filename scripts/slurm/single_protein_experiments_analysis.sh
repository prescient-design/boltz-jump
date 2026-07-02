#!/bin/bash
#SBATCH --job-name=boltz_hypersweep_analysis
#SBATCH --output=logs/boltz_sweep_analysis_%A_%a.out
#SBATCH --error=logs/boltz_sweep_analysis_%A_%a.err
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

# Get all subdirectories as an array
mapfile -t subdirs < <(find ${NAME}_experiments -mindepth 1 -maxdepth 1 -type d | sort)

# Get the subdirectory for this array task
exp_dir=${subdirs[$SLURM_ARRAY_TASK_ID]}

# Set output directory for this combination
OUT_DIR="${exp_dir}/boltz_jump_results_${NAME}"

echo "Processing array task ${SLURM_ARRAY_TASK_ID}: ${exp_dir}"

python scripts/eval/analyse_ensembles.py \
    --atlas_dir ${ATLAS_DIR} \
    --pred_dir ${OUT_DIR}/predictions \
    --name ${NAME}

echo "Analysis completed for: ${exp_dir}"