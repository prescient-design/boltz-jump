#!/bin/bash
#SBATCH --job-name=boltz_walk_denoise
#SBATCH --output=logs/boltz_walk_denoise_%A_%a.out
#SBATCH --error=logs/boltz_walk_denoise_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --array=0-39

# Walk-denoise: walk-jump whose jump runs NUM_DENOISING_STEPS reverse-diffusion
# steps from sigma instead of one Tweedie step. Submit with, e.g.,
#   scripts/slurm/submit.sh gpu scripts/slurm/boltz_walk_denoise.sh --export=ALL,DATASET=atlas_test_msa,NUM_DENOISING_STEPS=10
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f .env ] && { set -a; source .env; set +a; }
source .venv/bin/activate
: "${DATASET:?set DATASET (e.g. atlas_test_msa or mdcath_msa)}"
: "${NUM_DENOISING_STEPS:?set NUM_DENOISING_STEPS}"

CONFIG_FILES=($(find "configs/${DATASET}/configs" -name "*.yaml" -type f | sort))
if [ ${SLURM_ARRAY_TASK_ID} -ge ${#CONFIG_FILES[@]} ]; then exit 0; fi
CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="${RESULTS_DIR}/${DATASET}"
mkdir -p "$OUT_DIR"

boltz-jump predict "$CONFIG_FILE" --no_kernels --override --out_dir "$OUT_DIR" \
    --walk_jump_steps 100 --num_denoising_steps "$NUM_DENOISING_STEPS"
