#!/bin/bash
#SBATCH --job-name=abl_kinase_states
#SBATCH --output=logs/abl_kinase_states_%A_%a.out
#SBATCH --error=logs/abl_kinase_states_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --array=0-2

# Does Boltz-2 diffusion prefer the active (6XR6) or inactive I1 (6XR7) Abl kinase
# state, and does unguided walk-jump seeded from either state visit the other?
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f .env ] && { set -a; source .env; set +a; }
source .venv/bin/activate
OUT_DIR="${RESULTS_DIR}/abl_kinase_states"
mkdir -p "$OUT_DIR"

case $SLURM_ARRAY_TASK_ID in
  0) boltz predict configs/abl_kinase/abl_kinase.yaml --no_kernels --override \
       --out_dir "$OUT_DIR" --diffusion_samples 100 --max_parallel_samples 1 ;;
  1) boltz-jump predict configs/abl_kinase/abl_kinase_from_6XR6.yaml --no_kernels --override \
       --out_dir "$OUT_DIR" --walk_jump_steps 1000 ;;
  2) boltz-jump predict configs/abl_kinase/abl_kinase_from_6XR7.yaml --no_kernels --override \
       --out_dir "$OUT_DIR" --walk_jump_steps 1000 ;;
esac
