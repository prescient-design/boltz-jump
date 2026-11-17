#!/bin/bash
#SBATCH --job-name=docking_walkjump
#SBATCH --output=logs/docking_walkjump_%A_%a.out
#SBATCH --error=logs/docking_walkjump_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Walk-jump pose refinement over a task list written by
# scripts/docking/prepare_walkjump.py: one tab-separated line per run,
#   <config yaml>  <sigma>  <seed>  <output dir>
# Submit with --array=0-<number of lines - 1>, e.g.
#   TASKS=$DOCKING_DIR/posebusters/walkjump_tasks.tsv \
#       scripts/slurm/submit.sh gpu scripts/slurm/docking_walkjump.sh --array=0-<n-1>
# The step size is 0.2 sigma and the friction 2.0, with 200 frames per run.
# ACCELERATOR=cpu runs on the CPU partition when GPUs are contended.

mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f .env ] && { set -a; source .env; set +a; }
source .venv/bin/activate

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${TASKS}")
if [ -z "${LINE}" ]; then
    echo "Array index ${SLURM_ARRAY_TASK_ID} exceeds the task list ${TASKS}"
    exit 0
fi
IFS=$'\t' read -r CONFIG_FILE SIGMA RUN_SEED OUT_DIR <<< "${LINE}"
DELTA=$(python -c "print(round(0.2 * ${SIGMA}, 4))")
mkdir -p ${OUT_DIR}

CMD="boltz-jump predict ${CONFIG_FILE} --no_kernels --override --out_dir ${OUT_DIR} \
    --walk_jump_sigma ${SIGMA} --walk_jump_delta ${DELTA} --walk_jump_friction 2.0 \
    --walk_jump_steps 200 --walk_jump_save_every_n_steps 1 --seed ${RUN_SEED} --accelerator ${ACCELERATOR:-gpu}"
echo "Running command: ${CMD}"
if ${CMD}; then
    echo "Successfully processed ${CONFIG_FILE} sigma=${SIGMA} seed=${RUN_SEED}"
else
    echo "Failed to process ${CONFIG_FILE} sigma=${SIGMA} seed=${RUN_SEED}"
    exit 1
fi
