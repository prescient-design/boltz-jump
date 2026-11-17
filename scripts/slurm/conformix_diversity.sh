#!/bin/bash
#SBATCH --job-name=conformix_diversity
#SBATCH --output=logs/conformix_diversity_%A_%a.out
#SBATCH --error=logs/conformix_diversity_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Walk-jump exploration strategies on ConforMix, one task-list line per run
# (written by conformix_benchmark/scripts/make_diversity_tasks.py):
#   <config>  <out_dir>  <seed>  <extra boltz-jump arguments>
#   TASKS=conformix_benchmark/diversity_tasks_dev.tsv \
#       scripts/slurm/submit.sh gpu scripts/slurm/conformix_diversity.sh --array=0-<n-1>
# When GPUs are contended, ACCELERATOR=cpu runs on the CPU partition instead:
#   ACCELERATOR=cpu TASKS=... scripts/slurm/submit.sh cpu scripts/slurm/conformix_diversity.sh \
#       --cpus-per-task=16 --mem=64G --time=12:00:00 --array=0-<n-1>

mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f .env ] && { set -a; source .env; set +a; }
source .venv/bin/activate

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${TASKS}")
if [ -z "${LINE}" ]; then
    echo "Array index ${SLURM_ARRAY_TASK_ID} exceeds the task list ${TASKS}"
    exit 0
fi
IFS=$'\t' read -r CONFIG_FILE OUT_DIR RUN_SEED EXTRA <<< "${LINE}"
NAME=$(basename "${CONFIG_FILE}" .yaml)
mkdir -p "${OUT_DIR}"

CMD="boltz-jump predict ${CONFIG_FILE} --out_dir ${OUT_DIR} --seed ${RUN_SEED} --accelerator ${ACCELERATOR:-gpu} ${EXTRA}"
echo "Running command: ${CMD}"
set -o pipefail
if ! ${CMD}; then
    echo "Failed to process ${NAME}"
    exit 1
fi
if [ -z "$(ls ${OUT_DIR}/boltz_jump_results_*_${NAME}/predictions/${NAME}/${NAME}_model_0.xtc 2>/dev/null)" ]; then
    echo "Failed to process ${NAME}: no trajectory written"
    exit 1
fi
echo "Successfully processed ${NAME}"
