#!/bin/bash
#SBATCH --job-name=convergence
#SBATCH --output=logs/convergence_%A_%a.out
#SBATCH --error=logs/convergence_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# RMWD vs number of samples (scripts/eval/convergence.py), one target per task:
# SEEDED=1 scores only atoms present in the walk-jump seed (writes convergence_seeded/).
# indices 0-39 are ATLAS, 40-79 mdCATH. Writes results_analysis/convergence/<dataset>/<target>.csv.
#   scripts/slurm/submit.sh cpu scripts/slurm/convergence.sh --array=0-79

mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f .env ] && { set -a; source .env; set +a; }
source .venv/bin/activate

if [ "${SLURM_ARRAY_TASK_ID}" -lt 40 ]; then
    DATASET=atlas_test; PRED=results_organized/atlas_test_msa; REF=${ATLAS_DIR}; IDX=${SLURM_ARRAY_TASK_ID}
else
    DATASET=mdcath; PRED=results_organized/mdcath_msa; REF=${MDCATH_DIR}; IDX=$((SLURM_ARRAY_TASK_ID - 40))
fi
NAME=$(ls ${PRED} | sed -n "$((IDX + 1))p")
OUT=results_analysis/convergence${SEEDED:+_seeded}/${DATASET}/${NAME}.csv
mkdir -p $(dirname ${OUT})
[ -s ${OUT} ] && { echo "exists: ${OUT}"; exit 0; }
cd scripts/eval
python convergence.py --dataset ${DATASET} --name ${NAME} --reference_dir ${REF} \
    --predictions_dir ../../${PRED} --out ../../${OUT} ${SEEDED:+--seeded_atoms_only}
