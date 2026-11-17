#!/bin/bash
#SBATCH --job-name=analyse_ensembles_mdcath
#SBATCH --output=logs/analyse_ensembles_mdcath_%A_%a.out
#SBATCH --error=logs/analyse_ensembles_mdcath_%A_%a.err
#SBATCH --array=0-50
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Create logs and output directories
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f .env ] && { set -a; source .env; set +a; }
source .venv/bin/activate

# Number of entries to process per job
ENTRIES_PER_JOB=1

# Directories
DATASET="mdcath"
REFERENCE_DIR="${MDCATH_DIR}"
PRED_DIR="results_organized/mdcath_msa/"
OUTPUT_DIR="results_analysis/"

mkdir -p "$OUTPUT_DIR/"

# List of models to analyze
MODELS=(
    "alphaflow_md_base"
    "alphaflow_md_distilled"
    "bioemu"
    "boltz2_jump"
    "boltz2_diffusion"
    "boltz2_diffusion_empty_msa"
    "boltz2_msa_subsample_32"
    "boltz2_msa_subsample_64"
    "boltz2_full_msa"
    "boltz2_jump_msa"
    "boltz2_jump_empty_msa"
    "boltz2_jump_msa_no_init"
    "boltz2_jump_empty_msa_no_init"
    "boltz2_jump_empty_msa_boltz2_init"
    "boltz2_jump_msa_d2_100"
    "boltz2_jump_empty_msa_d2_100"
    "boltz2_jump_msa_no_init_d2_100"
    "boltz2_jump_empty_msa_no_init_d2_100"
    "boltz2_jump_empty_msa_boltz2_init_d2_100"
    "boltz2_jump_msa_d1"
    "boltz2_jump_empty_msa_d1"
    "boltz2_jump_msa_no_init_d1"
    "boltz2_jump_empty_msa_no_init_d1"
    "boltz2_jump_empty_msa_boltz2_init_d1"
    "boltz2_noised_seed_sde"
    "boltz2_noised_seed_ode"
    "boltz2_noised_seed_one_step"
    "boltz2_jump_empty_msa_boltz2_init_d1_v2"
    "boltz2_noised_seed_sde_v2"
    "boltz2_noised_seed_ode_v2"
    "boltz2_jump_msa_d1_rep1"
    "boltz2_jump_msa_d1_rep2"
    "boltz2_walk_denoise_n10"
    "boltz2_walk_denoise_n53"
)

# ===================================================================
# DYNAMIC TARGET LIST GENERATION
# ===================================================================
echo "Generating target list from directories in $PRED_DIR..."

# 1. Find all subdirectories (-type d)
# 2. Print only the folder name (%f), not full path
# 3. Sort them so every array job sees the same order
mapfile -t TARGETS < <(find "$PRED_DIR" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)

TOTAL_TARGETS=${#TARGETS[@]}
echo "Found $TOTAL_TARGETS targets."

# Calculate start and end indices for this specific job
# Bash arrays are 0-indexed
START_IDX=$(( SLURM_ARRAY_TASK_ID * ENTRIES_PER_JOB ))
END_IDX=$(( START_IDX + ENTRIES_PER_JOB - 1 ))

echo "==================================="
echo "Job ${SLURM_ARRAY_TASK_ID}"
echo "Processing indices ${START_IDX} to ${END_IDX}"
echo "Models to process: ${#MODELS[@]}"
echo "==================================="

# Counters
SUCCESS_COUNT=0
FAILED_COUNT=0

# Loop through the array indices calculated above
for (( i=START_IDX; i<=END_IDX; i++ )); do
    
    # Check if index is out of bounds (past the last folder found)
    if [ "$i" -ge "$TOTAL_TARGETS" ]; then
        echo "Index $i is out of bounds (Total: $TOTAL_TARGETS). Stopping."
        break
    fi

    # Get the name from the array
    NAME="${TARGETS[$i]}"

    echo ""
    echo "-----------------------------------"
    echo "Processing index ${i}: $NAME"
    echo "-----------------------------------"
    
    # Flag to track if all models for this entry succeeded
    ENTRY_SUCCESS=true
    
    # Loop through the defined models
    for MODEL in "${MODELS[@]}"; do
        OUTPUT_FILE="${OUTPUT_DIR}/${DATASET}/${NAME}/${MODEL}/analysis.pkl"
        if [ -f "$OUTPUT_FILE" ]; then
            echo ">> Skip (exists): $MODEL for $NAME"
            continue
        fi

        echo "Running analysis for $NAME using model: $MODEL..."

        if python scripts/eval/analyse_ensembles.py \
            --dataset "$DATASET" \
            --reference_dir "$REFERENCE_DIR" \
            --predictions_dir "$PRED_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --name "${NAME}" \
            --model "$MODEL"; then

            echo ">> Success: $MODEL for $NAME"
        else
            echo ">> Failed: $MODEL for $NAME"
            ENTRY_SUCCESS=false
        fi
    done
    
    # Update counters based on overall entry success
    if [ "$ENTRY_SUCCESS" = true ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "Successfully processed all models for $NAME"
    else
        FAILED_COUNT=$((FAILED_COUNT + 1))
        echo "Failures detected for $NAME"
    fi
done

echo ""
echo "==================================="
echo "Job ${SLURM_ARRAY_TASK_ID} complete"
echo "Successful entries: ${SUCCESS_COUNT}"
echo "Failed entries: ${FAILED_COUNT}"
echo "==================================="

# Exit with error if any failures occurred
if [ ${FAILED_COUNT} -gt 0 ]; then
    exit 1
fi