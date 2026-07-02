#!/bin/bash
#SBATCH --job-name=atlas_array
#SBATCH --output=logs/atlas_%A_%a.out
#SBATCH --error=logs/atlas_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=cpu
#SBATCH --array=1-40%50

# Create logs directory if it doesn't exist
mkdir -p logs

# Load any required modules (adjust as needed for your system)
# module load python/3.9
# module load conda

# Activate your conda environment if needed
# conda activate your_env_name

# Set up paths
ATLAS_DIR="${ATLAS_DIR}"
ATLAS_MSA_DIR="${ATLAS_MSA_DIR}"
OUTPUT_DIR="atlas_msa"

# Number of entries to process per job
ENTRIES_PER_JOB=50

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}/configs"

cd ${BOLTZ_JUMP_DIR}/

# Get list of all entries (directories only)
ENTRIES=($(find "${ATLAS_DIR}" -maxdepth 1 -type d -exec basename {} \; | sort | tail -n +2))

# Calculate total number of entries
TOTAL_ENTRIES=${#ENTRIES[@]}

echo "Total entries: ${TOTAL_ENTRIES}"
echo "Processing job ${SLURM_ARRAY_TASK_ID}"

# Calculate the start and end indices for this job
START_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) * ENTRIES_PER_JOB ))
END_IDX=$(( START_IDX + ENTRIES_PER_JOB - 1 ))

# Don't go past the total number of entries
if [ ${END_IDX} -ge ${TOTAL_ENTRIES} ]; then
    END_IDX=$((TOTAL_ENTRIES - 1))
fi

echo "Processing entries ${START_IDX} to ${END_IDX}"

# Process entries in the range
FAILED_COUNT=0
SUCCESS_COUNT=0

for i in $(seq ${START_IDX} ${END_IDX}); do
    # Check if index is within bounds
    if [ ${i} -ge ${TOTAL_ENTRIES} ]; then
        break
    fi
    
    ENTRY=${ENTRIES[$i]}
    
    echo "Processing entry $((i + 1))/${TOTAL_ENTRIES}: ${ENTRY}"
    
    # Check if output already exists (optional skip logic)
    # if [ -f "${OUTPUT_DIR}/configs/${ENTRY}.yaml" ]; then
    #     echo "Skipping ${ENTRY} as output already exists."
    #     continue
    # fi
    
    # Run the command for this specific entry
    CMD="python scripts/eval/create_atlas_yaml.py ${ENTRY} --output-dir ${OUTPUT_DIR} --atlas-dir ${ATLAS_DIR} --atlas-msa-dir ${ATLAS_MSA_DIR}"
    echo "Running command: ${CMD}"
    
    if ${CMD}; then
        echo "✓ Successfully processed ${ENTRY}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "✗ Failed to process ${ENTRY}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

echo "==================================="
echo "Job ${SLURM_ARRAY_TASK_ID} complete"
echo "Successful: ${SUCCESS_COUNT}"
echo "Failed: ${FAILED_COUNT}"
echo "==================================="

# Exit with error if any failures occurred
if [ ${FAILED_COUNT} -gt 0 ]; then
    exit 1
fi