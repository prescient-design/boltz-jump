#!/bin/bash
#SBATCH --job-name=docking_msa_cpu
#SBATCH --output=logs/docking_msa_cpu_%j.out
#SBATCH --error=logs/docking_msa_cpu_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=340G
#SBATCH --requeue

# CPU variant of docking_msa.sh: same ColabFold search without the GPU prefilter
# servers, for when GPU nodes are contended. Submit with `submit.sh cpu`.
# MSAs for the docking targets with mmseqs against the ColabFold databases
# (the commands of the paper appendix, "MSA Generation"). FASTA headers are
# sequence hashes, so each MSA is written as <OUT_DIR>/<hash>.a3m.
# Pass every FASTA at once in FASTA (space separated): the run time is dominated
# by scanning the databases, not by the number of queries, so one job for all
# targets is far cheaper than one per dataset.
#   FASTA="$DOCKING_DIR/posebusters/sequences.fasta $DOCKING_DIR/runs_n_poses/sequences.fasta" \
#   OUT_DIR=$DOCKING_DIR/msa scripts/slurm/submit.sh gpu scripts/slurm/docking_msa.sh
# Needs COLABFOLD_DB_DIR (directory with colabfold_envdb_202108_db and
# uniref30_2302_db), mmseqs in mmseqs/bin and colabfold_search in .venv_colabfold.

set -e
mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f .env ] && { set -a; source .env; set +a; }
export PATH=$PWD/mmseqs/bin:$PATH
mkdir -p "${OUT_DIR}"


# Deduplicate the sequences of all input FASTAs (headers are sequence hashes).
QUERY="${OUT_DIR}/all_sequences.fasta"
awk '/^>/ {h=$0; next} {if (!(h in seen)) {seen[h]=1; print h; print $0}}' ${FASTA} > "${QUERY}"
echo "querying $(grep -c '>' ${QUERY}) unique sequences with ${SLURM_CPUS_PER_TASK} threads"

.venv_colabfold/bin/colabfold_search "${QUERY}" ${COLABFOLD_DB_DIR}/ "${OUT_DIR}" \
    --gpu 0 --db-load-mode 2 --mmseqs mmseqs/bin/mmseqs \
    --threads ${SLURM_CPUS_PER_TASK}
echo "MSAs written: $(ls ${OUT_DIR}/*.a3m | wc -l) of $(grep -c '>' ${QUERY}) sequences"
