#!/bin/bash

if [[ $# < 1 ]]; then
  echo "Must pass name of FASTA file to $0"
  exit
fi
echo "Reading sequences from $1"

export PATH=${BOLTZ_JUMP_DIR}/mmseqs/bin:$PATH

FASTA=$1
OUTDIR="${FASTA%.*}_mmseqs"
mkdir -p $OUTDIR

source ${BOLTZ_JUMP_DIR}/.venv/bin/activate

# rsync -av --progress ${COLABFOLD_DB_DIR}/colabfold_envdb_202108_db* /scratch/
# rsync -av --progress ${COLABFOLD_DB_DIR}/uniref30_2302_db* /scratch/

mmseqs gpuserver ${COLABFOLD_DB_DIR}/colabfold_envdb_202108_db --max-seqs 10000 --db-load-mode 3 --prefilter-mode 1 & PID1=$! \
&& mmseqs gpuserver ${COLABFOLD_DB_DIR}/uniref30_2302_db --max-seqs 10000 --db-load-mode 3 --prefilter-mode 1 & PID2=$! \
&& sleep 60


colabfold_search $FASTA ${COLABFOLD_DB_DIR}/ $OUTDIR --gpu 1 --gpu-server 1 --db-load-mode 3