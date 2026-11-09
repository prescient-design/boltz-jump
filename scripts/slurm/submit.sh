#!/bin/bash
# Submit a job script with cluster-specific SLURM settings taken from .env.
#
# Usage (from anywhere):
#   scripts/slurm/submit.sh gpu scripts/slurm/boltz_jump_atlas.sh [extra sbatch args]
#   scripts/slurm/submit.sh cpu scripts/slurm/analyse_ensembles_atlas.sh
#
# Reads SLURM_ACCOUNT, SLURM_QOS, SLURM_GPU_PARTITION, SLURM_GPU_GRES and
# SLURM_CPU_PARTITION from .env (see .env.example); unset values are omitted.
# Jobs run from the repository root.
set -euo pipefail

if [ $# -lt 2 ] || { [ "$1" != gpu ] && [ "$1" != cpu ]; }; then
    sed -n '2,10p' "$0"
    exit 1
fi
kind=$1
script=$(realpath "$2")
shift 2

cd "$(dirname "$(realpath "$0")")/../.."
[ -f .env ] && { set -a; source .env; set +a; }

args=()
[ -n "${SLURM_ACCOUNT:-}" ] && args+=(--account "$SLURM_ACCOUNT")
[ -n "${SLURM_QOS:-}" ] && args+=(--qos "$SLURM_QOS")
if [ "$kind" = gpu ]; then
    [ -n "${SLURM_GPU_PARTITION:-}" ] && args+=(--partition "$SLURM_GPU_PARTITION")
    args+=(--gres "${SLURM_GPU_GRES:-gpu:1}")
else
    [ -n "${SLURM_CPU_PARTITION:-}" ] && args+=(--partition "$SLURM_CPU_PARTITION")
fi

mkdir -p logs
sbatch "${args[@]}" "$@" "$script"
