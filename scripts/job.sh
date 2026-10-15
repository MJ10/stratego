#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

echo "Host: $(hostname)  Cluster: ${SLURM_CLUSTER_NAME:-?}  Job: ${SLURM_JOB_ID:-local}  Commit: $(git rev-parse --short HEAD)"
nvidia-smi -L || true

echo "Running: $*"
srun uv run --no-sync "$@"
