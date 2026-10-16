#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

echo "Host: $(hostname)  Cluster: ${SLURM_CLUSTER_NAME:-?}  Job: ${SLURM_JOB_ID:-local}  Commit: $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

echo "Running: $*"
PYTHONUNBUFFERED=1 uv run --no-sync "$@" 2>&1 | tee "$SLURM_TMPDIR/train.log"

output_path="$(scontrol show job "${SLURM_JOB_ID:?not running under Slurm}" -o 2>/dev/null \
    | tr ' ' '\n' | sed -n 's/^StdOut=//p' | head -n 1 || true)"
output_path="${output_path:-${SBATCH_OUTPUT:-}}"
run_dir="${output_path%/*}"
run_dir="${run_dir//%j/${SLURM_JOB_ID}}"
mkdir -p "$run_dir"
cp "$SLURM_TMPDIR/train.log" "$run_dir/train.log"
