#!/bin/bash
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=safw-run
#SBATCH --output=logs/run_%j.out
#SBATCH --error=logs/run_%j.err

# Thin cluster wrapper. All arguments go to scripts/run.py, which owns
# argument parsing, model resolution, cross-validation, and the task
# table. Example:
#   sbatch scripts/run.sh --method safw --lang bo --scale 32B --host cpt
#   TASKS are declared with --tasks rc rs

set -euo pipefail

module purge
module load rhel8/default-amp
source ~/.bashrc
conda activate safw_env

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p logs
python -m scripts.run "$@"
