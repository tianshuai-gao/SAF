#!/bin/bash
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=safw
#SBATCH --output=logs/safw_%j.out
#SBATCH --error=logs/safw_%j.err

# Run SAF-W inference on the MiLiC-Eval tasks for one language and scale.
# Outputs land in the canonical results tree. scripts/canonical.py is the
# single source of truth for paths, and the filename encodes method,
# family, language, task, scale, and host direction.
#
# Usage:
#   sbatch scripts/run_safw.sh <eval_lang> <host_model> <scorer_model>
# For the uniform-averaging reduction, set METHOD=safw_fixed.

set -euo pipefail

module purge
module load rhel8/default-amp
source ~/.bashrc
conda activate safw_env

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

EVAL_LANG=${1:-bo}
HOST_MODEL=${2:-Qwen/Qwen2.5-32B-Instruct}
SCORER_MODEL=${3:-pkupie/Qwen2.5-1.5B-${EVAL_LANG}-cpt}
PROMPT_LANG=en
METHOD=${METHOD:-safw}
BETA_FIXED=${BETA_FIXED:-0.5}
DATA_ROOT=${DATA_ROOT:-data}
mkdir -p logs

stem () {
    python -m scripts.canonical --method "${METHOD}" --lang "${EVAL_LANG}" \
        --task "$1" --models "${HOST_MODEL}" "${SCORER_MODEL}"
}

run_task () {
    local short=$1 task=$2 subdir=$3 nex=$4 mnt=$5
    local STEM
    STEM=$(stem "${short}")
    mkdir -p "$(dirname "${STEM}")"
    python -m scripts.infer \
        --method "${METHOD}" \
        --host_model "${HOST_MODEL}" \
        --scorer_model "${SCORER_MODEL}" \
        --beta_fixed "${BETA_FIXED}" \
        --task "${task}" \
        --eval_lang "${EVAL_LANG}" \
        --prompt_lang "${PROMPT_LANG}" \
        --num_exemplar "${nex}" \
        --max_new_tokens "${mnt}" \
        --input_file "${DATA_ROOT}/${subdir}/${EVAL_LANG}/test.json" \
        --exemplar_file "${DATA_ROOT}/${subdir}/${EVAL_LANG}/train_1.json" \
        --output_file "${STEM}_preds.json"
}

run_task rc    reading_comprehension reading_comprehension 5 2
run_task rs    response_selection    response_selection    5 2
run_task title title_generation      title_generation_200  3 250
run_task math  math                  math                  5 250

run_mt () {
    local short=$1 src=$2 tgt=$3
    local STEM
    STEM=$(stem "${short}")
    mkdir -p "$(dirname "${STEM}")"
    python -m scripts.infer --method "${METHOD}" \
        --host_model "${HOST_MODEL}" --scorer_model "${SCORER_MODEL}" \
        --beta_fixed "${BETA_FIXED}" \
        --task translation --eval_lang "${EVAL_LANG}" --prompt_lang "${PROMPT_LANG}" \
        --src_lang "${src}" --tgt_lang "${tgt}" --num_exemplar 5 --max_new_tokens 200 \
        --input_file "${DATA_ROOT}/translation_dialogue/${EVAL_LANG}/test.json" \
        --exemplar_file "${DATA_ROOT}/translation_dialogue/${EVAL_LANG}/train_1.json" \
        --output_file "${STEM}_preds.json"
}

run_mt xx2en "${EVAL_LANG}" en
run_mt en2xx en "${EVAL_LANG}"

echo "inference done for ${EVAL_LANG} (method=${METHOD})"
