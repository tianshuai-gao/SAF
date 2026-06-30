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
#
# SAF-W fuses a host (large instruction model) and a scorer (small CPT model)
# at decoding time. The host and scorer roles are chosen per task and language
# on a development set. This script runs the endorsement method (method=safw).
# For the uniform-averaging reduction used on math, set METHOD=safw_fixed and
# BETA_FIXED=0.5.
#
# Usage:
#   sbatch scripts/run_safw.sh <eval_lang> <host_model> <scorer_model>
# Example:
#   sbatch scripts/run_safw.sh bo Qwen/Qwen2.5-32B-Instruct pkupie/Qwen2.5-1.5B-bo-cpt

set -euo pipefail

module purge
module load rhel8/default-amp
source ~/.bashrc
conda activate safw_env

# Deterministic decoding and offline model loading on the cluster.
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
OUT_ROOT=${OUT_ROOT:-results/${METHOD}/${EVAL_LANG}}
mkdir -p "${OUT_ROOT}" logs

# task | data subdir | num_exemplar | max_new_tokens
run_task () {
    local task=$1 subdir=$2 nex=$3 mnt=$4
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
        --output_file "${OUT_ROOT}/${task}.json"
}

run_task reading_comprehension reading_comprehension 5 2
run_task response_selection    response_selection    5 2
run_task title_generation      title_generation_200  3 250
run_task math                  math                  5 250

# Translation runs in both directions.
python -m scripts.infer --method "${METHOD}" \
    --host_model "${HOST_MODEL}" --scorer_model "${SCORER_MODEL}" \
    --beta_fixed "${BETA_FIXED}" \
    --task translation --eval_lang "${EVAL_LANG}" --prompt_lang "${PROMPT_LANG}" \
    --src_lang "${EVAL_LANG}" --tgt_lang en --num_exemplar 5 --max_new_tokens 200 \
    --input_file "${DATA_ROOT}/translation_dialogue/${EVAL_LANG}/test.json" \
    --exemplar_file "${DATA_ROOT}/translation_dialogue/${EVAL_LANG}/train_1.json" \
    --output_file "${OUT_ROOT}/translation_${EVAL_LANG}2en.json"

python -m scripts.infer --method "${METHOD}" \
    --host_model "${HOST_MODEL}" --scorer_model "${SCORER_MODEL}" \
    --beta_fixed "${BETA_FIXED}" \
    --task translation --eval_lang "${EVAL_LANG}" --prompt_lang "${PROMPT_LANG}" \
    --src_lang en --tgt_lang "${EVAL_LANG}" --num_exemplar 5 --max_new_tokens 200 \
    --input_file "${DATA_ROOT}/translation_dialogue/${EVAL_LANG}/test.json" \
    --exemplar_file "${DATA_ROOT}/translation_dialogue/${EVAL_LANG}/train_1.json" \
    --output_file "${OUT_ROOT}/translation_en2${EVAL_LANG}.json"

echo "SAF-W inference done for ${EVAL_LANG} (method=${METHOD})"
