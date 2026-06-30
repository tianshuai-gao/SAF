#!/bin/bash
# Evaluate all MiLiC-Eval tasks for one language and method.
#
# Reads the prediction files written by run_safw.sh under
# results/<method>/<lang>/ and writes a .metrics.json next to each one.
#
# Usage:
#   bash scripts/eval.sh <eval_lang> [method]
# Example:
#   bash scripts/eval.sh bo safw

set -euo pipefail

EVAL_LANG=${1:-bo}
METHOD=${2:-safw}
DATA_ROOT=${DATA_ROOT:-data}
OUT_ROOT=${OUT_ROOT:-results/${METHOD}/${EVAL_LANG}}

# task | data subdir
eval_task () {
    local task=$1 subdir=$2
    python -m scripts.evaluate \
        --task "${task}" \
        --input_file "${DATA_ROOT}/${subdir}/${EVAL_LANG}/test.json" \
        --pred_file "${OUT_ROOT}/${task}.json" \
        --metrics_output_file "${OUT_ROOT}/${task}.metrics.json"
}

eval_task reading_comprehension reading_comprehension
eval_task response_selection    response_selection
eval_task title_generation      title_generation_200
eval_task math                  math

# Translation, both directions; tgt_lang differs per direction.
python -m scripts.evaluate --task translation \
    --input_file "${DATA_ROOT}/translation_dialogue/${EVAL_LANG}/test.json" \
    --pred_file "${OUT_ROOT}/translation_${EVAL_LANG}2en.json" \
    --metrics_output_file "${OUT_ROOT}/translation_${EVAL_LANG}2en.metrics.json" \
    --tgt_lang en

python -m scripts.evaluate --task translation \
    --input_file "${DATA_ROOT}/translation_dialogue/${EVAL_LANG}/test.json" \
    --pred_file "${OUT_ROOT}/translation_en2${EVAL_LANG}.json" \
    --metrics_output_file "${OUT_ROOT}/translation_en2${EVAL_LANG}.metrics.json" \
    --tgt_lang "${EVAL_LANG}"

echo "evaluation done for ${EVAL_LANG} (method=${METHOD})"
