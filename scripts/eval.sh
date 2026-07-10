#!/bin/bash
# Evaluate every non-empty *_preds.json in the canonical tree that does
# not yet have a *_metrics.json next to it. Task and language are read
# from the path, so no model arguments are needed.
#
# Usage:
#   bash scripts/eval.sh [tree_root]

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-data}
ROOT=${1:-results/test_outputs}

subdir_of () {
    case $1 in
        rc) echo reading_comprehension ;;
        rs) echo response_selection ;;
        title) echo title_generation_200 ;;
        math) echo math ;;
        xx2en|en2xx) echo translation_dialogue ;;
    esac
}

task_of () {
    case $1 in
        rc) echo reading_comprehension ;;
        rs) echo response_selection ;;
        title) echo title_generation ;;
        math) echo math ;;
        xx2en|en2xx) echo translation ;;
    esac
}

find "${ROOT}" -name "*_preds.json" -size +0c | sort | while read -r PRED; do
    METR="${PRED%_preds.json}_metrics.json"
    [ -s "${METR}" ] && continue
    DIR=$(dirname "${PRED}")
    SHORT=$(basename "${DIR}")
    LANG_=$(basename "$(dirname "${DIR}")")
    TGT=en
    [ "${SHORT}" = "en2xx" ] && TGT="${LANG_}"
    python -m scripts.evaluate \
        --task "$(task_of "${SHORT}")" \
        --input_file "${DATA_ROOT}/$(subdir_of "${SHORT}")/${LANG_}/test.json" \
        --pred_file "${PRED}" \
        --metrics_output_file "${METR}" \
        --tgt_lang "${TGT}" \
        --source_run "$(basename "${PRED}")"
done
