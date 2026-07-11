#!/bin/bash
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=safw-run
#SBATCH --output=logs/run_%j.out
#SBATCH --error=logs/run_%j.err

# One batch entry for every decoding method. infer.py is the single
# parametrised engine underneath. This wrapper adds what the cluster
# needs, namely the SBATCH header, the environment, and the task table.
# scripts/canonical.py decides where every output lands.
#
# Usage:
#   sbatch scripts/run.sh safw       <lang> [host] [scorer]
#   sbatch scripts/run.sh safw_fixed <lang> [host] [scorer]
#   sbatch scripts/run.sh proxy      <lang> [base] [expert] [antiexpert]
#   sbatch scripts/run.sh trimix     <lang> [base] [expert] [antiexpert]
#
# TriMix weights follow the per-task perplexity heuristic. Restrict the
# tasks and set the weights per invocation:
#   TASKS="rc rs" BASE_WEIGHT=0.1 sbatch scripts/run.sh trimix bo

set -euo pipefail

module purge
module load rhel8/default-amp
source ~/.bashrc
conda activate safw_env

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

METHOD=${1:?method: safw, safw_fixed, proxy, or trimix}
EVAL_LANG=${2:-bo}
PROMPT_LANG=en
DATA_ROOT=${DATA_ROOT:-data}
TASKS=${TASKS:-rc rs title math xx2en en2xx}
mkdir -p logs

case "${METHOD}" in
  safw|safw_fixed)
    HOST_MODEL=${3:-Qwen/Qwen2.5-32B-Instruct}
    SCORER_MODEL=${4:-pkupie/Qwen2.5-1.5B-${EVAL_LANG}-cpt}
    MODELS=("${HOST_MODEL}" "${SCORER_MODEL}")
    METHOD_ARGS=(--host_model "${HOST_MODEL}"
                 --scorer_model "${SCORER_MODEL}"
                 --beta_fixed "${BETA_FIXED:-0.5}")
    ;;
  proxy|trimix)
    BASE_MODEL=${3:-Qwen/Qwen2.5-7B-Instruct}
    EXPERT_MODEL=${4:-pkupie/Qwen2.5-1.5B-${EVAL_LANG}-cpt}
    ANTI_MODEL=${5:-Qwen/Qwen2.5-1.5B}
    MODELS=("${BASE_MODEL}" "${EXPERT_MODEL}" "${ANTI_MODEL}")
    METHOD_ARGS=(--base_model "${BASE_MODEL}"
                 --expert_model "${EXPERT_MODEL}"
                 --antiexpert_model "${ANTI_MODEL}"
                 --alpha "${ALPHA:-1.0}"
                 --base_weight "${BASE_WEIGHT:-1.0}"
                 --expert_weight "${EXPERT_WEIGHT:-1.0}"
                 --plausibility_alpha "${PLAUSIBILITY_ALPHA:-0.1}")
    ;;
  *)
    echo "unknown method ${METHOD}" >&2
    exit 1
    ;;
esac

want () {
    case " ${TASKS} " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}

stem () {
    python -m scripts.canonical --method "${METHOD}" --lang "${EVAL_LANG}" \
        --task "$1" --models "${MODELS[@]}"
}

run_one () {
    local short=$1 task=$2 subdir=$3 nex=$4 mnt=$5
    shift 5
    want "${short}" || return 0
    local STEM
    STEM=$(stem "${short}")
    mkdir -p "$(dirname "${STEM}")"
    python -m scripts.infer \
        --method "${METHOD}" "${METHOD_ARGS[@]}" \
        --task "${task}" \
        --eval_lang "${EVAL_LANG}" \
        --prompt_lang "${PROMPT_LANG}" \
        --num_exemplar "${nex}" \
        --max_new_tokens "${mnt}" \
        "$@" \
        --input_file "${DATA_ROOT}/${subdir}/${EVAL_LANG}/test.json" \
        --exemplar_file "${DATA_ROOT}/${subdir}/${EVAL_LANG}/train_1.json" \
        --output_file "${STEM}_preds.json"
}

run_one rc    reading_comprehension reading_comprehension 5 2
run_one rs    response_selection    response_selection    5 2
run_one title title_generation      title_generation_200  3 250
run_one math  math                  math                  5 250
run_one xx2en translation translation_dialogue 5 200 --src_lang "${EVAL_LANG}" --tgt_lang en
run_one en2xx translation translation_dialogue 5 200 --src_lang en --tgt_lang "${EVAL_LANG}"

echo "inference done: method=${METHOD} lang=${EVAL_LANG}"
