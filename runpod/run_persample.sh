#!/bin/bash
# Per-sample host selection experiment (7B host + 1.5B-bo-cpt scorer).
# Usage: bash runpod/run_persample.sh [task] [lang] [n_examples] [batch_size]
#
# Pipeline:
#   1. build prompts (exemplars + input, no answer)
#   2. fused_ppl.py: per-sample fused perplexity, pick lower-ppl host per example
#   3. decode BOTH assignments (A-led: host=7B, B-led: host=cpt)
#   4. assemble per-sample prediction from the chosen host
#   5. assemble ORACLE prediction (uses gold; upper bound only)
#   6. score per-sample vs A-led vs B-led vs oracle, plus host agreement
set -e

TASK=${1:-reading_comprehension}
LANG=${2:-bo}
NMAX=${3:-50}
BATCH=${4:-8}

HOST7B=./models/Qwen2.5-7B-Instruct
CPT=./models/Qwen2.5-1.5B-bo-cpt
DATA=data/${TASK}/${LANG}
OUT=results/persample/${TASK}_${LANG}
mkdir -p ${OUT}

echo "=== [1] build prompts (no answer) ==="
python3 -c "
import json, sys; sys.path.insert(0,'.')
from safw.prompts import TASK_BUILDERS
test = json.load(open('${DATA}/test.json'))[:${NMAX}]
ex   = json.load(open('${DATA}/train_1.json'))
conv = TASK_BUILDERS['${TASK}'](test, ex, eval_lang='${LANG}', num_exemplar=3, prompt_lang='en')
json.dump([{'id':c['id'],'input':c['input']} for c in conv], open('${OUT}/prompts.json','w'), ensure_ascii=False)
json.dump({c['id']:c['gold'] for c in conv}, open('${OUT}/gold.json','w'), ensure_ascii=False)
print('built', len(conv), 'prompts')
"

echo "=== [2] per-sample host selection by fused ppl ==="
python3 fused_ppl.py \
    --model_a ${HOST7B} --model_b ${CPT} \
    --prompts_file ${OUT}/prompts.json \
    --output_file ${OUT}/selection.json \
    --device cuda

echo "=== [3a] decode A-led (host=7B, scorer=cpt) ==="
python3 -m scripts.infer --method safw \
    --host_model ${HOST7B} --scorer_model ${CPT} \
    --task ${TASK} --eval_lang ${LANG} --prompt_lang en \
    --num_exemplar 3 --max_new_tokens 2 --batch_size ${BATCH} \
    --max_test_example_num ${NMAX} \
    --input_file ${DATA}/test.json --exemplar_file ${DATA}/train_1.json \
    --output_file ${OUT}/pred_A_led.json

echo "=== [3b] decode B-led (host=cpt, scorer=7B) ==="
python3 -m scripts.infer --method safw \
    --host_model ${CPT} --scorer_model ${HOST7B} \
    --task ${TASK} --eval_lang ${LANG} --prompt_lang en \
    --num_exemplar 3 --max_new_tokens 2 --batch_size ${BATCH} \
    --max_test_example_num ${NMAX} \
    --input_file ${DATA}/test.json --exemplar_file ${DATA}/train_1.json \
    --output_file ${OUT}/pred_B_led.json

echo "=== [4+5+6] assemble per-sample & oracle, then score ==="
python3 -c "
import json
sel  = {r['id']:r['host'] for r in json.load(open('${OUT}/selection.json'))}
pa   = json.load(open('${OUT}/pred_A_led.json'))
pb   = json.load(open('${OUT}/pred_B_led.json'))
gold = json.load(open('${OUT}/gold.json'))
def norm(x):
    x = (x or '').strip()
    return x[0] if x and x[0] in 'ABCDE' else x
ids = list(gold)
def acc(predmap):
    return sum(1 for i in ids if norm(predmap[i])==gold[i])/len(ids)
per_sample = {}; oracle = {}; match_oracle = 0
for i in ids:
    chosen = sel[i]
    per_sample[i] = pa[i] if chosen=='A' else pb[i]
    a_ok = norm(pa[i])==gold[i]; b_ok = norm(pb[i])==gold[i]
    oracle[i] = pa[i] if a_ok else (pb[i] if b_ok else pa[i])
    oracle_host = 'A' if a_ok else ('B' if b_ok else 'A')
    if chosen == oracle_host: match_oracle += 1
print('--- results (${TASK}/${LANG}, n=%d) ---' % len(ids))
print('A-led (per-task host=7B) acc : %.3f' % acc(pa))
print('B-led (per-task host=cpt) acc: %.3f' % acc(pb))
print('per-sample (fused ppl)   acc : %.3f' % acc(per_sample))
print('oracle (upper bound)     acc : %.3f' % acc(oracle))
print('fused-ppl host == oracle host: %.1f%%' % (100*match_oracle/len(ids)))
"
echo "=== done. results in ${OUT}/ ==="
