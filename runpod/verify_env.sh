#!/bin/bash
# One-shot environment check: run a single SAF-W decode on bo rc examples.
# If this prints non-empty answers in seconds (not minutes), the 4.47.0 KV
# cache works and the GPU path is correct. This is the gate before any real run.
set -e

echo "[verify] running SAF-W decode (should be seconds on GPU) ..."
python3 -m scripts.infer --method safw \
    --host_model ./models/Qwen2.5-7B-Instruct \
    --scorer_model ./models/Qwen2.5-1.5B-bo-cpt \
    --task reading_comprehension --eval_lang bo --prompt_lang en \
    --num_exemplar 3 --max_new_tokens 2 \
    --max_test_example_num 2 \
    --input_file data/reading_comprehension/bo/test.json \
    --exemplar_file data/reading_comprehension/bo/train_1.json \
    --output_file /tmp/verify_safw.json

echo "[verify] output:"
cat /tmp/verify_safw.json
echo ""
echo "[verify] if answers are non-empty and this finished fast, env is correct."
