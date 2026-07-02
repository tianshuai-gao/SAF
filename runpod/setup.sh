#!/bin/bash
# RunPod environment setup for SAF-W.
# Pins transformers to 4.47.0 so the cached generation loop in safw/dexperts.py
# works (the _supports_cache_class hack and prepare_inputs_for_generation
# interface changed in newer versions and break KV caching).
set -e

echo "[setup] installing pinned dependencies ..."
pip install -q torch==2.5.1
pip install -q transformers==4.47.0 accelerate sentencepiece numpy sacrebleu tqdm
# multilingual ROUGE (MiLiC-Eval parity); overrides official rouge-score
pip install -q pyonmttok
pip install -q "git+https://github.com/csebuetnlp/xl-sum.git#subdirectory=multilingual_rouge_scoring"
pip install -q huggingface_hub

echo "[setup] verifying versions ..."
python3 -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__); assert transformers.__version__ == '4.47.0', 'WRONG transformers version'; print('OK pinned versions correct')"

echo "[setup] installing the safw package (editable) ..."
pip install -q -e .

echo "[setup] done."
