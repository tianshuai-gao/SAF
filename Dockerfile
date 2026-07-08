# Reproducible environment for SAF-W.
#
# Built on the NVIDIA CUDA runtime so the image runs on GPU nodes. It installs
# the Python dependencies and the SAF-W package. Model weights are NOT baked
# into the image; mount a Hugging Face cache at run time.
#
# Build:
#   docker build -t safw:latest .
# Run (mount a model cache and the working directory):
#   docker run --gpus all -v $PWD:/workspace \
#       -v /path/to/hf_cache:/root/.cache/huggingface \
#       safw:latest python -m scripts.infer --help

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /workspace

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .
RUN python -m pip install -e .

# Multilingual ROUGE (xl-sum fork) must override any official rouge-score
# pulled in transitively. Fail the build if the wrong one is present.
RUN pip uninstall -y rouge-score rouge_score || true
RUN pip install --no-cache-dir pyonmttok \
    "git+https://github.com/csebuetnlp/xl-sum.git#subdirectory=multilingual_rouge_scoring" \
    pytest
RUN python -c "from rouge_score import rouge_scorer; import inspect; sig = str(inspect.signature(rouge_scorer.RougeScorer.__init__)); assert 'kwargs' in sig, 'official rouge-score detected: ' + sig; print('rouge check OK')"

# Deterministic decoding on Ampere and other CUDA GPUs.
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

CMD ["python", "-m", "scripts.infer", "--help"]
