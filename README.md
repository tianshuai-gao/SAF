# Constrained Decoding for Large Language Models

**Test-Time Logit Fusion for Low-Resource Language Adaptation**

This repository holds the data analysis pipeline for the research project
of the MPhil in Data Intensive Science at the University of Cambridge.
The associated project report and executive summary are submitted
separately. The report describes the method and the results. This README
describes how to install, run, and reproduce everything.

The project studies decoding-time logit fusion for adapting large
language models to four low-resource languages. It reproduces two
published fusion baselines, Proxy Tuning and TriMix, and proposes
SAF-W. SAF-W fuses a host and a scorer model at every decoding step
with a per-token endorsement weight. No parameters are updated and no
extra model is trained.

## Table of Contents

- [Data Availability](#data-availability)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Layout](#repository-layout)
- [Testing](#testing)
- [Documentation](#documentation)
- [Use of Auto-generation Tools](#use-of-auto-generation-tools)
- [Support](#support)
- [License](#license)
- [Authors and Acknowledgment](#authors-and-acknowledgment)

## Data Availability

The MiLiC-Eval splits ship with the repository under `data/`, one
directory per task and language. Each language directory holds a
`test.json` and the few-shot exemplar files `train_<seed>.json`.

All models are frozen public checkpoints from Hugging Face. The Qwen
experiments use `Qwen/Qwen2.5-{7B,14B,32B}-Instruct` as hosts or bases,
`pkupie/Qwen2.5-1.5B-<lang>-cpt` as scorers or experts, and
`Qwen/Qwen2.5-1.5B` as the antiexpert for the baselines. On a cluster
without internet access, set `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` and point the Hugging Face cache at a
pre-downloaded directory.

Results live under `results/` in a canonical tree described below.
Prediction and metric files are populated from the CSD3 runs. Files
start as zero-byte placeholders, and a zero-byte file never blocks a
run and never counts as evaluated. Audit the unfilled files with:

```bash
find results/test_outputs -name "*.json" -size 0 | wc -l
```

## Installation

### Requirements

- Python 3.10 or newer
- A CUDA-capable GPU for inference
- Docker, if you prefer the containerised setup

### Local setup

```bash
git clone https://github.com/tianshuai-gao/SAF.git
cd SAF
pip install -e .
```

Title generation is scored with the multilingual ROUGE fork from
XL-Sum. It installs as `rouge_score` and must override the official
package:

```bash
pip install pyonmttok
pip install "git+https://github.com/csebuetnlp/xl-sum.git#subdirectory=multilingual_rouge_scoring"
```

Do not install the official `rouge-score` package. It shares the import
name and silently deflates the title scores. Verify the correct scorer:

```bash
python -c "from rouge_score import rouge_scorer; import inspect; \
assert 'kwargs' in str(inspect.signature(rouge_scorer.RougeScorer.__init__)), 'wrong rouge'; \
print('multilingual fork OK')"
```

### Docker

```bash
docker build -t safw:latest .
docker run --gpus all -v $PWD:/workspace \
    -v /path/to/hf_cache:/root/.cache/huggingface \
    safw:latest python -m scripts.run --help
```

The build installs the ROUGE fork and fails if the wrong scorer is
present. The image was built and CPU-tested on macOS. The unit tests
pass inside the container. On an HPC system without Docker, the image
runs through Apptainer with the `--nv` flag.

## Usage

### One cell in three commands

A cell is one method, one language, one scale, and one task.
`scripts/run.py` declares every run explicitly and cross-checks the
declarations against the model names before anything loads.

```bash
python -m scripts.run --method safw --lang bo --scale 7B --host ins --tasks rc --dry_run
python -m scripts.run --method safw --lang bo --scale 7B --host ins --tasks rc
bash scripts/eval.sh
```

The first command prints the plan and the output path, then exits. A
declaration mismatch stops with a message and a non-zero exit code. The
second command runs inference. The third walks the results tree and
writes a metrics file with a metadata header next to every non-empty
predictions file.

### The declaration driver

| Flag | Meaning |
| --- | --- |
| `--method` | `safw`, `safw_fixed`, `proxy`, or `trimix` |
| `--family` | `qwen` (default) or `gemma` |
| `--lang` | `bo`, `ug`, `mn`, or `kk` |
| `--scale` | `7B`, `14B`, `32B`, `12B`, or `27B` |
| `--host` | `ins` or `cpt`. SAF-W only |
| `--tasks` | any of `rc rs title math xx2en en2xx`. Defaults to all six |

Model paths resolve from a built-in zoo and can be overridden. Every
declaration is validated against the model names, so a wrong scale, a
wrong language, or a model in the wrong slot refuses to run.

### Methods

SAF-W needs a host direction. Uniform averaging is the symmetric
`beta = 0.5` reduction of SAF-W, decodes the math task, and takes no
host. TriMix takes per-task weights from its perplexity heuristic,
computed by `scripts/get_perplexity.py`:

```bash
python -m scripts.run --method safw       --lang bo --scale 32B --host cpt
python -m scripts.run --method safw_fixed --lang bo --scale 32B --tasks math
python -m scripts.run --method proxy      --lang bo --scale 7B
python -m scripts.run --method trimix     --lang bo --scale 7B --tasks rc rs --base_weight 0.1
```

The deployed SAF-W host per cell follows the devx selection recorded
under `results/devx` and reported in the paper.

### On the cluster

`scripts/run.sh` is a thin SBATCH wrapper around the driver. One
submission covers one method, language, and scale across all six tasks:

```bash
sbatch scripts/run.sh --method safw --lang bo --scale 32B --host cpt
```

### The results tree

`scripts/canonical.py` is the single source of truth for output paths:

```text
results/test_outputs/{method}/{family}/{lang}/{task}/
    {method}_{family}_{lang}_{task}_{key}_preds.json
    {method}_{family}_{lang}_{task}_{key}_metrics.json
```

The key is `ins{scale}` or `cpt{size}` for single models, `{scale}` for
the three-model baselines and uniform averaging, and `inshost{scale}`
or `cpthost{scale}` for SAF-W. Every metrics file carries a metadata
header with the method, family, language, task, key, example count, and
source filename.

## Repository Layout

```text
safw/            the package: decoders, generation loop, prompts, evaluation
scripts/         run.py (driver), infer.py (engine), canonical.py (paths),
                 evaluate.py, eval.sh, run.sh, get_perplexity.py, make_devx.py
data/            MiLiC-Eval splits, one directory per task and language
results/         devx/ (host-selection scores) and test_outputs/ (canonical tree)
exploration/     development-time diagnostic scripts and their records/
docs/            Sphinx documentation
tests/           unit tests for the decoders and the driver checks
runpod/          environment setup for ad-hoc GPU pods
```

`exploration/records/` keeps the development diagnostics for
provenance. Paper-facing numbers live only under `results/`.

## Testing

```bash
pip install -e ".[test]"
pytest tests/ -q
```

Eighteen tests cover the fusion rules, the first-token anchor, and the
twelve declaration checks of the driver. The GitLab CI runs the same
suite.

## Documentation

```bash
pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html
```

The documentation covers the method with full equations, installation,
a quickstart, the reproduction guide, the data layout, and the API
reference.

## Use of Auto-generation Tools

Claude (Anthropic) assisted with code, experiment orchestration,
debugging, and report drafting. Every generated artefact was reviewed,
tested, and verified by the author. The full declaration is in the
appendix of the project report.

## Support

For questions, contact tg561@cam.ac.uk.

## License

This project is licensed under the MIT License. See the LICENSE file.

## Authors and Acknowledgment

Tianshuai Gao, Sidney Sussex College, MPhil in Data Intensive Science,
supervised by Dr Weiwei Sun. The decoding framework extends the
released TriMix codebase, and the evaluation uses the MiLiC-Eval
benchmark. The CPT checkpoints follow the TriMix release.
