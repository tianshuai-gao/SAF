Installation
============

SAF-W requires Python 3.10 or newer and a CUDA-capable GPU for inference. The
package and its dependencies install with pip.

From source
-----------

.. code-block:: bash

   git clone <repository-url> SAF
   cd SAF
   pip install -e .

This installs the ``safw`` package together with its runtime dependencies:
PyTorch, Transformers, Accelerate, SentencePiece, NumPy, sacrebleu, and
rouge-score.

Optional dependency groups
---------------------------

.. code-block:: bash

   pip install -e ".[docs]"   # Sphinx and the Read the Docs theme
   pip install -e ".[test]"   # pytest

Docker
------

A CUDA runtime image is provided for reproducible GPU runs. Model weights are
not baked into the image; mount a Hugging Face cache at run time.

.. code-block:: bash

   docker build -t safw:latest .
   docker run --gpus all -v $PWD:/workspace \
       -v /path/to/hf_cache:/root/.cache/huggingface \
       safw:latest python -m scripts.infer --help

Determinism
-----------

For reproducible greedy decoding on Ampere and other CUDA GPUs, set

.. code-block:: bash

   export CUBLAS_WORKSPACE_CONFIG=:4096:8

The Docker image sets this automatically.
