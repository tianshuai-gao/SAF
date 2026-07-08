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
PyTorch, Transformers, Accelerate, SentencePiece, NumPy, sacrebleu, and tqdm.

The ROUGE scorer is installed separately (next section) because it is a fork
that replaces the official package.

Multilingual ROUGE (required)
-----------------------------

Title generation is scored with the multilingual ROUGE fork from XL-Sum, which
installs *as* ``rouge_score`` and overrides the official package. Install it
after the package itself:

.. code-block:: bash

   pip install pyonmttok
   pip install "git+https://github.com/csebuetnlp/xl-sum.git#subdirectory=multilingual_rouge_scoring"

.. warning::

   Do not install the official ``rouge-score`` package. It shares the import
   name ``rouge_score`` and silently replaces the fork, which deflates title
   ROUGE-L scores on the four evaluation languages by roughly a factor of
   three. To verify the correct scorer is active:

   .. code-block:: bash

      python -c "from rouge_score import rouge_scorer; import inspect; \
      assert 'kwargs' in str(inspect.signature(rouge_scorer.RougeScorer.__init__)), 'wrong rouge'; \
      print('multilingual fork OK')"

   The Docker build runs this assertion automatically and fails if the wrong
   scorer is present.

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
