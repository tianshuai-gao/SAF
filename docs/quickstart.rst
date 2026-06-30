Quickstart
==========

This page shows how to run each decoding method on a single task. All methods
share one entry point, ``scripts/infer.py``; the ``--method`` flag selects the
decoder. Models load in bfloat16 by default; pass ``--load_in_8bit`` to load in
8-bit instead.

SAF-W
-----

Run SAF-W on Tibetan reading comprehension with a 32B host and a 1.5B scorer:

.. code-block:: bash

   python -m scripts.infer --method safw \
       --host_model Qwen/Qwen2.5-32B-Instruct \
       --scorer_model pkupie/Qwen2.5-1.5B-bo-cpt \
       --task reading_comprehension --eval_lang bo --prompt_lang en \
       --num_exemplar 5 --max_new_tokens 2 \
       --input_file data/reading_comprehension/bo/test.json \
       --exemplar_file data/reading_comprehension/bo/train_1.json \
       --output_file results/safw/bo/reading_comprehension.json

Uniform averaging (the math setting)
------------------------------------

Use ``--method safw_fixed`` with ``--beta_fixed 0.5`` for the
uniform-averaging reduction:

.. code-block:: bash

   python -m scripts.infer --method safw_fixed --beta_fixed 0.5 \
       --host_model Qwen/Qwen2.5-32B-Instruct \
       --scorer_model pkupie/Qwen2.5-1.5B-bo-cpt \
       --task math --eval_lang bo --prompt_lang en \
       --num_exemplar 5 --max_new_tokens 250 \
       --input_file data/math/bo/test.json \
       --exemplar_file data/math/bo/train_1.json \
       --output_file results/safw_fixed/bo/math.json

TriMix
------

.. code-block:: bash

   python -m scripts.infer --method trimix \
       --base_model Qwen/Qwen2.5-7B-Instruct \
       --expert_model pkupie/Qwen2.5-1.5B-bo-cpt \
       --antiexpert_model Qwen/Qwen2.5-1.5B \
       --base_weight 0.1 --expert_weight 1.0 \
       --task reading_comprehension --eval_lang bo --prompt_lang en \
       --num_exemplar 5 --max_new_tokens 2 \
       --input_file data/reading_comprehension/bo/test.json \
       --exemplar_file data/reading_comprehension/bo/train_1.json \
       --output_file results/trimix/bo/reading_comprehension.json

Proxy Tuning
------------

.. code-block:: bash

   python -m scripts.infer --method proxy \
       --base_model Qwen/Qwen2.5-7B-Instruct \
       --expert_model pkupie/Qwen2.5-1.5B-bo-cpt \
       --antiexpert_model Qwen/Qwen2.5-1.5B \
       --alpha 1.0 \
       --task reading_comprehension --eval_lang bo --prompt_lang en \
       --num_exemplar 5 --max_new_tokens 2 \
       --input_file data/reading_comprehension/bo/test.json \
       --exemplar_file data/reading_comprehension/bo/train_1.json \
       --output_file results/proxy/bo/reading_comprehension.json

Evaluating predictions
----------------------

Each run writes a dictionary of predictions keyed by example id. Score it with

.. code-block:: bash

   python -m scripts.evaluate --task reading_comprehension \
       --input_file data/reading_comprehension/bo/test.json \
       --pred_file results/safw/bo/reading_comprehension.json \
       --metrics_output_file results/safw/bo/reading_comprehension.metrics.json
