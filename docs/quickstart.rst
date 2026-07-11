Quickstart
==========

One cell means one method, one language, one scale, and one task. The
driver ``scripts/run.py`` declares every run explicitly and checks the
declarations against the model names before anything loads.

Preview the plan
----------------

.. code-block:: bash

   python -m scripts.run --method safw --lang bo --scale 7B \
       --host ins --tasks rc --dry_run

The driver prints one plan line per task and exits. A declaration
mismatch stops here with a message and a non-zero exit code.

Run and evaluate
----------------

.. code-block:: bash

   python -m scripts.run --method safw --lang bo --scale 7B \
       --host ins --tasks rc
   bash scripts/eval.sh

``eval.sh`` walks ``results/test_outputs``, evaluates every non-empty
predictions file without a metrics file, and writes the metric with a
metadata header next to it.

.. code-block:: bash

   cat results/test_outputs/saf/qwen/bo/rc/saf_qwen_bo_rc_inshost7B_metrics.json

Single runs without the driver
------------------------------

``scripts/infer.py`` is the parametrised engine underneath. It runs one
task with explicit paths and suits custom settings:

.. code-block:: bash

   python -m scripts.infer --method safw \
       --host_model Qwen/Qwen2.5-7B-Instruct \
       --scorer_model pkupie/Qwen2.5-1.5B-bo-cpt \
       --task reading_comprehension --eval_lang bo --prompt_lang en \
       --num_exemplar 5 --max_new_tokens 2 \
       --input_file data/reading_comprehension/bo/test.json \
       --exemplar_file data/reading_comprehension/bo/train_1.json \
       --output_file /tmp/bo_rc_preds.json
