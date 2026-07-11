Reproducing the experiments
===========================

Results layout
--------------

Every run lands in one canonical tree. ``scripts/canonical.py`` is the
single source of truth for the mapping:

.. code-block:: text

   results/test_outputs/{method}/{family}/{lang}/{task}/
       {method}_{family}_{lang}_{task}_{key}_preds.json
       {method}_{family}_{lang}_{task}_{key}_metrics.json

The key is ``ins{scale}`` or ``cpt{size}`` for single models,
``{scale}`` for the three-model baselines and uniform averaging, and
``inshost{scale}`` or ``cpthost{scale}`` for SAF-W. Files start as
zero-byte placeholders. A zero-byte file never blocks a run and never
counts as evaluated. Audit the unfilled files with:

.. code-block:: bash

   find results/test_outputs -name "*.json" -size 0 | wc -l

Task settings
-------------

The six tasks and their generation settings follow the MiLiC-Eval protocol.

.. list-table::
   :header-rows: 1
   :widths: 30 30 15 15

   * - Task
     - Data subdirectory
     - Exemplars
     - Max new tokens
   * - Reading comprehension
     - ``reading_comprehension``
     - 5
     - 2
   * - Response selection
     - ``response_selection``
     - 5
     - 2
   * - Title generation
     - ``title_generation_200``
     - 3
     - 250
   * - Math
     - ``math``
     - 5
     - 250
   * - Translation (xx2en)
     - ``translation_dialogue``
     - 5
     - 200
   * - Translation (en2xx)
     - ``translation_dialogue``
     - 5
     - 200

Full grid
---------

On the cluster, one submission covers one method, language, and scale
across all six tasks. ``scripts/run.sh`` is a thin SBATCH wrapper
around the driver:

.. code-block:: bash

   for LANG in bo ug mn kk; do
     sbatch scripts/run.sh --method single --lang $LANG --host cpt
     for SCALE in 7B 14B 32B; do
       sbatch scripts/run.sh --method single --lang $LANG --scale $SCALE --host ins
       sbatch scripts/run.sh --method safw --lang $LANG --scale $SCALE --host ins
       sbatch scripts/run.sh --method safw --lang $LANG --scale $SCALE --host cpt
       sbatch scripts/run.sh --method safw_fixed --lang $LANG --scale $SCALE
       sbatch scripts/run.sh --method proxy --lang $LANG --scale $SCALE
     done
   done

The deployed SAF-W host per cell follows the devx selection recorded
under ``results/devx`` and reported in the paper.

TriMix weights
--------------

TriMix uses per-task weights from its perplexity heuristic. Restrict
the tasks and set the weights per submission:

.. code-block:: bash

   sbatch scripts/run.sh --method trimix --lang bo --scale 7B \
       --tasks rc rs --base_weight 0.1 --expert_weight 1.0

Uniform averaging and math
--------------------------

Math decodes with uniform averaging. The mode is symmetric, so no host
is declared:

.. code-block:: bash

   sbatch scripts/run.sh --method safw_fixed --lang bo --scale 7B --tasks math

Evaluation
----------

.. code-block:: bash

   bash scripts/eval.sh

The sweep is idempotent. It skips every cell with a non-empty metrics
file already in place.
