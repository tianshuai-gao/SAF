Reproducing the experiments
===========================

This page describes how to reproduce the thesis results across all six
MiLiC-Eval tasks and four low-resource languages (Tibetan ``bo``, Uyghur
``ug``, Mongolian ``mn``, Kazakh ``kk``).

Tasks and settings
-------------------

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

Running a full language with SAF-W
----------------------------------

``scripts/run_safw.sh`` runs all six tasks for one language on a SLURM cluster.
It takes the evaluation language, the host model, and the scorer model:

.. code-block:: bash

   sbatch scripts/run_safw.sh bo \
       Qwen/Qwen2.5-32B-Instruct \
       pkupie/Qwen2.5-1.5B-bo-cpt

For the math task, run the uniform-averaging reduction by setting the
environment variables ``METHOD=safw_fixed`` and ``BETA_FIXED=0.5`` before
submitting.

Evaluation
----------

``scripts/eval.sh`` scores every prediction file for one language and writes a
metrics file beside each one:

.. code-block:: bash

   bash scripts/eval.sh bo safw

Multiple-choice tasks report accuracy, translation reports chrF++, and title
generation reports ROUGE-L. The metric per task is fixed in
:mod:`safw.eval`.

Selecting TriMix weights by perplexity
--------------------------------------

The TriMix baseline selects its base and expert weights per task and language
on a development set. ``scripts/get_perplexity.py`` sweeps the weight grid and
reports the lowest-perplexity combination:

.. code-block:: bash

   python -m scripts.get_perplexity \
       --base_model_name_or_path Qwen/Qwen2.5-7B-Instruct \
       --expert_model_name_or_path pkupie/Qwen2.5-1.5B-bo-cpt \
       --antiexpert_model_name_or_path Qwen/Qwen2.5-1.5B \
       --task_name reading_comprehension --lang bo --prompt_lang en \
       --input_file data/reading_comprehension/bo/test.json \
       --exemplar_file data/reading_comprehension/bo/train_1.json \
       --output_file results/ppl/bo_rc.json

Scales
------

The Qwen experiments use instruction models at three scales (7B, 14B, 32B) as
the host or base, with a 1.5B continually-pretrained model as the scorer or
expert and the 1.5B base model as the antiexpert. Pass the relevant model
paths to the host/scorer (SAF-W) or base/expert/antiexpert (baselines)
arguments.
