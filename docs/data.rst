Data and models
===============

Datasets
--------

The experiments use the MiLiC-Eval benchmark across four low-resource
languages: Tibetan (``bo``), Uyghur (``ug``), Mongolian (``mn``), and Kazakh
(``kk``). The expected layout under ``data/`` is one directory per task, then
one directory per language, each holding a ``test.json`` and few-shot exemplar
files ``train_<seed>.json``:

.. code-block:: text

   data/
     reading_comprehension/<lang>/{test.json, train_1.json}
     response_selection/<lang>/{test.json, train_1.json}
     title_generation_200/<lang>/{test.json, train_1.json}
     math/<lang>/{test.json, train_1.json}
     translation_dialogue/<lang>/{test.json, train_1.json}

Each example has an ``id`` field and the task-specific fields the prompt
builders in :mod:`safw.prompts` expect (for example ``context``, ``options``,
and ``answer`` for the multiple-choice tasks; ``question`` and ``answer`` for
math; the source and target language keys for translation).

Models
------

The Qwen experiments use the following frozen checkpoints:

* Hosts / bases: ``Qwen/Qwen2.5-7B-Instruct``, ``Qwen/Qwen2.5-14B-Instruct``,
  ``Qwen/Qwen2.5-32B-Instruct``.
* Scorers / experts: ``pkupie/Qwen2.5-1.5B-<lang>-cpt`` (continually
  pretrained on each language).
* Antiexpert (baselines only): ``Qwen/Qwen2.5-1.5B``.

Models load in bfloat16 by default. On a cluster without internet access, set
``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1`` and point the Hugging Face
cache at a pre-downloaded model directory.
