API reference
=============

Decoders
--------

The Proxy Tuning, TriMix, and SAF-W decoders share one KV-cached generation
loop and differ only in their fusion rule.

.. automodule:: safw.dexperts
   :members:
   :undoc-members:
   :show-inheritance:

Generation and model loading
----------------------------

.. automodule:: safw.utils
   :members:
   :undoc-members:

Prompts
-------

.. automodule:: safw.prompts
   :members:
   :undoc-members:

Evaluation
----------

.. automodule:: safw.eval
   :members:
   :undoc-members:

Pipeline scripts
----------------

The batch driver, the canonical path resolver, the inference engine,
and the evaluator live under ``scripts``.

.. automodule:: scripts.canonical
   :members:

.. automodule:: scripts.run
   :members:

.. automodule:: scripts.infer
   :members:

.. automodule:: scripts.evaluate
   :members:
