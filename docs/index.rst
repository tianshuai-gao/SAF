SAF-W documentation
===================

Scorer-Adaptive Fusion with Weighting (SAF-W) is a training-free, test-time
logit-fusion decoder for adapting large language models to low-resource
languages. At every decoding step it combines the next-token logits of two
frozen models, a host and a scorer, using a per-token endorsement weight. No
parameters are updated and no extra model is trained.

This documentation covers the method, installation, a quickstart for each
decoding method, full instructions for reproducing the experiments, the data
and model layout, and the API reference.

.. toctree::
   :maxdepth: 2
   :caption: Guide

   method
   installation
   quickstart
   reproduction
   data
   citing

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

The core mechanism
------------------

At each step the host proposes its top token. The scorer reports the
probability it assigns to that token; this endorsement sets the mixing weight.
High endorsement leans the fused distribution toward the host, low endorsement
toward the scorer, and an endorsement of one half averages the two models.
Uniform averaging is recovered as the special case used for the math task. The
Proxy Tuning and TriMix baselines reuse the same generation loop and differ
only in their fusion rule. See :doc:`method` for the full description.

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
