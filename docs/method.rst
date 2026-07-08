Method
======

This page describes the SAF-W decoding rule and the two baselines it is
compared against. All three methods fuse the next-token logits of frozen
models at every decoding step and differ only in how the logits are combined.
The generation loop, prompt construction, and evaluation are shared.

Notation
--------

Let :math:`L_{\text{host}}` and :math:`L_{\text{scorer}}` be the next-token
logit vectors of the two SAF-W models at a given step, and let
:math:`p_{\text{scorer}} = \operatorname{softmax}(L_{\text{scorer}})` be the
scorer's next-token distribution. For the three-model baselines, write
:math:`L_{\text{base}}`, :math:`L_{\text{expert}}`, and
:math:`L_{\text{anti}}` for the base (large instruction model), expert
(small continually-pretrained model), and antiexpert (small base model).

SAF-W
-----

SAF-W combines a **host** and a **scorer**. The host proposes its most likely
token

.. math::

   t^\star = \arg\max_v L_{\text{host}}[v].

The scorer reports the probability it assigns to that token, the
**endorsement**

.. math::

   e = p_{\text{scorer}}[t^\star].

The scorer weight is

.. math::

   \beta = \min(1,\ 1 - e + \delta),

where :math:`\delta \ge 0` is a per-task offset. The fused logits are

.. math::

   L_{\text{fused}} = (1 - \beta)\, L_{\text{host}} + \beta\, L_{\text{scorer}}.

When the scorer strongly endorses the host's token, :math:`e \to 1`, the
weight is small and the fused distribution follows the host. When the scorer
rejects it, :math:`e \to 0`, the weight approaches one and the fused
distribution follows the scorer. The offset raises the scorer's floor weight
on tasks that benefit from stronger correction, and the ``min`` keeps
:math:`\beta` a valid mixture coefficient.

The host and scorer roles are not fixed. They are chosen per task, language,
and scale on held-out training data; see the host selection protocol in the
reproduction guide. Either model can take either role.

First-token language anchor
---------------------------

On generation tasks the first decoded token fixes the output language, and a
host that prefers the wrong language can derail the whole sequence. SAF-W
therefore supports a first-token anchor. With ``anchor_lang="lrl"`` the first
token comes from the CPT model alone, with ``anchor_lang="en"`` from the
instruction model alone, and fusion resumes from the second token. The anchor
follows the required output language, not the host direction.

Uniform averaging as a special case
------------------------------------

Fixing the scorer weight to a constant :math:`\beta = 0.5` removes the
dependence on the endorsement and reduces SAF-W to a uniform average of the
two models:

.. math::

   L_{\text{fused}} = \tfrac{1}{2} L_{\text{host}} + \tfrac{1}{2} L_{\text{scorer}}.

This :math:`\delta = 0` reduction is used for the math task. Trusting the host
on math suppresses the chain-of-thought reasoning the scorer contributes, so
uniform averaging, which keeps both models in the mix at every step, is the
appropriate setting there. The endorsement form and the uniform-averaging
reduction are the same method at two settings of one weight; the latter is a
degenerate special case, not a separate baseline.

Proxy Tuning (baseline)
-----------------------

Proxy Tuning adds the expert/antiexpert residual to the base logits:

.. math::

   L_{\text{fused}} = L_{\text{base}}
   + \alpha\, (L_{\text{expert}} - L_{\text{anti}}).

Implemented by :class:`safw.dexperts.DExpertsLlama`.

TriMix (baseline)
-----------------

TriMix takes a weighted combination of the three models,

.. math::

   L_{\text{fused}} = w_b\, L_{\text{base}} + w_e\, L_{\text{expert}}
   + (1 - w_b - w_e)\, L_{\text{anti}},

and applies a plausibility constraint that masks tokens the plausibility model
finds unlikely (logit below :math:`\log \alpha` under the row maximum). The
weights :math:`w_b` and :math:`w_e` are selected per task and language on a
development set by perplexity, using ``scripts/get_perplexity.py`` (see
:doc:`reproduction`). Implemented by :class:`safw.dexperts.TriMixQwen`.
