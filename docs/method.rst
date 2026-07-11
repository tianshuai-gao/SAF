The SAF-W method
================

SAF-W fuses two frozen models at every decoding step. The host proposes
its top token. The scorer reports the probability it assigns to that
token. This probability is the endorsement ``e``. The scorer weight is
``beta = 1 - e`` and the fused logits are
``(1 - beta) * L_host + beta * L_scorer``. High endorsement keeps the
host in charge. Low endorsement hands weight to the scorer.

Anchor
------

Generation tasks pin the first token to the model that owns the output
language. Title generation and en2xx translation anchor on the CPT
model. xx2en translation anchors on the instruction model. The anchor
fires only at the first step and only on generation tasks.

Host selection
--------------

Which model hosts is a per task and per language choice. The choice is
made once on the devx set. The devx set contains all annotated data
outside the prompt exemplars, namely dev, train_2, and train_3. The
selection scores live under ``results/devx``. The report carries the
full analysis.

Uniform averaging
-----------------

A constant ``beta = 0.5`` reduces SAF-W to uniform averaging. The
reduction is symmetric in the two models, so no host choice applies.
Math decodes in this mode.
