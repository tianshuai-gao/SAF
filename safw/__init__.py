"""SAF-W and baseline logit-fusion decoding for low-resource language adaptation.

The package implements the method of this thesis, Scorer-Adaptive Fusion with
Weighting (:class:`SAFW`), together with the Proxy Tuning
(:class:`DExpertsLlama`) and TriMix (:class:`TriMixQwen`) baselines. All three
share one KV-cached generation loop and differ only in their fusion rule. The
package also provides batched generation, prompt builders, and the evaluation
metrics for the MiLiC-Eval tasks.
"""

from . import prompts, eval
from .dexperts import DExpertsLlama, TriMixQwen, TriMixGemma, SAFW
from .utils import (
    generate_completions,
    add_pad_token,
    load_trimix,
    load_proxy,
    load_safw,
    KeyWordsCriteria,
)

__all__ = [
    "prompts",
    "eval",
    "DExpertsLlama",
    "TriMixQwen",
    "TriMixGemma",
    "SAFW",
    "generate_completions",
    "add_pad_token",
    "load_trimix",
    "load_proxy",
    "load_safw",
    "KeyWordsCriteria",
]
__version__ = "1.0.0"
