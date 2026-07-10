#!/usr/bin/env python3
"""Resolve the canonical results path for one run.

Every run lands under one tree, and the filename carries the full key,
so no two runs can collide:

    results/test_outputs/{method}/{family}/{lang}/{task}/
        {method}_{family}_{lang}_{task}_{key}_preds.json

The key is ins{scale} or cpt{size} for single models, {scale} for the
three-model baselines and for uniform, and inshost{scale} or
cpthost{scale} for SAF-W. Pass the host or base model first. Both the
run scripts and the eval scripts call this module, so the mapping lives
in one place.

Example::

    python -m scripts.canonical --method safw --lang bo --task rc \
        --models pkupie/Qwen2.5-1.5B-bo-cpt Qwen/Qwen2.5-32B-Instruct
    results/test_outputs/saf/qwen/bo/rc/saf_qwen_bo_rc_cpthost32B
"""
from __future__ import annotations

import argparse
import re

METHOD_DIR = {
    "safw": "saf",
    "safw_fixed": "uniform",
    "proxy": "proxy_tuning",
    "trimix": "trimix",
    "single": "single",
}


def family_of(names):
    joined = " ".join(n.lower() for n in names)
    if "qwen" in joined:
        return "qwen"
    if "gemma" in joined:
        return "gemma"
    raise ValueError(f"cannot infer family from {names}")


def size_of(name):
    match = re.search(r"(\d+(?:\.\d+)?)[bB]", name)
    if not match:
        raise ValueError(f"no parameter size in {name}")
    return match.group(1) + "B"


def is_cpt(name):
    return "cpt" in name.lower()


def key_of(method, models):
    if method == "single":
        tag = "cpt" if is_cpt(models[0]) else "ins"
        return tag + size_of(models[0])
    large = next((n for n in models if not is_cpt(n)), models[0])
    scale = size_of(large)
    if method == "safw":
        tag = "cpthost" if is_cpt(models[0]) else "inshost"
        return tag + scale
    return scale


def resolve(method, lang, task, models):
    mdir = METHOD_DIR[method]
    fam = family_of(models)
    key = key_of(method, models)
    stem = f"{mdir}_{fam}_{lang}_{task}_{key}"
    return f"results/test_outputs/{mdir}/{fam}/{lang}/{task}/{stem}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=sorted(METHOD_DIR))
    p.add_argument("--lang", required=True)
    p.add_argument("--task", required=True,
                   choices=["rc", "rs", "title", "xx2en", "en2xx", "math"])
    p.add_argument("--models", nargs="+", required=True,
                   help="Host or base model first.")
    args = p.parse_args()
    print(resolve(args.method, args.lang, args.task, args.models))


if __name__ == "__main__":
    main()
