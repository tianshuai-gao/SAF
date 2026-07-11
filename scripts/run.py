#!/usr/bin/env python3
"""Batch driver for the MiLiC-Eval runs.

Every run is declared explicitly. The driver takes --method, --family,
--lang, --scale, --host, and --tasks. It resolves model paths from a
small zoo and cross-checks every declaration against the model names.
Any mismatch stops the run before it starts. Model paths can be
overridden, and the same checks then apply to the overrides.

The driver prints one plan line per task and calls scripts.infer for
each. Use --dry_run to see the plan without running anything.

Examples::

    python -m scripts.run --method safw --lang bo --scale 32B --host cpt
    python -m scripts.run --method trimix --lang bo --scale 7B \
        --tasks rc rs --base_weight 0.1 --dry_run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from scripts.canonical import resolve, size_of, is_cpt

INSTRUCT = {
    ("qwen", "7B"): "Qwen/Qwen2.5-7B-Instruct",
    ("qwen", "14B"): "Qwen/Qwen2.5-14B-Instruct",
    ("qwen", "32B"): "Qwen/Qwen2.5-32B-Instruct",
    ("gemma", "12B"): "google/gemma-3-12b-it",
    ("gemma", "27B"): "google/gemma-3-27b-it",
}
CPT = {"qwen": "pkupie/Qwen2.5-1.5B-{lang}-cpt"}
ANTI = {"qwen": "Qwen/Qwen2.5-1.5B"}

# short task: infer task, data subdir, exemplars, max_new_tokens, extra args
TASK_TABLE = {
    "rc":    ("reading_comprehension", "reading_comprehension", 5, 2, []),
    "rs":    ("response_selection", "response_selection", 5, 2, []),
    "title": ("title_generation", "title_generation_200", 3, 250, []),
    "math":  ("math", "math", 5, 250, []),
    "xx2en": ("translation", "translation_dialogue", 5, 200,
              ["--src_lang", "{lang}", "--tgt_lang", "en"]),
    "en2xx": ("translation", "translation_dialogue", 5, 200,
              ["--src_lang", "en", "--tgt_lang", "{lang}"]),
}
TASK_ORDER = ["rc", "rs", "title", "math", "xx2en", "en2xx"]


def fail(msg):
    sys.exit(f"declaration mismatch: {msg}")


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True,
                   choices=["safw", "safw_fixed", "proxy", "trimix"])
    p.add_argument("--family", default="qwen", choices=["qwen", "gemma"])
    p.add_argument("--lang", required=True, choices=["bo", "ug", "mn", "kk"])
    p.add_argument("--scale", required=True,
                   choices=["7B", "14B", "32B", "12B", "27B"])
    p.add_argument("--host", choices=["ins", "cpt"],
                   help="SAF-W only. Which model leads the decoding.")
    p.add_argument("--tasks", nargs="+", default=TASK_ORDER,
                   choices=TASK_ORDER)
    p.add_argument("--host_model")
    p.add_argument("--scorer_model")
    p.add_argument("--base_model")
    p.add_argument("--expert_model")
    p.add_argument("--antiexpert_model")
    p.add_argument("--beta_fixed", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--base_weight", type=float, default=1.0)
    p.add_argument("--expert_weight", type=float, default=1.0)
    p.add_argument("--plausibility_alpha", type=float, default=0.1)
    p.add_argument("--prompt_lang", default="en")
    p.add_argument("--data_root", default="data")
    p.add_argument("--dry_run", action="store_true")
    return p


def default_instruct(args):
    key = (args.family, args.scale)
    if key not in INSTRUCT:
        fail(f"no default instruct model for {key}. Pass the model path.")
    return INSTRUCT[key]


def default_cpt(args):
    if args.family not in CPT:
        fail(f"no default cpt model for family {args.family}. Pass the model path.")
    return CPT[args.family].format(lang=args.lang)


def validate_common(args, instruct, cpt):
    for m in (instruct, cpt):
        if args.family not in m.lower():
            fail(f"--family {args.family} but model is {m}")
    if size_of(instruct) != args.scale:
        fail(f"--scale {args.scale} but instruct model is {instruct}")
    if is_cpt(instruct):
        fail(f"instruct slot holds a cpt model: {instruct}")
    if not is_cpt(cpt):
        fail(f"cpt slot holds a non-cpt model: {cpt}")
    if f"-{args.lang}-" not in cpt:
        fail(f"--lang {args.lang} but cpt model is {cpt}")


def resolve_models(args):
    """Return (models_in_canonical_order, method_args)."""
    if args.method in ("safw", "safw_fixed"):
        if args.method == "safw" and args.host is None:
            fail("--host is required for safw")
        if args.method == "safw_fixed" and args.host is not None:
            fail("--host does not apply to safw_fixed, which is symmetric")
        if args.base_model or args.expert_model or args.antiexpert_model:
            fail("base/expert/antiexpert models do not apply to SAF-W")
        host_is_cpt = args.host == "cpt"
        instruct = args.scorer_model if host_is_cpt else args.host_model
        cpt = args.host_model if host_is_cpt else args.scorer_model
        instruct = instruct or default_instruct(args)
        cpt = cpt or default_cpt(args)
        validate_common(args, instruct, cpt)
        host, scorer = (cpt, instruct) if host_is_cpt else (instruct, cpt)
        margs = ["--host_model", host, "--scorer_model", scorer,
                 "--beta_fixed", str(args.beta_fixed)]
        return [host, scorer], margs
    # proxy / trimix
    if args.host is not None:
        fail("--host applies only to safw")
    if args.host_model or args.scorer_model:
        fail("host/scorer models do not apply to three-model baselines")
    base = args.base_model or default_instruct(args)
    expert = args.expert_model or default_cpt(args)
    anti = args.antiexpert_model
    if anti is None:
        if args.family not in ANTI:
            fail(f"no default antiexpert for family {args.family}. Pass the model path.")
        anti = ANTI[args.family]
    validate_common(args, base, expert)
    if is_cpt(anti):
        fail(f"antiexpert slot holds a cpt model: {anti}")
    margs = ["--base_model", base, "--expert_model", expert,
             "--antiexpert_model", anti,
             "--alpha", str(args.alpha),
             "--base_weight", str(args.base_weight),
             "--expert_weight", str(args.expert_weight),
             "--plausibility_alpha", str(args.plausibility_alpha)]
    return [base, expert, anti], margs


def main():
    args = build_argparser().parse_args()
    models, method_args = resolve_models(args)
    ordered = [t for t in TASK_ORDER if t in args.tasks]
    for short in ordered:
        task, subdir, nex, mnt, extra = TASK_TABLE[short]
        stem = resolve(args.method, args.lang, short, models)
        host_note = f" host={args.host}" if args.method == "safw" else ""
        print(f"plan: method={args.method} family={args.family} "
              f"lang={args.lang} scale={args.scale}{host_note} "
              f"task={short} -> {stem}_preds.json", flush=True)
        if args.dry_run:
            continue
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        extra = [e.format(lang=args.lang) for e in extra]
        cmd = [sys.executable, "-m", "scripts.infer",
               "--method", args.method, *method_args,
               "--task", task,
               "--eval_lang", args.lang,
               "--prompt_lang", args.prompt_lang,
               "--num_exemplar", str(nex),
               "--max_new_tokens", str(mnt),
               *extra,
               "--input_file", f"{args.data_root}/{subdir}/{args.lang}/test.json",
               "--exemplar_file", f"{args.data_root}/{subdir}/{args.lang}/train_1.json",
               "--output_file", f"{stem}_preds.json"]
        subprocess.run(cmd, check=True)
    print(f"done: method={args.method} lang={args.lang} scale={args.scale}")


if __name__ == "__main__":
    main()
