#!/usr/bin/env python3
"""Unified inference entry point for SAF-W, Proxy Tuning, and TriMix.

One script runs every method, following the MiLiC-Eval / TriMix protocol.
``--method`` selects the decoder:

- ``safw``: SAF-W (host + scorer endorsement fusion);
- ``safw_fixed``: the uniform-averaging reduction of SAF-W (constant beta);
- ``proxy``: Proxy Tuning (base + expert/antiexpert residual);
- ``trimix``: TriMix (weighted three-model sum with plausibility constraint).

The prompt construction, batched generation, stopping criteria, output format,
and "keep the first line" post-processing are shared across all methods.

Examples
--------
SAF-W on Tibetan reading comprehension (bf16; omit ``--load_in_8bit``)::

    python -m scripts.infer --method safw \\
        --host_model Qwen/Qwen2.5-32B-Instruct \\
        --scorer_model pkupie/Qwen2.5-1.5B-bo-cpt \\
        --task reading_comprehension --eval_lang bo --prompt_lang en \\
        --num_exemplar 5 --max_new_tokens 2 \\
        --input_file data/reading_comprehension/bo/test.json \\
        --exemplar_file data/reading_comprehension/bo/train_1.json \\
        --output_file results/safw/bo_rc.json

TriMix on the same task::

    python -m scripts.infer --method trimix \\
        --base_model Qwen/Qwen2.5-7B-Instruct \\
        --expert_model pkupie/Qwen2.5-1.5B-bo-cpt \\
        --antiexpert_model Qwen/Qwen2.5-1.5B \\
        --base_weight 0.1 --expert_weight 1.0 \\
        --task reading_comprehension --eval_lang bo --prompt_lang en ...
"""

from __future__ import annotations

import argparse
import json
import os

from safw import (
    generate_completions, add_pad_token,
    load_safw, load_proxy, load_trimix,
)
from safw.prompts import TASK_BUILDERS, remove_special_tokens

# Stop sequences used by TriMix: end generation at a blank line or EOS.
STOP_STRINGS = ["\n\n", "\n", ".\n\n", "!\n\n", "?\n\n", "<|endoftext|>"]


def build_argparser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    p = argparse.ArgumentParser(description="SAF-W / Proxy / TriMix inference.")
    p.add_argument("--method", required=True,
                   choices=["safw", "safw_fixed", "proxy", "trimix"])
    # two-model (SAF-W)
    p.add_argument("--host_model")
    p.add_argument("--scorer_model")
    p.add_argument("--beta_fixed", type=float, default=0.5)
    # three-model (Proxy / TriMix)
    p.add_argument("--base_model")
    p.add_argument("--expert_model")
    p.add_argument("--antiexpert_model")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--base_weight", type=float, default=1.0)
    p.add_argument("--expert_weight", type=float, default=1.0)
    p.add_argument("--plausibility_model", default="expert")
    p.add_argument("--plausibility_alpha", type=float, default=0.1)
    p.add_argument("--load_in_8bit", action="store_true")
    # task
    p.add_argument("--task", required=True, choices=list(TASK_BUILDERS))
    p.add_argument("--eval_lang", default="bo")
    p.add_argument("--prompt_lang", default="en")
    p.add_argument("--num_exemplar", type=int, default=5)
    p.add_argument("--src_lang", default=None)
    p.add_argument("--tgt_lang", default=None)
    p.add_argument("--max_passage_len", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_test_example_num", type=int, default=-1)
    # io
    p.add_argument("--input_file", required=True)
    p.add_argument("--exemplar_file", default=None)
    p.add_argument("--output_file", required=True)
    return p


def build_converted(args, input_data, exemplar_data):
    """Build prompts for the requested task."""
    builder = TASK_BUILDERS[args.task]
    if args.task == "translation":
        return builder(input_data, exemplar_data, src_lang=args.src_lang,
                       tgt_lang=args.tgt_lang, num_exemplar=args.num_exemplar,
                       prompt_lang=args.prompt_lang)
    if args.task == "title_generation":
        return builder(input_data, exemplar_data, eval_lang=args.eval_lang,
                       num_exemplar=args.num_exemplar,
                       max_passage_len=args.max_passage_len,
                       prompt_lang=args.prompt_lang)
    return builder(input_data, exemplar_data, eval_lang=args.eval_lang,
                   num_exemplar=args.num_exemplar, prompt_lang=args.prompt_lang)


def load_model(args):
    """Load the decoder and tokenizer for the requested method."""
    if args.method == "safw":
        return load_safw(args.host_model, args.scorer_model,
                         fixed_beta=False, load_in_8bit=args.load_in_8bit)
    if args.method == "safw_fixed":
        return load_safw(args.host_model, args.scorer_model,
                         beta_fixed=args.beta_fixed, fixed_beta=True,
                         load_in_8bit=args.load_in_8bit)
    if args.method == "proxy":
        return load_proxy(args.base_model, args.expert_model,
                          args.antiexpert_model, alpha=args.alpha,
                          load_in_8bit=args.load_in_8bit)
    if args.method == "trimix":
        return load_trimix(args.base_model, args.expert_model,
                           args.antiexpert_model, base_weight=args.base_weight,
                           expert_weight=args.expert_weight,
                           plausibility_model=args.plausibility_model,
                           plausibility_alpha=args.plausibility_alpha,
                           load_in_8bit=args.load_in_8bit)
    raise ValueError(f"unknown method {args.method!r}")


def main() -> None:
    """Parse arguments, run the chosen method, write predictions."""
    args = build_argparser().parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(args.output_file) and os.path.getsize(args.output_file) > 0:
        print(f"Output {args.output_file} exists. Stopping.")
        return

    input_data = json.load(open(args.input_file, encoding="utf-8"))
    exemplar_data = (json.load(open(args.exemplar_file, encoding="utf-8"))
                     if args.exemplar_file else None)
    converted = build_converted(args, input_data, exemplar_data)
    if args.max_test_example_num > 0:
        converted = converted[: args.max_test_example_num]

    model, tokenizer = load_model(args)
    print(f"loaded method={args.method}")

    # Stop strings to token-id sequences (no special tokens).
    stop_id_sequences = [
        tokenizer.encode(s, add_special_tokens=False) for s in STOP_STRINGS
    ]
    stop_id_sequences = [s for s in stop_id_sequences if len(s) > 0]

    prompts = [item["input"] for item in converted]
    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=args.batch_size,
        stop_id_sequences=stop_id_sequences,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    results = {}
    for item, out in zip(converted, outputs):
        text = remove_special_tokens(out.strip().split("\n")[0])
        results[item["id"]] = text

    json.dump(results, open(args.output_file, "w", encoding="utf-8"),
              indent=4, ensure_ascii=False)
    print(f"wrote {len(results)} predictions to {args.output_file}")


if __name__ == "__main__":
    main()
