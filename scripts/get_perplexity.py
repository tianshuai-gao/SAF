#!/usr/bin/env python3
"""Select TriMix fusion weights by perplexity on a development set.

TriMix sets its base and expert weights per task and language. Following the
original protocol, this script sweeps all ``(base_weight, expert_weight)``
combinations on a grid, computes the perplexity of the weighted three-model
logit combination on a sampled development set, and reports the combination
with the lowest perplexity.

The three models are the base (large-ins), the expert (small-cpt), and the
antiexpert (small-base). Models load in bfloat16 by default; pass
``--load_in_8bit`` to load in 8-bit as in the original release.

The prompt construction is shared with the main pipeline via
:data:`safw.prompts.TASK_BUILDERS`.
"""

from __future__ import annotations

import argparse
import json
import math
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from safw.prompts import TASK_BUILDERS

DEVICE = "cuda"

WEIGHT_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


@torch.no_grad()
def get_logits_and_perplexity(model, tokenizer, text, device=DEVICE):
    """Compute shifted logits, labels, and perplexity for one text.

    :param model: A causal LM.
    :param tokenizer: The tokenizer.
    :param text: The input string.
    :param device: Device for the forward pass.
    :returns: ``(shift_logits_cpu, shift_labels_cpu, perplexity)``. Logits are
        returned on CPU to bound memory when accumulating over a dataset.
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    outputs = model(input_ids)
    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    shift_logits = shift_logits.to(torch.float32)
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    nll = -token_log_probs.mean()
    perplexity = torch.exp(nll).item()

    return shift_logits.to("cpu"), shift_labels.to("cpu"), perplexity


def build_converted(task_name, input_dataset, exemplar_dataset, lang, prompt_lang):
    """Build dev-set prompts for the given task, matching TriMix exemplar counts.

    :param task_name: Task name; ``translation_{src}2{tgt}`` encodes directions.
    :param input_dataset: The dev input data.
    :param exemplar_dataset: The exemplar data.
    :param lang: Evaluation language code.
    :param prompt_lang: Prompt instruction language.
    :returns: The list of converted prompt records.
    """
    if task_name == "reading_comprehension":
        return TASK_BUILDERS["reading_comprehension"](
            input_dataset, exemplar_dataset, eval_lang=lang, num_exemplar=3,
            prompt_lang=prompt_lang)
    if task_name == "response_selection":
        return TASK_BUILDERS["response_selection"](
            input_dataset, exemplar_dataset, eval_lang=lang, num_exemplar=3,
            prompt_lang=prompt_lang)
    if task_name == "math":
        return TASK_BUILDERS["math"](
            input_dataset, exemplar_dataset, eval_lang=lang, num_exemplar=3,
            prompt_lang=prompt_lang)
    if task_name == "title_generation_200":
        return TASK_BUILDERS["title_generation"](
            input_dataset, exemplar_dataset, eval_lang=lang, num_exemplar=3,
            max_passage_len=256, prompt_lang=prompt_lang)
    if "translation" in task_name:
        direction = task_name.split("_")[1]
        src_lang, tgt_lang = direction.split("2")
        return TASK_BUILDERS["translation"](
            input_dataset, exemplar_dataset, src_lang=src_lang,
            tgt_lang=tgt_lang, num_exemplar=3, prompt_lang=prompt_lang)
    raise ValueError(f"unknown task {task_name!r}")


def main() -> None:
    """Sweep weight combinations and report the lowest-perplexity one."""
    parser = argparse.ArgumentParser(
        description="Select TriMix weights by dev-set perplexity.")
    parser.add_argument("--base_model_name_or_path", type=str, required=True)
    parser.add_argument("--expert_model_name_or_path", type=str, required=True)
    parser.add_argument("--antiexpert_model_name_or_path", type=str, required=True)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--max_test_example_num", type=int, default=50)
    parser.add_argument("--prompt_lang", type=str, default="en")
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--exemplar_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path, trust_remote_code=True)

    print("Loading models...")
    load_args = {"trust_remote_code": True, "device_map": "auto"}
    if args.load_in_8bit:
        load_args["load_in_8bit"] = True
    else:
        load_args["torch_dtype"] = torch.bfloat16

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path, **load_args)
    base_model.eval()
    expert_model = AutoModelForCausalLM.from_pretrained(
        args.expert_model_name_or_path, **load_args)
    expert_model.eval()
    antiexpert_model = AutoModelForCausalLM.from_pretrained(
        args.antiexpert_model_name_or_path, **load_args)
    antiexpert_model.eval()

    base_vocab = base_model.get_input_embeddings().weight.size(0)
    if expert_model.get_input_embeddings().weight.size(0) != base_vocab:
        expert_model.resize_token_embeddings(base_vocab)
        antiexpert_model.resize_token_embeddings(base_vocab)
    print("Models loaded.")

    combinations = []
    for bw in WEIGHT_GRID:
        for ew in WEIGHT_GRID:
            combinations.append((bw, ew, 1 - bw - ew))

    input_dataset = json.load(open(args.input_file, encoding="utf-8"))
    exemplar_dataset = json.load(open(args.exemplar_file, encoding="utf-8"))
    converted = build_converted(args.task_name, input_dataset,
                                exemplar_dataset, args.lang, args.prompt_lang)
    converted = random.sample(converted, args.max_test_example_num)

    base_ppl_sum = expert_ppl_sum = antiexpert_ppl_sum = 0.0
    total = 0
    nll_sums = {combo: 0.0 for combo in combinations}

    print("Computing perplexities...")
    for item in tqdm(converted, desc="samples"):
        text = item["input"]
        base_logits, shift_labels, base_ppl = get_logits_and_perplexity(
            base_model, tokenizer, text)
        expert_logits, _, expert_ppl = get_logits_and_perplexity(
            expert_model, tokenizer, text)
        antiexpert_logits, _, antiexpert_ppl = get_logits_and_perplexity(
            antiexpert_model, tokenizer, text)

        base_ppl_sum += base_ppl
        expert_ppl_sum += expert_ppl
        antiexpert_ppl_sum += antiexpert_ppl
        total += 1

        base_logits = base_logits.to(torch.float32)
        expert_logits = expert_logits.to(torch.float32)
        antiexpert_logits = antiexpert_logits.to(torch.float32)
        shift_labels = shift_labels.to(DEVICE)

        for bw, ew, aw in combinations:
            combined = (bw * base_logits + ew * expert_logits
                        + aw * antiexpert_logits).to(DEVICE)
            log_probs = torch.log_softmax(combined, dim=-1)
            tok_log_probs = log_probs.gather(
                -1, shift_labels.unsqueeze(-1)).squeeze(-1)
            nll_sums[(bw, ew, aw)] += (-tok_log_probs.mean()).item()

        del base_logits, expert_logits, antiexpert_logits, shift_labels
        torch.cuda.empty_cache()

    avg_base = base_ppl_sum / total
    avg_expert = expert_ppl_sum / total
    avg_antiexpert = antiexpert_ppl_sum / total

    output = []
    for combo, total_nll in nll_sums.items():
        bw, ew, aw = combo
        output.append({
            "base_weight": bw,
            "expert_weight": ew,
            "antiexpert_weight": aw,
            "avg_base_perplexity": avg_base,
            "avg_expert_perplexity": avg_expert,
            "avg_antiexpert_perplexity": avg_antiexpert,
            "avg_combined_perplexity": math.exp(total_nll / total),
        })

    json.dump(output, open(args.output_file, "w", encoding="utf-8"), indent=4)
    best = min(output, key=lambda x: x["avg_combined_perplexity"])
    print(f"Best config for {args.task_name} ({args.lang}):\n{best}")


if __name__ == "__main__":
    main()
