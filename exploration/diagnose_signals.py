#!/usr/bin/env python3
"""Diagnose per-sample host-selection signals on a multiple-choice task.

For each rc example we run BOTH host assignments (ins-led: host=7B/scorer=cpt,
and cpt-led: host=cpt/scorer=7B). At the answer position we record two candidate
signals and whether that assignment answered correctly:

  * e   -- endorsement: scorer probability of the host's top-1 token
  * p   -- host confidence: host probability of the fused next token

We compare, within each assignment, the mean signal on CORRECT vs WRONG answers.
If correct answers carry a clearly higher signal, that signal has discriminative
power and could drive per-sample host selection. Read-only diagnostic.
"""
from __future__ import annotations
import argparse, json, sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
from safw.prompts import TASK_BUILDERS


def load(name, device):
    return AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True).eval()


def endorsement_and_conf(host_logits, scorer_logits):
    v = min(host_logits.shape[-1], scorer_logits.shape[-1])
    l_host = host_logits[..., :v]
    l_scorer = scorer_logits[..., :v]
    p_scorer = F.softmax(l_scorer, dim=-1)
    host_top1 = l_host.argmax(dim=-1, keepdim=True)
    e = p_scorer.gather(-1, host_top1).squeeze(-1)
    beta = (1.0 - e).unsqueeze(-1)
    fused = (1.0 - beta) * l_host + beta * l_scorer
    fused_token = fused.argmax(dim=-1)
    p_host = F.softmax(l_host, dim=-1).gather(
        -1, fused_token.unsqueeze(-1)).squeeze(-1)
    return e.item(), fused_token.item(), p_host.item()


@torch.no_grad()
def answer_letter(host_m, scorer_m, tok, prompt, device):
    ids = tok(prompt, return_tensors="pt").input_ids.to(host_m.device)
    h = host_m(ids).logits[:, -1, :]
    s = scorer_m(ids.to(scorer_m.device)).logits[:, -1, :].to(h.device)
    e, fused_token, p_host = endorsement_and_conf(h, s)
    letter = tok.decode([fused_token]).strip()
    letter = letter[0] if letter and letter[0] in "ABCDE" else letter
    return letter, e, p_host


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", required=True)
    ap.add_argument("--model_b", required=True)
    ap.add_argument("--task", default="reading_comprehension")
    ap.add_argument("--subdir", default="reading_comprehension")
    ap.add_argument("--lang", default="bo")
    ap.add_argument("--num_exemplar", type=int, default=5)
    ap.add_argument("--nmax", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    test = json.load(open(f"data/{args.subdir}/{args.lang}/test.json"))[: args.nmax]
    ex = json.load(open(f"data/{args.subdir}/{args.lang}/train_1.json"))
    conv = TASK_BUILDERS[args.task](
        test, ex, eval_lang=args.lang, num_exemplar=args.num_exemplar,
        prompt_lang="en")
    gold = {c["id"]: c["gold"] for c in conv}

    tok = AutoTokenizer.from_pretrained(args.model_a, trust_remote_code=True)
    A = load(args.model_a, args.device)
    B = load(args.model_b, args.device)

    stats = {"ins": {"e_ok": [], "e_no": [], "p_ok": [], "p_no": []},
             "cpt": {"e_ok": [], "e_no": [], "p_ok": [], "p_no": []}}

    for c in conv:
        g = gold[c["id"]]
        li, ei, pi = answer_letter(A, B, tok, c["input"], args.device)
        ok = (li == g)
        stats["ins"]["e_ok" if ok else "e_no"].append(ei)
        stats["ins"]["p_ok" if ok else "p_no"].append(pi)
        lc, ec, pc = answer_letter(B, A, tok, c["input"], args.device)
        ok = (lc == g)
        stats["cpt"]["e_ok" if ok else "e_no"].append(ec)
        stats["cpt"]["p_ok" if ok else "p_no"].append(pc)

    def mean(x):
        return sum(x) / len(x) if x else float("nan")

    print(f"\n=== signal diagnostic ({args.task}/{args.lang}, n={len(conv)}) ===")
    for cfg in ("ins", "cpt"):
        s = stats[cfg]
        host = "7B" if cfg == "ins" else "cpt"
        print(f"\n[{cfg}-led | host={host}]  correct={len(s['e_ok'])}  wrong={len(s['e_no'])}")
        print(f"  endorsement e : correct={mean(s['e_ok']):.3f}  wrong={mean(s['e_no']):.3f}"
              f"  gap={mean(s['e_ok'])-mean(s['e_no']):+.3f}")
        print(f"  host conf   p : correct={mean(s['p_ok']):.3f}  wrong={mean(s['p_no']):.3f}"
              f"  gap={mean(s['p_ok'])-mean(s['p_no']):+.3f}")
    print("\ngap>0 且明显 = 信号能区分对错 = 有望选host")
    print("gap approx 0 = 没区分度 = 没用")


if __name__ == "__main__":
    main()
