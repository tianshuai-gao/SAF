#!/usr/bin/env python3
"""Test whether endorsement (e) or host confidence (p) can pick the host.

For each example we run both assignments and record, at the answer position,
the endorsement e and host confidence p, plus each assignment's predicted
letter. We build per-sample predictions and compare accuracy against fixed
hosts and the oracle upper bound. If e- or p-per-sample beats the better fixed
host and approaches oracle, the signal can drive per-sample host selection.
"""
from __future__ import annotations
import argparse, json, sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
from safw.prompts import TASK_BUILDERS


def load(name):
    return AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True).eval()


def fuse_probe(host_logits, scorer_logits):
    v = min(host_logits.shape[-1], scorer_logits.shape[-1])
    lh = host_logits[..., :v]; ls = scorer_logits[..., :v]
    ps = F.softmax(ls, dim=-1)
    top1 = lh.argmax(dim=-1, keepdim=True)
    e = ps.gather(-1, top1).squeeze(-1)
    beta = (1.0 - e).unsqueeze(-1)
    fused = (1.0 - beta) * lh + beta * ls
    ft = fused.argmax(dim=-1)
    p = F.softmax(lh, dim=-1).gather(-1, ft.unsqueeze(-1)).squeeze(-1)
    return e.item(), ft.item(), p.item()


@torch.no_grad()
def probe(host_m, scorer_m, tok, prompt):
    ids = tok(prompt, return_tensors="pt").input_ids.to(host_m.device)
    h = host_m(ids).logits[:, -1, :]
    s = scorer_m(ids.to(scorer_m.device)).logits[:, -1, :].to(h.device)
    e, ft, p = fuse_probe(h, s)
    letter = tok.decode([ft]).strip()
    letter = letter[0] if letter and letter[0] in "ABCDE" else letter
    return letter, e, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", required=True)
    ap.add_argument("--model_b", required=True)
    ap.add_argument("--task", default="reading_comprehension")
    ap.add_argument("--subdir", default="reading_comprehension")
    ap.add_argument("--lang", default="bo")
    ap.add_argument("--num_exemplar", type=int, default=5)
    ap.add_argument("--nmax", type=int, default=200)
    args = ap.parse_args()

    test = json.load(open(f"data/{args.subdir}/{args.lang}/test.json"))[: args.nmax]
    ex = json.load(open(f"data/{args.subdir}/{args.lang}/train_1.json"))
    conv = TASK_BUILDERS[args.task](test, ex, eval_lang=args.lang,
                                    num_exemplar=args.num_exemplar, prompt_lang="en")
    gold = {c["id"]: c["gold"] for c in conv}

    tok = AutoTokenizer.from_pretrained(args.model_a, trust_remote_code=True)
    A = load(args.model_a); B = load(args.model_b)

    n = 0
    acc = {"ins": 0, "cpt": 0, "e_ps": 0, "p_ps": 0, "oracle": 0}
    pick = {"e_ins": 0, "e_cpt": 0, "p_ins": 0, "p_cpt": 0}
    for c in conv:
        g = gold[c["id"]]
        li, ei, pi = probe(A, B, tok, c["input"])
        lc, ec, pc = probe(B, A, tok, c["input"])
        ok_i = int(li == g); ok_c = int(lc == g)
        n += 1
        acc["ins"] += ok_i
        acc["cpt"] += ok_c
        acc["oracle"] += 1 if (ok_i or ok_c) else 0
        if ei >= ec:
            acc["e_ps"] += ok_i; pick["e_ins"] += 1
        else:
            acc["e_ps"] += ok_c; pick["e_cpt"] += 1
        if pi >= pc:
            acc["p_ps"] += ok_i; pick["p_ins"] += 1
        else:
            acc["p_ps"] += ok_c; pick["p_cpt"] += 1

    print(f"\n=== select-host test ({args.task}/{args.lang}, n={n}) ===")
    print(f"  fixed ins (host=7B)  : {acc['ins']/n:.3f}")
    print(f"  fixed cpt (host=cpt) : {acc['cpt']/n:.3f}")
    print(f"  e-per-sample         : {acc['e_ps']/n:.3f}   (picked ins {pick['e_ins']}, cpt {pick['e_cpt']})")
    print(f"  p-per-sample         : {acc['p_ps']/n:.3f}   (picked ins {pick['p_ins']}, cpt {pick['p_cpt']})")
    print(f"  oracle (upper bound) : {acc['oracle']/n:.3f}")
    best_fixed = max(acc['ins'], acc['cpt'])/n
    print(f"\n  better fixed host    : {best_fixed:.3f}")
    print(f"  e-per-sample vs fixed: {acc['e_ps']/n - best_fixed:+.3f}")
    print(f"  p-per-sample vs fixed: {acc['p_ps']/n - best_fixed:+.3f}")
    print("  >0 = 选host有效; <=0 = 信号选不对host")


if __name__ == "__main__":
    main()
