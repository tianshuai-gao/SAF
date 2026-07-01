#!/usr/bin/env python3
"""Test three richer signals for per-sample host selection on disagree cases.

Endorsement e and host confidence p are sign-locked (absolute values depend on
which model is host/scorer, not on which assignment is correct). Here we test
entropy H, KL(host||scorer), and margin M, and check on the DISAGREE subset
whether each separates 'ins-led right' from 'cpt-led right'. Read-only.

Intended pick directions:
  lower H  = more certain  -> pick lower-entropy assignment
  lower KL = host agrees with scorer -> pick lower-KL assignment
  higher M = more certain  -> pick higher-margin assignment
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


def signals(host_logits, scorer_logits):
    v = min(host_logits.shape[-1], scorer_logits.shape[-1])
    lh = host_logits[..., :v].float()
    ls = scorer_logits[..., :v].float()
    ph = F.softmax(lh, dim=-1)
    logph = F.log_softmax(lh, dim=-1)
    logpsc = F.log_softmax(ls, dim=-1)
    H = -(ph * logph).sum(dim=-1)
    KL = (ph * (logph - logpsc)).sum(dim=-1)
    top2 = ph.topk(2, dim=-1).values
    M = (top2[..., 0] - top2[..., 1])
    ft = lh.argmax(dim=-1)
    return ft.item(), H.item(), KL.item(), M.item()


@torch.no_grad()
def probe(host_m, scorer_m, tok, prompt):
    ids = tok(prompt, return_tensors="pt").input_ids.to(host_m.device)
    h = host_m(ids).logits[:, -1, :]
    s = scorer_m(ids.to(scorer_m.device)).logits[:, -1, :].to(h.device)
    ft, H, KL, M = signals(h, s)
    letter = tok.decode([ft]).strip()
    letter = letter[0] if letter and letter[0] in "ABCDE" else letter
    return letter, H, KL, M


def mean(x):
    return sum(x) / len(x) if x else float("nan")


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

    R = {k: {"ins": [], "cpt": []} for k in ("H", "KL", "M")}
    W = {k: {"ins": [], "cpt": []} for k in ("H", "KL", "M")}
    n_dis = ins_right = cpt_right = both_wrong = n_agree = 0
    acc = {"ins": 0, "cpt": 0, "oracle": 0, "H": 0, "KL": 0, "M": 0}
    N = 0

    for c in conv:
        g = gold[c["id"]]
        li, Hi, KLi, Mi = probe(A, B, tok, c["input"])
        lc, Hc, KLc, Mc = probe(B, A, tok, c["input"])
        ok_i = int(li == g); ok_c = int(lc == g)
        N += 1
        acc["ins"] += ok_i; acc["cpt"] += ok_c
        acc["oracle"] += 1 if (ok_i or ok_c) else 0
        acc["H"]  += ok_i if Hi  <= Hc  else ok_c
        acc["KL"] += ok_i if KLi <= KLc else ok_c
        acc["M"]  += ok_i if Mi  >= Mc  else ok_c
        if li == lc:
            n_agree += 1
        else:
            n_dis += 1
            if ok_i:
                ins_right += 1
                for k, vi, vc in (("H",Hi,Hc),("KL",KLi,KLc),("M",Mi,Mc)):
                    R[k]["ins"].append(vi); R[k]["cpt"].append(vc)
            elif ok_c:
                cpt_right += 1
                for k, vi, vc in (("H",Hi,Hc),("KL",KLi,KLc),("M",Mi,Mc)):
                    W[k]["ins"].append(vi); W[k]["cpt"].append(vc)
            else:
                both_wrong += 1

    print(f"\n=== multi-signal ({args.task}/{args.lang}, n={N}) ===")
    print(f"AGREE {n_agree}, DISAGREE {n_dis} (ins_right {ins_right}, cpt_right {cpt_right}, both_wrong {both_wrong})")
    print("\n--- separation on DISAGREE (want ordering to FLIP with correctness) ---")
    for k, want in (("H","lower picks"),("KL","lower picks"),("M","higher picks")):
        print(f"\n[{k}] ({want})")
        print(f"  when INS right: {k}_ins={mean(R[k]['ins']):.3f}  {k}_cpt={mean(R[k]['cpt']):.3f}")
        print(f"  when CPT right: {k}_ins={mean(W[k]['ins']):.3f}  {k}_cpt={mean(W[k]['cpt']):.3f}")
    print("\n--- actual pick accuracy (full set) ---")
    best_fixed = max(acc['ins'], acc['cpt'])/N
    print(f"  fixed ins {acc['ins']/N:.3f}  fixed cpt {acc['cpt']/N:.3f}  oracle {acc['oracle']/N:.3f}")
    for k in ("H","KL","M"):
        print(f"  {k}-per-sample {acc[k]/N:.3f}  vs best fixed {acc[k]/N-best_fixed:+.3f}")
    print("\n关系翻转 = 信号能选host; 不翻转 = 恒偏没用")


if __name__ == "__main__":
    main()
