#!/usr/bin/env python3
"""Design 1: symmetric joint-probability judge for per-sample answer selection.

Confidence signals (e, p, entropy, KL, margin) are sign-locked: the larger
model is always surer regardless of correctness. The fix is a SYMMETRIC judge
that scores both candidate answers with the SAME yardstick.

  A = ins-led fused token (host=7B, scorer=cpt)
  B = cpt-led fused token (host=cpt, scorer=7B)
  s(x) = log P_7B(x) + log P_cpt(x)   -- same judge for A and B, comparable

Pick the answer with higher joint score. Reports FLIP on disagree + accuracy.
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


def fused_argmax(lh, ls):
    ps = F.softmax(ls, dim=-1)
    top1 = lh.argmax(dim=-1, keepdim=True)
    e = ps.gather(-1, top1).squeeze(-1)
    beta = (1.0 - e).unsqueeze(-1)
    fused = (1.0 - beta) * lh + beta * ls
    return fused.argmax(dim=-1).item()


def letter_of(tok, tid):
    s = tok.decode([tid]).strip()
    return s[0] if s and s[0] in "ABCDE" else s


def mean(x):
    return sum(x) / len(x) if x else float("nan")


@torch.no_grad()
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

    N = 0
    acc = {"ins": 0, "cpt": 0, "joint": 0, "oracle": 0}
    dis = {"n": 0, "sA_when_A": [], "sB_when_A": [], "sA_when_B": [], "sB_when_B": []}
    n_agree = 0

    for c in conv:
        g = gold[c["id"]]
        ids = tok(c["input"], return_tensors="pt").input_ids
        l7 = A(ids.to(A.device)).logits[:, -1, :].float()
        lc = B(ids.to(B.device)).logits[:, -1, :].float().to(l7.device)
        v = min(l7.shape[-1], lc.shape[-1])
        l7 = l7[..., :v]; lc = lc[..., :v]

        tokA = fused_argmax(l7, lc)
        tokB = fused_argmax(lc, l7)
        la = letter_of(tok, tokA); lb = letter_of(tok, tokB)
        ok_i = int(la == g); ok_c = int(lb == g)
        N += 1
        acc["ins"] += ok_i; acc["cpt"] += ok_c
        acc["oracle"] += 1 if (ok_i or ok_c) else 0

        lp7 = F.log_softmax(l7, dim=-1).squeeze(0)
        lpc = F.log_softmax(lc, dim=-1).squeeze(0)
        sA = (lp7[tokA] + lpc[tokA]).item()
        sB = (lp7[tokB] + lpc[tokB]).item()
        acc["joint"] += ok_i if sA >= sB else ok_c

        if la == lb:
            n_agree += 1
        else:
            dis["n"] += 1
            if ok_i:
                dis["sA_when_A"].append(sA); dis["sB_when_A"].append(sB)
            elif ok_c:
                dis["sA_when_B"].append(sA); dis["sB_when_B"].append(sB)

    print(f"\n=== joint-score design1 ({args.task}/{args.lang}, n={N}) ===")
    print(f"AGREE {n_agree}, DISAGREE {dis['n']}")
    print("\n--- on DISAGREE, does joint score FLIP with correctness? ---")
    a_ok = mean(dis['sA_when_A']) > mean(dis['sB_when_A'])
    b_ok = mean(dis['sB_when_B']) > mean(dis['sA_when_B'])
    print(f"  when INS right: sA={mean(dis['sA_when_A']):.3f} sB={mean(dis['sB_when_A']):.3f}  ({'sA>sB OK' if a_ok else 'NOT'})")
    print(f"  when CPT right: sA={mean(dis['sA_when_B']):.3f} sB={mean(dis['sB_when_B']):.3f}  ({'sB>sA OK' if b_ok else 'NOT'})")
    print(f"  FLIP: {'YES (comparable!)' if (a_ok and b_ok) else 'NO (still locked)'}")
    print("\n--- actual pick accuracy ---")
    best = max(acc['ins'], acc['cpt'])/N
    print(f"  fixed ins {acc['ins']/N:.3f}  fixed cpt {acc['cpt']/N:.3f}  oracle {acc['oracle']/N:.3f}")
    print(f"  joint pick {acc['joint']/N:.3f}  vs best fixed {acc['joint']/N-best:+.3f}")


if __name__ == "__main__":
    main()
