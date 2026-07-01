#!/usr/bin/env python3
"""Symmetric cross-rank signal for per-sample selection.

Plain rank is locked toward the 7B answer (large-model choices rank high under
both models). This variant scores each candidate ONLY under the OTHER model:

  A = ins-led token (7B chose) -> rank of A under cpt   (cross)
  B = cpt-led token (cpt chose) -> rank of B under 7B   (cross)

Lower cross-rank = other model also ranks it high. Removes the self-rank term.
Reports FLIP + accuracy, alongside the prob-version agree signal.
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


def rank_of(logits_1d, tid):
    return (logits_1d > logits_1d[tid]).sum().item()


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
    accfix = {"ins": 0, "cpt": 0, "oracle": 0}
    accsig = {"xrank": 0, "agree": 0}
    sep = {k: {"A_insR": [], "B_insR": [], "A_cptR": [], "B_cptR": []} for k in ("xrank", "agree")}
    n_dis = 0

    for c in conv:
        g = gold[c["id"]]
        ids = tok(c["input"], return_tensors="pt").input_ids
        l7 = A(ids.to(A.device)).logits[:, -1, :].float().squeeze(0)
        lc = B(ids.to(B.device)).logits[:, -1, :].float().to(l7.device).squeeze(0)
        v = min(l7.shape[-1], lc.shape[-1])
        l7 = l7[:v]; lc = lc[:v]
        p7 = F.softmax(l7, dim=-1); pc = F.softmax(lc, dim=-1)

        tokA = fused_argmax(l7.unsqueeze(0), lc.unsqueeze(0))
        tokB = fused_argmax(lc.unsqueeze(0), l7.unsqueeze(0))
        la = letter_of(tok, tokA); lb = letter_of(tok, tokB)
        ok_i = int(la == g); ok_c = int(lb == g)
        N += 1
        accfix["ins"] += ok_i; accfix["cpt"] += ok_c
        accfix["oracle"] += 1 if (ok_i or ok_c) else 0

        xr_A = -rank_of(lc, tokA)
        xr_B = -rank_of(l7, tokB)
        ag_A = pc[tokA].item()
        ag_B = p7[tokB].item()

        accsig["xrank"] += ok_i if xr_A >= xr_B else ok_c
        accsig["agree"] += ok_i if ag_A >= ag_B else ok_c

        if la != lb:
            n_dis += 1
            for k, sA, sB in (("xrank", xr_A, xr_B), ("agree", ag_A, ag_B)):
                if ok_i:
                    sep[k]["A_insR"].append(sA); sep[k]["B_insR"].append(sB)
                elif ok_c:
                    sep[k]["A_cptR"].append(sA); sep[k]["B_cptR"].append(sB)

    print(f"\n=== rank-sym ({args.task}/{args.lang}, n={N}) ===")
    print(f"fixed ins {accfix['ins']/N:.3f}  fixed cpt {accfix['cpt']/N:.3f}  oracle {accfix['oracle']/N:.3f}  DISAGREE {n_dis}")
    best = max(accfix['ins'], accfix['cpt'])/N
    for k in ("xrank", "agree"):
        a_ok = mean(sep[k]["A_insR"]) > mean(sep[k]["B_insR"])
        b_ok = mean(sep[k]["B_cptR"]) > mean(sep[k]["A_cptR"])
        flip = "YES" if (a_ok and b_ok) else "no"
        pick = accsig[k]/N
        print(f"\n[{k}] pick={pick:.3f} vs_fix={pick-best:+.3f} FLIP={flip}")
        print(f"   when INS right: A={mean(sep[k]['A_insR']):.2f} B={mean(sep[k]['B_insR']):.2f} (want A>B)")
        print(f"   when CPT right: A={mean(sep[k]['A_cptR']):.2f} B={mean(sep[k]['B_cptR']):.2f} (want B>A)")
    print("\nFLIP=YES 且 vs_fix>0 = 可比信号")


if __name__ == "__main__":
    main()
