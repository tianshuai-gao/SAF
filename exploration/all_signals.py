#!/usr/bin/env python3
"""Screen many candidate signals for per-sample host/answer selection at once.

Screens SEVEN signals in one pass; for each reports whether it FLIPS with
correctness on the disagree subset and its pick accuracy vs fixed hosts/oracle.

Signals (higher score -> that candidate picked, unless noted):
  joint   : logP7(x)+logPc(x)                        (design 1)
  rank    : -(rank7(x)+rankc(x))  top=0              (design 5)
  optnorm : renorm over {A,B,C,D}, sum of two models' prob of x  (design 3)
  agree   : prob the OTHER model gives x             (design 4)
  geo     : sqrt(P7*Pc)
  minp    : min(P7,Pc)
  disag   : -|logP7-logPc|  (models agree on x)
"""
from __future__ import annotations
import argparse, json, sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
from safw.prompts import TASK_BUILDERS

LETTERS = "ABCDE"


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
    return s[0] if s and s[0] in LETTERS else s


def rank_of(logits_1d, tid):
    return (logits_1d > logits_1d[tid]).sum().item()


def mean(x):
    return sum(x) / len(x) if x else float("nan")


SIGNAL_NAMES = ["joint", "rank", "optnorm", "agree", "geo", "minp", "disag"]


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

    opt_ids = []
    for L in "ABCD":
        tid = tok.encode(L, add_special_tokens=False)
        if len(tid) == 1:
            opt_ids.append(tid[0])

    N = 0
    accfix = {"ins": 0, "cpt": 0, "oracle": 0}
    accsig = {k: 0 for k in SIGNAL_NAMES}
    sep = {k: {"A_insR": [], "B_insR": [], "A_cptR": [], "B_cptR": []} for k in SIGNAL_NAMES}
    n_dis = 0

    for c in conv:
        g = gold[c["id"]]
        ids = tok(c["input"], return_tensors="pt").input_ids
        l7 = A(ids.to(A.device)).logits[:, -1, :].float().squeeze(0)
        lc = B(ids.to(B.device)).logits[:, -1, :].float().to(l7.device).squeeze(0)
        v = min(l7.shape[-1], lc.shape[-1])
        l7 = l7[:v]; lc = lc[:v]
        lp7 = F.log_softmax(l7, dim=-1)
        lpc = F.log_softmax(lc, dim=-1)
        p7 = lp7.exp(); pc = lpc.exp()

        tokA = fused_argmax(l7.unsqueeze(0), lc.unsqueeze(0))
        tokB = fused_argmax(lc.unsqueeze(0), l7.unsqueeze(0))
        la = letter_of(tok, tokA); lb = letter_of(tok, tokB)
        ok_i = int(la == g); ok_c = int(lb == g)
        N += 1
        accfix["ins"] += ok_i; accfix["cpt"] += ok_c
        accfix["oracle"] += 1 if (ok_i or ok_c) else 0

        if opt_ids:
            z7 = p7[opt_ids].sum().clamp_min(1e-9)
            zc = pc[opt_ids].sum().clamp_min(1e-9)

        def score(x):
            joint = (lp7[x] + lpc[x]).item()
            rank = -(rank_of(l7, x) + rank_of(lc, x))
            optnorm = (p7[x] / z7 + pc[x] / zc).item() if opt_ids else joint
            geo = torch.sqrt(p7[x] * pc[x]).item()
            minp = torch.min(p7[x], pc[x]).item()
            disag = -abs((lp7[x] - lpc[x]).item())
            return {"joint": joint, "rank": rank, "optnorm": optnorm,
                    "agree": None, "geo": geo, "minp": minp, "disag": disag}

        sc_A = score(tokA); sc_B = score(tokB)
        sc_A["agree"] = pc[tokA].item()
        sc_B["agree"] = p7[tokB].item()

        for k in SIGNAL_NAMES:
            accsig[k] += ok_i if sc_A[k] >= sc_B[k] else ok_c

        if la != lb:
            n_dis += 1
            for k in SIGNAL_NAMES:
                if ok_i:
                    sep[k]["A_insR"].append(sc_A[k]); sep[k]["B_insR"].append(sc_B[k])
                elif ok_c:
                    sep[k]["A_cptR"].append(sc_A[k]); sep[k]["B_cptR"].append(sc_B[k])

    print(f"\n=== all-signals ({args.task}/{args.lang}, n={N}) ===")
    print(f"fixed ins {accfix['ins']/N:.3f}  fixed cpt {accfix['cpt']/N:.3f}  oracle {accfix['oracle']/N:.3f}  DISAGREE {n_dis}")
    best = max(accfix['ins'], accfix['cpt'])/N
    print(f"\n{'signal':8} {'pick':>6} {'vs_fix':>7}  {'FLIP?':>5}   detail")
    for k in SIGNAL_NAMES:
        a_ok = mean(sep[k]["A_insR"]) > mean(sep[k]["B_insR"])
        b_ok = mean(sep[k]["B_cptR"]) > mean(sep[k]["A_cptR"])
        flip = "YES" if (a_ok and b_ok) else "no"
        pick = accsig[k]/N
        print(f"{k:8} {pick:6.3f} {pick-best:+7.3f}  {flip:>5}   "
              f"insR:A={mean(sep[k]['A_insR']):.2f},B={mean(sep[k]['B_insR']):.2f} | "
              f"cptR:A={mean(sep[k]['A_cptR']):.2f},B={mean(sep[k]['B_cptR']):.2f}")
    print("\nFLIP=YES 且 vs_fix>0 = 找到可比信号")


if __name__ == "__main__":
    main()
