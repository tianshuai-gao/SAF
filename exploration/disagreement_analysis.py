#!/usr/bin/env python3
"""Analyze agreement vs disagreement between the two host assignments.

Per-sample host selection only matters where ins-led and cpt-led DISAGREE.
This splits examples into AGREE / DISAGREE and reports where the headroom is
and whether any signal separates 'ins is right' from 'cpt is right' on the
disagree subset. Read-only diagnostic.
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

    agree = {"n": 0, "correct": 0}
    dis = {"n": 0, "ins_right": 0, "cpt_right": 0, "both_wrong": 0,
           "e_ins_when_ins_right": [], "e_cpt_when_ins_right": [],
           "e_ins_when_cpt_right": [], "e_cpt_when_cpt_right": [],
           "p_ins_when_ins_right": [], "p_cpt_when_ins_right": [],
           "p_ins_when_cpt_right": [], "p_cpt_when_cpt_right": []}

    for c in conv:
        g = gold[c["id"]]
        li, ei, pi = probe(A, B, tok, c["input"])
        lc, ec, pc = probe(B, A, tok, c["input"])
        if li == lc:
            agree["n"] += 1
            agree["correct"] += int(li == g)
        else:
            dis["n"] += 1
            ins_ok = (li == g); cpt_ok = (lc == g)
            if ins_ok:
                dis["ins_right"] += 1
                dis["e_ins_when_ins_right"].append(ei); dis["e_cpt_when_ins_right"].append(ec)
                dis["p_ins_when_ins_right"].append(pi); dis["p_cpt_when_ins_right"].append(pc)
            elif cpt_ok:
                dis["cpt_right"] += 1
                dis["e_ins_when_cpt_right"].append(ei); dis["e_cpt_when_cpt_right"].append(ec)
                dis["p_ins_when_cpt_right"].append(pi); dis["p_cpt_when_cpt_right"].append(pc)
            else:
                dis["both_wrong"] += 1

    N = agree["n"] + dis["n"]
    print(f"\n=== disagreement analysis ({args.task}/{args.lang}, n={N}) ===")
    print(f"\nAGREE   : {agree['n']}/{N}, accuracy {agree['correct']}/{agree['n']} = {agree['correct']/max(agree['n'],1):.3f}")
    print(f"DISAGREE: {dis['n']}/{N}  <- host choice only matters here")
    print(f"  ins-led right {dis['ins_right']}, cpt-led right {dis['cpt_right']}, both wrong {dis['both_wrong']}")
    print("\n--- on DISAGREE, does e/p separate 'ins right' from 'cpt right'? ---")
    print(f"  when INS right (n={dis['ins_right']}): e_ins={mean(dis['e_ins_when_ins_right']):.3f} e_cpt={mean(dis['e_cpt_when_ins_right']):.3f}  (want e_ins>e_cpt)")
    print(f"  when CPT right (n={dis['cpt_right']}): e_ins={mean(dis['e_ins_when_cpt_right']):.3f} e_cpt={mean(dis['e_cpt_when_cpt_right']):.3f}  (want e_cpt>e_ins)")
    print(f"  when INS right (n={dis['ins_right']}): p_ins={mean(dis['p_ins_when_ins_right']):.3f} p_cpt={mean(dis['p_cpt_when_ins_right']):.3f}  (want p_ins>p_cpt)")
    print(f"  when CPT right (n={dis['cpt_right']}): p_ins={mean(dis['p_ins_when_cpt_right']):.3f} p_cpt={mean(dis['p_cpt_when_cpt_right']):.3f}  (want p_cpt>p_ins)")
    print("\n关系相反(ins对时e_ins高, cpt对时e_cpt高) -> 信号能选host")
    print("关系一样(恒偏一方) -> 选不出")


if __name__ == "__main__":
    main()
