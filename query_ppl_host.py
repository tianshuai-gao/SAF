"""Per-sample host selection via query-only conditional PPL (exemplars as
condition, not scored). Direction: weaker-as-host. rc/rs, first-token answer.
Memory-safe: never materializes full-sequence float32 logits."""
import argparse, json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safw.prompts import convert_comprehension, convert_response

TASKS = {
    "rc": ("reading_comprehension", convert_comprehension),
    "rs": ("response_selection", convert_response),
}

@torch.no_grad()
def score_model(model, ids, start, chunk=256):
    """Return (mean NLL of tokens after `start`, last-position logits on cpu)."""
    out = model(ids.to(model.device)).logits[0]  # (L, V) bf16, stays on GPU
    L = out.shape[0]
    tgt = ids[0].to(out.device)
    nll_sum, cnt = 0.0, 0
    for s in range(start, L - 1, chunk):
        e = min(s + chunk, L - 1)
        lp = F.log_softmax(out[s:e].float(), dim=-1)
        got = lp.gather(-1, tgt[s + 1:e + 1].unsqueeze(-1)).squeeze(-1)
        nll_sum += -got.sum().item()
        cnt += got.numel()
        del lp, got
    last = out[-1].float().cpu()
    del out
    torch.cuda.empty_cache()
    return nll_sum / max(cnt, 1), last

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ins", required=True)
    ap.add_argument("--cpt", required=True)
    ap.add_argument("--task", default="rc", choices=["rc", "rs"])
    ap.add_argument("--lang", default="bo")
    ap.add_argument("--prompt_lang", default="en")
    ap.add_argument("--num_exemplar", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"qppl_{args.task}_{args.lang}.json"

    dirname, builder = TASKS[args.task]
    d = f"data/{dirname}/{args.lang}"
    test = json.load(open(f"{d}/test.json"))
    exem = json.load(open(f"{d}/train_1.json"))
    full = builder(test, exem, eval_lang=args.lang,
                   num_exemplar=args.num_exemplar, prompt_lang=args.prompt_lang)
    bare = builder(test, None, eval_lang=args.lang,
                   num_exemplar=args.num_exemplar, prompt_lang=args.prompt_lang)

    tok = AutoTokenizer.from_pretrained(args.ins)
    kw = dict(torch_dtype=torch.bfloat16, device_map="auto")
    m_ins = AutoModelForCausalLM.from_pretrained(args.ins, **kw).eval()
    m_cpt = AutoModelForCausalLM.from_pretrained(args.cpt, **kw).eval()

    def letter(t):
        s = tok.decode([t]).strip()
        return s[0] if s else ""

    recs = []
    for it_full, it_bare in zip(full, bare):
        prompt = it_full["input"]
        prefix = prompt[: len(prompt) - len(it_bare["input"])]
        n_prefix = tok(prefix, return_tensors="pt").input_ids.shape[1]
        ids = tok(prompt, return_tensors="pt").input_ids
        start = max(n_prefix - 1, 0)  # logits[i] predicts token i+1

        nll_ins, l_ins = score_model(m_ins, ids, start)
        nll_cpt, l_cpt = score_model(m_cpt, ids, start)

        n = min(l_ins.size(0), l_cpt.size(0))
        l_ins, l_cpt = l_ins[:n], l_cpt[:n]
        p_ins = F.log_softmax(l_ins, -1)
        p_cpt = F.log_softmax(l_cpt, -1)

        v = l_ins.argmax().item()
        e = p_cpt[v].exp().item()
        ansA = letter((e * l_ins + (1 - e) * l_cpt).argmax().item())  # ins-led

        v2 = l_cpt.argmax().item()
        e2 = p_ins[v2].exp().item()
        ansB = letter((e2 * l_cpt + (1 - e2) * l_ins).argmax().item())  # cpt-led

        # weaker-as-host: lower query-NLL model = stronger = scorer
        weak = ansB if nll_ins < nll_cpt else ansA
        strong = ansA if nll_ins < nll_cpt else ansB

        recs.append({"id": it_full["id"], "gold": it_full["gold"],
                     "insled": ansA, "cptled": ansB,
                     "weak_host": weak, "strong_host": strong,
                     "nll_ins": round(nll_ins, 4), "nll_cpt": round(nll_cpt, 4)})

    json.dump(recs, open(out, "w"), ensure_ascii=False, indent=1)

    N = len(recs)
    acc = lambda k: sum(r[k] == r["gold"] for r in recs) / N
    agree = sum(r["insled"] == r["cptled"] for r in recs)
    orc = sum(r["insled"] == r["gold"] or r["cptled"] == r["gold"]
              for r in recs) / N
    print(f"task={args.task} lang={args.lang}  n={N}  agree={agree} ({agree/N:.1%})")
    print(f"acc  ins-led={acc('insled'):.3f}  cpt-led={acc('cptled'):.3f}")
    print(f"per-sample  weaker-as-host={acc('weak_host'):.3f}  "
          f"stronger-as-host={acc('strong_host'):.3f}  oracle={orc:.3f}")
    print(f"saved -> {out}")

if __name__ == "__main__":
    main()
