"""NLL sign distribution on title_generation query segment.
Only prompt forwards, no decoding. Tells us whether weaker-as-host
would ever pick cpt-led on title."""
import argparse, json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safw.prompts import convert_title

@torch.no_grad()
def query_nll(model, ids, start, chunk=256):
    out = model(ids.to(model.device)).logits[0]
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
    del out
    torch.cuda.empty_cache()
    return nll_sum / max(cnt, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ins", required=True)
    ap.add_argument("--cpt", required=True)
    ap.add_argument("--lang", default="bo")
    ap.add_argument("--prompt_lang", default="en")
    ap.add_argument("--num_exemplar", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"titlenll_{args.lang}.json"

    d = f"data/title_generation_200/{args.lang}"
    test = json.load(open(f"{d}/test.json"))
    exem = json.load(open(f"{d}/train_1.json"))
    full = convert_title(test, exem, eval_lang=args.lang,
                         num_exemplar=args.num_exemplar,
                         prompt_lang=args.prompt_lang)
    bare = convert_title(test, None, eval_lang=args.lang,
                         num_exemplar=args.num_exemplar,
                         prompt_lang=args.prompt_lang)

    tok = AutoTokenizer.from_pretrained(args.ins)
    kw = dict(torch_dtype=torch.bfloat16, device_map="auto")
    m_ins = AutoModelForCausalLM.from_pretrained(args.ins, **kw).eval()
    m_cpt = AutoModelForCausalLM.from_pretrained(args.cpt, **kw).eval()

    recs = []
    for it_full, it_bare in zip(full, bare):
        prompt = it_full["input"]
        prefix = prompt[: len(prompt) - len(it_bare["input"])]
        n_prefix = tok(prefix, return_tensors="pt").input_ids.shape[1]
        ids = tok(prompt, return_tensors="pt").input_ids
        start = max(n_prefix - 1, 0)
        nll_ins = query_nll(m_ins, ids, start)
        nll_cpt = query_nll(m_cpt, ids, start)
        recs.append({"id": it_full["id"],
                     "nll_ins": round(nll_ins, 4),
                     "nll_cpt": round(nll_cpt, 4),
                     "len": ids.shape[1]})

    json.dump(recs, open(out, "w"), ensure_ascii=False, indent=1)
    N = len(recs)
    cpt_lower = sum(r["nll_cpt"] < r["nll_ins"] for r in recs)
    gaps = sorted(r["nll_ins"] - r["nll_cpt"] for r in recs)
    print(f"lang={args.lang}  n={N}")
    print(f"cpt_lower_nll (rule -> ins-led): {cpt_lower}/{N}")
    print(f"ins_lower_nll (rule -> cpt-led): {N - cpt_lower}/{N}")
    print(f"nll gap (ins - cpt): min={gaps[0]:.3f} "
          f"median={gaps[N//2]:.3f} max={gaps[-1]:.3f}")
    print(f"saved -> {out}")

if __name__ == "__main__":
    main()
