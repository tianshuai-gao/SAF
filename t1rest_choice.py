"""Host selection scores on the UNUSED tail of train_1 (exemplars are the
first num_exemplar items; we evaluate items[num_exemplar:]). rc/rs only,
first-token answer via single forward per model (same method as judge_rc,
which reproduced the paper numbers)."""
import argparse, json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safw.prompts import convert_comprehension, convert_response

TASKS = {
    "rc": ("reading_comprehension", convert_comprehension),
    "rs": ("response_selection", convert_response),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ins", required=True)
    ap.add_argument("--cpt", required=True)
    ap.add_argument("--task", required=True, choices=["rc", "rs"])
    ap.add_argument("--lang", required=True)
    ap.add_argument("--scale", required=True, help="7B/14B/32B tag for filename")
    ap.add_argument("--prompt_lang", default="en")
    ap.add_argument("--num_exemplar", type=int, default=5)
    ap.add_argument("--load_in_8bit", action="store_true")
    args = ap.parse_args()
    out = f"t1rest_{args.task}_{args.lang}_{args.scale}.json"

    dirname, builder = TASKS[args.task]
    d = f"data/{dirname}/{args.lang}"
    train1 = json.load(open(f"{d}/train_1.json"))
    exem = train1[: args.num_exemplar]
    evalset = train1[args.num_exemplar:]
    print(f"train1 total={len(train1)}  exemplars={len(exem)}  eval={len(evalset)}")

    items = builder(evalset, exem, eval_lang=args.lang,
                    num_exemplar=args.num_exemplar, prompt_lang=args.prompt_lang)

    tok = AutoTokenizer.from_pretrained(args.ins)
    if args.load_in_8bit:
        kw = dict(load_in_8bit=True, device_map="auto")
        kw_small = dict(torch_dtype=torch.bfloat16, device_map="auto")
    else:
        kw = dict(torch_dtype=torch.bfloat16, device_map="auto")
        kw_small = kw
    m_ins = AutoModelForCausalLM.from_pretrained(args.ins, **kw).eval()
    m_cpt = AutoModelForCausalLM.from_pretrained(args.cpt, **kw_small).eval()

    def letter(t):
        s = tok.decode([t]).strip()
        return s[0] if s else ""

    recs = []
    for it in items:
        ids = tok(it["input"], return_tensors="pt").input_ids.to(m_ins.device)
        with torch.no_grad():
            l_ins = m_ins(ids).logits[0, -1].float()
            l_cpt = m_cpt(ids.to(m_cpt.device)).logits[0, -1].float()
        n = min(l_ins.size(0), l_cpt.size(0))
        l_ins, l_cpt = l_ins[:n].cpu(), l_cpt[:n].cpu()
        lp_ins = F.log_softmax(l_ins, -1)
        lp_cpt = F.log_softmax(l_cpt, -1)

        v = l_ins.argmax().item()
        e = lp_cpt[v].exp().item()
        ansA = letter((e * l_ins + (1 - e) * l_cpt).argmax().item())

        v2 = l_cpt.argmax().item()
        e2 = lp_ins[v2].exp().item()
        ansB = letter((e2 * l_cpt + (1 - e2) * l_ins).argmax().item())

        recs.append({"id": it["id"], "gold": it["gold"],
                     "insled": ansA, "cptled": ansB})

    json.dump(recs, open(out, "w"), ensure_ascii=False, indent=1)
    N = len(recs)
    ci = sum(r["insled"] == r["gold"] for r in recs)
    cc = sum(r["cptled"] == r["gold"] for r in recs)
    print(f"task={args.task} lang={args.lang} scale={args.scale} n={N}")
    print(f"ins-led correct={ci}/{N}  cpt-led correct={cc}/{N}")
    print(f"saved -> {out}")

if __name__ == "__main__":
    main()
