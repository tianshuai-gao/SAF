"""Per-sample host selection on rc/rs via delta=0 judge. One forward per model."""
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
    ap.add_argument("--task", default="rc", choices=["rc", "rs"])
    ap.add_argument("--lang", default="bo")
    ap.add_argument("--prompt_lang", default="en")
    ap.add_argument("--num_exemplar", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"judge_{args.task}_{args.lang}.json"

    dirname, builder = TASKS[args.task]
    d = f"data/{dirname}/{args.lang}"
    test = json.load(open(f"{d}/test.json"))
    exem = json.load(open(f"{d}/train_1.json"))
    items = builder(test, exem, eval_lang=args.lang,
                    num_exemplar=args.num_exemplar, prompt_lang=args.prompt_lang)

    tok = AutoTokenizer.from_pretrained(args.ins)
    kw = dict(torch_dtype=torch.bfloat16, device_map="auto")
    m_ins = AutoModelForCausalLM.from_pretrained(args.ins, **kw).eval()
    m_cpt = AutoModelForCausalLM.from_pretrained(args.cpt, **kw).eval()

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
        tA = (e * l_ins + (1 - e) * l_cpt).argmax().item()

        v2 = l_cpt.argmax().item()
        e2 = lp_ins[v2].exp().item()
        tB = (e2 * l_cpt + (1 - e2) * l_ins).argmax().item()

        JA = 0.5 * (lp_ins[tA] + lp_cpt[tA]).item()
        JB = 0.5 * (lp_ins[tB] + lp_cpt[tB]).item()
        pick = tA if JA >= JB else tB

        recs.append({"id": it["id"], "gold": it["gold"],
                     "A": letter(tA), "B": letter(tB), "pick": letter(pick),
                     "JA": round(JA, 4), "JB": round(JB, 4),
                     "e_insled": round(e, 4), "e_cptled": round(e2, 4)})

    json.dump(recs, open(out, "w"), ensure_ascii=False, indent=1)

    N = len(recs)
    agree = sum(r["A"] == r["B"] for r in recs)
    accA = sum(r["A"] == r["gold"] for r in recs) / N
    accB = sum(r["B"] == r["gold"] for r in recs) / N
    accJ = sum(r["pick"] == r["gold"] for r in recs) / N
    dis = [r for r in recs if r["A"] != r["B"]]
    disJ = sum(r["pick"] == r["gold"] for r in dis)
    disO = sum(r["A"] == r["gold"] or r["B"] == r["gold"] for r in dis)
    print(f"task={args.task} lang={args.lang}")
    print(f"n={N}  agree={agree} ({agree/N:.1%})")
    print(f"acc  ins-led={accA:.3f}  cpt-led={accB:.3f}  judge={accJ:.3f}")
    print(f"disagree n={len(dis)}  judge_correct={disJ}  oracle_correct={disO}")
    print(f"saved -> {out}")

if __name__ == "__main__":
    main()
