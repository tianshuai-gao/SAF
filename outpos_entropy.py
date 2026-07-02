"""Entropy of each model's next-token distribution at the generation position.
Cheap probe: does the entropy-difference sign flip between tasks?"""
import argparse, json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from safw.prompts import convert_comprehension, convert_response, convert_title

TASKS = {
    "rc": ("reading_comprehension", convert_comprehension, {}),
    "rs": ("response_selection", convert_response, {}),
    "title": ("title_generation_200", convert_title, {}),
}
NEX = {"rc": 5, "rs": 5, "title": 3}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ins", required=True)
    ap.add_argument("--cpt", required=True)
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--lang", default="ug")
    ap.add_argument("--prompt_lang", default="en")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"outent_{args.task}_{args.lang}.json"

    dirname, builder, _ = TASKS[args.task]
    d = f"data/{dirname}/{args.lang}"
    test = json.load(open(f"{d}/test.json"))
    exem = json.load(open(f"{d}/train_1.json"))
    items = builder(test, exem, eval_lang=args.lang,
                    num_exemplar=NEX[args.task], prompt_lang=args.prompt_lang)

    tok = AutoTokenizer.from_pretrained(args.ins)
    kw = dict(torch_dtype=torch.bfloat16, device_map="auto")
    m_ins = AutoModelForCausalLM.from_pretrained(args.ins, **kw).eval()
    m_cpt = AutoModelForCausalLM.from_pretrained(args.cpt, **kw).eval()

    @torch.no_grad()
    def last_entropy(model, ids):
        lg = model(ids.to(model.device)).logits[0, -1].float()
        lp = F.log_softmax(lg, -1)
        h = -(lp.exp() * lp).sum().item()
        del lg, lp
        return h

    recs = []
    for it in tqdm(items, desc=f"{args.task} {args.lang}"):
        ids = tok(it["input"], return_tensors="pt").input_ids
        recs.append({"id": it["id"],
                     "H_ins": round(last_entropy(m_ins, ids), 4),
                     "H_cpt": round(last_entropy(m_cpt, ids), 4)})

    json.dump(recs, open(out, "w"), ensure_ascii=False, indent=1)
    N = len(recs)
    ins_lower = sum(r["H_ins"] < r["H_cpt"] for r in recs)
    gaps = sorted(r["H_ins"] - r["H_cpt"] for r in recs)
    print(f"task={args.task} lang={args.lang}  n={N}")
    print(f"H_ins < H_cpt (ins more confident): {ins_lower}/{N}")
    print(f"H_cpt < H_ins (cpt more confident): {N - ins_lower}/{N}")
    print(f"H gap (ins - cpt): min={gaps[0]:.3f} median={gaps[N//2]:.3f} max={gaps[-1]:.3f}")
    print(f"saved -> {out}")

if __name__ == "__main__":
    main()
