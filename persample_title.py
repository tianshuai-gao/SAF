"""Per-sample host selection on title via output-position entropy.
Loads one SAFW (ins as base slot, cpt as expert slot). A `swap` flag on a
subclass exchanges the two roles in fuse_logits, which equals swapping
host/scorer without reloading. Entropy is read from outent_title_<lang>.json.
Rule: lower-entropy model is the host."""
import argparse, json, os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from safw.dexperts import SAFW
from safw.utils import add_pad_token, generate_completions, _model_kwargs
from safw.prompts import convert_title, remove_special_tokens
from safw.eval import rouge_title

STOP_STRINGS = ["\n\n", "\n", ".\n\n", "!\n\n", "?\n\n", "<|endoftext|>"]


class SAFWSwap(SAFW):
    """SAFW with a runtime host/scorer swap flag."""
    swap = False

    def fuse_logits(self, base_logits, expert_logits, antiexpert_logits):
        if self.swap:
            base_logits, expert_logits = expert_logits, base_logits
        return super().fuse_logits(base_logits, expert_logits, antiexpert_logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ins", required=True)
    ap.add_argument("--cpt", required=True)
    ap.add_argument("--lang", default="ug")
    ap.add_argument("--prompt_lang", default="en")
    ap.add_argument("--num_exemplar", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=250)
    ap.add_argument("--entropy_file", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ent_file = args.entropy_file or f"outent_title_{args.lang}.json"
    out = args.out or f"persample_title_{args.lang}_entropyhost.json"
    if os.path.exists(out):
        print(f"Output {out} exists. Stopping.")
        return

    ent = {r["id"]: r for r in json.load(open(ent_file))}

    d = f"data/title_generation_200/{args.lang}"
    test = json.load(open(f"{d}/test.json"))
    exem = json.load(open(f"{d}/train_1.json"))
    items = convert_title(test, exem, eval_lang=args.lang,
                          num_exemplar=args.num_exemplar,
                          max_passage_len=1024,
                          prompt_lang=args.prompt_lang)

    tokenizer = AutoTokenizer.from_pretrained(args.ins)
    tokenizer = add_pad_token(tokenizer, "left")
    model = SAFWSwap(
        host_model_name_or_path=args.ins,
        scorer_model_name_or_path=args.cpt,
        tokenizer=tokenizer,
        fixed_beta=False,
        model_kwargs=_model_kwargs(False),
    )

    stop_id_sequences = [
        tokenizer.encode(s, add_special_tokens=False) for s in STOP_STRINGS
    ]
    stop_id_sequences = [s for s in stop_id_sequences if len(s) > 0]

    preds, choices = {}, {}
    for it in tqdm(items, desc=f"persample title {args.lang}"):
        e = ent[it["id"]]
        # lower-entropy model is host; base slot holds ins
        model.swap = not (e["H_ins"] < e["H_cpt"])  # swap=True -> cpt-led
        outs = generate_completions(
            model=model, tokenizer=tokenizer, prompts=[it["input"]],
            batch_size=1, stop_id_sequences=stop_id_sequences,
            max_new_tokens=args.max_new_tokens, do_sample=False,
            disable_tqdm=True,
        )
        text = remove_special_tokens(outs[0].strip().split("\n")[0])
        preds[it["id"]] = text
        choices[it["id"]] = "cpt-led" if model.swap else "ins-led"

    json.dump({"predictions": preds, "choices": choices},
              open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    score = rouge_title(test, preds)["rouge-l"] * 100
    n_cpt = sum(v == "cpt-led" for v in choices.values())
    print(f"lang={args.lang}  n={len(preds)}  "
          f"cpt-led chosen: {n_cpt}/{len(preds)}")
    print(f"per-sample entropy-host ROUGE-L: {score:.1f}")
    print("reference (ug 7B): ins-led=16.9  cpt-led=20.0  oracle=23.3")
    print(f"saved -> {out}")

if __name__ == "__main__":
    main()
