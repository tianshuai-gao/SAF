"""Both-direction SAF-W scores on the t1rest+t23 selection set for the
generation tasks (title, translation). Full decoding with the first-token
language anchor, matching deployment exactly. Scored with the multilingual
ROUGE fork (title) or chrF++ (translation)."""
import argparse, json, os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from safw.dexperts import SAFW
from safw.utils import add_pad_token, generate_completions, _model_kwargs
from safw.prompts import convert_title, convert_translation, remove_special_tokens

STOP_STRINGS = ["\n\n", "\n", ".\n\n", "!\n\n", "?\n\n", "<|endoftext|>"]

def build_eval(d, nex, builder, **kw):
    train1 = json.load(open(f"{d}/train_1.json"))
    exem = train1[:nex]
    evalset = (train1[nex:]
               + json.load(open(f"{d}/train_2.json"))
               + json.load(open(f"{d}/train_3.json")))
    items = builder(evalset, exem, num_exemplar=nex, **kw)
    return evalset, items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ins", required=True)
    ap.add_argument("--cpt", required=True)
    ap.add_argument("--task", required=True,
                    choices=["title", "xx2en", "en2xx"])
    ap.add_argument("--lang", required=True)
    ap.add_argument("--scale", required=True)
    ap.add_argument("--host", required=True, choices=["ins", "cpt"])
    ap.add_argument("--prompt_lang", default="en")
    args = ap.parse_args()
    out = f"gensel_{args.task}_{args.lang}_{args.scale}_{args.host}host.json"
    if os.path.exists(out):
        print(f"Output {out} exists. Stopping.")
        return

    if args.task == "title":
        d = f"data/title_generation_200/{args.lang}"
        evalset, items = build_eval(
            d, 3, convert_title, lang=args.lang,
            max_passage_len=1024, prompt_lang=args.prompt_lang)
        anchor, mnt = "lrl", 250
    else:
        d = f"data/translation_dialogue/{args.lang}"
        src, tgt = (args.lang, "en") if args.task == "xx2en" else ("en", args.lang)
        evalset, items = build_eval(
            d, 5, convert_translation, src_lang=src, tgt_lang=tgt,
            prompt_lang=args.prompt_lang)
        anchor, mnt = ("en" if args.task == "xx2en" else "lrl"), 150

    host_path = args.ins if args.host == "ins" else args.cpt
    scorer_path = args.cpt if args.host == "ins" else args.ins

    tokenizer = AutoTokenizer.from_pretrained(host_path)
    tokenizer = add_pad_token(tokenizer, "left")
    model = SAFW(
        host_model_name_or_path=host_path,
        scorer_model_name_or_path=scorer_path,
        tokenizer=tokenizer,
        fixed_beta=False,
        model_kwargs=_model_kwargs(False),
    )
    model.anchor_lang = anchor
    model.ins_in_base = (args.host == "ins")

    stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in STOP_STRINGS]
    stop_ids = [s for s in stop_ids if len(s) > 0]

    preds = {}
    for it in tqdm(items, desc=f"{args.task} {args.lang} {args.scale} {args.host}-host"):
        outs = generate_completions(
            model=model, tokenizer=tokenizer, prompts=[it["input"]],
            batch_size=1, stop_id_sequences=stop_ids,
            max_new_tokens=mnt, do_sample=False, disable_tqdm=True)
        preds[it["id"]] = remove_special_tokens(outs[0].strip().split("\n")[0])

    if args.task == "title":
        from safw.eval import rouge_title
        score = rouge_title(evalset, preds)["rouge-l"] * 100
        metric = "rouge-l"
    else:
        from safw.eval import chrf_mt
        score = chrf_mt(evalset, preds, tgt_lang=tgt)
        metric = "chrf++"

    json.dump({"predictions": preds, "score": score, "metric": metric},
              open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"task={args.task} lang={args.lang} scale={args.scale} "
          f"host={args.host}  n={len(preds)}  {metric}={score:.1f}")
    print(f"saved -> {out}")

if __name__ == "__main__":
    main()
