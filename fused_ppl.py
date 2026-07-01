#!/usr/bin/env python3
"""Per-sample host selection by fused perplexity (SAF-W endorsement fusion)."""

import argparse
import json
import torch
import torch.nn.functional as F


def fuse_logits(host_logits, scorer_logits):
    """SAF-W endorsement fusion at one position. Same rule as SAFW.fuse_logits."""
    # Host and scorer may have different vocab sizes (e.g. 7B has 152064, the
    # 1.5B cpt has 151936). Truncate both to the smaller size before fusing,
    # matching the vocab handling in the decoder.
    v = min(host_logits.shape[-1], scorer_logits.shape[-1])
    host_logits = host_logits[..., :v]
    scorer_logits = scorer_logits[..., :v]
    p_scorer = F.softmax(scorer_logits, dim=-1)
    host_top1 = host_logits.argmax(dim=-1, keepdim=True)
    e = p_scorer.gather(-1, host_top1).squeeze(-1)
    beta = (1.0 - e).unsqueeze(-1)
    return (1.0 - beta) * host_logits + beta * scorer_logits


@torch.no_grad()
def fused_perplexity(host_model, scorer_model, tokenizer, text, device):
    """Perplexity of text under the SAF-W fused distribution for one assignment."""
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(host_model.device)
    host_logits_all = host_model(input_ids).logits[0]
    scorer_logits_all = scorer_model(input_ids).logits[0]
    v = host_logits_all.shape[-1]
    scorer_logits_all = scorer_logits_all[:, :v]

    seq_len = input_ids.shape[1]
    logps = []
    for i in range(seq_len - 1):
        fused = fuse_logits(host_logits_all[i], scorer_logits_all[i])
        logp = F.log_softmax(fused, dim=-1)
        true_token = input_ids[0, i + 1]
        logps.append(logp[true_token].item())

    nll = -sum(logps) / len(logps)
    return float(torch.exp(torch.tensor(nll)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", required=True)
    parser.add_argument("--model_b", required=True)
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model_a, trust_remote_code=True)
    A = AutoModelForCausalLM.from_pretrained(args.model_a, torch_dtype=torch.bfloat16,
                                             trust_remote_code=True, device_map="auto").eval()
    B = AutoModelForCausalLM.from_pretrained(args.model_b, torch_dtype=torch.bfloat16,
                                             trust_remote_code=True, device_map="auto").eval()

    data = json.load(open(args.prompts_file, encoding="utf-8"))
    results = []
    for item in data:
        text = item["input"]
        ppl_a_led = fused_perplexity(A, B, tok, text, args.device)
        ppl_b_led = fused_perplexity(B, A, tok, text, args.device)
        host = "A" if ppl_a_led < ppl_b_led else "B"
        results.append({"id": item["id"], "ppl_a_led": ppl_a_led,
                        "ppl_b_led": ppl_b_led, "host": host})
        print(f"  {item['id']}: A-led={ppl_a_led:.3f} B-led={ppl_b_led:.3f} -> host={host}")

    json.dump(results, open(args.output_file, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    n_a = sum(1 for r in results if r["host"] == "A")
    print(f"\nwrote {len(results)} selections: {n_a} A-led, {len(results)-n_a} B-led")


if __name__ == "__main__":
    main()
