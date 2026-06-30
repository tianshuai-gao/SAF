"""Prompt construction for the MiLiC-Eval tasks.

Each task renders few-shot exemplars followed by the query into a single
prompt string. The templates follow the MiLiC-Eval / TriMix evaluation
protocol and support English and Chinese instruction languages. Every builder
returns a list of dicts with keys ``id``, ``input`` (the prompt), and ``gold``
(the reference answer).
"""

from __future__ import annotations

import random

abbr_to_lang_en = {
    "zh": "Chinese", "en": "English", "mn": "Mongolian", "kk": "Kazakh",
    "ug": "Uyghur", "bo": "Tibetan", "ben": "Bengali", "tam": "Tamil",
    "tel": "Telugu", "ori": "Odia",
}
abbr_to_lang_zh = {
    "zh": "\u6c49\u8bed", "en": "\u82f1\u8bed", "mn": "\u8499\u53e4\u8bed",
    "kk": "\u54c8\u8428\u514b\u8bed", "ug": "\u7ef4\u543e\u5c14\u8bed",
    "bo": "\u85cf\u8bed",
}


def remove_special_tokens(text: str) -> str:
    """Strip residual special tokens from a generation.

    :param text: Raw decoded text.
    :returns: Text with ``<pad>``/``<s>``/``</s>``/``<unk>``/``<extra_id_0>``
        removed and surrounding whitespace stripped.
    """
    for tok in ("<pad>", "<s>", "</s>", "<unk>", "<extra_id_0>"):
        text = text.replace(tok, "")
    return text.strip()


def convert_translation(input_dataset, exemplar_dataset=None, src_lang="zh",
                        tgt_lang="mn", num_exemplar=3, prompt_lang="zh"):
    """Build translation prompts (source language to target language)."""
    out = []
    prefix = ""
    if exemplar_dataset is not None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            ex = exemplar_dataset[i]
            if prompt_lang == "en":
                prefix += (f"Please translate the following "
                           f"{abbr_to_lang_en[src_lang]} text into "
                           f"{abbr_to_lang_en[tgt_lang]}.\n"
                           f"{abbr_to_lang_en[src_lang]}: {ex[src_lang]}\n"
                           f"{abbr_to_lang_en[tgt_lang]}: {ex[tgt_lang]}\n\n")
            else:
                prefix += (f"\u8bf7\u5c06\u4e0b\u9762\u7684{abbr_to_lang_zh[src_lang]}"
                           f"\u6587\u672c\u7ffb\u8bd1\u6210{abbr_to_lang_zh[tgt_lang]}\u3002\n"
                           f"{abbr_to_lang_zh[src_lang]}\uff1a{ex[src_lang]}\n"
                           f"{abbr_to_lang_zh[tgt_lang]}\uff1a{ex[tgt_lang]}\n\n")
    for item in input_dataset:
        p = prefix
        if prompt_lang == "en":
            p += (f"Please translate the following {abbr_to_lang_en[src_lang]} "
                  f"text into {abbr_to_lang_en[tgt_lang]}.\n"
                  f"{abbr_to_lang_en[src_lang]}: {item[src_lang]}\n"
                  f"{abbr_to_lang_en[tgt_lang]}: ")
        else:
            p += (f"\u8bf7\u5c06\u4e0b\u9762\u7684{abbr_to_lang_zh[src_lang]}\u6587\u672c"
                  f"\u7ffb\u8bd1\u6210{abbr_to_lang_zh[tgt_lang]}\u3002\n"
                  f"{abbr_to_lang_zh[src_lang]}\uff1a{item[src_lang]}\n"
                  f"{abbr_to_lang_zh[tgt_lang]}\uff1a")
        out.append({"id": item["id"], "input": p, "gold": item[tgt_lang]})
    return out


def convert_title(input_dataset, exemplar_dataset=None, eval_lang="zh",
                  num_exemplar=3, max_passage_len=1024, prompt_lang="zh"):
    """Build title-generation prompts."""
    out = []
    prefix = ""
    if exemplar_dataset is not None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            ex = exemplar_dataset[i]
            if prompt_lang == "en":
                prefix += (f"Please write a title for the following article in "
                           f"{abbr_to_lang_en[eval_lang]}.\n"
                           f"Article: {ex['content'][:max_passage_len]}\n"
                           f"Title: {ex['title']}\n\n")
            else:
                prefix += (f"\u8bf7\u4e3a\u4ee5\u4e0b{abbr_to_lang_zh[eval_lang]}"
                           f"\u6587\u7ae0\u5199\u4e00\u4e2a\u6807\u9898\u3002\n"
                           f"\u6587\u7ae0\uff1a{ex['content'][:max_passage_len]}\n"
                           f"\u6807\u9898\uff1a{ex['title']}\n\n")
    for item in input_dataset:
        p = prefix
        if prompt_lang == "en":
            p += (f"Please write a title for the following article in "
                  f"{abbr_to_lang_en[eval_lang]}.\n"
                  f"Article: {item['content'][:max_passage_len]}\nTitle: ")
        else:
            p += (f"\u8bf7\u4e3a\u4ee5\u4e0b{abbr_to_lang_zh[eval_lang]}\u6587\u7ae0"
                  f"\u5199\u4e00\u4e2a\u6807\u9898\u3002\n"
                  f"\u6587\u7ae0\uff1a{item['content'][:max_passage_len]}\n\u6807\u9898\uff1a")
        out.append({"id": item["id"], "input": p, "gold": item["title"]})
    return out


def convert_response(input_dataset, exemplar_dataset=None, eval_lang="mn",
                     num_exemplar=3, prompt_lang="zh"):
    """Build response-selection prompts (multiple choice)."""
    out = []
    prefix = ""

    def render(item, with_ans):
        if prompt_lang == "en":
            s = (f"Please select an appropriate response for the following "
                 f"{abbr_to_lang_en[eval_lang]} dialogue.\nContext:\n")
            s += "".join(f"{c}\n" for c in item["context"])
            s += "Options:\n"
            s += "".join(f"{chr(ord('A') + j)}. {o}\n"
                         for j, o in enumerate(item["options"]))
            s += f"Answer: {item['answer']}\n\n" if with_ans else "Answer:"
        else:
            s = (f"\u8bf7\u4e3a\u4ee5\u4e0b{abbr_to_lang_zh[eval_lang]}\u5bf9\u8bdd"
                 f"\u9009\u62e9\u5408\u9002\u7684\u56de\u7b54\u3002\n\u5bf9\u8bdd\u5185\u5bb9\uff1a\n")
            s += "".join(f"{c}\n" for c in item["context"])
            s += "\u9009\u9879\uff1a\n"
            s += "".join(f"{chr(ord('A') + j)}. {o}\n"
                         for j, o in enumerate(item["options"]))
            s += (f"\u7b54\u6848\uff1a{item['answer']}\n\n" if with_ans
                  else "\u7b54\u6848\uff1a")
        return s

    if exemplar_dataset is not None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            prefix += render(exemplar_dataset[i], True)
    for item in input_dataset:
        out.append({"id": item["id"], "input": prefix + render(item, False),
                    "gold": item["answer"]})
    return out


def convert_comprehension(input_dataset, exemplar_dataset=None, eval_lang="mn",
                          num_exemplar=3, prompt_lang="zh"):
    """Build reading-comprehension prompts (multiple choice)."""
    out = []
    prefix = ""

    def render(item, with_ans):
        if prompt_lang == "en":
            s = (f"Please read the following {abbr_to_lang_en[eval_lang]} "
                 f"dialogue and answer the question.\nContext:\n")
            s += "".join(f"{c}\n" for c in item["context"])
            s += f"Question: {item['question']}\nOptions:\n"
            s += "".join(f"{chr(ord('A') + j)}. {o}\n"
                         for j, o in enumerate(item["options"]))
            s += f"Answer: {item['answer']}\n\n" if with_ans else "Answer:"
        else:
            s = (f"\u8bf7\u9605\u8bfb\u4ee5\u4e0b{abbr_to_lang_zh[eval_lang]}\u5bf9\u8bdd"
                 f"\u5e76\u56de\u7b54\u95ee\u9898\u3002\n\u5bf9\u8bdd\u5185\u5bb9\uff1a\n")
            s += "".join(f"{c}\n" for c in item["context"])
            s += f"\u95ee\u9898\uff1a{item['question']}\n\u9009\u9879\uff1a\n"
            s += "".join(f"{chr(ord('A') + j)}. {o}\n"
                         for j, o in enumerate(item["options"]))
            s += (f"\u7b54\u6848\uff1a{item['answer']}\n\n" if with_ans
                  else "\u7b54\u6848\uff1a")
        return s

    if exemplar_dataset is not None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            prefix += render(exemplar_dataset[i], True)
    for item in input_dataset:
        out.append({"id": item["id"], "input": prefix + render(item, False),
                    "gold": item["answer"]})
    return out


def convert_math(input_dataset, exemplar_dataset=None, eval_lang="bo",
                 num_exemplar=3, prompt_lang="zh"):
    """Build step-by-step math prompts."""
    out = []
    prefix = ""
    if exemplar_dataset is not None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            ex = exemplar_dataset[i]
            if prompt_lang == "en":
                prefix += (f"Please solve the following {abbr_to_lang_en[eval_lang]}"
                           f" math problem step by step.\n"
                           f"Problem: {ex['question']}\n"
                           f"Step-by-step solution: {ex['cot'][prompt_lang]}\n\n")
            else:
                prefix += (f"\u8bf7\u5206\u6b65\u89e3\u7b54\u4e0b\u9762\u7684"
                           f"{abbr_to_lang_zh[eval_lang]}\u6570\u5b66\u95ee\u9898\u3002\n"
                           f"\u95ee\u9898\uff1a{ex['question']}\n"
                           f"\u5206\u6b65\u89e3\u7b54\uff1a{ex['cot'][prompt_lang]}\n\n")
    for item in input_dataset:
        p = prefix
        if prompt_lang == "en":
            p += (f"Please solve the following {abbr_to_lang_en[eval_lang]} math "
                  f"problem step by step.\nProblem: {item['question']}\n"
                  f"Step-by-step solution: ")
        else:
            p += (f"\u8bf7\u5206\u6b65\u89e3\u7b54\u4e0b\u9762\u7684"
                  f"{abbr_to_lang_zh[eval_lang]}\u6570\u5b66\u95ee\u9898\u3002\n"
                  f"\u95ee\u9898\uff1a{item['question']}\n\u5206\u6b65\u89e3\u7b54\uff1a")
        out.append({"id": item["id"], "input": p, "gold": item["answer"]})
    return out


TASK_BUILDERS = {
    "translation": convert_translation,
    "title_generation": convert_title,
    "response_selection": convert_response,
    "reading_comprehension": convert_comprehension,
    "math": convert_math,
}
