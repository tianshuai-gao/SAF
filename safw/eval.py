"""Evaluation metrics for the MiLiC-Eval tasks.

Each task has its own metric, following the MiLiC-Eval / TriMix protocol:

- reading comprehension, response selection: accuracy on the multiple-choice
  letter (the first character of the prediction is taken as the answer);
- translation: chrF++ (character n-gram F-score with word order 2);
- title generation: ROUGE-L F-measure;
- math: accuracy on the last number in the prediction.

Predictions are a dict mapping example id to the generated string; references
come from the input dataset.
"""

from __future__ import annotations

import re

import numpy as np


def _to_pred_dict(predictions):
    """Accept either a dict id->text or a list of records and return a dict.

    :param predictions: A dict or a list of ``{"id"/"index", "prediction"}``.
    :returns: A dict mapping id to prediction text.
    """
    if isinstance(predictions, dict):
        return predictions
    out = {}
    for rec in predictions:
        key = rec.get("id", rec.get("index"))
        out[key] = rec.get("prediction", "")
    return out


def accuracy_choice(input_data, predictions) -> dict:
    """Accuracy for multiple-choice tasks (reading comp, response selection).

    :param input_data: List of examples with ``id`` and ``answer``.
    :param predictions: Predictions dict or list.
    :returns: ``{"accuracy": value}``.
    """
    pred = _to_pred_dict(predictions)
    correct = 0
    for item in input_data:
        p = pred.get(item["id"], "")
        if len(p) > 0 and p[0] in ("A", "B", "C", "D", "E"):
            p = p[0]
        if item["answer"] == p:
            correct += 1
    return {"accuracy": correct / len(input_data)}


def chrf_translation(input_data, predictions, tgt_lang="en") -> dict:
    """chrF++ for translation.

    :param input_data: List with ``id`` and the target-language reference.
    :param predictions: Predictions dict or list.
    :param tgt_lang: Target language key in the references.
    :returns: ``{"chrf++": value}`` with the score scaled to ``[0, 1]``.
    """
    from sacrebleu.metrics import CHRF

    pred = _to_pred_dict(predictions)
    refs = [d[tgt_lang] for d in input_data]
    preds = [pred.get(d["id"], "") for d in input_data]
    chrfpp = CHRF(word_order=2, lowercase=True)
    return {"chrf++": chrfpp.corpus_score(preds, [refs]).score / 100}


def rouge_title(input_data, predictions) -> dict:
    """ROUGE-L F-measure for title generation.

    :param input_data: List with ``id`` and ``title``.
    :param predictions: Predictions dict or list.
    :returns: ``{"rouge-l": value}``.
    """
    from rouge_score import rouge_scorer

    pred = _to_pred_dict(predictions)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
    scores = [scorer.score(item["title"], pred.get(item["id"], ""))
              for item in input_data]
    return {"rouge-l": float(np.mean([s["rougeL"].fmeasure for s in scores]))}


def _extract_last_number(s: str) -> float:
    """Extract the last number in a string, or -1 if none.

    :param s: The string to search.
    :returns: The last number as a float, or ``-1``.
    """
    s = s.replace(",", "")
    if len(s) > 0 and s[-1] == ".":
        s = s[:-1]
    nums = re.findall(r"\d+\.\d+|\d+", s)
    if not nums:
        return -1
    return float(nums[-1])


def accuracy_math(input_data, predictions) -> dict:
    """Accuracy for math: compare the last number against the gold answer.

    :param input_data: List with ``id`` and numeric ``answer``.
    :param predictions: Predictions dict or list.
    :returns: ``{"accuracy": value}``.
    """
    pred = _to_pred_dict(predictions)
    correct = 0
    for item in input_data:
        p = _extract_last_number(pred.get(item["id"], ""))
        try:
            if float(item["answer"]) == float(p):
                correct += 1
        except (TypeError, ValueError):
            pass
    return {"accuracy": correct / len(input_data)}


TASK_METRICS = {
    "reading_comprehension": accuracy_choice,
    "response_selection": accuracy_choice,
    "title_generation": rouge_title,
    "math": accuracy_math,
}


def evaluate(task, input_data, predictions, tgt_lang="en") -> dict:
    """Dispatch to the correct metric for a task.

    :param task: Task name.
    :param input_data: List of reference examples.
    :param predictions: Predictions dict or list.
    :param tgt_lang: Target language for translation.
    :returns: A metrics dict.
    """
    if task == "translation":
        return chrf_translation(input_data, predictions, tgt_lang=tgt_lang)
    if task in TASK_METRICS:
        return TASK_METRICS[task](input_data, predictions)
    raise ValueError(f"unknown task {task!r}")
