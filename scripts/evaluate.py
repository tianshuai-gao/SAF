#!/usr/bin/env python3
"""Evaluate prediction files against references for the MiLiC-Eval tasks.

Given a task, a reference (input) file, and a prediction file, this script
computes the task metric using :mod:`safw.eval` and writes it to a metrics
file. Predictions are a dict mapping example id to the generated string, as
written by ``scripts/infer.py``.

Example::

    python -m scripts.evaluate --task reading_comprehension \\
        --input_file data/reading_comprehension/bo/test.json \\
        --pred_file results/safw/bo/reading_comprehension.json \\
        --metrics_output_file results/safw/bo/reading_comprehension.metrics.json
"""

from __future__ import annotations

import argparse
import json

from safw.eval import evaluate


def main() -> None:
    """Parse arguments, compute the metric, and write it out."""
    parser = argparse.ArgumentParser(description="Evaluate predictions.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--metrics_output_file", required=True)
    parser.add_argument("--tgt_lang", default="en",
                        help="Target language for translation.")
    args = parser.parse_args()

    input_data = json.load(open(args.input_file, encoding="utf-8"))
    predictions = json.load(open(args.pred_file, encoding="utf-8"))

    # Fill missing predictions with empty strings so counts line up.
    for item in input_data:
        if item["id"] not in predictions:
            print(f"missing prediction for id {item['id']}")
            predictions[item["id"]] = ""

    metrics = evaluate(args.task, input_data, predictions, tgt_lang=args.tgt_lang)
    print(metrics)
    json.dump(metrics, open(args.metrics_output_file, "w", encoding="utf-8"),
              indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
