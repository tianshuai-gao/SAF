#!/usr/bin/env python3
"""Build an expanded development set (devx) by merging dev and exemplar splits.

The standard development set is small. For SAF-W host selection it is expanded
by merging the dev split with two exemplar splits (``train_2`` and
``train_3``), leaving ``train_1`` untouched so it can still serve as the
few-shot exemplar file at test time without leakage. The merged set is written
as ``devx.json``.

Records are merged in order and de-duplicated by ``id``, so any overlap between
splits is dropped rather than double-counted. This expansion is applied to the
multiple-choice tasks (reading comprehension, response selection, math) where a
larger sample makes the host choice more stable.

Example::

    python -m scripts.make_devx \\
        --dev_file data/reading_comprehension/bo/dev.json \\
        --extra_files data/reading_comprehension/bo/train_2.json \\
                      data/reading_comprehension/bo/train_3.json \\
        --output_file data/reading_comprehension/bo/devx.json
"""

from __future__ import annotations

import argparse
import json


def merge_dedup(files):
    """Merge JSON record lists in order, dropping duplicate ids.

    :param files: Paths to JSON files, each a list of records with an ``id``.
    :returns: The merged list with the first occurrence of each id kept.
    """
    seen = set()
    merged = []
    for path in files:
        records = json.load(open(path, encoding="utf-8"))
        for rec in records:
            key = rec.get("id")
            if key in seen:
                continue
            seen.add(key)
            merged.append(rec)
    return merged


def main() -> None:
    """Merge the dev and extra splits into a devx file."""
    parser = argparse.ArgumentParser(
        description="Build an expanded dev set (devx) from dev + extra splits.")
    parser.add_argument("--dev_file", required=True,
                        help="The base development split.")
    parser.add_argument("--extra_files", nargs="+", default=[],
                        help="Extra splits to merge in (e.g. train_2, train_3).")
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    merged = merge_dedup([args.dev_file] + args.extra_files)
    json.dump(merged, open(args.output_file, "w", encoding="utf-8"),
              indent=4, ensure_ascii=False)
    print(f"wrote {len(merged)} records to {args.output_file}")


if __name__ == "__main__":
    main()
