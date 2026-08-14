#!/usr/bin/env python3
"""Validate a KLEOS dataset against the data contract.

Checks schema conformance, duplicate ids, coverage gaps, placeholder text,
sensitive-content patterns, policy-vs-fact phrasing, and (optionally) leakage
between splits.

Usage::

    python scripts/validate_dataset.py --dataset data/examples
    python scripts/validate_dataset.py --dataset data/examples --no-strict
    python scripts/validate_dataset.py --dataset data/examples \
                                       --leakage-report reports/leakage
    python scripts/validate_dataset.py --eval data/examples/synthetic_eval.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import add_common_arguments, run, setup_logging
from kleos_models.data.leakage import check_leakage, enforce_leakage_policy
from kleos_models.data.loaders import load_examples
from kleos_models.data.schemas import EvaluationExample, TrainingExample
from kleos_models.data.validation import validate_examples
from kleos_models.logging_utils import get_logger

logger = get_logger("validate_dataset")


def _discover_splits(dataset_dir: Path) -> dict[str, Path]:
    """Find the split files present in a dataset directory."""
    candidates = {
        "train": ["train.jsonl", "synthetic_train.jsonl"],
        "validation": ["validation.jsonl", "valid.jsonl"],
        "test": ["test.jsonl"],
    }
    found: dict[str, Path] = {}
    for split, names in candidates.items():
        for name in names:
            path = dataset_dir / name
            if path.exists():
                found[split] = path
                break
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, help="Dataset directory containing *.jsonl splits.")
    parser.add_argument("--eval", type=Path, dest="eval_path", help="Evaluation JSONL to validate.")
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Report all invalid examples rather than the first.",
    )
    parser.add_argument(
        "--require-reviewed", action="store_true", help="Treat unreviewed examples as errors."
    )
    parser.add_argument(
        "--max-tokens", type=int, help="Flag examples longer than this (approximate tokens)."
    )
    parser.add_argument(
        "--leakage-report", type=Path, help="Directory to write the leakage report into."
    )
    parser.add_argument(
        "--leakage-policy",
        choices=["none", "fatal", "any"],
        default="fatal",
        help="Which cross-split findings cause a non-zero exit (default: fatal).",
    )
    parser.add_argument(
        "--no-content-scan", action="store_true", help="Skip sensitive-content heuristics."
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    if not args.dataset and not args.eval_path:
        parser.error("supply --dataset and/or --eval")

    exit_code = 0
    splits: dict[str, list] = {}

    # --- training splits ----------------------------------------------------
    if args.dataset:
        found = _discover_splits(args.dataset)
        if not found:
            print(f"\n✗ No split files found in {args.dataset}", file=sys.stderr)
            print("  Expected train.jsonl (or synthetic_train.jsonl).\n", file=sys.stderr)
            return 1

        for split_name, path in found.items():
            print(f"\n── {split_name}: {path.name} " + "─" * max(0, 48 - len(path.name)))
            examples, load_report = load_examples(
                path, model=TrainingExample, strict=not args.no_strict
            )
            if load_report.errors:
                print(f"  {len(load_report.errors)} example(s) failed schema validation:")
                for problem in load_report.errors[:10]:
                    fields = ", ".join(
                        f"{e['field']}: {e['message']}" for e in problem["errors"][:3]
                    )
                    print(f"    line {problem['line']} (id={problem['id']}): {fields}")
                exit_code = 1

            report = validate_examples(
                examples,
                split_name=split_name,
                max_tokens=args.max_tokens,
                require_reviewed=args.require_reviewed,
                scan_content=not args.no_content_scan,
            )
            print(report.render())
            if not report.ok:
                exit_code = 1
            splits[split_name] = examples

    # --- evaluation benchmark ----------------------------------------------
    if args.eval_path:
        print(f"\n── evaluation: {args.eval_path.name} " + "─" * 30)
        eval_examples, eval_load = load_examples(
            args.eval_path, model=EvaluationExample, strict=not args.no_strict
        )
        if eval_load.errors:
            print(f"  {len(eval_load.errors)} example(s) failed schema validation:")
            for problem in eval_load.errors[:10]:
                fields = ", ".join(f"{e['field']}: {e['message']}" for e in problem["errors"][:3])
                print(f"    line {problem['line']} (id={problem['id']}): {fields}")
            exit_code = 1
        else:
            print(f"  ✓ {len(eval_examples)} evaluation example(s) valid")
            tagged = sum(1 for e in eval_examples if e.split_tag == "ood")
            if tagged == 0:
                print("  ! No example is tagged split_tag='ood'.")
                print("    OOD evaluation will not be measurable, so no generalization")
                print("    claim can be made from this benchmark.")
            else:
                print(f"  ✓ {tagged} OOD-tagged example(s)")
        splits["evaluation"] = eval_examples

    # --- leakage ------------------------------------------------------------
    if len(splits) >= 1:
        print("\n── leakage " + "─" * 52)
        leakage = check_leakage(splits)
        counts = leakage.counts()
        print(f"  examined : {leakage.examined}")
        print(f"  findings : {counts['total']} total, {counts['cross_split']} cross-split")
        for kind, count in counts.items():
            if kind not in ("total", "cross_split") and count:
                print(f"    {kind:<24} {count}")
        if leakage.clean:
            print("  ✓ no leakage detected")

        if args.leakage_report:
            paths = leakage.write(args.leakage_report)
            print(f"  report   : {paths['markdown']}")

        try:
            enforce_leakage_policy(leakage, fail_on=args.leakage_policy)
        except Exception as exc:
            print(f"\n✗ {exc}\n", file=sys.stderr)
            exit_code = 1

    print()
    if exit_code == 0:
        print("✓ Dataset validation passed.\n")
    else:
        print("✗ Dataset validation failed. See findings above.\n", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run(main))
