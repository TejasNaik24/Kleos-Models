#!/usr/bin/env python3
"""Compare base and fine-tuned evaluation results (spec §54).

Reports per-task base, fine-tuned, absolute delta, relative delta, OOD delta and
consistency delta. Deltas are paired per example with bootstrap confidence
intervals, and an improvement that is not statistically significant is reported as
such rather than as a win.

There is no blended aggregate unless you ask for one with --show-aggregate: a
single number across tasks hides exactly the per-task trade-offs that matter.

Usage::

    python scripts/compare.py --base outputs/base_results.json \
                              --finetuned outputs/finetuned_results.json

    python scripts/compare.py --base outputs/base_results.json \
                              --finetuned outputs/finetuned_results.json \
                              --report reports/experiment-001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import add_common_arguments, run, setup_logging
from kleos_models.evaluation.reports import compare_results, write_experiment_report
from kleos_models.evaluation.runner import load_evaluation_result
from kleos_models.logging_utils import get_logger

logger = get_logger("compare")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base", type=Path, required=True, help="Base-arm results JSON.")
    parser.add_argument(
        "--finetuned", type=Path, required=True, help="Fine-tuned-arm results JSON."
    )
    parser.add_argument(
        "--report", type=Path, help="Directory to write summary.md and metrics.json into."
    )
    parser.add_argument(
        "--experiment-id", default="unknown", help="Experiment id for the report header."
    )
    parser.add_argument(
        "--dataset-version", default="unknown", help="Dataset version for the report header."
    )
    parser.add_argument(
        "--show-aggregate",
        action="store_true",
        help="Also print a single blended score across tasks.",
    )
    parser.add_argument(
        "--no-significance", action="store_true", help="Skip bootstrap significance testing."
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    base = load_evaluation_result(args.base)
    finetuned = load_evaluation_result(args.finetuned)

    comparison = compare_results(base, finetuned, compute_significance=not args.no_significance)
    print("\n" + comparison.render(show_aggregate=args.show_aggregate))

    if args.report:
        paths = write_experiment_report(
            comparison,
            args.report,
            experiment_id=args.experiment_id,
            dataset_version=args.dataset_version,
            config_summary={
                "base_arm": comparison.base_arm,
                "finetuned_arm": comparison.finetuned_arm,
                "base_model": comparison.base_model.get("base_model", "-"),
                "adapter": comparison.finetuned_model.get("adapter_path", "-"),
            },
        )
        print(f"✓ Report written to {paths['summary']}")
        print(f"  metrics    {paths['metrics']}\n")

    # Non-zero exit when the fine-tuned arm regressed, so CI or a driver script
    # can notice. A regression is a valid result, not an error, hence exit 0 by
    # default unless --fail-on-regression is used by the caller.
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
