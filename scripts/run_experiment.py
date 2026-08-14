#!/usr/bin/env python3
"""Run a full experiment: train, evaluate both arms, compare, report.

Chains the individual scripts so a complete controlled comparison is one command.
Every stage writes its own artifacts, so a failure part-way still leaves a usable
record.

Usage::

    python scripts/run_experiment.py --config configs/training/qlora_small.yaml
    python scripts/run_experiment.py --config configs/training/qlora_small.yaml \
                                     --dataset /path/to/private/dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import (
    add_common_arguments,
    add_config_arguments,
    print_header,
    run,
    setup_logging,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

from kleos_models.config import load_config
from kleos_models.experiments.registry import ExperimentRegistry
from kleos_models.logging_utils import get_logger

logger = get_logger("run_experiment")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_arguments(parser)
    parser.add_argument(
        "--skip-training", action="store_true", help="Evaluate an existing adapter."
    )
    parser.add_argument("--adapter", type=Path, help="Existing adapter, with --skip-training.")
    parser.add_argument(
        "--report", type=Path, help="Report directory (default: reports/<experiment-id>)."
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    config = load_config(
        args.config, overrides=args.overrides, dataset_path=args.dataset, output_dir=args.output_dir
    )

    print_header("KLEOS experiment")
    print(f"  config : {args.config}")
    print(f"  model  : {config.model.base_model}")
    print(f"  arms   : {config.evaluation.arms}")

    # --- 1. train -----------------------------------------------------------
    adapter_path = args.adapter
    experiment_id = "unknown"
    if not args.skip_training:
        import train as train_script

        train_argv = ["--config", str(args.config)]
        for override in args.overrides:
            train_argv += ["--set", override]
        if args.dataset:
            train_argv += ["--dataset", str(args.dataset)]
        print("\n▶ Stage 1/4: training")
        code = train_script.main(train_argv)
        if code != 0:
            print("\n✗ Training failed; experiment aborted.", file=sys.stderr)
            return code

        registry = ExperimentRegistry(config.training.output_dir)
        entries = registry.filter(kind="training")
        if not entries:
            print("\n✗ No training run found after training.", file=sys.stderr)
            return 1
        latest = entries[0]
        experiment_id = latest.experiment_id
        adapter_path = latest.directory / "adapter"
    elif adapter_path is None:
        parser.error("--skip-training requires --adapter")

    # --- 2-3. evaluate both arms -------------------------------------------
    import evaluate as evaluate_script

    output_root = Path(config.training.output_dir)
    base_results = output_root / f"{experiment_id}_base_results.json"
    ft_results = output_root / f"{experiment_id}_finetuned_results.json"

    print("\n▶ Stage 2/4: evaluating the base arm")
    code = evaluate_script.main(
        ["--config", str(args.config), "--arm", "arm0_base", "--output", str(base_results)]
    )
    if code != 0:
        return code

    print("\n▶ Stage 3/4: evaluating the fine-tuned arm")
    code = evaluate_script.main(
        [
            "--config",
            str(args.config),
            "--arm",
            "arm2_finetuned",
            "--adapter",
            str(adapter_path),
            "--output",
            str(ft_results),
        ]
    )
    if code != 0:
        return code

    # --- 4. compare ---------------------------------------------------------
    import compare as compare_script

    report_dir = args.report or (REPO_ROOT / "reports" / experiment_id)
    print("\n▶ Stage 4/4: comparing")
    code = compare_script.main(
        [
            "--base",
            str(base_results),
            "--finetuned",
            str(ft_results),
            "--report",
            str(report_dir),
            "--experiment-id",
            experiment_id,
        ]
    )
    if code != 0:
        return code

    print("\n" + "=" * 72)
    print("Experiment complete")
    print("=" * 72)
    print(f"  experiment id : {experiment_id}")
    print(f"  adapter       : {adapter_path}")
    print(f"  report        : {report_dir}/summary.md")
    print("\n  Read the report before drawing a conclusion. A training loss curve")
    print("  is not a result.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
