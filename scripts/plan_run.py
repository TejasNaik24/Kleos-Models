#!/usr/bin/env python3
"""Check whether a training run fits the available GPU before launching it.

Answers "will this work on the runtime I have been assigned?" without downloading
any weights. On a free Colab tier this is the difference between finding out in
two seconds and finding out after a 28GB download.

Usage::

    python scripts/plan_run.py --config configs/training/qlora_small.yaml
    python scripts/plan_run.py --all-models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import add_common_arguments, print_header, run, setup_logging

REPO_ROOT = Path(__file__).resolve().parent.parent

from kleos_models.config import TrainingConfig, load_config, load_model_config
from kleos_models.logging_utils import get_logger
from kleos_models.models.adapters import get_adapter
from kleos_models.models.feasibility import (
    assess_feasibility,
    probe_gpu,
    render_feasibility_table,
)

logger = get_logger("plan_run")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="Experiment config to assess.")
    parser.add_argument(
        "--all-models", action="store_true", help="Assess every model config in configs/models/."
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--json", type=Path, help="Write the assessment as JSON.")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    if not args.config and not args.all_models:
        parser.error("supply --config or --all-models")

    gpu = probe_gpu()
    print_header("Run feasibility")
    print(f"Detected: {gpu.render()}\n")

    if not gpu.available:
        print("No CUDA GPU is available, so nothing can be trained here.")
        print("Memory estimates below assume the model would need to fit on one GPU.\n")

    reports = []
    if args.all_models:
        training = TrainingConfig()
        for path in sorted((REPO_ROOT / "configs" / "models").glob("*.yaml")):
            model_config = load_model_config(path)
            adapter = get_adapter(model_config)
            reports.append(
                assess_feasibility(
                    model_config, training, gpu=gpu, target_modules=adapter.default_target_modules
                )
            )
        print(render_feasibility_table(reports))
    else:
        config = load_config(args.config, overrides=args.overrides)
        adapter = get_adapter(config.model)
        report = assess_feasibility(
            config.model, config.training, gpu=gpu, target_modules=adapter.default_target_modules
        )
        reports.append(report)
        print(report.render())
        print()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([r.to_dict() for r in reports], indent=2, default=str), encoding="utf-8"
        )
        print(f"Wrote assessment to {args.json}\n")

    # Exit non-zero if the single assessed config cannot be trained, so a driver
    # script can branch on it.
    if args.config and not reports[0].fits:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
