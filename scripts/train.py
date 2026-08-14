#!/usr/bin/env python3
"""Train a KLEOS LoRA/QLoRA adapter (spec §15).

Executes the pipeline in the order the specification defines: load and validate
config, inspect the environment, load tokenizer and quantized model, prepare
PEFT, load and validate the dataset, format examples, train, evaluate, save the
adapter, tokenizer, effective config and environment metadata, write the
experiment manifest, and print the artifact paths.

Usage::

    python scripts/train.py --config configs/training/qlora_small.yaml

    # Consume the externally produced private dataset artifact:
    python scripts/train.py --config configs/training/qlora_small.yaml \\
                            --dataset /path/to/private/dataset

    # Resume after a Colab runtime reset:
    python scripts/train.py --config configs/training/qlora_small.yaml \\
                            --resume-from-checkpoint auto

    # See whether it fits before committing a GPU session:
    python scripts/train.py --config configs/training/qlora_small.yaml --dry-run
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
from kleos_models.config import load_config
from kleos_models.constants import TRAINING_LOG_FILENAME
from kleos_models.data.leakage import check_leakage, enforce_leakage_policy
from kleos_models.data.loaders import load_dataset_bundle
from kleos_models.data.validation import validate_examples
from kleos_models.experiments.environment import set_global_seed
from kleos_models.experiments.manifest import build_manifest
from kleos_models.logging_utils import get_logger

logger = get_logger("train")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_arguments(parser)
    parser.add_argument(
        "--resume-from-checkpoint",
        metavar="PATH|auto",
        help="Resume from a checkpoint path, or 'auto' for the newest valid one.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, dataset and feasibility, then stop before loading weights.",
    )
    parser.add_argument(
        "--leakage-policy",
        choices=["none", "fatal", "any"],
        default="fatal",
        help="Which cross-split leakage findings abort training (default: fatal).",
    )
    parser.add_argument(
        "--skip-gradient-check",
        action="store_true",
        help="Skip the pre-training forward/backward verification.",
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    # --- 1-2. load and validate config -------------------------------------
    print_header("KLEOS training")
    config = load_config(
        args.config,
        overrides=args.overrides,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
    )
    print(f"  config      : {args.config}")
    print(f"  experiment  : {config.name}")
    print(f"  model       : {config.model.name} ({config.model.base_model})")
    print(f"  task        : {config.task or 'not restricted to a single task'}")
    print(f"  config hash : {config.config_hash[:16]}")

    if config.dataset is None:
        print("\n✗ No dataset configured. Pass --dataset or set dataset.path.\n", file=sys.stderr)
        return 1

    seeding = set_global_seed(config.seed)

    # --- 7-8. dataset -------------------------------------------------------
    print("\n── dataset " + "─" * 52)
    bundle = load_dataset_bundle(config.dataset, strict=True, require_splits=("train",))
    print(f"  version : {bundle.version}")
    print(f"  counts  : {bundle.counts}")
    print(f"  hash    : {bundle.dataset_hash()[:16]}")

    report = validate_examples(
        bundle.train,
        split_name="train",
        max_tokens=config.model.max_seq_length,
        require_reviewed=not config.dataset.allow_unreviewed,
    )
    if not report.ok:
        print("\n" + report.render(), file=sys.stderr)
        print("\n✗ Dataset validation failed. Training aborted.\n", file=sys.stderr)
        print("  Run: python scripts/validate_dataset.py --dataset <dir>\n", file=sys.stderr)
        return 1
    if report.warnings:
        print(f"  {len(report.warnings)} validation warning(s):")
        for warning in report.warnings[:5]:
            print(f"    ! {warning.message}")

    splits = {
        name: bundle.split(name) for name in ("train", "validation", "test") if bundle.split(name)
    }
    if len(splits) > 1:
        leakage = check_leakage(splits)
        cross = leakage.cross_split_findings
        print(f"  leakage : {len(cross)} cross-split finding(s)")
        enforce_leakage_policy(leakage, fail_on=args.leakage_policy)

    # --- 3. environment and feasibility ------------------------------------
    manifest = build_manifest(
        config,
        kind="training",
        dataset_version=bundle.version,
        dataset_hash=bundle.dataset_hash(),
        dataset_counts=bundle.counts,
    )
    manifest.seeding = seeding
    print(f"\n  experiment id : {manifest.experiment_id}")

    from kleos_models.models.adapters import get_adapter
    from kleos_models.models.feasibility import assess_feasibility, enforce_feasibility

    adapter = get_adapter(config.model)
    feasibility = assess_feasibility(
        config.model,
        config.training,
        target_modules=adapter.default_target_modules,
    )
    manifest.feasibility = feasibility.to_dict()
    print("\n" + feasibility.render())

    if args.dry_run:
        output_dir = Path(config.training.output_dir) / manifest.experiment_id
        manifest.note("Dry run: no weights were loaded and no training was performed.")
        manifest.save(output_dir)
        print(f"\n✓ Dry run complete. Manifest written to {output_dir}/manifest.json")
        print("  Remove --dry-run to train.\n")
        return 0

    adjustments = enforce_feasibility(feasibility, config.training)
    for adjustment in adjustments:
        manifest.add_adjustment(
            adjustment.field, adjustment.original, adjustment.adjusted, adjustment.reason
        )
        # Apply the recorded adjustment to the live config.
        if adjustment.field == "model.max_seq_length":
            config.model.max_seq_length = adjustment.adjusted
        elif adjustment.field == "training.gradient_checkpointing":
            config.training.gradient_checkpointing = adjustment.adjusted
        elif adjustment.field == "training.per_device_train_batch_size":
            original = config.training.per_device_train_batch_size
            config.training.per_device_train_batch_size = adjustment.adjusted
            # Preserve the effective batch size so learning dynamics are unchanged.
            factor = max(1, original // max(1, adjustment.adjusted))
            config.training.gradient_accumulation_steps *= factor

    # --- 4-17. train --------------------------------------------------------
    output_dir = Path(config.training.output_dir) / manifest.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args, log_file=output_dir / TRAINING_LOG_FILENAME)

    from kleos_models.training.trainer import run_training

    result = run_training(
        config,
        bundle,
        manifest,
        resume_from_checkpoint=args.resume_from_checkpoint,
        verify_gradients=not args.skip_gradient_check,
    )

    # --- 18. report artifacts ----------------------------------------------
    print("\n" + result.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
