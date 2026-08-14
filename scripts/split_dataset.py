#!/usr/bin/env python3
"""Split a dataset into train/validation/test (spec §12).

Random splitting is development-only. Generalization claims need a held-out
strategy, because a random split of a dataset containing paraphrases puts
near-copies on both sides of the boundary.

Usage::

    python scripts/split_dataset.py --input data/examples/synthetic_train.jsonl \
                                    --output data/processed/v1 \
                                    --strategy entity_holdout --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import add_common_arguments, run, setup_logging
from kleos_models.config import SplitConfig
from kleos_models.constants import SPLIT_STRATEGIES
from kleos_models.data.leakage import check_leakage
from kleos_models.data.loaders import file_sha256, load_examples, write_jsonl
from kleos_models.data.schemas import DatasetManifest, SplitCounts, TrainingExample
from kleos_models.data.splitting import split_examples
from kleos_models.logging_utils import get_logger

logger = get_logger("split_dataset")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True, help="JSONL file to split.")
    parser.add_argument("--output", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument("--strategy", choices=list(SPLIT_STRATEGIES), default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--group-key", help="Metadata key for group/scenario strategies.")
    parser.add_argument(
        "--holdout", nargs="*", default=[], help="Values to hold out for *_holdout strategies."
    )
    parser.add_argument(
        "--version", default="kleos-policy-v0.1.0", help="Dataset version to write."
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing version directory."
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        print(f"\n✗ {args.output} already exists and is not empty.", file=sys.stderr)
        print("  Dataset versions are immutable: silently overwriting one breaks", file=sys.stderr)
        print("  every comparison that referenced it. Use a new --version, or", file=sys.stderr)
        print("  pass --force if you are certain.\n", file=sys.stderr)
        return 1

    examples, _ = load_examples(args.input, model=TrainingExample, strict=True)
    print(f"Loaded {len(examples)} example(s) from {args.input}")

    config = SplitConfig(
        strategy=args.strategy,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        group_key=args.group_key,
        holdout_values=args.holdout,
    )
    result = split_examples(examples, config)

    print(f"\nStrategy : {result.strategy} (seed {result.seed})")
    for name, count in result.counts.items():
        share = 100.0 * count / max(result.total, 1)
        print(f"  {name:<12} {count:>5}  ({share:5.1f}%)")
    if result.holdout_values:
        print(f"  held out : {result.holdout_values}")
    for note in result.notes:
        print(f"  note     : {note}")

    if args.strategy == "random":
        print("\n! Random split: development use only. Near-duplicate scenarios can")
        print("  land on both sides, which inflates apparent generalization.")

    # Verify the split did not leak before writing it.
    leakage = check_leakage(
        {name: result.split(name) for name in ("train", "validation", "test") if result.split(name)}
    )
    cross = leakage.cross_split_findings
    print(f"\nLeakage check: {len(cross)} cross-split finding(s)")
    if cross:
        print("  ! The split leaks. Fix this before training:")
        for finding in cross[:5]:
            print(f"    {finding.kind.value}: {finding.left_id} ↔ {finding.right_id}")

    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ("train", "validation", "test"):
        rows = result.split(name)
        if not rows:
            continue
        path = write_jsonl(rows, args.output / f"{name}.jsonl")
        hashes[f"{name}.jsonl"] = file_sha256(path)

    manifest = DatasetManifest(
        version=args.version,
        source="split",
        description=f"Split from {args.input.name} with strategy {result.strategy}",
        example_count=result.total,
        splits=SplitCounts(**result.counts),
        task_distribution=dict(Counter(e.task for e in examples).most_common()),
        domain_distribution=dict(Counter(e.variation_axes.domain for e in examples).most_common()),
        split_strategy=result.strategy,
        split_seed=result.seed,
        holdout_values=result.holdout_values,
        file_hashes=hashes,
    ).finalize()

    (args.output / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8"
    )

    print(f"\n✓ Wrote dataset version {args.version} to {args.output}")
    print(f"  content hash: {manifest.content_hash[:16]}")
    print(
        f"\nNext: python scripts/train.py --config configs/training/qlora_small.yaml --dataset {args.output}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
