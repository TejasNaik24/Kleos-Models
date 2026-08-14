#!/usr/bin/env python3
"""Print a dataset quality and coverage report (spec §25).

Reports example counts, per-task and per-domain distributions, token statistics,
missing metadata, and variation-axis coverage — including which situation types
have too few examples to support a claim.

Usage::

    python scripts/inspect_dataset.py --dataset data/examples
    python scripts/inspect_dataset.py --dataset data/examples --coverage
    python scripts/inspect_dataset.py --dataset data/examples --tokenizer Qwen/Qwen3-8B
    python scripts/inspect_dataset.py --dataset data/examples --show-masking
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import add_common_arguments, run, setup_logging
from kleos_models.data.coverage import (
    DEFAULT_JOINT_AXES,
    build_coverage_report,
    render_coverage_report,
)
from kleos_models.data.loaders import load_examples, load_manifest
from kleos_models.data.schemas import TrainingExample
from kleos_models.data.validation import build_quality_report, render_quality_report
from kleos_models.logging_utils import get_logger

logger = get_logger("inspect_dataset")


def _load_all(dataset_dir: Path) -> tuple[list[TrainingExample], dict[str, int]]:
    """Load every training split found in a dataset directory."""
    names = ["train.jsonl", "synthetic_train.jsonl", "validation.jsonl", "test.jsonl"]
    examples: list[TrainingExample] = []
    counts: dict[str, int] = {}
    for name in names:
        path = dataset_dir / name
        if not path.exists():
            continue
        loaded, _ = load_examples(path, model=TrainingExample, strict=False)
        split = name.replace("synthetic_", "").replace(".jsonl", "")
        counts[split] = len(loaded)
        examples.extend(loaded)
    return examples, counts


def _build_token_counter(tokenizer_id: str):
    """Build a real token counter, or fall back with a clear message."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        print(f"Using tokenizer {tokenizer_id} for exact token counts.\n")
        return lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    except Exception as exc:
        print(f"Could not load tokenizer {tokenizer_id} ({type(exc).__name__}).")
        print("Falling back to approximate token counts.\n")
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset directory.")
    parser.add_argument(
        "--coverage", action="store_true", help="Include the variation-axis coverage report."
    )
    parser.add_argument(
        "--joint-axes",
        nargs="+",
        default=list(DEFAULT_JOINT_AXES),
        help="Axes defining a situation type.",
    )
    parser.add_argument(
        "--min-cell-count", type=int, default=3, help="Cells below this are reported as thin."
    )
    parser.add_argument("--tokenizer", help="Hugging Face tokenizer id for exact token counts.")
    parser.add_argument(
        "--show-masking",
        action="store_true",
        help="Show which tokens would be supervised (needs --tokenizer).",
    )
    parser.add_argument("--json", type=Path, help="Also write the report as JSON.")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    examples, counts = _load_all(args.dataset)
    if not examples:
        print(f"\n✗ No training examples found in {args.dataset}\n", file=sys.stderr)
        return 1

    manifest = load_manifest(args.dataset)
    version = manifest.version if manifest else "unversioned"

    token_counter = _build_token_counter(args.tokenizer) if args.tokenizer else None

    quality = build_quality_report(
        examples, version=version, counts=counts, token_counter=token_counter
    )
    print(render_quality_report(quality))

    coverage = None
    if args.coverage:
        coverage = build_coverage_report(
            examples, joint_axes=args.joint_axes, min_cell_count=args.min_cell_count
        )
        print(render_coverage_report(coverage))

    if args.show_masking:
        if not args.tokenizer:
            print("--show-masking requires --tokenizer\n", file=sys.stderr)
            return 1
        from transformers import AutoTokenizer

        from kleos_models.data.formatting import ConversationFormatter

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        formatter = ConversationFormatter(tokenizer, max_seq_length=4096)
        print("Assistant-token masking (supervised / total tokens)")
        print("  Counts only — example text is never printed.\n")
        for example in examples[:20]:
            print("   ", formatter.describe_masking(example))
        print()

    if manifest and manifest.contains_private_data:
        print("!" * 72)
        print("This dataset is flagged contains_private_data=true.")
        print("It must NOT be committed to this public repository.")
        print("!" * 72)
        print()

    if args.json:
        payload = {"quality": quality.to_dict()}
        if coverage:
            payload["coverage"] = coverage.to_dict()
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to {args.json}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
