#!/usr/bin/env python3
"""Derive an evaluation benchmark from a sealed KLEOS dataset release.

The sealed release stores its test split as ``TrainingExample`` rows: a
conversation ending in the assistant turn, with no grader reference attached.
The evaluation harness consumes ``EvaluationExample`` rows, which reject that
shape (``domain`` is not a field there) and require an explicit ``reference``.
Without this step the held-out split cannot be evaluated at all.

The conversion is mechanical and lossless in the direction that matters: the
prompt is carried across verbatim and the reference is *read out of the sealed
target*, never authored here.

    prompt      = messages[:-1]            (verbatim, system + user)
    reference   = json.loads(messages[-1]) → ranking / deciding_factor /
                  confident / next_step
    grader      = kleos_policy
    split_tag   = ood
    ood_shift   = unseen_formats

THE SEALED RELEASE IS NEVER WRITTEN TO. Its hashes are verified against
RELEASE.lock before anything is read, and the benchmark is written to a separate
directory with its own manifest, so a reviewer can re-run this and compare.

Usage::

    python scripts/build_benchmark.py --dataset data/processed/kleos-policy-v0.0.6
    python scripts/build_benchmark.py --dataset <dir> --output <dir> --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import add_common_arguments, print_header, run, setup_logging
from kleos_models.data.loaders import load_examples, write_jsonl
from kleos_models.data.schemas import EvaluationExample, TrainingExample
from kleos_models.errors import DatasetIntegrityError
from kleos_models.logging_utils import get_logger

logger = get_logger("build_benchmark")

RELEASE_LOCK_FILENAME = "RELEASE.lock"
BENCHMARK_FILENAME = "benchmark.jsonl"
BENCHMARK_MANIFEST_FILENAME = "benchmark_manifest.json"

#: The held-out axis in a ``format_holdout`` release. Declared explicitly so the
#: OOD report can never silently treat this benchmark as in-distribution.
OOD_SHIFT = "unseen_formats"
DEFAULT_GRADER = "kleos_policy"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sealed_release(dataset_dir: Path) -> dict[str, Any]:
    """Refuse to build from a release whose bytes do not match its own lock."""
    lock_path = dataset_dir / RELEASE_LOCK_FILENAME
    if not lock_path.exists():
        raise DatasetIntegrityError(
            f"No {RELEASE_LOCK_FILENAME} in {dataset_dir}.",
            suggestions=["Point --dataset at a sealed release directory."],
        )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    mismatches = []
    for filename, expected in sorted(lock.get("all_file_hashes", {}).items()):
        target = dataset_dir / filename
        if not target.exists():
            mismatches.append(f"{filename}: missing")
            continue
        actual = file_sha256(target)
        if actual != expected:
            mismatches.append(f"{filename}: {actual[:16]} != {expected[:16]}")

    if mismatches:
        raise DatasetIntegrityError(
            "Sealed release does not match RELEASE.lock; refusing to build a benchmark.",
            details={"mismatches": mismatches},
            suggestions=["Re-copy the release from the immutable original."],
        )
    logger.info(
        "Verified %d file(s) against %s",
        len(lock.get("all_file_hashes", {})),
        RELEASE_LOCK_FILENAME,
    )
    return lock


def build_reference(example: TrainingExample) -> dict[str, Any]:
    """Read the grader reference out of the sealed target turn.

    Raises:
        DatasetIntegrityError: when the target is not the JSON object the
            ``format_holdout`` test split is defined to contain. Guessing a
            reference from prose would invent ground truth.
    """
    target = example.messages[-1].content
    try:
        payload = json.loads(target)
    except json.JSONDecodeError as exc:
        raise DatasetIntegrityError(
            f"Target of {example.id} is not JSON; cannot derive a reference.",
            details={"error": str(exc), "target_preview": target[:200]},
            suggestions=["This script expects a format_holdout test split of JSON targets."],
        ) from exc

    if not isinstance(payload, dict) or "ranking" not in payload:
        raise DatasetIntegrityError(
            f"Target of {example.id} has no 'ranking'; cannot derive a reference.",
            details={
                "keys": sorted(payload) if isinstance(payload, dict) else type(payload).__name__
            },
        )

    reference: dict[str, Any] = {"ranking": [str(item) for item in payload["ranking"]]}

    # 'label' is the key every classification-style grader reads. The deciding
    # factor is the policy KLEOS is being taught, so it is the label here.
    if isinstance(payload.get("deciding_factor"), str):
        reference["label"] = payload["deciding_factor"]
    if isinstance(payload.get("confident"), bool):
        reference["confident"] = payload["confident"]
    if isinstance(payload.get("next_step"), str):
        reference["next_step"] = payload["next_step"]
    if isinstance(payload.get("reasons"), dict):
        reference["reasons"] = payload["reasons"]
    return reference


def to_evaluation_example(
    example: TrainingExample, *, options: list[str], grader: str
) -> EvaluationExample:
    """Convert one sealed training row into a benchmark row.

    The prompt is carried across verbatim, minus the target turn. Nothing about
    the conversation is rewritten, so the model sees exactly what the release
    specifies.
    """
    reference = build_reference(example)
    if "label" in reference:
        reference["options"] = options

    return EvaluationExample(
        id=example.id,
        version=example.version,
        task=example.task,
        messages=example.messages[:-1],
        variation_axes=example.variation_axes,
        metadata=example.metadata,
        reference=reference,
        grader=grader,
        split_tag="ood",
        ood_shift=OOD_SHIFT,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Sealed release directory (read-only)."
    )
    parser.add_argument(
        "--split", default="test", help="Split to convert (default: test). Never 'train'."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (default: <dataset>-benchmark, alongside the release).",
    )
    parser.add_argument(
        "--grader", default=DEFAULT_GRADER, help=f"Grader (default: {DEFAULT_GRADER})."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Rebuild and compare against an existing benchmark instead of writing.",
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("KLEOS benchmark build")

    dataset_dir = args.dataset.resolve()
    if args.split == "train":
        print("\n✗ Refusing to build a benchmark from the training split.\n", file=sys.stderr)
        return 1

    lock = verify_sealed_release(dataset_dir)
    release = lock.get("version", "unknown")
    print(f"  release      : {release}")
    print(f"  content hash : {lock.get('content_hash', '?')[:16]}")
    print(f"  source split : {args.split}.jsonl (read-only)")

    source = dataset_dir / f"{args.split}.jsonl"
    examples, report = load_examples(source, model=TrainingExample, strict=True)
    if report.errors:
        print(f"\n✗ {len(report.errors)} invalid row(s) in {source}.\n", file=sys.stderr)
        return 1
    print(f"  loaded       : {len(examples)} example(s)")

    # A stable, sorted option list keeps the classification reference identical
    # across rebuilds regardless of row order.
    options = sorted(
        {
            reference["label"]
            for reference in (build_reference(example) for example in examples)
            if "label" in reference
        }
    )
    print(f"  label space  : {len(options)} deciding factor(s)")

    benchmark = [
        to_evaluation_example(example, options=options, grader=args.grader) for example in examples
    ]
    if len(benchmark) != len(examples):
        print("\n✗ Example count changed during conversion.\n", file=sys.stderr)
        return 1

    output_dir = args.output or dataset_dir.parent / f"{dataset_dir.name}-benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = output_dir / BENCHMARK_FILENAME

    if args.verify:
        if not benchmark_path.exists():
            print(f"\n✗ Nothing to verify at {benchmark_path}.\n", file=sys.stderr)
            return 1
        existing = file_sha256(benchmark_path)
        scratch = output_dir / f".{BENCHMARK_FILENAME}.rebuild"
        write_jsonl(benchmark, scratch)
        rebuilt = file_sha256(scratch)
        scratch.unlink()
        if existing != rebuilt:
            print(f"\n✗ Benchmark is not reproducible: {existing[:16]} != {rebuilt[:16]}\n")
            return 1
        print(f"\n✓ Benchmark reproduces exactly ({existing[:16]}).\n")
        return 0

    write_jsonl(benchmark, benchmark_path)

    # Round-trip through the loader the harness itself uses. Writing a file the
    # evaluator cannot read is the exact failure this script exists to remove.
    reloaded, reload_report = load_examples(benchmark_path, model=EvaluationExample, strict=True)
    if reload_report.errors or len(reloaded) != len(examples):
        print(
            f"\n✗ Benchmark did not round-trip: {len(reloaded)}/{len(examples)} loaded, "
            f"{len(reload_report.errors)} error(s).\n",
            file=sys.stderr,
        )
        return 1

    manifest = {
        "benchmark_of": release,
        "source_split": args.split,
        "source_content_hash": lock.get("content_hash"),
        "source_file_hash": lock.get("all_file_hashes", {}).get(f"{args.split}.jsonl"),
        "example_count": len(benchmark),
        "grader": args.grader,
        "split_tag": "ood",
        "ood_shift": OOD_SHIFT,
        "deciding_factor_options": options,
        "benchmark_sha256": file_sha256(benchmark_path),
        "built_at": datetime.now(UTC).isoformat(),
        "built_by": "scripts/build_benchmark.py",
        "note": (
            "Derived mechanically from the sealed release; the release itself is "
            "read-only and unmodified. Held-out evaluation only: never train on "
            "this file and never use it to select hyperparameters."
        ),
    }
    manifest_path = output_dir / BENCHMARK_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    counts: dict[str, int] = {}
    for item in benchmark:
        counts[item.task] = counts.get(item.task, 0) + 1

    print(f"\n  → {benchmark_path}")
    print(f"  → {manifest_path}")
    print(f"  sha256       : {manifest['benchmark_sha256'][:16]}")
    print(f"  round-trip   : {len(reloaded)}/{len(examples)} loaded as EvaluationExample ✓")
    print("\n  per task:")
    for task, count in sorted(counts.items()):
        print(f"    {task:32s} {count:4d}")
    print(
        f"\n✓ Benchmark built. The sealed release at {dataset_dir} was not modified.\n"
        f"  This split is held out: evaluation only, never training or tuning.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
