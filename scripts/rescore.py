#!/usr/bin/env python3
"""Re-grade a stored evaluation result offline, without re-running inference.

Evaluation results retain the model's raw responses, so a grader defect can be
corrected after the fact by re-scoring the saved artifact. No GPU, no model, no
regenerated text — the responses are exactly the ones the model produced.

    python scripts/rescore.py --results eval_arm2_full.json \\
                              --benchmark benchmark.jsonl \\
                              --output eval_arm2_full.rescored.json

**The input file is opened read-only and never written back.** The re-scored copy
goes to a new path, so the original stays frozen and its hash stays valid. Both
arms of a comparison must be re-scored with the same grader version, or the
comparison is meaningless — the script records the grader in the output so that
is checkable.

Scores are recomputed with ``summarize``, the same function ``run_evaluation``
uses, so the output is structurally identical to a fresh evaluation and
``compare.py`` consumes it unchanged.
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
from kleos_models.errors import EvaluationError
from kleos_models.evaluation.graders import get_grader
from kleos_models.evaluation.metrics import summarize
from kleos_models.logging_utils import get_logger

logger = get_logger("rescore")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_references(benchmark: Path) -> dict[str, dict[str, Any]]:
    """Map example id → grader reference, read from the benchmark."""
    references: dict[str, dict[str, Any]] = {}
    with benchmark.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            references[row["id"]] = row.get("reference", {})
    return references


def rescore(
    payload: dict[str, Any], references: dict[str, dict[str, Any]], *, default_grader: str
) -> tuple[dict[str, Any], list[tuple[str, float, float]]]:
    """Return a re-scored copy of ``payload`` plus the per-example score changes.

    Raises:
        EvaluationError: when a row has no stored response, or no reference. Both
            mean the re-score would be inventing data rather than recomputing it.
    """
    rows = payload.get("results") or []
    if not rows:
        raise EvaluationError("Results payload contains no examples to re-score.")

    graders: dict[str, Any] = {}
    changes: list[tuple[str, float, float]] = []
    rescored_rows: list[dict[str, Any]] = []

    for row in rows:
        example_id = row["example_id"]
        if "response" not in row:
            raise EvaluationError(
                f"Example {example_id} has no stored response, so it cannot be re-graded.",
                suggestions=[
                    "Re-scoring needs results saved with responses included "
                    "(evaluate.py without --no-responses)."
                ],
            )
        reference = references.get(example_id)
        if reference is None:
            raise EvaluationError(
                f"Example {example_id} is not present in the benchmark.",
                details={"benchmark_examples": len(references)},
                suggestions=["Point --benchmark at the file this run was evaluated against."],
            )

        name = row.get("grader") or default_grader
        if name not in graders:
            graders[name] = get_grader(name)
        graded = graders[name].grade(row["response"], reference)

        updated = dict(row)
        before = round(float(row.get("score", 0.0)), 4)
        after = round(graded.score, 4)
        updated["score"] = after
        updated["sub_scores"] = {k: round(v, 4) for k, v in graded.sub_scores.items()}
        updated["parse_failed"] = graded.parse_failed
        rescored_rows.append(updated)
        # Compare at stored precision: the saved score is already rounded, so an
        # exact comparison would report every row as changed by ~1e-5.
        if before != after:
            changes.append((example_id, before, after))

    result = dict(payload)
    result["results"] = rescored_rows

    # Recomputed with the same helper run_evaluation uses, so the shape matches.
    scores = [r["score"] for r in rescored_rows]
    metrics: dict[str, Any] = {"overall": summarize("overall", scores).to_dict()}
    sub_names = {k for r in rescored_rows for k in r.get("sub_scores", {})}
    for name in sorted(sub_names):
        values = [r["sub_scores"][name] for r in rescored_rows if name in r.get("sub_scores", {})]
        metrics[name] = summarize(name, values).to_dict()
    result["metrics"] = metrics

    per_task: dict[str, Any] = {}
    for task in sorted({r.get("task", "") for r in rescored_rows}):
        scores = [r["score"] for r in rescored_rows if r.get("task") == task]
        per_task[task] = summarize(task, scores).to_dict()
    result["per_task"] = per_task

    result["rescored"] = {
        "at": datetime.now(UTC).isoformat(),
        "by": "scripts/rescore.py",
        "graders": sorted(graders),
        "examples_changed": len(changes),
        "note": (
            "Re-graded from stored responses. No inference was re-run and the "
            "source artifact was not modified. Compare only against another arm "
            "re-scored with the same grader version."
        ),
    }
    return result, changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", type=Path, required=True, help="Stored evaluation JSON.")
    parser.add_argument("--benchmark", type=Path, required=True, help="Benchmark JSONL.")
    parser.add_argument(
        "--output", type=Path, required=True, help="Where to write the re-scored copy."
    )
    parser.add_argument(
        "--grader", default="kleos_policy", help="Grader for rows that do not name one."
    )
    parser.add_argument(
        "--show", type=int, default=10, help="How many changed examples to list (default 10)."
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("KLEOS re-score")

    if args.output.resolve() == args.results.resolve():
        print(
            "\n✗ Refusing to overwrite the source artifact. Choose a new --output.\n",
            file=sys.stderr,
        )
        return 1

    source_hash = file_sha256(args.results)
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    references = load_references(args.benchmark)

    print(f"  source     : {args.results.name}")
    print(f"  source sha : {source_hash[:16]}")
    print(f"  arm        : {payload.get('arm')}")
    print(f"  benchmark  : {args.benchmark.name} ({len(references)} reference(s))")

    before_overall = (payload.get("metrics") or {}).get("overall", {}).get("mean")
    rescored, changes = rescore(payload, references, default_grader=args.grader)
    after_overall = rescored["metrics"]["overall"]["mean"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rescored, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # The source must be byte-identical to how we found it.
    if file_sha256(args.results) != source_hash:
        print("\n✗ Source artifact changed during re-scoring. This is a bug.\n", file=sys.stderr)
        return 1

    print(f"\n  examples     : {len(rescored['results'])}")
    print(f"  changed      : {len(changes)}")
    print(f"  overall      : {before_overall} → {after_overall}")
    if changes:
        worst = sorted(changes, key=lambda c: abs(c[2] - c[1]), reverse=True)[: args.show]
        print(f"\n  largest changes (showing {len(worst)} of {len(changes)}):")
        for example_id, before, after in worst:
            print(f"    {example_id:34s} {before:.4f} → {after:.4f}  ({after - before:+.4f})")

    print(f"\n  → {args.output}")
    print(f"  source unmodified (sha {source_hash[:16]}) ✓\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
