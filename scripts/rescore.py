#!/usr/bin/env python3
"""Re-grade or annotate a stored evaluation result offline, without re-running inference.

Evaluation results retain the model's raw responses, so a stored artifact can be
re-examined after the fact. No GPU, no model, no regenerated text — the responses
are exactly the ones the model produced.

Two modes:

``--mode regrade`` (default)
    Re-grade every response with the current grader and write a copy whose
    scores, sub-scores, details, decisions and aggregates reflect it. For
    correcting a grader defect.

``--mode annotate``
    Keep every original field and value exactly as stored, and add beside them
    what the current code measures: per-record ``group_id``, ``subset`` and grader
    ``details``, the benchmark's sha256, ``generation_stats`` and the ``corrected``
    block (consistency by group, subsets, cluster intervals; findings H-F11 to
    H-F14). The stored scores must reproduce under the current grader, or the run
    stops: an annotation must not sit beside numbers it cannot account for.

    python scripts/rescore.py --mode annotate \\
        --results eval_arm2_full.json --benchmark benchmark.jsonl \\
        --output eval_arm2_full.annotated.json \\
        --gold-targets /path/to/kleos-policy-v0.0.6/test.jsonl \\
        --summary eval_arm2_full.annotated.md

    ``--gold-targets`` runs the fabricated-citation heuristic over the gold
    answers of the same items and records only counts: the heuristic's floor.

**The input file is opened read-only and never written back.** The output goes to
a new path, so the original stays frozen and its hash stays valid. Both arms of a
comparison must be processed with the same code, or the comparison is
meaningless; the script records what it did in the output so that is checkable.
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
from kleos_models.evaluation.consistency import build_consistency_groups
from kleos_models.evaluation.corrections import (
    annotate_records,
    benchmark_index,
    build_corrected,
    generation_stats,
    render_corrected,
)
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


def load_benchmark_rows(benchmark: Path) -> list[dict[str, Any]]:
    """The benchmark's rows, as stored."""
    rows: list[dict[str, Any]] = []
    with benchmark.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_references(benchmark: Path) -> dict[str, dict[str, Any]]:
    """Map example id → grader reference, read from the benchmark."""
    return {row["id"]: row.get("reference", {}) for row in load_benchmark_rows(benchmark)}


def _decision(details: dict[str, Any], response: str) -> str:
    """The decision the runner records, from grader details (runner._extract_decision)."""
    for key in ("predicted_label", "predicted_ranking", "predicted_items"):
        value = details.get(key)
        if value:
            return json.dumps(value, sort_keys=True) if isinstance(value, list) else str(value)
    return response.strip()[:200]


def _reference_decision(reference: dict[str, Any]) -> str | None:
    """The reference decision the runner records (runner._reference_decision)."""
    for key in ("label", "expected_decision", "text"):
        if key in reference:
            return str(reference[key])
    if "ranking" in reference and isinstance(reference["ranking"], list):
        return json.dumps([str(i) for i in reference["ranking"]], sort_keys=True)
    return None


def _grade_rows(
    rows: list[dict[str, Any]], references: dict[str, dict[str, Any]], *, default_grader: str
) -> list[tuple[dict[str, Any], Any]]:
    """Grade every stored response against its reference.

    Raises:
        EvaluationError: when a row has no stored response, or no reference. Both
            mean the re-score would be inventing data rather than recomputing it.
    """
    graders: dict[str, Any] = {}
    graded: list[tuple[dict[str, Any], Any]] = []
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
        graded.append((row, graders[name].grade(row["response"], reference)))
    return graded


def rescore(
    payload: dict[str, Any], references: dict[str, dict[str, Any]], *, default_grader: str
) -> tuple[dict[str, Any], list[tuple[str, float, float]]]:
    """Return a re-scored copy of ``payload`` plus the per-example score changes."""
    rows = payload.get("results") or []
    if not rows:
        raise EvaluationError("Results payload contains no examples to re-score.")

    changes: list[tuple[str, float, float]] = []
    rescored_rows: list[dict[str, Any]] = []
    graders: set[str] = set()

    for row, graded in _grade_rows(rows, references, default_grader=default_grader):
        graders.add(row.get("grader") or default_grader)
        updated = dict(row)
        before = round(float(row.get("score", 0.0)), 4)
        after = round(graded.score, 4)
        updated["score"] = after
        updated["sub_scores"] = {k: round(v, 4) for k, v in graded.sub_scores.items()}
        updated["parse_failed"] = graded.parse_failed
        updated["details"] = json.loads(json.dumps(graded.details, default=str))
        # The decision follows the grade; a stale one would skew consistency.
        updated["decision"] = _decision(graded.details, row["response"])
        rescored_rows.append(updated)
        # Compare at stored precision: the saved score is already rounded, so an
        # exact comparison would report every row as changed by ~1e-5.
        if before != after:
            changes.append((row["example_id"], before, after))

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

    # The stored consistency block was computed from the old decisions.
    stored = payload.get("consistency") or {}
    key = stored.get("group_key") or "scenario_family"
    grouped = [r for r in rescored_rows if r.get(key)]
    if stored and grouped:
        result["consistency"] = build_consistency_groups(
            example_ids=[r["example_id"] for r in grouped],
            group_ids=[str(r[key]) for r in grouped],
            decisions=[r["decision"] for r in grouped],
            scores=[r["score"] for r in grouped],
            perturbation_kinds=[r.get("perturbation_kind") or "" for r in grouped],
            reference_decisions=[r.get("reference_decision") for r in grouped],
            min_group_size=2,
            group_key=key,
        ).to_dict()

    result["rescored"] = {
        "at": datetime.now(UTC).isoformat(),
        "by": "scripts/rescore.py --mode regrade",
        "graders": sorted(graders),
        "examples_changed": len(changes),
        "note": (
            "Re-graded from stored responses. No inference was re-run and the "
            "source artifact was not modified. Compare only against another arm "
            "re-scored with the same grader version."
        ),
    }
    return result, changes


def gold_floor(payload: dict[str, Any], benchmark: Path, gold_files: list[Path]) -> dict[str, Any]:
    """Run the fabricated-citation heuristic over the gold answers: counts only.

    The gold answer is the final assistant message of the same item in the
    sealed split files. The context and provided evidence come from the
    benchmark item, exactly as for the model's responses.
    """
    from kleos_models.data.loaders import load_evaluation_examples
    from kleos_models.evaluation.faithfulness import assess_faithfulness
    from kleos_models.inference.generate import collect_evidence_ids, extract_context_text

    gold: dict[str, str] = {}
    for path in gold_files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                answers = [m for m in row.get("messages", []) if m.get("role") == "assistant"]
                if answers:
                    gold[str(row["id"])] = str(answers[-1].get("content", ""))

    examples = {e.id: e for e in load_evaluation_examples(benchmark, strict=True)}
    evaluated = sorted({str(r["example_id"]) for r in payload.get("results", [])})
    found = flagged = 0
    for example_id in evaluated:
        if example_id not in gold or example_id not in examples:
            continue
        example = examples[example_id]
        found += 1
        assessment = assess_faithfulness(
            gold[example_id],
            context_text=extract_context_text(example),
            provided_evidence_ids=collect_evidence_ids(example),
            decisive_evidence_ids=[str(e) for e in example.reference.get("evidence_ids", [])],
        )
        if assessment.fabricated_ids:
            flagged += 1
    return {
        "gold_answers_found": found,
        "gold_answers_flagged": flagged,
        "gold_flag_rate": round(flagged / found, 4) if found else None,
        "note": (
            "The same heuristic applied to the reference answers. A model scoring "
            "near this floor is doing what the gold data does, not fabricating more."
        ),
    }


def annotate(
    payload: dict[str, Any],
    benchmark: Path,
    *,
    default_grader: str,
    allow_score_drift: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an annotated copy of ``payload`` and a summary of what was checked.

    Raises:
        EvaluationError: when the benchmark is not the one the results were
            graded against, or the stored scores do not reproduce (unless
            ``allow_score_drift``).
    """
    rows = payload.get("results") or []
    if not rows:
        raise EvaluationError("Results payload contains no examples to annotate.")
    benchmark_rows = load_benchmark_rows(benchmark)
    references = {row["id"]: row.get("reference", {}) for row in benchmark_rows}
    index = benchmark_index(benchmark_rows)

    # The benchmark must be the one these results were graded against: every
    # stored target must match it (finding H-F5: a path proves nothing).
    mismatched = [
        r["example_id"]
        for r in rows
        if r["example_id"] in references
        and r.get("reference_decision") is not None
        and r.get("reference_decision") != _reference_decision(references[r["example_id"]])
    ]
    if mismatched:
        raise EvaluationError(
            f"{len(mismatched)} stored target(s) differ from this benchmark's.",
            details={"first": ", ".join(mismatched[:5])},
            suggestions=["Point --benchmark at the file these results were evaluated against."],
        )

    drift: list[tuple[str, float, float]] = []
    details_by_row: list[dict[str, Any]] = []
    for row, graded in _grade_rows(rows, references, default_grader=default_grader):
        stored, regraded = round(float(row["score"]), 4), round(graded.score, 4)
        if stored != regraded:
            drift.append((row["example_id"], stored, regraded))
        details_by_row.append(json.loads(json.dumps(graded.details, default=str)))
    if drift and not allow_score_drift:
        raise EvaluationError(
            f"{len(drift)} stored score(s) do not reproduce under the current grader.",
            details={
                "first": ", ".join(f"{i} {a}→{b}" for i, a, b in drift[:5]),
            },
            suggestions=[
                "The grader changed since this result was produced. Annotations would "
                "sit beside numbers the current code cannot account for.",
                "Use --mode regrade for a re-graded copy, or --allow-score-drift to "
                "annotate anyway with the drift recorded.",
            ],
        )

    annotated_rows = annotate_records(rows, index)
    for row, details in zip(annotated_rows, details_by_row, strict=True):
        # Only added where absent: a stored value is never replaced.
        row.setdefault("details", details)

    max_new_tokens = (payload.get("generation") or {}).get("max_new_tokens")
    result = dict(payload)
    result["results"] = annotated_rows
    result.setdefault("schema_version", 1)
    result["benchmark_sha256"] = result.get("benchmark_sha256") or file_sha256(benchmark)
    result["generation_stats"] = generation_stats(annotated_rows, max_new_tokens=max_new_tokens)
    result["corrected"] = build_corrected(annotated_rows, index=index)
    result["annotated"] = {
        "at": datetime.now(UTC).isoformat(),
        "by": "scripts/rescore.py --mode annotate",
        "benchmark": benchmark.name,
        "scores_reproduced": not drift,
        "score_drift": [{"example_id": i, "stored": a, "regraded": b} for i, a, b in drift],
        "note": (
            "Every original field and value is unchanged. Added beside them: "
            "per-record group_id, subset and details; benchmark_sha256; "
            "generation_stats; corrected."
        ),
    }
    summary = {"examples": len(rows), "score_drift": len(drift)}
    return result, summary


def render_summary(original: dict[str, Any], annotated: dict[str, Any], source: str) -> str:
    """Markdown: the original numbers beside the corrected ones."""
    corrected = annotated["corrected"]
    overall = corrected.get("overall") or {}
    lines = [
        f"# Original vs corrected: `{source}`",
        "",
        f"Arm `{original.get('arm')}`. Original fields are unchanged; corrected measures "
        "are computed beside them (docs/evaluation.md).",
        "",
        "| Measure | Original | Corrected |",
        "| --- | --- | --- |",
    ]
    original_overall = (original.get("metrics") or {}).get("overall") or {}
    lines.append(
        f"| overall mean, 95% CI | {original_overall.get('mean', '-')} "
        f"({original_overall.get('ci95_low', '-')}-{original_overall.get('ci95_high', '-')}, "
        f"examples as independent) | {overall.get('mean', '-')} "
        f"({overall.get('ci95_low', '-')}-{overall.get('ci95_high', '-')}, "
        f"{overall.get('clusters', '-')} groups resampled) |"
    )
    stored = original.get("consistency") or {}
    by_family = corrected["consistency"]["scenario_family"]
    by_group = corrected["consistency"]["group_id"]
    if stored:
        lines.append(
            f"| consistency agreement | {stored.get('agreement_rate', '-')} over "
            f"{stored.get('evaluated_groups', '-')} {stored.get('group_key', 'scenario_family')} "
            f"groups (oracle {by_family.get('oracle_agreement_rate', '-')}) | "
            f"{by_group.get('agreement_rate', '-')} over {by_group.get('evaluated_groups', '-')} "
            f"group_id groups (oracle {by_group.get('oracle_agreement_rate', '-')}) |"
        )
    for subset, block in (corrected.get("subsets") or {}).items():
        if subset == "unassigned":
            continue
        lines.append(
            f"| {subset} | (not reported) | {block.get('mean', '-')} (n={block.get('n')}, "
            f"{block.get('clusters', '-')} groups, CI {block.get('ci95_low', '-')}-"
            f"{block.get('ci95_high', '-')}) |"
        )
    faith = corrected.get("faithfulness") or {}
    floor = faith.get("gold_floor") or {}
    lines.append(
        f"| fabricated citations | {faith.get('responses_with_fabricated_citations', '-')} "
        "response(s) | UNCALIBRATED; gold floor "
        + (
            f"{floor.get('gold_answers_flagged')}/{floor.get('gold_answers_found')}"
            if floor
            else "not computed (--gold-targets)"
        )
        + " |"
    )
    if faith.get("evidence_coverage"):
        lines.append(f"| evidence coverage | as stored | {faith['evidence_coverage']} |")
    lines.extend(["", "```", render_corrected(corrected), "```", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", type=Path, required=True, help="Stored evaluation JSON.")
    parser.add_argument("--benchmark", type=Path, required=True, help="Benchmark JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the copy.")
    parser.add_argument(
        "--mode",
        choices=["regrade", "annotate"],
        default="regrade",
        help="regrade: re-score with the current grader. annotate: add corrected "
        "measures beside unchanged originals.",
    )
    parser.add_argument(
        "--grader", default="kleos_policy", help="Grader for rows that do not name one."
    )
    parser.add_argument(
        "--gold-targets",
        type=Path,
        nargs="+",
        metavar="SPLIT.jsonl",
        help="annotate: sealed split files holding the gold answers, for the citation "
        "heuristic's gold floor (counts only).",
    )
    parser.add_argument(
        "--summary", type=Path, help="annotate: write an original-vs-corrected Markdown table."
    )
    parser.add_argument(
        "--allow-score-drift",
        action="store_true",
        help="annotate: continue when stored scores do not reproduce, recording the drift.",
    )
    parser.add_argument(
        "--show", type=int, default=10, help="How many changed examples to list (default 10)."
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header(f"KLEOS re-score ({args.mode})")

    protected = {args.results.resolve(), args.benchmark.resolve()}
    protected.update(p.resolve() for p in args.gold_targets or [])
    for target in (args.output, args.summary):
        if target is not None and target.resolve() in protected:
            print(
                f"\n✗ Refusing to overwrite an input ({target}). Choose a new path.\n",
                file=sys.stderr,
            )
            return 1
    if args.mode == "regrade" and (args.gold_targets or args.summary or args.allow_score_drift):
        parser.error("--gold-targets, --summary and --allow-score-drift need --mode annotate")

    source_hash = file_sha256(args.results)
    payload = json.loads(args.results.read_text(encoding="utf-8"))

    print(f"  source     : {args.results.name}")
    print(f"  source sha : {source_hash[:16]}")
    print(f"  arm        : {payload.get('arm')}")
    print(f"  benchmark  : {args.benchmark.name}")

    if args.mode == "regrade":
        references = load_references(args.benchmark)
        before_overall = (payload.get("metrics") or {}).get("overall", {}).get("mean")
        output_payload, changes = rescore(payload, references, default_grader=args.grader)
        after_overall = output_payload["metrics"]["overall"]["mean"]
    else:
        output_payload, summary = annotate(
            payload,
            args.benchmark,
            default_grader=args.grader,
            allow_score_drift=args.allow_score_drift,
        )
        if args.gold_targets:
            output_payload["corrected"]["faithfulness"]["gold_floor"] = gold_floor(
                payload, args.benchmark, list(args.gold_targets)
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.mode == "annotate" and args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            render_summary(payload, output_payload, args.results.name), encoding="utf-8"
        )

    # The source must be byte-identical to how we found it.
    if file_sha256(args.results) != source_hash:
        print("\n✗ Source artifact changed during re-scoring. This is a bug.\n", file=sys.stderr)
        return 1

    if args.mode == "regrade":
        print(f"\n  examples     : {len(output_payload['results'])}")
        print(f"  changed      : {len(changes)}")
        print(f"  overall      : {before_overall} → {after_overall}")
        if changes:
            worst = sorted(changes, key=lambda c: abs(c[2] - c[1]), reverse=True)[: args.show]
            print(f"\n  largest changes (showing {len(worst)} of {len(changes)}):")
            for example_id, before, after in worst:
                print(f"    {example_id:34s} {before:.4f} → {after:.4f}  ({after - before:+.4f})")
    else:
        print(f"\n  examples     : {summary['examples']}")
        print(
            f"  score drift  : {summary['score_drift']} (stored scores reproduce: "
            f"{'yes' if not summary['score_drift'] else 'NO'})"
        )
        print("\n" + render_corrected(output_payload["corrected"]))
        floor = output_payload["corrected"]["faithfulness"].get("gold_floor")
        if floor:
            print(
                f"\n  citation heuristic gold floor: {floor['gold_answers_flagged']} of "
                f"{floor['gold_answers_found']} gold answers flagged"
            )
        if args.summary:
            print(f"\n  summary → {args.summary}")

    print(f"\n  → {args.output}")
    print(f"  source unmodified (sha {source_hash[:16]}) ✓\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
