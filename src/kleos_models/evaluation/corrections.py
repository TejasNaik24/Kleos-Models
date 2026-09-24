"""Corrected measures, reported beside the originals (findings H-F11 to H-F14).

The audit of the Hermes evaluation found four ways the original numbers mislead
on the KLEOS benchmark. None of them is fixed by changing an original field: every
v1 key keeps its definition and its value, so published numbers stay reproducible.
The corrections are computed *beside* them, from per-example records, by this one
code path for fresh runs (``runner.py``) and for results stored before it existed
(``scripts/rescore.py --mode annotate``).

H-F11, the consistency unit
    Consistency grouped by ``scenario_family`` mixes cases whose correct answers
    differ, so even a perfect model "flips" (on v0.0.6 an oracle scores 9 of 15
    families). ``group_id`` is the unit of logical equivalence: perturbations of
    one case, label-identical by construction. Both are reported, each with the
    score an oracle would get, so a metric's ceiling is visible next to it.
H-F12, the citation heuristic
    The fabricated-citation check flags gold answers too. It is reported with its
    gold floor (``rescore.py --gold-targets``) and marked uncalibrated.
H-F13, evidence coverage
    No benchmark item carries ``reference.evidence_ids``, so every response scores
    1.0: the metric is vacuous here and is labelled so.
H-F14, clustering
    349 examples come in 78 groups. Intervals that resample examples treat them
    as independent and come out too narrow; the corrected intervals resample
    groups (``metrics.cluster_bootstrap_mean``).

Subsets: a benchmark item whose reference says ``confident: false`` should be
declined. On v0.0.6 those 78 cases carry the four decline labels that never occur
in training, so the two subsets measure different things and are reported apart.

This module imports no torch.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from kleos_models.evaluation.consistency import ConsistencyReport, build_consistency_groups
from kleos_models.evaluation.metrics import cluster_bootstrap_mean

#: Layout version of the ``corrected`` block.
CORRECTIONS_VERSION = 1

ANSWERABLE = "answerable"
SHOULD_DECLINE = "should_decline"
SUBSETS = (ANSWERABLE, SHOULD_DECLINE)

#: The cluster unit for corrected intervals and the corrected consistency.
CLUSTER_KEY = "group_id"


def subset_of(reference: Mapping[str, Any]) -> str | None:
    """The subset a benchmark item belongs to, read from its reference."""
    confident = reference.get("confident")
    if confident is True:
        return ANSWERABLE
    if confident is False:
        return SHOULD_DECLINE
    return None


def benchmark_index(examples: Iterable[Any]) -> dict[str, dict[str, Any]]:
    """Map example id to its grouping facts.

    Accepts ``EvaluationExample`` objects or raw benchmark rows (dicts), so a
    stored result can be annotated from the JSONL file without validation
    dependencies.
    """
    index: dict[str, dict[str, Any]] = {}
    for example in examples:
        if isinstance(example, Mapping):
            example_id = str(example["id"])
            metadata = example.get("metadata") or {}
            reference = example.get("reference") or {}
            task = example.get("task")
        else:
            example_id = example.id
            metadata = example.metadata.model_dump()
            reference = example.reference
            task = example.task
        index[example_id] = {
            "group_id": metadata.get("group_id"),
            "scenario_family": metadata.get("scenario_family"),
            "subset": subset_of(reference),
            "task": task,
            "has_evidence_ids": bool(reference.get("evidence_ids")),
        }
    return index


def annotate_records(
    records: Sequence[Mapping[str, Any]], index: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Copies of ``records`` with ``group_id`` and ``subset`` joined from the benchmark.

    Raises:
        ValueError: when a record is missing from the benchmark, or already
            carries a value the benchmark contradicts. Either means the results
            and the benchmark are not the same evaluation.
    """
    annotated: list[dict[str, Any]] = []
    for record in records:
        example_id = str(record["example_id"])
        facts = index.get(example_id)
        if facts is None:
            raise ValueError(f"example {example_id} is not in the benchmark")
        if record.get("task") not in (None, facts["task"]):
            raise ValueError(
                f"example {example_id}: task {record.get('task')!r} in the results, "
                f"{facts['task']!r} in the benchmark"
            )
        row = dict(record)
        for key in ("group_id", "subset"):
            present = record.get(key)
            if present is not None and present != facts[key]:
                raise ValueError(
                    f"example {example_id}: {key} {present!r} in the results, "
                    f"{facts[key]!r} in the benchmark"
                )
            row[key] = facts[key]
        annotated.append(row)
    return annotated


def targets_fingerprint(records: Iterable[Mapping[str, Any]]) -> str:
    """sha256 over the evaluated targets: sorted (example id, task, reference decision).

    Identifies *what was graded* from a results file alone, so two results can be
    checked for the same benchmark even when neither recorded the file's hash
    (finding H-F5). Duplicate rows from several seeds collapse to one.
    """
    rows = sorted(
        {
            (
                str(r["example_id"]),
                str(r.get("task")),
                str(r.get("reference_decision")),
            )
            for r in records
        }
    )
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def consistency_summary(
    records: Sequence[Mapping[str, Any]],
    key: str,
    *,
    subset: str | None = None,
    min_group_size: int = 2,
) -> dict[str, Any]:
    """Consistency grouped by ``key``, with the score an oracle would get.

    The oracle answers every item with its reference decision. Under a sound
    grouping it agrees within every group (``oracle_agreement_rate`` 1.0); below
    that, the grouping itself mixes different correct answers and the model's
    agreement rate cannot be read as stability.
    """
    rows = [r for r in records if r.get(key) and (subset is None or r.get("subset") == subset)]
    summary: dict[str, Any] = {"key": key, "subset": subset or "all", "examples": len(rows)}
    if not rows:
        summary.update(evaluated_groups=0, reason=f"no record carries {key}")
        return summary

    def grouped(decisions: list[str]) -> ConsistencyReport:
        return build_consistency_groups(
            example_ids=[str(r["example_id"]) for r in rows],
            group_ids=[str(r[key]) for r in rows],
            decisions=decisions,
            perturbation_kinds=[r.get("perturbation_kind") or "" for r in rows],
            reference_decisions=[r.get("reference_decision") for r in rows],
            min_group_size=min_group_size,
            group_key=key,
        )

    model = grouped([r.get("decision") or "" for r in rows])
    oracle = grouped([r.get("reference_decision") or "" for r in rows])
    summary.update(
        evaluated_groups=model.evaluated_groups,
        agreement_rate=round(model.agreement_rate, 4),
        correct_agreement_rate=round(model.correct_agreement_rate, 4),
        mean_majority_share=round(model.mean_majority_share, 4),
        inconsistent_groups=len(model.inconsistent_groups),
        flips=sum(model.flips_by_perturbation().values()),
        oracle_agreement_rate=round(oracle.agreement_rate, 4),
        oracle_flips=sum(oracle.flips_by_perturbation().values()),
    )
    return summary


def _score_block(
    rows: Sequence[Mapping[str, Any]], *, iterations: int, seed: int
) -> dict[str, Any]:
    """Mean score with a cluster interval, plus sub-score means."""
    missing = sum(1 for r in rows if not r.get(CLUSTER_KEY))
    if missing:
        return {
            "n": len(rows),
            "mean": round(sum(float(r["score"]) for r in rows) / len(rows), 4),
            "estimable": False,
            "reason": f"{missing} record(s) carry no {CLUSTER_KEY}",
        }
    block = cluster_bootstrap_mean(
        [float(r["score"]) for r in rows],
        [str(r[CLUSTER_KEY]) for r in rows],
        iterations=iterations,
        seed=seed,
    )
    names = sorted({name for r in rows for name in (r.get("sub_scores") or {})})
    block["sub_scores"] = {
        name: round(
            sum(float(r["sub_scores"][name]) for r in rows if name in (r.get("sub_scores") or {}))
            / sum(1 for r in rows if name in (r.get("sub_scores") or {})),
            4,
        )
        for name in names
    }
    return block


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile; 0 for an empty sequence."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(fraction * len(ordered) + 0.5)))
    return float(ordered[rank - 1])


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": round(sum(values) / len(values), 3),
        "p50": round(_percentile(values, 0.50), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "max": round(max(values), 3),
    }


def generation_stats(
    records: Sequence[Mapping[str, Any]], *, max_new_tokens: int | None = None
) -> dict[str, Any]:
    """What generation cost and where it hit limits.

    ``hit_max_new_tokens`` counts responses that used the whole budget: those
    were probably cut off, and a truncated answer is graded as if complete.
    """
    completion = [int(r.get("completion_tokens") or 0) for r in records]
    latency = [float(r.get("latency_seconds") or 0.0) for r in records]
    stats: dict[str, Any] = {
        "responses": len(records),
        "completion_tokens": _distribution(completion),
        "prompt_tokens": _distribution([int(r.get("prompt_tokens") or 0) for r in records]),
        "latency_seconds": {**_distribution(latency), "total": round(sum(latency), 1)},
        "max_new_tokens": max_new_tokens,
        "hit_max_new_tokens": (
            sum(1 for c in completion if c >= max_new_tokens) if max_new_tokens else None
        ),
        "parse_failures": sum(1 for r in records if r.get("parse_failed")),
    }
    if all("response" in r for r in records):
        stats["empty_responses"] = sum(1 for r in records if not str(r["response"]).strip())
    return stats


def faithfulness_caveats(
    records: Sequence[Mapping[str, Any]],
    *,
    index: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """The state of the two faithfulness measures the audit found unreliable."""
    assessed = [r for r in records if r.get("faithfulness")]
    caveats: dict[str, Any] = {
        "responses_assessed": len(assessed),
        "responses_with_fabricated_citations": sum(
            1 for r in assessed if (r["faithfulness"] or {}).get("fabricated_ids")
        ),
        "citation_heuristic": (
            "UNCALIBRATED: it flags gold answers as well (H-F12). Read it against its "
            "gold floor, computed by scripts/rescore.py --gold-targets."
        ),
    }
    if index is not None:
        with_ids = sum(1 for facts in index.values() if facts.get("has_evidence_ids"))
        caveats["items_with_evidence_ids"] = with_ids
        caveats["evidence_coverage"] = (
            "vacuous: no benchmark item carries reference.evidence_ids, so every "
            "response scores 1.0 (H-F13)"
            if with_ids == 0
            else "measured"
        )
    return caveats


def build_corrected(
    records: Sequence[Mapping[str, Any]],
    *,
    index: Mapping[str, Mapping[str, Any]] | None = None,
    min_group_size: int = 2,
    iterations: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """The ``corrected`` block: every correction, from per-example records.

    Records need ``group_id`` and ``subset`` (fresh runs carry them; stored ones
    get them from :func:`annotate_records`). Without them the affected parts say
    so rather than fall back silently.
    """
    corrected: dict[str, Any] = {
        "version": CORRECTIONS_VERSION,
        "cluster_key": CLUSTER_KEY,
        "consistency": {
            "group_id": consistency_summary(records, "group_id", min_group_size=min_group_size),
            "group_id_answerable": consistency_summary(
                records, "group_id", subset=ANSWERABLE, min_group_size=min_group_size
            ),
            "scenario_family": consistency_summary(
                records, "scenario_family", min_group_size=min_group_size
            ),
        },
        "overall": _score_block(records, iterations=iterations, seed=seed) if records else {},
        "per_task": {},
        "subsets": {},
        "faithfulness": faithfulness_caveats(records, index=index),
    }
    for task in sorted({str(r.get("task")) for r in records}):
        rows = [r for r in records if str(r.get("task")) == task]
        corrected["per_task"][task] = _score_block(rows, iterations=iterations, seed=seed)
    for subset in SUBSETS:
        rows = [r for r in records if r.get("subset") == subset]
        if rows:
            block = _score_block(rows, iterations=iterations, seed=seed)
            block["per_task"] = {
                task: {
                    "n": sum(1 for r in rows if r.get("task") == task),
                    "mean": round(
                        sum(float(r["score"]) for r in rows if r.get("task") == task)
                        / sum(1 for r in rows if r.get("task") == task),
                        4,
                    ),
                }
                for task in sorted({str(r.get("task")) for r in rows})
            }
            corrected["subsets"][subset] = block
    unassigned = sum(1 for r in records if r.get("subset") is None)
    if unassigned:
        corrected["subsets"]["unassigned"] = {"n": unassigned}
    return corrected


def render_corrected(corrected: Mapping[str, Any]) -> str:
    """Human-readable summary of a ``corrected`` block."""
    lines = ["Corrected measures (beside the originals; docs/evaluation.md)"]
    overall = corrected.get("overall") or {}
    if overall:
        lines.append(f"  overall                 : {_render_block(overall)}")
    for subset, block in (corrected.get("subsets") or {}).items():
        if subset == "unassigned":
            lines.append(f"  {subset:<24}: n={block['n']}")
            continue
        lines.append(f"  {subset:<24}: {_render_block(block)}")
    lines.append("  consistency (model | oracle agreement):")
    for name, block in (corrected.get("consistency") or {}).items():
        if not block.get("evaluated_groups"):
            lines.append(f"    {name:<22} not measured ({block.get('reason', 'no groups')})")
            continue
        lines.append(
            f"    {name:<22} {block['agreement_rate']:.3f} | {block['oracle_agreement_rate']:.3f}"
            f"  over {block['evaluated_groups']} group(s), {block['flips']} flip(s)"
            f" (oracle {block['oracle_flips']})"
        )
    faith = corrected.get("faithfulness") or {}
    if faith.get("evidence_coverage"):
        lines.append(f"  evidence coverage       : {faith['evidence_coverage']}")
    if faith:
        lines.append(
            f"  fabricated citations    : {faith.get('responses_with_fabricated_citations', 0)} "
            "response(s), UNCALIBRATED (see its gold floor)"
        )
    return "\n".join(lines)


def _render_block(block: Mapping[str, Any]) -> str:
    text = f"{block.get('mean', 0.0):.4f} (n={block.get('n', 0)}"
    if block.get("clusters") is not None:
        text += f", {block['clusters']} groups"
    text += ")"
    if block.get("estimable"):
        text += f"  cluster 95% CI {block['ci95_low']:.4f}-{block['ci95_high']:.4f}"
    elif block.get("reason"):
        text += f"  CI not estimable: {block['reason']}"
    return text
