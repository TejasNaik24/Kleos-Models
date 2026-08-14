"""Comparison and report generation (spec sections 53, 54, 36).

Two jobs: compare arms honestly, and write the experiment report.

Design constraints taken directly from the specification:

* **No single misleading aggregate** unless explicitly requested (section 54). The
  default output is per-task, with OOD, consistency and capability reported as
  their own columns.
* **Report where it failed**, not only where it won (section 53).
* **Do not claim superiority from a small or noisy difference** (sections 35, 36).
  Deltas come with a paired bootstrap confidence interval, and the verdict wording
  is driven by that interval rather than by the sign of the mean.

This module imports no torch.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kleos_models.evaluation.metrics import bootstrap_difference, summarize
from kleos_models.logging_utils import format_table, get_logger

logger = get_logger(__name__)

#: Below this absolute delta, a difference is treated as noise regardless of sign.
NEGLIGIBLE_DELTA = 0.01


@dataclass
class TaskComparison:
    """Base vs fine-tuned on one task."""

    task: str
    base_score: float
    finetuned_score: float
    count: int
    significance: dict[str, Any] = field(default_factory=dict)

    @property
    def absolute_delta(self) -> float:
        return self.finetuned_score - self.base_score

    @property
    def relative_delta(self) -> float:
        if self.base_score == 0:
            return 0.0
        return self.absolute_delta / self.base_score

    @property
    def verdict(self) -> str:
        """Plain-language outcome, driven by the confidence interval."""
        if abs(self.absolute_delta) < NEGLIGIBLE_DELTA:
            return "no change"
        significant = self.significance.get("significant_at_05")
        direction = "improved" if self.absolute_delta > 0 else "regressed"
        if significant is False:
            return f"{direction} (not significant)"
        if significant is True:
            return f"{direction} (p={self.significance.get('p_value')})"
        return direction

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "base": round(self.base_score, 4),
            "finetuned": round(self.finetuned_score, 4),
            "absolute_delta": round(self.absolute_delta, 4),
            "relative_delta": round(self.relative_delta, 4),
            "count": self.count,
            "verdict": self.verdict,
            "significance": self.significance,
        }


@dataclass
class ComparisonReport:
    """A full base-vs-fine-tuned comparison."""

    base_arm: str
    finetuned_arm: str
    base_model: dict[str, Any] = field(default_factory=dict)
    finetuned_model: dict[str, Any] = field(default_factory=dict)
    per_task: list[TaskComparison] = field(default_factory=list)
    overall: TaskComparison | None = None
    ood_delta: dict[str, Any] = field(default_factory=dict)
    consistency_delta: dict[str, Any] = field(default_factory=dict)
    faithfulness_delta: dict[str, Any] = field(default_factory=dict)
    capability_delta: dict[str, Any] = field(default_factory=dict)
    comparability_warnings: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def regressed_tasks(self) -> list[TaskComparison]:
        """Tasks where the fine-tuned model got worse.

        Surfaced prominently: a mixed result across tasks may be the most
        informative outcome the experiment can produce (spec section 58).
        """
        return [t for t in self.per_task if t.absolute_delta < -NEGLIGIBLE_DELTA]

    @property
    def improved_tasks(self) -> list[TaskComparison]:
        return [t for t in self.per_task if t.absolute_delta > NEGLIGIBLE_DELTA]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "base_arm": self.base_arm,
            "finetuned_arm": self.finetuned_arm,
            "base_model": self.base_model,
            "finetuned_model": self.finetuned_model,
            "overall": self.overall.to_dict() if self.overall else None,
            "per_task": [t.to_dict() for t in self.per_task],
            "ood_delta": self.ood_delta,
            "consistency_delta": self.consistency_delta,
            "faithfulness_delta": self.faithfulness_delta,
            "capability_delta": self.capability_delta,
            "comparability_warnings": self.comparability_warnings,
            "improved_task_count": len(self.improved_tasks),
            "regressed_task_count": len(self.regressed_tasks),
        }

    def render(self, *, show_aggregate: bool = False) -> str:
        """Render the comparison table (spec section 54)."""
        lines = [
            "=" * 88,
            f"Comparison: {self.base_arm} vs {self.finetuned_arm}",
            "=" * 88,
            "",
            f"  base model      : {self.base_model.get('base_model', '-')}",
            f"  fine-tuned      : {self.finetuned_model.get('base_model', '-')}",
            f"  adapter         : {self.finetuned_model.get('adapter_path', '-')}",
            "",
        ]

        if self.comparability_warnings:
            lines.append("  COMPARABILITY WARNINGS")
            lines.extend(f"    ! {w}" for w in self.comparability_warnings)
            lines.append("")

        rows = [
            {
                "task": t.task,
                "base": t.base_score,
                "finetuned": t.finetuned_score,
                "abs delta": t.absolute_delta,
                "rel delta": f"{t.relative_delta:+.1%}",
                "n": t.count,
                "verdict": t.verdict,
            }
            for t in self.per_task
        ]
        if rows:
            lines.append("Per task")
            lines.append(format_table(rows))
            lines.append("")

        if show_aggregate and self.overall:
            lines.extend(
                [
                    "Overall aggregate (requested explicitly)",
                    f"  base {self.overall.base_score:.4f} → fine-tuned "
                    f"{self.overall.finetuned_score:.4f} "
                    f"({self.overall.absolute_delta:+.4f}, {self.overall.verdict})",
                    "",
                    "  A single aggregate across tasks hides per-task trade-offs and is",
                    "  reported here only because it was explicitly requested.",
                    "",
                ]
            )

        if self.ood_delta:
            lines.extend(
                [
                    "Out-of-distribution",
                    f"  in-distribution delta : {self.ood_delta.get('in_distribution_delta', 0):+.4f}",
                    f"  OOD delta             : {self.ood_delta.get('ood_delta', 0):+.4f}",
                    f"  generalization gap Δ  : "
                    f"{self.ood_delta.get('generalization_gap_delta', 0):+.4f}",
                    f"  verdict               : {self.ood_delta.get('verdict', '-')}",
                    "",
                ]
            )

        if self.consistency_delta:
            lines.extend(
                [
                    "Consistency",
                    f"  agreement rate        : "
                    f"{self.consistency_delta.get('base_agreement', 0):.3f} → "
                    f"{self.consistency_delta.get('finetuned_agreement', 0):.3f} "
                    f"({self.consistency_delta.get('agreement_delta', 0):+.3f})",
                    f"  correct agreement     : "
                    f"{self.consistency_delta.get('base_correct_agreement', 0):.3f} → "
                    f"{self.consistency_delta.get('finetuned_correct_agreement', 0):.3f} "
                    f"({self.consistency_delta.get('correct_agreement_delta', 0):+.3f})",
                    "",
                ]
            )

        if self.capability_delta:
            lines.extend(
                [
                    "General capability",
                    f"  base {self.capability_delta.get('base_score', 0):.4f} → "
                    f"fine-tuned {self.capability_delta.get('fine_tuned_score', 0):.4f} "
                    f"({self.capability_delta.get('delta', 0):+.4f}, "
                    f"{self.capability_delta.get('relative_delta', 0):+.2%})",
                    "",
                ]
            )

        lines.extend(["Where it failed", ""])
        if self.regressed_tasks:
            for task in self.regressed_tasks:
                lines.append(
                    f"  - {task.task}: {task.base_score:.4f} → {task.finetuned_score:.4f} "
                    f"({task.absolute_delta:+.4f})"
                )
        else:
            lines.append("  No task regressed beyond the noise threshold.")

        lines.extend(["", self._conclusion(), ""])
        return "\n".join(lines)

    def _conclusion(self) -> str:
        """State the outcome without overclaiming (spec sections 35, 36)."""
        improved, regressed = len(self.improved_tasks), len(self.regressed_tasks)

        if improved and regressed:
            return (
                "Conclusion: MIXED. Fine-tuning improved "
                f"{improved} task(s) and regressed {regressed}. Per spec section 58 this "
                "may be the most interesting outcome available — report both halves and "
                "investigate what distinguishes them."
            )
        if regressed and not improved:
            return (
                f"Conclusion: fine-tuning REGRESSED {regressed} task(s) and improved none. "
                "This is a valid, publishable result. Analyse data quality, policy "
                "learnability, capacity and evaluation design before discarding it."
            )
        if improved and not regressed:
            significant = [
                t for t in self.improved_tasks if t.significance.get("significant_at_05")
            ]
            if not significant:
                return (
                    f"Conclusion: fine-tuning improved {improved} task(s), but no "
                    "improvement is statistically significant at p<0.05. Treat this as "
                    "suggestive, not demonstrated: gather more evaluation examples or "
                    "more seeds before claiming a win."
                )
            return (
                f"Conclusion: fine-tuning improved {improved} task(s), "
                f"{len(significant)} significant at p<0.05. Check the OOD and capability "
                "rows before describing this as a general improvement."
            )
        return (
            "Conclusion: no meaningful difference between the arms. Fine-tuning neither "
            "helped nor harmed on this benchmark. That is a valid result."
        )

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return target


def _per_example_scores(
    results: Sequence[dict[str, Any]], *, task: str | None = None
) -> dict[str, float]:
    """Map example id → score, optionally restricted to one task."""
    return {
        record["example_id"]: float(record["score"])
        for record in results
        if task is None or record.get("task") == task
    }


def compare_results(
    base: dict[str, Any],
    finetuned: dict[str, Any],
    *,
    compute_significance: bool = True,
) -> ComparisonReport:
    """Compare two saved evaluation-result payloads (spec section 54).

    Scores are paired by ``example_id`` so the bootstrap is a genuine paired test
    rather than a comparison of two unrelated samples.
    """
    report = ComparisonReport(
        base_arm=base.get("arm", "base"),
        finetuned_arm=finetuned.get("arm", "finetuned"),
        base_model=base.get("model", {}),
        finetuned_model=finetuned.get("model", {}),
    )

    base_results = base.get("results", [])
    finetuned_results = finetuned.get("results", [])

    # --- comparability checks ----------------------------------------------
    if base.get("benchmark_path") != finetuned.get("benchmark_path"):
        report.comparability_warnings.append(
            f"Different benchmarks: {base.get('benchmark_path')} vs "
            f"{finetuned.get('benchmark_path')}. The scores are not comparable."
        )
    if base.get("generation") != finetuned.get("generation"):
        report.comparability_warnings.append(
            "Decoding settings differ between the arms; part of any difference may be "
            "due to temperature or max_new_tokens rather than the model."
        )
    base_model_id = base.get("model", {}).get("base_model")
    ft_model_id = finetuned.get("model", {}).get("base_model")
    if base_model_id != ft_model_id:
        report.comparability_warnings.append(
            f"Different base models ({base_model_id} vs {ft_model_id}). This is a "
            "cross-model comparison, not a fine-tuning effect."
        )
    if finetuned.get("model", {}).get("adapter_path") is None:
        report.comparability_warnings.append(
            "The fine-tuned arm reports no adapter path — it may be the base model evaluated twice."
        )

    base_ids = {r["example_id"] for r in base_results}
    ft_ids = {r["example_id"] for r in finetuned_results}
    shared = base_ids & ft_ids
    if len(shared) < min(len(base_ids), len(ft_ids)):
        report.comparability_warnings.append(
            f"Only {len(shared)} example(s) are shared between the arms "
            f"({len(base_ids)} vs {len(ft_ids)} evaluated). Comparison uses the "
            "intersection."
        )
    if not shared:
        report.comparability_warnings.append(
            "The arms share no example ids; no paired comparison is possible."
        )
        return report

    # --- per task -----------------------------------------------------------
    tasks = sorted(
        {r.get("task", "unknown") for r in base_results}
        | {r.get("task", "unknown") for r in finetuned_results}
    )
    for task in tasks:
        base_scores = _per_example_scores(base_results, task=task)
        ft_scores = _per_example_scores(finetuned_results, task=task)
        paired_ids = sorted(set(base_scores) & set(ft_scores))
        if not paired_ids:
            continue
        left = [base_scores[i] for i in paired_ids]
        right = [ft_scores[i] for i in paired_ids]
        report.per_task.append(
            TaskComparison(
                task=task,
                base_score=summarize(task, left).mean,
                finetuned_score=summarize(task, right).mean,
                count=len(paired_ids),
                significance=(bootstrap_difference(left, right) if compute_significance else {}),
            )
        )

    # --- overall -------------------------------------------------------------
    base_all = _per_example_scores(base_results)
    ft_all = _per_example_scores(finetuned_results)
    paired_ids = sorted(set(base_all) & set(ft_all))
    if paired_ids:
        left = [base_all[i] for i in paired_ids]
        right = [ft_all[i] for i in paired_ids]
        report.overall = TaskComparison(
            task="overall",
            base_score=summarize("overall", left).mean,
            finetuned_score=summarize("overall", right).mean,
            count=len(paired_ids),
            significance=bootstrap_difference(left, right) if compute_significance else {},
        )

    # --- OOD -----------------------------------------------------------------
    base_ood, ft_ood = base.get("ood"), finetuned.get("ood")
    if base_ood and ft_ood and base_ood.get("measurable") and ft_ood.get("measurable"):
        in_delta = ft_ood["in_distribution_score"] - base_ood["in_distribution_score"]
        ood_delta = ft_ood["ood_score"] - base_ood["ood_score"]
        verdict = "no meaningful change"
        if in_delta > NEGLIGIBLE_DELTA and ood_delta < -NEGLIGIBLE_DELTA:
            verdict = (
                "improved in-distribution but degraded OOD — consistent with fitting "
                "surface features rather than learning a transferable policy"
            )
        elif in_delta > NEGLIGIBLE_DELTA and ood_delta > NEGLIGIBLE_DELTA:
            verdict = "improved both in-distribution and out-of-distribution"
        elif ood_delta < -NEGLIGIBLE_DELTA:
            verdict = "degraded out-of-distribution"
        report.ood_delta = {
            "in_distribution_delta": round(in_delta, 4),
            "ood_delta": round(ood_delta, 4),
            "generalization_gap_delta": round(
                ft_ood["generalization_gap"] - base_ood["generalization_gap"], 4
            ),
            "verdict": verdict,
        }

    # --- consistency ---------------------------------------------------------
    base_consistency, ft_consistency = base.get("consistency"), finetuned.get("consistency")
    if base_consistency and ft_consistency:
        report.consistency_delta = {
            "base_agreement": base_consistency.get("agreement_rate", 0.0),
            "finetuned_agreement": ft_consistency.get("agreement_rate", 0.0),
            "agreement_delta": round(
                ft_consistency.get("agreement_rate", 0.0)
                - base_consistency.get("agreement_rate", 0.0),
                4,
            ),
            "base_correct_agreement": base_consistency.get("correct_agreement_rate", 0.0),
            "finetuned_correct_agreement": ft_consistency.get("correct_agreement_rate", 0.0),
            "correct_agreement_delta": round(
                ft_consistency.get("correct_agreement_rate", 0.0)
                - base_consistency.get("correct_agreement_rate", 0.0),
                4,
            ),
        }

    # --- faithfulness --------------------------------------------------------
    base_faith, ft_faith = base.get("faithfulness"), finetuned.get("faithfulness")
    if base_faith and ft_faith and base_faith.get("count") and ft_faith.get("count"):
        report.faithfulness_delta = {
            "base_score": base_faith.get("mean_score", 0.0),
            "finetuned_score": ft_faith.get("mean_score", 0.0),
            "delta": round(ft_faith.get("mean_score", 0.0) - base_faith.get("mean_score", 0.0), 4),
            "unsupported_claim_rate_delta": round(
                ft_faith.get("mean_unsupported_claim_rate", 0.0)
                - base_faith.get("mean_unsupported_claim_rate", 0.0),
                4,
            ),
        }

    # --- capability ----------------------------------------------------------
    capability = finetuned.get("capability")
    if capability:
        report.capability_delta = capability

    for warning in report.comparability_warnings:
        logger.warning("Comparability: %s", warning)
    return report


def render_markdown_report(
    comparison: ComparisonReport,
    *,
    experiment_id: str = "unknown",
    dataset_version: str = "unknown",
    config_summary: dict[str, Any] | None = None,
) -> str:
    """Render the experiment report (spec section 53).

    Answers the questions the specification requires: what changed, what dataset,
    what model, what configuration, what metrics, did it improve, where did it
    fail, what changed OOD, what changed in general capability.
    """
    lines = [
        f"# Experiment report — {experiment_id}",
        "",
        f"Generated: {comparison.generated_at}",
        "",
        "## What was compared",
        "",
        f"- base arm: `{comparison.base_arm}`",
        f"- fine-tuned arm: `{comparison.finetuned_arm}`",
        f"- base model: `{comparison.base_model.get('base_model', '-')}`",
        f"- model family: `{comparison.base_model.get('family', '-')}`",
        f"- adapter: `{comparison.finetuned_model.get('adapter_path', '-')}`",
        f"- dataset version: `{dataset_version}`",
        "",
    ]

    if config_summary:
        lines.extend(["## Configuration", "", "| Setting | Value |", "| --- | --- |"])
        lines.extend(f"| `{key}` | {value} |" for key, value in sorted(config_summary.items()))
        lines.append("")

    if comparison.comparability_warnings:
        lines.extend(["## Comparability warnings", ""])
        lines.extend(f"- {w}" for w in comparison.comparability_warnings)
        lines.append("")

    lines.extend(
        [
            "## Did it improve?",
            "",
            "| Task | Base | Fine-tuned | Abs Δ | Rel Δ | n | Verdict |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for task in comparison.per_task:
        lines.append(
            f"| `{task.task}` | {task.base_score:.4f} | {task.finetuned_score:.4f} | "
            f"{task.absolute_delta:+.4f} | {task.relative_delta:+.1%} | {task.count} | "
            f"{task.verdict} |"
        )
    lines.append("")

    if comparison.ood_delta:
        lines.extend(
            [
                "## What changed out of distribution?",
                "",
                f"- in-distribution delta: **{comparison.ood_delta['in_distribution_delta']:+.4f}**",
                f"- OOD delta: **{comparison.ood_delta['ood_delta']:+.4f}**",
                f"- generalization gap delta: "
                f"**{comparison.ood_delta['generalization_gap_delta']:+.4f}**",
                "",
                f"> {comparison.ood_delta['verdict']}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## What changed out of distribution?",
                "",
                "_Not measured._ No OOD-tagged examples were present in this benchmark.",
                "",
                "**No generalization claim can be made from this run** (spec section 36).",
                "",
            ]
        )

    if comparison.consistency_delta:
        delta = comparison.consistency_delta
        lines.extend(
            [
                "## Consistency",
                "",
                f"- agreement rate: {delta['base_agreement']:.3f} → "
                f"{delta['finetuned_agreement']:.3f} ({delta['agreement_delta']:+.3f})",
                f"- correct agreement: {delta['base_correct_agreement']:.3f} → "
                f"{delta['finetuned_correct_agreement']:.3f} "
                f"({delta['correct_agreement_delta']:+.3f})",
                "",
            ]
        )

    if comparison.capability_delta:
        capability = comparison.capability_delta
        lines.extend(
            [
                "## What changed in general capability?",
                "",
                f"- base: {capability.get('base_score', 0):.4f}",
                f"- fine-tuned: {capability.get('fine_tuned_score', 0):.4f}",
                f"- delta: **{capability.get('delta', 0):+.4f}** "
                f"({capability.get('relative_delta', 0):+.2%})",
                "",
            ]
        )
        if capability.get("regressed"):
            lines.extend(
                [
                    "> **General capability regressed.** Any task-specific gain above is a",
                    "> trade-off, not a free improvement.",
                    "",
                ]
            )

    lines.extend(["## Where did it fail?", ""])
    if comparison.regressed_tasks:
        for task in comparison.regressed_tasks:
            lines.append(
                f"- `{task.task}`: {task.base_score:.4f} → {task.finetuned_score:.4f} "
                f"({task.absolute_delta:+.4f})"
            )
    else:
        lines.append("- No task regressed beyond the noise threshold.")

    lines.extend(["", "## Conclusion", "", comparison._conclusion(), ""])
    lines.extend(
        [
            "---",
            "",
            "_Generated by `scripts/compare.py`. Deltas are paired per example and",
            "accompanied by bootstrap confidence intervals. A difference that is not",
            "significant is reported as such rather than as a win._",
            "",
        ]
    )
    return "\n".join(lines)


def write_experiment_report(
    comparison: ComparisonReport,
    output_dir: Path | str,
    *,
    experiment_id: str = "unknown",
    dataset_version: str = "unknown",
    config_summary: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write ``summary.md`` and ``metrics.json`` for a comparison."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    summary_path = target / "summary.md"
    metrics_path = target / "metrics.json"

    summary_path.write_text(
        render_markdown_report(
            comparison,
            experiment_id=experiment_id,
            dataset_version=dataset_version,
            config_summary=config_summary,
        ),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(comparison.to_dict(), indent=2, default=str), encoding="utf-8"
    )

    logger.info("Wrote experiment report to %s", target)
    return {"summary": summary_path, "metrics": metrics_path}
