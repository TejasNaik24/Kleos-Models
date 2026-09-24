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

Beside the original example-level bootstrap, every delta also gets a *cluster*
interval that resamples whole groups (``group_id``), because perturbations of one
case are not independent (finding H-F14), and the answerable and should-decline
subsets are compared separately. Benchmark identity is checked by content, never
by path (finding H-F5). ``cross_model=True`` compares two models on one arm (for
example Hermes arm2 against Logos arm2), where differing base models are the point
rather than a warning.

This module imports no torch.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kleos_models.evaluation.corrections import SUBSETS
from kleos_models.evaluation.metrics import (
    bootstrap_difference,
    paired_cluster_bootstrap_difference,
    render_p_value,
    summarize,
)
from kleos_models.logging_utils import format_table, get_logger

logger = get_logger(__name__)

#: Below this absolute delta, a difference is treated as noise regardless of sign.
NEGLIGIBLE_DELTA = 0.01

#: Why a cluster interval is missing when the records carry no group ids.
_NO_GROUPS = (
    "records carry no group_id; annotate stored results with scripts/rescore.py --mode annotate"
)


def _verdict(delta: float, significance: dict[str, Any]) -> str:
    """Plain-language outcome of one delta, driven by its interval."""
    if abs(delta) < NEGLIGIBLE_DELTA:
        return "no change"
    significant = significance.get("significant_at_05")
    direction = "improved" if delta > 0 else "regressed"
    if significant is False:
        return f"{direction} (not significant)"
    if significant is True:
        p_text = render_p_value(significance.get("p_value"), significance.get("iterations"))
        return f"{direction} ({p_text})"
    return direction


@dataclass
class TaskComparison:
    """Base vs fine-tuned on one task."""

    task: str
    base_score: float
    finetuned_score: float
    count: int
    significance: dict[str, Any] = field(default_factory=dict)
    #: Paired bootstrap resampling groups rather than examples (H-F14).
    cluster_significance: dict[str, Any] = field(default_factory=dict)

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
        return _verdict(self.absolute_delta, self.significance)

    @property
    def cluster_verdict(self) -> str:
        """The same, driven by the cluster interval; ``n/a`` without group ids."""
        if not self.cluster_significance.get("estimable"):
            return "n/a"
        return _verdict(self.absolute_delta, self.cluster_significance)

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
            "cluster_verdict": self.cluster_verdict,
            "cluster_significance": self.cluster_significance,
        }


@dataclass
class ComparisonReport:
    """A full base-vs-fine-tuned comparison.

    In cross-model mode "base" and "fine-tuned" read as "left" and "right": the
    two models being compared, on the same arm.
    """

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
    mode: str = "fine_tuning"
    #: How the two results were established to use the same benchmark.
    benchmark_identity: dict[str, Any] = field(default_factory=dict)
    #: Set when the results must not be compared at all.
    incomparable: bool = False
    subsets: list[TaskComparison] = field(default_factory=list)
    #: The pre-registered primary comparison, when one was requested.
    primary: dict[str, Any] = field(default_factory=dict)
    corrected_consistency: dict[str, Any] = field(default_factory=dict)

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
            "mode": self.mode,
            "incomparable": self.incomparable,
            "benchmark_identity": self.benchmark_identity,
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
            "subsets": [t.to_dict() for t in self.subsets],
            "primary": self.primary,
            "corrected_consistency": self.corrected_consistency,
        }

    def render(self, *, show_aggregate: bool = False) -> str:
        """Render the comparison table (spec section 54)."""
        left, right = self.labels
        lines = [
            "=" * 88,
            f"Comparison: {self.base_arm} vs {self.finetuned_arm}"
            + ("  (cross-model)" if self.mode == "cross_model" else ""),
            "=" * 88,
            "",
            f"  {left:<15} : {self.base_model.get('base_model', '-')}",
            f"  {right:<15} : {self.finetuned_model.get('base_model', '-')}",
            f"  adapter         : {self.finetuned_model.get('adapter_path', '-')}",
            f"  benchmark       : {self.benchmark_identity.get('detail', 'not checked')}",
            "",
        ]
        if self.incomparable:
            lines.append("  NOT COMPARABLE — see the warnings below. No delta is reported.")
            lines.append("")

        if self.comparability_warnings:
            lines.append("  COMPARABILITY WARNINGS")
            lines.extend(f"    ! {w}" for w in self.comparability_warnings)
            lines.append("")

        rows = [
            {
                "task": t.task,
                left: t.base_score,
                right: t.finetuned_score,
                "abs delta": t.absolute_delta,
                "rel delta": f"{t.relative_delta:+.1%}",
                "n": t.count,
                "verdict": t.verdict,
                "cluster verdict": t.cluster_verdict,
            }
            for t in self.per_task
        ]
        if rows:
            lines.append(
                "Per task (verdict: examples resampled; cluster verdict: groups resampled)"
            )
            lines.append(format_table(rows))
            lines.append("")

        if self.subsets:
            lines.append("By subset (paired, groups resampled)")
            lines.append(
                format_table(
                    [
                        {
                            "subset": t.task,
                            left: t.base_score,
                            right: t.finetuned_score,
                            "abs delta": t.absolute_delta,
                            "n": t.count,
                            "groups": t.cluster_significance.get("clusters", "-"),
                            "cluster 95% CI": _render_ci(t.cluster_significance),
                            "cluster verdict": t.cluster_verdict,
                        }
                        for t in self.subsets
                    ]
                )
            )
            lines.append("")

        if self.primary:
            lines.extend(_render_primary(self.primary, left, right))

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

        if self.corrected_consistency:
            lines.append("Consistency by group_id (corrected unit; oracle ceiling in brackets)")
            for name, pair in self.corrected_consistency.items():
                lines.append(
                    f"  {name:<22} {pair['left']:.3f} → {pair['right']:.3f} "
                    f"({pair['right'] - pair['left']:+.3f})  [oracle {pair['oracle']:.3f}]"
                )
            lines.append("")

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

    @property
    def labels(self) -> tuple[str, str]:
        """Column names for the two sides."""
        if self.mode == "cross_model":
            return (
                _short_model(self.base_model.get("base_model")),
                _short_model(self.finetuned_model.get("base_model")),
            )
        return "base", "finetuned"

    def _conclusion(self) -> str:
        """State the outcome without overclaiming (spec sections 35, 36)."""
        if self.incomparable:
            return "Conclusion: none. The two results are not comparable."
        if self.mode == "cross_model":
            if self.primary:
                return (
                    f"Conclusion (primary, pre-registered): {self.primary['verdict'].upper()} "
                    f"on the {self.primary['subset']} subset "
                    f"({self.primary.get('explanation', '')}). Per-task rows are secondary."
                )
            return (
                "Conclusion: cross-model comparison without a primary subset; read the "
                "per-task and subset rows, and do not summarise them as a single winner."
            )
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


def _short_model(model_id: Any) -> str:
    """The last path segment of a model id, for column headers."""
    text = str(model_id or "-")
    return text.rsplit("/", 1)[-1][:24]


def _render_ci(significance: dict[str, Any]) -> str:
    if not significance.get("estimable"):
        return "n/a"
    return f"{significance['ci95_low']:+.4f} to {significance['ci95_high']:+.4f}"


def _render_primary(primary: dict[str, Any], left: str, right: str) -> list[str]:
    return [
        f"PRIMARY: {right} vs {left} on the {primary['subset']} subset",
        f"  difference     : {primary.get('difference', 0):+.4f} "
        f"(n={primary.get('n', 0)}, {primary.get('clusters', 0)} groups)",
        f"  cluster 95% CI : {_render_ci(primary)}",
        f"  margin         : ±{primary['margin']}",
        f"  verdict        : {primary['verdict'].upper()} — {primary.get('explanation', '')}",
        "",
    ]


def primary_verdict(significance: dict[str, Any], margin: float) -> tuple[str, str]:
    """Classify a difference against an equivalence margin by its cluster interval.

    better: the whole interval is above zero. worse: the whole interval is below
    zero. equivalent: the whole interval lies within ±margin. Otherwise
    inconclusive: the data cannot tell these apart.
    """
    if not significance.get("estimable"):
        return "inconclusive", f"interval not estimable: {significance.get('reason', 'unknown')}"
    low, high = significance["ci95_low"], significance["ci95_high"]
    if low > 0:
        note = ", but within the equivalence margin" if high <= margin else ""
        return "better", f"95% CI {low:+.4f} to {high:+.4f} excludes zero{note}"
    if high < 0:
        note = ", but within the equivalence margin" if low >= -margin else ""
        return "worse", f"95% CI {low:+.4f} to {high:+.4f} excludes zero{note}"
    if -margin <= low and high <= margin:
        return "equivalent", f"95% CI {low:+.4f} to {high:+.4f} lies within ±{margin}"
    return "inconclusive", f"95% CI {low:+.4f} to {high:+.4f} spans zero and exceeds ±{margin}"


def check_benchmark_identity(
    left: dict[str, Any], right: dict[str, Any]
) -> tuple[bool, dict[str, Any], list[str]]:
    """Whether two results were graded against the same benchmark.

    By content: the benchmark file's sha256 when both recorded it, otherwise the
    targets of the examples both evaluated (id, task, reference decision). A path
    identifies nothing (H-F5). Returns ``(comparable, identity, warnings)``.
    """
    warnings: list[str] = []
    left_rows = {r["example_id"]: r for r in left.get("results", [])}
    right_rows = {r["example_id"]: r for r in right.get("results", [])}
    shared = sorted(set(left_rows) & set(right_rows))
    disagreeing = [
        i
        for i in shared
        if (left_rows[i].get("task"), left_rows[i].get("reference_decision"))
        != (right_rows[i].get("task"), right_rows[i].get("reference_decision"))
    ]
    targets_agree = bool(shared) and not disagreeing

    left_sha, right_sha = left.get("benchmark_sha256"), right.get("benchmark_sha256")
    if left_sha and right_sha and left_sha == right_sha:
        return True, {"method": "sha256", "detail": f"sha256 {left_sha[:16]}… on both"}, warnings
    if left_sha and right_sha:
        if targets_agree:
            warnings.append(
                "The benchmark files differ byte for byte, but every shared example has "
                "the same task and target. Compared on the shared examples."
            )
            return (
                True,
                {"method": "targets", "detail": "files differ; shared targets identical"},
                warnings,
            )
        return (
            False,
            {"method": "sha256", "detail": f"sha256 {left_sha[:16]}… vs {right_sha[:16]}…"},
            [*warnings, "Different benchmarks (sha256 and targets differ). Not comparable."],
        )
    if targets_agree:
        return (
            True,
            {
                "method": "targets",
                "detail": f"{len(shared)} shared example(s) with identical targets "
                "(a file hash was not recorded on both sides)",
            },
            warnings,
        )
    if not shared:
        return False, {"method": "targets", "detail": "no shared examples"}, warnings
    return (
        False,
        {"method": "targets", "detail": f"{len(disagreeing)} shared example(s) differ in target"},
        [
            *warnings,
            f"{len(disagreeing)} example(s) have different tasks or targets in the two "
            "results: they were graded against different benchmarks. Not comparable.",
        ],
    )


def _groups_for(results: Sequence[dict[str, Any]]) -> dict[str, str | None]:
    return {r["example_id"]: r.get("group_id") for r in results}


def _paired(
    base_results: Sequence[dict[str, Any]],
    finetuned_results: Sequence[dict[str, Any]],
    *,
    name: str,
    keep: Any,
    compute_significance: bool,
) -> TaskComparison | None:
    """One paired comparison over the examples ``keep(record)`` selects."""
    base_scores = {r["example_id"]: float(r["score"]) for r in base_results if keep(r)}
    ft_scores = {r["example_id"]: float(r["score"]) for r in finetuned_results if keep(r)}
    paired_ids = sorted(set(base_scores) & set(ft_scores))
    if not paired_ids:
        return None
    left = [base_scores[i] for i in paired_ids]
    right = [ft_scores[i] for i in paired_ids]
    comparison = TaskComparison(
        task=name,
        base_score=summarize(name, left).mean,
        finetuned_score=summarize(name, right).mean,
        count=len(paired_ids),
    )
    if compute_significance:
        comparison.significance = bootstrap_difference(left, right)
        groups = _groups_for(base_results)
        clusters = [groups.get(i) for i in paired_ids]
        if all(clusters):
            comparison.cluster_significance = paired_cluster_bootstrap_difference(
                left, right, [str(c) for c in clusters]
            )
        else:
            comparison.cluster_significance = {"estimable": False, "reason": _NO_GROUPS}
    return comparison


def compare_results(
    base: dict[str, Any],
    finetuned: dict[str, Any],
    *,
    compute_significance: bool = True,
    cross_model: bool = False,
    primary_subset: str | None = None,
    equivalence_margin: float | None = None,
) -> ComparisonReport:
    """Compare two saved evaluation-result payloads (spec section 54).

    Scores are paired by ``example_id`` so the bootstrap is a genuine paired test
    rather than a comparison of two unrelated samples.

    Args:
        cross_model: ``base`` and ``finetuned`` are two models on the same arm,
            not one model before and after fine-tuning.
        primary_subset: Subset (``answerable`` or ``should_decline``) whose
            cluster interval decides the primary verdict, with
            ``equivalence_margin``.
    """
    report = ComparisonReport(
        base_arm=base.get("arm", "base"),
        finetuned_arm=finetuned.get("arm", "finetuned"),
        base_model=base.get("model", {}),
        finetuned_model=finetuned.get("model", {}),
        mode="cross_model" if cross_model else "fine_tuning",
    )

    base_results = base.get("results", [])
    finetuned_results = finetuned.get("results", [])

    # --- comparability checks ----------------------------------------------
    comparable, identity, identity_warnings = check_benchmark_identity(base, finetuned)
    report.benchmark_identity = identity
    report.comparability_warnings.extend(identity_warnings)
    if not comparable:
        report.incomparable = True
    if base.get("generation") != finetuned.get("generation"):
        report.comparability_warnings.append(
            "Decoding settings differ between the arms; part of any difference may be "
            "due to temperature or max_new_tokens rather than the model."
        )
    base_model_id = base.get("model", {}).get("base_model")
    ft_model_id = finetuned.get("model", {}).get("base_model")
    if cross_model:
        if report.base_arm != report.finetuned_arm:
            report.comparability_warnings.append(
                f"Cross-model comparison of different arms ({report.base_arm} vs "
                f"{report.finetuned_arm}); arm conditions differ as well as models."
            )
    else:
        if base_model_id != ft_model_id:
            report.comparability_warnings.append(
                f"Different base models ({base_model_id} vs {ft_model_id}). This is a "
                "cross-model comparison, not a fine-tuning effect; use cross_model mode."
            )
        if finetuned.get("model", {}).get("adapter_path") is None:
            report.comparability_warnings.append(
                "The fine-tuned arm reports no adapter path — it may be the base model "
                "evaluated twice."
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
        report.incomparable = True
        return report
    if report.incomparable:
        for warning in report.comparability_warnings:
            logger.warning("Comparability: %s", warning)
        return report

    # --- per task -----------------------------------------------------------
    tasks = sorted(
        {r.get("task", "unknown") for r in base_results}
        | {r.get("task", "unknown") for r in finetuned_results}
    )
    for task in tasks:
        comparison = _paired(
            base_results,
            finetuned_results,
            name=task,
            keep=lambda r, task=task: r.get("task", "unknown") == task,
            compute_significance=compute_significance,
        )
        if comparison is not None:
            report.per_task.append(comparison)

    # --- overall -------------------------------------------------------------
    report.overall = _paired(
        base_results,
        finetuned_results,
        name="overall",
        keep=lambda r: True,
        compute_significance=compute_significance,
    )

    # --- subsets (answerable / should-decline) -------------------------------
    for subset in SUBSETS:
        comparison = _paired(
            base_results,
            finetuned_results,
            name=subset,
            keep=lambda r, subset=subset: r.get("subset") == subset,
            compute_significance=compute_significance,
        )
        if comparison is not None:
            report.subsets.append(comparison)

    if primary_subset is not None:
        margin = equivalence_margin if equivalence_margin is not None else 0.0
        chosen = next((t for t in report.subsets if t.task == primary_subset), None)
        if chosen is None:
            report.primary = {
                "subset": primary_subset,
                "margin": margin,
                "verdict": "inconclusive",
                "explanation": f"no record carries subset={primary_subset!r}; annotate the "
                "results with scripts/rescore.py --mode annotate",
            }
        else:
            verdict, explanation = primary_verdict(chosen.cluster_significance, margin)
            report.primary = {
                "subset": primary_subset,
                "margin": margin,
                **chosen.cluster_significance,
                "left_mean": round(chosen.base_score, 4),
                "right_mean": round(chosen.finetuned_score, 4),
                "verdict": verdict,
                "explanation": explanation,
            }

    # --- consistency by group_id, from the corrected blocks ------------------
    base_corrected = (base.get("corrected") or {}).get("consistency") or {}
    ft_corrected = (finetuned.get("corrected") or {}).get("consistency") or {}
    for name in ("group_id", "group_id_answerable"):
        left_block, right_block = base_corrected.get(name), ft_corrected.get(name)
        if (
            left_block
            and right_block
            and left_block.get("evaluated_groups")
            and right_block.get("evaluated_groups")
        ):
            report.corrected_consistency[name] = {
                "left": left_block["agreement_rate"],
                "right": right_block["agreement_rate"],
                "oracle": left_block["oracle_agreement_rate"],
                "groups": left_block["evaluated_groups"],
            }

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
    left, right = comparison.labels
    lines = [
        f"# Experiment report — {experiment_id}",
        "",
        f"Generated: {comparison.generated_at}",
        "",
        "## What was compared",
        "",
        f"- mode: `{comparison.mode}`",
        f"- {left} arm: `{comparison.base_arm}`",
        f"- {right} arm: `{comparison.finetuned_arm}`",
        f"- {left} model: `{comparison.base_model.get('base_model', '-')}`",
        f"- {right} model: `{comparison.finetuned_model.get('base_model', '-')}`",
        f"- model family: `{comparison.base_model.get('family', '-')}`",
        f"- adapter: `{comparison.finetuned_model.get('adapter_path', '-')}`",
        f"- dataset version: `{dataset_version}`",
        f"- benchmark identity: {comparison.benchmark_identity.get('detail', 'not checked')}",
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

    if comparison.incomparable:
        lines.extend(["## Not comparable", "", comparison._conclusion(), ""])
        return "\n".join(lines)

    if comparison.primary:
        primary = comparison.primary
        lines.extend(
            [
                "## Primary comparison (pre-registered)",
                "",
                f"- subset: `{primary['subset']}`, margin ±{primary['margin']}",
                f"- {left} {primary.get('left_mean', '-')} → {right} "
                f"{primary.get('right_mean', '-')} "
                f"(difference {primary.get('difference', 0):+.4f})",
                f"- cluster 95% CI: {_render_ci(primary)} "
                f"({primary.get('clusters', 0)} groups, n={primary.get('n', 0)})",
                f"- **verdict: {primary['verdict'].upper()}** — {primary.get('explanation', '')}",
                "",
            ]
        )

    lines.extend(
        [
            "## Did it improve?" if comparison.mode == "fine_tuning" else "## Per task",
            "",
            f"| Task | {left} | {right} | Abs Δ | Rel Δ | n | Verdict | Cluster verdict |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for task in comparison.per_task:
        lines.append(
            f"| `{task.task}` | {task.base_score:.4f} | {task.finetuned_score:.4f} | "
            f"{task.absolute_delta:+.4f} | {task.relative_delta:+.1%} | {task.count} | "
            f"{task.verdict} | {task.cluster_verdict} |"
        )
    lines.extend(
        [
            "",
            "_Verdict: examples resampled (the original method). Cluster verdict: "
            "whole groups resampled, which respects that perturbations of one case "
            "are not independent (H-F14)._",
            "",
        ]
    )

    if comparison.subsets:
        lines.extend(
            [
                "## By subset",
                "",
                f"| Subset | {left} | {right} | Abs Δ | n | Groups | Cluster 95% CI | Verdict |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for subset in comparison.subsets:
            lines.append(
                f"| `{subset.task}` | {subset.base_score:.4f} | {subset.finetuned_score:.4f} | "
                f"{subset.absolute_delta:+.4f} | {subset.count} | "
                f"{subset.cluster_significance.get('clusters', '-')} | "
                f"{_render_ci(subset.cluster_significance)} | {subset.cluster_verdict} |"
            )
        lines.append("")

    if comparison.corrected_consistency:
        lines.extend(["## Consistency by group_id", ""])
        for name, pair in comparison.corrected_consistency.items():
            lines.append(
                f"- {name}: {pair['left']:.3f} → {pair['right']:.3f} "
                f"({pair['right'] - pair['left']:+.3f}); an oracle scores {pair['oracle']:.3f} "
                f"over {pair['groups']} groups"
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
                "## Consistency (configured grouping key)",
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
            "accompanied by bootstrap confidence intervals, over examples and over",
            "groups. A difference that is not significant is reported as such rather",
            "than as a win._",
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
