"""Out-of-distribution evaluation (spec section 23).

Spec section 36 forbids claiming OOD generalization without OOD evaluation, and
spec section 23 forbids collapsing everything into one score. This module enforces
both by construction: it reports

``in_distribution_score``
``ood_score``
``generalization_gap``

as three separate numbers, plus a per-shift breakdown. There is deliberately no
"overall" field, because a single blended number is exactly what hides the case
that matters — a model that improves in-distribution while degrading OOD.

Shift kinds come from :data:`kleos_models.constants.OOD_SHIFT_KINDS`: unseen
entities, unseen domains, unseen formats, unseen source types, context-length
shift, reordered evidence, conflicting evidence.

This module imports no torch.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from kleos_models.constants import OOD_SHIFT_KINDS
from kleos_models.evaluation.metrics import MetricSummary, summarize
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ShiftResult:
    """Scores for one OOD shift kind."""

    shift: str
    summary: MetricSummary
    gap_vs_in_distribution: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "shift": self.shift,
            **self.summary.to_dict(),
            "gap_vs_in_distribution": round(self.gap_vs_in_distribution, 4),
        }


@dataclass
class OODReport:
    """Separate in-distribution, OOD and per-shift results."""

    in_distribution: MetricSummary
    ood: MetricSummary
    per_shift: list[ShiftResult] = field(default_factory=list)
    metric_name: str = "score"
    unregistered_shifts: list[str] = field(default_factory=list)

    @property
    def in_distribution_score(self) -> float:
        return self.in_distribution.mean

    @property
    def ood_score(self) -> float:
        return self.ood.mean

    @property
    def generalization_gap(self) -> float:
        """In-distribution minus OOD. Positive means performance drops OOD."""
        return self.in_distribution.mean - self.ood.mean

    @property
    def relative_gap(self) -> float:
        """Gap as a fraction of in-distribution performance."""
        if self.in_distribution.mean == 0:
            return 0.0
        return self.generalization_gap / self.in_distribution.mean

    @property
    def measurable(self) -> bool:
        """Whether both populations had examples. A gap needs both sides."""
        return self.in_distribution.count > 0 and self.ood.count > 0

    def worst_shift(self) -> ShiftResult | None:
        """The shift with the largest drop."""
        if not self.per_shift:
            return None
        return max(self.per_shift, key=lambda s: s.gap_vs_in_distribution)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric_name,
            "measurable": self.measurable,
            "in_distribution_score": round(self.in_distribution_score, 4),
            "ood_score": round(self.ood_score, 4),
            "generalization_gap": round(self.generalization_gap, 4),
            "relative_gap": round(self.relative_gap, 4),
            "in_distribution": self.in_distribution.to_dict(),
            "ood": self.ood.to_dict(),
            "per_shift": [s.to_dict() for s in self.per_shift],
            "unregistered_shifts": self.unregistered_shifts,
        }

    def render(self) -> str:
        if not self.measurable:
            missing = "OOD" if self.ood.count == 0 else "in-distribution"
            return (
                "OOD evaluation (spec section 23)\n"
                f"  NOT MEASURABLE — the {missing} population is empty.\n"
                "  Tag benchmark examples with split_tag='ood' and an ood_shift value.\n"
                "  No generalization claim can be made from this run."
            )

        lines = [
            "OOD evaluation (spec section 23)",
            f"  metric                  : {self.metric_name}",
            "",
            f"  in_distribution_score   : {self.in_distribution.render()}",
            f"  ood_score               : {self.ood.render()}",
            f"  generalization_gap      : {self.generalization_gap:+.4f} "
            f"({self.relative_gap:+.1%} relative)",
        ]
        if self.per_shift:
            lines.extend(
                [
                    "",
                    "  Per shift kind:",
                    f"    {'shift':<26} {'score':>8} {'gap':>9} {'n':>5}",
                    f"    {'-' * 26} {'-' * 8} {'-' * 9} {'-' * 5}",
                ]
            )
            for shift in sorted(self.per_shift, key=lambda s: -s.gap_vs_in_distribution):
                lines.append(
                    f"    {shift.shift:<26} {shift.summary.mean:>8.4f} "
                    f"{shift.gap_vs_in_distribution:>+9.4f} {shift.summary.count:>5}"
                )
            worst = self.worst_shift()
            if worst and worst.gap_vs_in_distribution > 0.05:
                lines.extend(
                    [
                        "",
                        f"  Largest drop: {worst.shift} ({worst.gap_vs_in_distribution:+.4f}).",
                    ]
                )
        if self.unregistered_shifts:
            lines.extend(["", f"  Unregistered shift kinds present: {self.unregistered_shifts}"])
        lines.extend(
            [
                "",
                "  These three numbers are reported separately on purpose. A single",
                "  blended score would hide a model that improves in-distribution",
                "  while degrading out-of-distribution.",
            ]
        )
        return "\n".join(lines)


def build_ood_report(
    *,
    scores: Sequence[float],
    split_tags: Sequence[str],
    shift_kinds: Sequence[str | None] | None = None,
    metric_name: str = "score",
) -> OODReport:
    """Partition per-example scores into in-distribution and OOD populations.

    Args:
        scores: One score per evaluated example.
        split_tags: ``"in_distribution"`` or ``"ood"`` per example. Anything else
            (including ``"capability"``) is excluded from both populations.
        shift_kinds: Optional OOD shift label per example.
        metric_name: Name of the metric being partitioned.

    Returns:
        An :class:`OODReport`.
    """
    if len(scores) != len(split_tags):
        raise ValueError(
            f"scores and split_tags must be the same length: {len(scores)} vs {len(split_tags)}"
        )

    in_distribution: list[float] = []
    ood: list[float] = []
    by_shift: dict[str, list[float]] = defaultdict(list)

    for index, (score, tag) in enumerate(zip(scores, split_tags, strict=True)):
        if tag == "ood":
            ood.append(score)
            raw_shift = shift_kinds[index] if shift_kinds and index < len(shift_kinds) else None
            by_shift[str(raw_shift) if raw_shift else "unspecified"].append(score)
        elif tag == "in_distribution":
            in_distribution.append(score)

    in_summary = summarize(f"{metric_name}_in_distribution", in_distribution)
    ood_summary = summarize(f"{metric_name}_ood", ood)

    per_shift = [
        ShiftResult(
            shift=shift,
            summary=summarize(f"{metric_name}_{shift}", values),
            gap_vs_in_distribution=in_summary.mean - (sum(values) / len(values)),
        )
        for shift, values in sorted(by_shift.items())
        if values
    ]

    unregistered = sorted({s.shift for s in per_shift} - set(OOD_SHIFT_KINDS) - {"unspecified"})
    if unregistered:
        logger.warning(
            "Unregistered OOD shift kind(s) %s; registered kinds are %s.",
            unregistered,
            list(OOD_SHIFT_KINDS),
        )

    report = OODReport(
        in_distribution=in_summary,
        ood=ood_summary,
        per_shift=per_shift,
        metric_name=metric_name,
        unregistered_shifts=unregistered,
    )

    if not report.measurable:
        logger.warning(
            "OOD evaluation is not measurable: in_distribution n=%d, ood n=%d. "
            "No generalization claim can be made.",
            in_summary.count,
            ood_summary.count,
        )
    else:
        logger.info(
            "OOD: in-distribution %.4f, ood %.4f, gap %+.4f",
            report.in_distribution_score,
            report.ood_score,
            report.generalization_gap,
        )
    return report


def compare_ood_reports(base: OODReport, finetuned: OODReport) -> dict[str, Any]:
    """Compare two OOD reports (base vs fine-tuned).

    Surfaces the case spec section 58 calls the most interesting result: an
    in-distribution gain that comes with an OOD loss.
    """
    in_delta = finetuned.in_distribution_score - base.in_distribution_score
    ood_delta = finetuned.ood_score - base.ood_score
    gap_delta = finetuned.generalization_gap - base.generalization_gap

    verdict = "no meaningful change"
    if in_delta > 0.01 and ood_delta > 0.01:
        verdict = "improved both in-distribution and out-of-distribution"
    elif in_delta > 0.01 >= ood_delta and ood_delta < -0.01:
        verdict = (
            "improved in-distribution but DEGRADED out-of-distribution — this is "
            "consistent with fitting surface features rather than learning a policy"
        )
    elif in_delta > 0.01:
        verdict = "improved in-distribution; out-of-distribution unchanged"
    elif in_delta < -0.01:
        verdict = "degraded in-distribution"

    return {
        "in_distribution_delta": round(in_delta, 4),
        "ood_delta": round(ood_delta, 4),
        "generalization_gap_delta": round(gap_delta, 4),
        "base": base.to_dict(),
        "finetuned": finetuned.to_dict(),
        "verdict": verdict,
    }
