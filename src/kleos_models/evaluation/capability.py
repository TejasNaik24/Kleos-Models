"""Capability preservation (spec section 24).

Fine-tuning on a narrow behavioural dataset can damage general ability —
catastrophic forgetting. A KLEOS adapter that improves notification
prioritization while making the model worse at ordinary reasoning is not a win,
and without this check it would look like one.

Method: a small **fixed** regression suite, run identically before and after
fine-tuning, reporting

``base_score``  ``fine_tuned_score``  ``delta``  ``relative_delta``

The suite does not need to reproduce a public benchmark. What it must be is
*constant across experiments* — spec section 24 is explicit that consistency
matters more than coverage here. Changing the suite invalidates comparison with
every earlier run, so the suite carries a version and the report records it.

This module imports no torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from kleos_models.evaluation.metrics import MetricSummary, bootstrap_difference, summarize
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)

#: Version of the capability suite. Bump ONLY when the suite content changes, and
#: understand that doing so breaks comparability with prior runs.
CAPABILITY_SUITE_VERSION = "kleos-capability-v0.1.0"


@dataclass
class CapabilityDelta:
    """Before/after comparison on the capability suite."""

    base: MetricSummary
    fine_tuned: MetricSummary
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    suite_version: str = CAPABILITY_SUITE_VERSION
    significance: dict[str, Any] = field(default_factory=dict)

    @property
    def base_score(self) -> float:
        return self.base.mean

    @property
    def fine_tuned_score(self) -> float:
        return self.fine_tuned.mean

    @property
    def delta(self) -> float:
        """Absolute change. Negative means capability was lost."""
        return self.fine_tuned.mean - self.base.mean

    @property
    def relative_delta(self) -> float:
        """Change as a fraction of the base score."""
        if self.base.mean == 0:
            return 0.0
        return self.delta / self.base.mean

    @property
    def regressed(self) -> bool:
        """Whether general capability measurably dropped.

        A 1% threshold keeps decoding noise from being reported as damage.
        """
        return self.relative_delta < -0.01

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_version": self.suite_version,
            "base_score": round(self.base_score, 4),
            "fine_tuned_score": round(self.fine_tuned_score, 4),
            "delta": round(self.delta, 4),
            "relative_delta": round(self.relative_delta, 4),
            "regressed": self.regressed,
            "base": self.base.to_dict(),
            "fine_tuned": self.fine_tuned.to_dict(),
            "per_category": self.per_category,
            "significance": self.significance,
        }

    def render(self) -> str:
        lines = [
            "Capability preservation (spec section 24)",
            f"  suite version        : {self.suite_version}",
            "",
            f"  base_score           : {self.base.render()}",
            f"  fine_tuned_score     : {self.fine_tuned.render()}",
            f"  delta                : {self.delta:+.4f}",
            f"  relative_delta       : {self.relative_delta:+.2%}",
        ]
        if self.significance:
            lines.append(
                f"  paired bootstrap     : p={self.significance.get('p_value')}, "
                f"95% CI [{self.significance.get('ci95_low')}, "
                f"{self.significance.get('ci95_high')}]"
            )
        if self.per_category:
            lines.extend(
                [
                    "",
                    f"  {'category':<26} {'base':>8} {'tuned':>8} {'delta':>9}",
                    f"  {'-' * 26} {'-' * 8} {'-' * 8} {'-' * 9}",
                ]
            )
            for category, scores in sorted(
                self.per_category.items(), key=lambda kv: kv[1].get("delta", 0.0)
            ):
                lines.append(
                    f"  {category:<26} {scores.get('base', 0):>8.4f} "
                    f"{scores.get('fine_tuned', 0):>8.4f} {scores.get('delta', 0):>+9.4f}"
                )
        lines.append("")
        if self.regressed:
            lines.extend(
                [
                    f"  REGRESSION: general capability dropped by {abs(self.relative_delta):.1%}.",
                    "  Report this alongside any task-specific gain. A model that wins on",
                    "  the target task while losing general ability is a trade-off, not an",
                    "  improvement, and the trade-off is the finding.",
                ]
            )
        else:
            lines.append("  No meaningful regression in general capability detected.")
        return "\n".join(lines)


def build_capability_delta(
    *,
    base_scores: Sequence[float],
    fine_tuned_scores: Sequence[float],
    categories: Sequence[str] | None = None,
    suite_version: str = CAPABILITY_SUITE_VERSION,
    compute_significance: bool = True,
) -> CapabilityDelta:
    """Compare base and fine-tuned scores on the capability suite.

    Args:
        base_scores: Per-item scores from the base model.
        fine_tuned_scores: Per-item scores from the fine-tuned model, in the same
            item order.
        categories: Optional per-item category label for a breakdown.
        suite_version: Version identifier of the suite used.
        compute_significance: Run a paired bootstrap on the difference.

    Raises:
        ValueError: when the two score vectors are not paired item-for-item.
    """
    if len(base_scores) != len(fine_tuned_scores):
        raise ValueError(
            "Capability comparison requires paired scores over the same items: "
            f"got {len(base_scores)} base and {len(fine_tuned_scores)} fine-tuned. "
            "Both arms must run the identical fixed suite."
        )

    base_summary = summarize("capability_base", base_scores)
    tuned_summary = summarize("capability_fine_tuned", fine_tuned_scores)

    per_category: dict[str, dict[str, float]] = {}
    if categories:
        if len(categories) != len(base_scores):
            raise ValueError("categories must be the same length as the score vectors")
        grouped: dict[str, tuple[list[float], list[float]]] = {}
        for index, category in enumerate(categories):
            bucket = grouped.setdefault(category, ([], []))
            bucket[0].append(base_scores[index])
            bucket[1].append(fine_tuned_scores[index])
        for category, (base_values, tuned_values) in grouped.items():
            base_mean = sum(base_values) / len(base_values)
            tuned_mean = sum(tuned_values) / len(tuned_values)
            per_category[category] = {
                "base": round(base_mean, 4),
                "fine_tuned": round(tuned_mean, 4),
                "delta": round(tuned_mean - base_mean, 4),
                "count": len(base_values),
            }

    significance: dict[str, Any] = {}
    if compute_significance and base_scores:
        significance = bootstrap_difference(base_scores, fine_tuned_scores)

    delta = CapabilityDelta(
        base=base_summary,
        fine_tuned=tuned_summary,
        per_category=per_category,
        suite_version=suite_version,
        significance=significance,
    )

    if delta.regressed:
        logger.warning(
            "Capability regression: %.4f → %.4f (%+.2f%%). Report this with any "
            "task-specific gain.",
            delta.base_score,
            delta.fine_tuned_score,
            delta.relative_delta * 100,
        )
    else:
        logger.info(
            "Capability preserved: %.4f → %.4f (%+.2f%%)",
            delta.base_score,
            delta.fine_tuned_score,
            delta.relative_delta * 100,
        )
    return delta
