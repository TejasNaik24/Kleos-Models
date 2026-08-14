"""Consistency testing (spec section 22).

The question: did the model learn a *decision policy*, or a surface pattern?

Method. Group examples that are logically equivalent — same underlying situation,
differing only in wording, evidence order, context order, irrelevant additions,
formatting, schema or length. A model that learned a policy reaches the same
decision across the whole group. A model that learned "answer the first-listed
item" or "match this phrasing" does not.

This is one of the few tests that can distinguish those two cases, which makes it
central rather than decorative. It is also cheap: no extra labels are needed,
only perturbations of scenarios that already exist.

Two numbers are reported, and conflating them would be a mistake:

``agreement_rate``
    How often the group reaches one decision. Measures stability.
``correct_agreement_rate``
    How often the group agrees **on the right answer**. A model that is
    consistently wrong scores 1.0 on the first and 0.0 on the second.

This module imports no torch.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from kleos_models.constants import PERTURBATION_KINDS
from kleos_models.evaluation.metrics import normalize_answer
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ConsistencyGroup:
    """One set of logically equivalent examples and the model's answers."""

    group_id: str
    example_ids: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    perturbation_kinds: list[str] = field(default_factory=list)
    reference_decision: str | None = None

    @property
    def size(self) -> int:
        return len(self.decisions)

    @property
    def distinct_decisions(self) -> int:
        return len({normalize_answer(d) for d in self.decisions})

    @property
    def is_consistent(self) -> bool:
        """Whether every perturbation produced the same decision."""
        return self.size > 1 and self.distinct_decisions == 1

    @property
    def majority_decision(self) -> str | None:
        if not self.decisions:
            return None
        counts = Counter(normalize_answer(d) for d in self.decisions)
        return counts.most_common(1)[0][0]

    @property
    def majority_share(self) -> float:
        """Fraction of answers agreeing with the majority.

        A softer measure than all-or-nothing consistency: 4/5 agreement is
        meaningfully different from 2/5, and binary consistency hides that.
        """
        if not self.decisions:
            return 0.0
        counts = Counter(normalize_answer(d) for d in self.decisions)
        return counts.most_common(1)[0][1] / len(self.decisions)

    @property
    def is_correct_and_consistent(self) -> bool:
        """Consistent *and* agreeing with the reference."""
        if not self.is_consistent or self.reference_decision is None:
            return False
        return normalize_answer(self.decisions[0]) == normalize_answer(self.reference_decision)

    def flipped_by(self) -> dict[str, int]:
        """Which perturbation kinds coincided with a minority answer.

        Attribution is approximate — several perturbations may combine — but it
        points at which axis the model is fragile along, which is the actionable
        part.
        """
        majority = self.majority_decision
        flips: dict[str, int] = {}
        for decision, kind in zip(self.decisions, self.perturbation_kinds, strict=False):
            if normalize_answer(decision) != majority:
                flips[kind or "unspecified"] = flips.get(kind or "unspecified", 0) + 1
        return flips

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "size": self.size,
            "distinct_decisions": self.distinct_decisions,
            "consistent": self.is_consistent,
            "majority_share": round(self.majority_share, 4),
            "correct_and_consistent": self.is_correct_and_consistent,
            "example_ids": self.example_ids,
            "flipped_by": self.flipped_by(),
        }


@dataclass
class ConsistencyReport:
    """Aggregate consistency results."""

    groups: list[ConsistencyGroup] = field(default_factory=list)
    skipped_singletons: int = 0
    group_key: str = "scenario_family"

    @property
    def evaluated_groups(self) -> int:
        return len(self.groups)

    @property
    def agreement_rate(self) -> float:
        """Fraction of groups reaching a single decision."""
        if not self.groups:
            return 0.0
        return sum(1 for g in self.groups if g.is_consistent) / len(self.groups)

    @property
    def mean_majority_share(self) -> float:
        """Average agreement within groups — a graded view of stability."""
        if not self.groups:
            return 0.0
        return sum(g.majority_share for g in self.groups) / len(self.groups)

    @property
    def correct_agreement_rate(self) -> float:
        """Fraction of groups that agree *and* are right."""
        gradeable = [g for g in self.groups if g.reference_decision is not None]
        if not gradeable:
            return 0.0
        return sum(1 for g in gradeable if g.is_correct_and_consistent) / len(gradeable)

    def flips_by_perturbation(self) -> dict[str, int]:
        """Total minority answers attributed to each perturbation kind."""
        totals: dict[str, int] = defaultdict(int)
        for group in self.groups:
            for kind, count in group.flipped_by().items():
                totals[kind] += count
        return dict(sorted(totals.items(), key=lambda kv: -kv[1]))

    @property
    def inconsistent_groups(self) -> list[ConsistencyGroup]:
        return [g for g in self.groups if not g.is_consistent]

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_key": self.group_key,
            "evaluated_groups": self.evaluated_groups,
            "skipped_singletons": self.skipped_singletons,
            "agreement_rate": round(self.agreement_rate, 4),
            "mean_majority_share": round(self.mean_majority_share, 4),
            "correct_agreement_rate": round(self.correct_agreement_rate, 4),
            "flips_by_perturbation": self.flips_by_perturbation(),
            "inconsistent_group_count": len(self.inconsistent_groups),
            "groups": [g.to_dict() for g in self.groups],
        }

    def render(self) -> str:
        lines = [
            "Consistency (spec section 22)",
            f"  grouping key            : {self.group_key}",
            f"  groups evaluated        : {self.evaluated_groups}",
            f"  singletons skipped      : {self.skipped_singletons}",
            "",
            f"  agreement rate          : {self.agreement_rate:.3f}  "
            "(same decision across all perturbations)",
            f"  mean majority share     : {self.mean_majority_share:.3f}  "
            "(graded within-group agreement)",
            f"  correct agreement rate  : {self.correct_agreement_rate:.3f}  "
            "(agrees AND matches the reference)",
        ]
        if self.evaluated_groups == 0:
            lines.extend(
                [
                    "",
                    "  No multi-example groups were found, so consistency was not measured.",
                    "  Populate metadata.scenario_family on perturbed examples to enable it.",
                ]
            )
            return "\n".join(lines)

        flips = self.flips_by_perturbation()
        if flips:
            lines.extend(["", "  Decision flips by perturbation kind:"])
            lines.extend(f"    {kind:<22} {count}" for kind, count in flips.items())
        if self.agreement_rate < 1.0:
            lines.extend(
                [
                    "",
                    f"  {len(self.inconsistent_groups)} group(s) changed decision under a "
                    "logically irrelevant perturbation.",
                    "  That is evidence of a surface pattern rather than a learned policy.",
                ]
            )
        return "\n".join(lines)


def build_consistency_groups(
    *,
    example_ids: Sequence[str],
    group_ids: Sequence[str],
    decisions: Sequence[str],
    scores: Sequence[float] | None = None,
    perturbation_kinds: Sequence[str] | None = None,
    reference_decisions: Sequence[str | None] | None = None,
    min_group_size: int = 2,
    group_key: str = "scenario_family",
) -> ConsistencyReport:
    """Group per-example decisions and compute consistency.

    Args:
        example_ids: Example identifiers.
        group_ids: Equivalence-class id per example.
        decisions: The decision extracted from each response.
        scores: Optional grader score per example.
        perturbation_kinds: Optional perturbation label per example.
        reference_decisions: Optional ground-truth decision per example.
        min_group_size: Groups smaller than this are skipped, not scored.
        group_key: Metadata key used, recorded in the report.

    Returns:
        A :class:`ConsistencyReport`.
    """
    lengths = {len(example_ids), len(group_ids), len(decisions)}
    if len(lengths) != 1:
        raise ValueError(
            "example_ids, group_ids and decisions must be the same length; "
            f"got {len(example_ids)}, {len(group_ids)}, {len(decisions)}"
        )

    grouped: dict[str, ConsistencyGroup] = {}
    for index, group_id in enumerate(group_ids):
        group = grouped.setdefault(group_id, ConsistencyGroup(group_id=group_id))
        group.example_ids.append(example_ids[index])
        group.decisions.append(decisions[index])
        if scores is not None and index < len(scores):
            group.scores.append(scores[index])
        group.perturbation_kinds.append(
            perturbation_kinds[index]
            if perturbation_kinds and index < len(perturbation_kinds)
            else ""
        )
        if reference_decisions and index < len(reference_decisions):
            reference = reference_decisions[index]
            if reference is not None and group.reference_decision is None:
                group.reference_decision = reference

    report = ConsistencyReport(group_key=group_key)
    for group in grouped.values():
        if group.size < min_group_size:
            report.skipped_singletons += 1
            continue
        report.groups.append(group)

    report.groups.sort(key=lambda g: g.group_id)

    if report.evaluated_groups == 0:
        logger.warning(
            "Consistency testing found no group with at least %d examples. Set "
            "metadata.scenario_family on perturbed examples so equivalent scenarios "
            "can be grouped.",
            min_group_size,
        )
    else:
        logger.info(
            "Consistency: %d group(s), agreement rate %.3f",
            report.evaluated_groups,
            report.agreement_rate,
        )
    return report


def validate_perturbation_kinds(kinds: Sequence[str]) -> list[str]:
    """Warn about perturbation kinds outside the registry."""
    unknown = sorted({k for k in kinds if k and k not in PERTURBATION_KINDS})
    if unknown:
        logger.warning(
            "Unregistered perturbation kind(s) %s; registered kinds are %s.",
            unknown,
            list(PERTURBATION_KINDS),
        )
    return unknown
