"""Variation-axis coverage reporting (spec section 10).

The question this module answers is deliberately narrow:

    How many examples cover each underlying situation type?

Not "how many examples are there". A dataset of 10,000 examples that all describe
one urgency level under one evidence-quality condition supports no claim about
generalization across urgency or evidence.

Coverage is reported at three levels:

1. **Marginal** — counts per value of each axis, independently.
2. **Joint** — counts per combination of a chosen set of axes (the "situation
   type"). This is where thin coverage actually shows up.
3. **Gaps** — combinations expected by the config that have zero or too few
   examples.

This module imports no torch.
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from kleos_models.constants import VARIATION_AXES
from kleos_models.data.schemas import TrainingExample
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)

#: Axes that define a "situation type" by default. Chosen because these are the
#: dimensions the KLEOS decision policies are supposed to be sensitive to.
DEFAULT_JOINT_AXES: tuple[str, ...] = ("domain", "urgency", "evidence_quality")

#: Below this many examples, a cell is reported as thin: present, but too sparse
#: to support a per-cell claim.
DEFAULT_MIN_CELL_COUNT = 3


@dataclass
class AxisCoverage:
    """Marginal coverage of one axis."""

    axis: str
    values: dict[str, int] = field(default_factory=dict)
    missing_count: int = 0

    @property
    def distinct_values(self) -> int:
        return len(self.values)

    @property
    def total(self) -> int:
        return sum(self.values.values())

    @property
    def is_constant(self) -> bool:
        """True when the axis takes exactly one value — no contrast available."""
        return self.distinct_values == 1

    @property
    def imbalance_ratio(self) -> float:
        """Largest value count divided by smallest. 1.0 is perfectly balanced."""
        if not self.values:
            return 0.0
        counts = list(self.values.values())
        smallest = min(counts)
        return max(counts) / smallest if smallest else float("inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "distinct_values": self.distinct_values,
            "values": self.values,
            "missing_count": self.missing_count,
            "is_constant": self.is_constant,
            "imbalance_ratio": round(self.imbalance_ratio, 2),
        }


@dataclass
class CoverageReport:
    """Full coverage picture for a dataset."""

    total_examples: int
    axes: dict[str, AxisCoverage] = field(default_factory=dict)
    joint_axes: list[str] = field(default_factory=list)
    joint_counts: dict[str, int] = field(default_factory=dict)
    empty_cells: list[str] = field(default_factory=list)
    thin_cells: dict[str, int] = field(default_factory=dict)
    min_cell_count: int = DEFAULT_MIN_CELL_COUNT

    @property
    def observed_cells(self) -> int:
        return len(self.joint_counts)

    @property
    def possible_cells(self) -> int:
        """Cartesian product size over observed values of the joint axes."""
        sizes = [
            max(self.axes[axis].distinct_values, 1) for axis in self.joint_axes if axis in self.axes
        ]
        if not sizes:
            return 0
        total = 1
        for size in sizes:
            total *= size
        return total

    @property
    def cell_fill_rate(self) -> float:
        """Fraction of possible situation types with at least one example."""
        possible = self.possible_cells
        return self.observed_cells / possible if possible else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_examples": self.total_examples,
            "joint_axes": self.joint_axes,
            "possible_cells": self.possible_cells,
            "observed_cells": self.observed_cells,
            "cell_fill_rate": round(self.cell_fill_rate, 4),
            "min_cell_count": self.min_cell_count,
            "empty_cells": self.empty_cells,
            "thin_cells": self.thin_cells,
            "joint_counts": self.joint_counts,
            "axes": {name: cov.to_dict() for name, cov in self.axes.items()},
        }


def _axis_value(example: TrainingExample, axis: str) -> str | None:
    """Read an axis value, falling back to task which lives on the example."""
    if axis == "task":
        return example.task
    value = example.variation_axes.as_dict().get(axis)
    return None if value is None else str(value)


def build_coverage_report(
    examples: Sequence[TrainingExample],
    *,
    joint_axes: Sequence[str] = DEFAULT_JOINT_AXES,
    min_cell_count: int = DEFAULT_MIN_CELL_COUNT,
    axes: Sequence[str] = VARIATION_AXES,
) -> CoverageReport:
    """Compute marginal and joint variation-axis coverage.

    Args:
        examples: Dataset to analyse.
        joint_axes: Axes whose combination defines a situation type.
        min_cell_count: Cells below this count are reported as thin.
        axes: Axes to compute marginal coverage for.
    """
    report = CoverageReport(
        total_examples=len(examples),
        joint_axes=list(joint_axes),
        min_cell_count=min_cell_count,
    )

    for axis in axes:
        counter: Counter[str] = Counter()
        missing = 0
        for example in examples:
            value = _axis_value(example, axis)
            if value is None:
                missing += 1
            else:
                counter[value] += 1
        if counter or missing:
            report.axes[axis] = AxisCoverage(
                axis=axis,
                values=dict(counter.most_common()),
                missing_count=missing,
            )

    # Joint coverage over the situation-defining axes.
    joint_counter: Counter[str] = Counter()
    for example in examples:
        parts = []
        for axis in joint_axes:
            value = _axis_value(example, axis)
            parts.append(f"{axis}={value if value is not None else '<missing>'}")
        joint_counter["|".join(parts)] += 1
    report.joint_counts = dict(joint_counter.most_common())

    # Which combinations of *observed* values never occur together.
    value_sets: list[list[str]] = []
    for axis in joint_axes:
        coverage = report.axes.get(axis)
        value_sets.append(
            sorted(coverage.values) if coverage and coverage.values else ["<missing>"]
        )

    for combination in itertools.product(*value_sets):
        key = "|".join(
            f"{axis}={value}" for axis, value in zip(joint_axes, combination, strict=True)
        )
        count = joint_counter.get(key, 0)
        if count == 0:
            report.empty_cells.append(key)
        elif count < min_cell_count:
            report.thin_cells[key] = count

    return report


def render_coverage_report(report: CoverageReport, *, max_rows: int = 25) -> str:
    """Render coverage as readable text."""
    lines = [
        "=" * 72,
        "Variation-axis coverage",
        "=" * 72,
        "",
        f"Examples          : {report.total_examples}",
        f"Situation axes    : {' × '.join(report.joint_axes)}",
        f"Situation types   : {report.observed_cells} observed of "
        f"{report.possible_cells} possible ({report.cell_fill_rate:.0%} filled)",
        "",
        "Marginal coverage",
    ]

    for axis, coverage in report.axes.items():
        flags = []
        if coverage.is_constant:
            flags.append("CONSTANT")
        if coverage.missing_count:
            flags.append(f"{coverage.missing_count} missing")
        if coverage.distinct_values > 1 and coverage.imbalance_ratio > 5:
            flags.append(f"imbalance {coverage.imbalance_ratio:.1f}×")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        preview = ", ".join(f"{k}={v}" for k, v in list(coverage.values.items())[:5])
        extra = f", +{coverage.distinct_values - 5} more" if coverage.distinct_values > 5 else ""
        lines.append(
            f"  {axis:<22} {coverage.distinct_values:>3} value(s): {preview}{extra}{suffix}"
        )

    lines.extend(["", f"Situation-type counts (top {max_rows})"])
    for key, count in list(report.joint_counts.items())[:max_rows]:
        marker = " ← thin" if count < report.min_cell_count else ""
        lines.append(f"  {count:>5}  {key}{marker}")
    if len(report.joint_counts) > max_rows:
        lines.append(f"  … {len(report.joint_counts) - max_rows} more combination(s)")

    if report.thin_cells:
        lines.extend(
            [
                "",
                f"Thin cells (< {report.min_cell_count} examples): {len(report.thin_cells)}",
                "  These situation types exist but are too sparse to support a per-cell claim.",
            ]
        )
    if report.empty_cells:
        lines.extend(
            [
                "",
                f"Empty cells: {len(report.empty_cells)} of {report.possible_cells}",
                "  Combinations of observed values with no examples at all:",
            ]
        )
        for key in report.empty_cells[:10]:
            lines.append(f"    {key}")
        if len(report.empty_cells) > 10:
            lines.append(f"    … {len(report.empty_cells) - 10} more")

    lines.extend(
        [
            "",
            "Reminder: a dataset is not diverse because it is large. Coverage, not",
            "count, is what supports a generalization claim.",
            "",
        ]
    )
    return "\n".join(lines)
