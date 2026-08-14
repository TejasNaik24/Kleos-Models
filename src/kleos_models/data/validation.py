"""Dataset validation and quality reporting (spec sections 25 and 30).

Schema validation lives in :mod:`kleos_models.data.schemas`. This module adds the
checks that need the *whole dataset* in view: duplicate ids, coverage gaps,
missing metadata, suspicious content, and the aggregate quality report.

This module imports no torch. Token statistics degrade to a whitespace-word
approximation when no tokenizer is supplied, so dataset work never requires a GPU
environment.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kleos_models.constants import (
    DATASET_SCHEMA_VERSION,
    REQUIRED_VARIATION_AXES,
    VARIATION_AXES,
)
from kleos_models.data.schemas import TrainingExample
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)


class Severity(str, Enum):
    """How much a finding matters."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Finding:
    """One validation result."""

    severity: Severity
    code: str
    message: str
    example_ids: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "example_ids": self.example_ids[:50],
            "affected_count": len(self.example_ids),
            "context": self.context,
        }


@dataclass
class ValidationReport:
    """Aggregated findings for a dataset."""

    findings: list[Finding] = field(default_factory=list)
    examples_checked: int = 0

    def add(
        self,
        severity: Severity,
        code: str,
        message: str,
        *,
        example_ids: Sequence[str] = (),
        **context: Any,
    ) -> None:
        self.findings.append(
            Finding(
                severity=severity,
                code=code,
                message=message,
                example_ids=list(example_ids),
                context=context,
            )
        )

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        """True when nothing at ERROR severity was found."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "examples_checked": self.examples_checked,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "findings": [f.to_dict() for f in self.findings],
        }

    def render(self) -> str:
        """Human-readable summary."""
        if not self.findings:
            return f"✓ {self.examples_checked} example(s) validated, no findings."
        lines = [f"Validation of {self.examples_checked} example(s):"]
        icons = {Severity.ERROR: "✗", Severity.WARNING: "!", Severity.INFO: "·"}
        for finding in sorted(self.findings, key=lambda f: list(Severity).index(f.severity)):
            lines.append(f"  {icons[finding.severity]} [{finding.code}] {finding.message}")
            if finding.example_ids:
                shown = ", ".join(finding.example_ids[:5])
                more = (
                    f" (+{len(finding.example_ids) - 5} more)"
                    if len(finding.example_ids) > 5
                    else ""
                )
                lines.append(f"      affected: {shown}{more}")
        lines.append("")
        lines.append(f"  {len(self.errors)} error(s), {len(self.warnings)} warning(s)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Content heuristics
# ---------------------------------------------------------------------------

#: Patterns suggesting private data reached a supposedly public dataset.
#: Deliberately conservative — false positives are cheap, a leaked secret is not.
_SENSITIVE_PATTERNS: dict[str, re.Pattern[str]] = {
    "email_address": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
    "us_phone": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    "bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{20,}"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "supabase_url": re.compile(r"https://[a-z0-9]{20}\.supabase\.co"),
    "private_key_block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}

#: Placeholder text that should never survive into a real dataset.
_PLACEHOLDER_PATTERNS = (
    re.compile(r"\bTODO\b"),
    re.compile(r"\bFIXME\b"),
    re.compile(r"\blorem ipsum\b", re.IGNORECASE),
    re.compile(r"^\s*\.\.\.\s*$", re.MULTILINE),
)


def scan_sensitive_content(text: str) -> list[str]:
    """Return the names of sensitive patterns matching the text."""
    return [name for name, pattern in _SENSITIVE_PATTERNS.items() if pattern.search(text)]


def approximate_token_count(text: str) -> int:
    """Rough token estimate without a tokenizer.

    Uses ~1.3 tokens per whitespace word, which is close enough for dataset
    triage. Real token statistics come from ``--tokenizer`` when available.
    """
    words = len(text.split())
    return int(words * 1.3) + 1


# ---------------------------------------------------------------------------
# Dataset-level validation
# ---------------------------------------------------------------------------


def validate_examples(
    examples: Sequence[TrainingExample],
    *,
    split_name: str = "dataset",
    max_tokens: int | None = None,
    token_counter: Callable[[str], int] | None = None,
    require_reviewed: bool = False,
    scan_content: bool = True,
) -> ValidationReport:
    """Run whole-dataset checks over already schema-valid examples.

    Args:
        examples: Parsed examples.
        split_name: Label used in messages.
        max_tokens: Flag examples longer than this.
        token_counter: Real tokenizer callable; defaults to the approximation.
        require_reviewed: Treat non-reviewed examples as errors.
        scan_content: Run the sensitive-content heuristics.
    """
    report = ValidationReport(examples_checked=len(examples))
    count_tokens = token_counter or approximate_token_count

    if not examples:
        report.add(Severity.ERROR, "empty_dataset", f"{split_name} contains no examples")
        return report

    # --- duplicate ids -----------------------------------------------------
    id_counts = Counter(e.id for e in examples)
    duplicates = [example_id for example_id, count in id_counts.items() if count > 1]
    if duplicates:
        report.add(
            Severity.ERROR,
            "duplicate_ids",
            f"{len(duplicates)} example id(s) appear more than once in {split_name}",
            example_ids=duplicates,
        )

    # --- schema version drift ----------------------------------------------
    versions = Counter(e.version for e in examples)
    if len(versions) > 1:
        report.add(
            Severity.WARNING,
            "mixed_schema_versions",
            f"{split_name} mixes schema versions: {dict(versions)}",
            versions=dict(versions),
        )
    unknown_versions = set(versions) - {DATASET_SCHEMA_VERSION}
    if unknown_versions:
        report.add(
            Severity.WARNING,
            "unexpected_schema_version",
            f"examples declare schema version(s) {sorted(unknown_versions)}, "
            f"current is {DATASET_SCHEMA_VERSION}",
        )

    # --- required variation axes -------------------------------------------
    for axis in REQUIRED_VARIATION_AXES:
        missing = [e.id for e in examples if not e.variation_axes.as_dict().get(axis)]
        if missing:
            report.add(
                Severity.ERROR,
                "missing_required_axis",
                f"{len(missing)} example(s) missing required variation axis {axis!r}",
                example_ids=missing,
            )

    # --- unregistered axes --------------------------------------------------
    unknown_axes: Counter[str] = Counter()
    for example in examples:
        unknown_axes.update(example.variation_axes.unknown_axes().keys())
    if unknown_axes:
        report.add(
            Severity.INFO,
            "unregistered_variation_axes",
            f"axes not in the registry: {sorted(unknown_axes)}",
            axes=dict(unknown_axes),
            hint="Promote them in kleos_models/constants.py once they are stable.",
        )

    # --- axis sparsity ------------------------------------------------------
    # A dataset is not diverse because it is large. Warn when an axis is
    # effectively constant, because it cannot support a generalization claim.
    for axis in VARIATION_AXES:
        values = [
            str(e.variation_axes.as_dict().get(axis))
            for e in examples
            if e.variation_axes.as_dict().get(axis) is not None
        ]
        if len(values) >= max(4, len(examples) // 2) and len(set(values)) == 1:
            report.add(
                Severity.WARNING,
                "constant_variation_axis",
                f"axis {axis!r} has a single value {values[0]!r} across "
                f"{len(values)} example(s); it cannot support a generalization claim",
                axis=axis,
                value=values[0],
            )

    # --- quality status -----------------------------------------------------
    status_counts = Counter(e.metadata.quality_status for e in examples)
    unreviewed = [e.id for e in examples if e.metadata.quality_status != "reviewed"]
    if unreviewed:
        report.add(
            Severity.ERROR if require_reviewed else Severity.WARNING,
            "unreviewed_examples",
            f"{len(unreviewed)} of {len(examples)} example(s) are not quality_status='reviewed'",
            example_ids=unreviewed,
            distribution=dict(status_counts),
        )
    rejected = [e.id for e in examples if e.metadata.quality_status == "rejected"]
    if rejected:
        report.add(
            Severity.ERROR,
            "rejected_examples_present",
            f"{len(rejected)} example(s) are marked 'rejected' but still in {split_name}",
            example_ids=rejected,
        )

    # --- length -------------------------------------------------------------
    if max_tokens is not None:
        too_long = [e.id for e in examples if count_tokens(e.conversation_text()) > max_tokens]
        if too_long:
            report.add(
                Severity.WARNING,
                "examples_exceed_max_tokens",
                f"{len(too_long)} example(s) exceed max_seq_length={max_tokens} and "
                "will be truncated, which can cut off the assistant target",
                example_ids=too_long,
                max_tokens=max_tokens,
            )

    # --- empty or placeholder targets ---------------------------------------
    placeholder_ids: list[str] = []
    for example in examples:
        for target in example.assistant_targets:
            if any(pattern.search(target) for pattern in _PLACEHOLDER_PATTERNS):
                placeholder_ids.append(example.id)
                break
    if placeholder_ids:
        report.add(
            Severity.ERROR,
            "placeholder_content",
            f"{len(placeholder_ids)} example(s) contain TODO/FIXME/lorem-ipsum placeholder text",
            example_ids=placeholder_ids,
        )

    # --- sensitive content --------------------------------------------------
    if scan_content:
        hits: dict[str, list[str]] = {}
        for example in examples:
            matched = scan_sensitive_content(example.conversation_text())
            for name in matched:
                hits.setdefault(name, []).append(example.id)
        for pattern_name, ids in hits.items():
            report.add(
                Severity.ERROR,
                "possible_sensitive_content",
                f"{len(ids)} example(s) match the {pattern_name!r} pattern; "
                "this repository is public and must not contain private data",
                example_ids=ids,
                pattern=pattern_name,
            )

    # --- policy-vs-fact heuristic -------------------------------------------
    report.findings.extend(_check_memorization_risk(examples).findings)

    return report


#: Phrasings that suggest an example teaches a private fact rather than a policy
#: (spec section 9). Heuristic and advisory: it flags for human review.
_FACT_TEACHING_HINTS = (
    re.compile(r"\bbecause (?:he|she|they|I) (?:work|works|worked) at\b", re.IGNORECASE),
    re.compile(r"\byour (?:resume|CV) (?:says|states|lists)\b", re.IGNORECASE),
    re.compile(r"\bas we discussed (?:last|on) \w+\b", re.IGNORECASE),
)


def _check_memorization_risk(examples: Sequence[TrainingExample]) -> ValidationReport:
    """Flag examples that look like they teach a fact rather than a policy.

    Spec section 9: the model should learn *prioritize the closest deadline with
    the strongest evidence*, not *this person works at company X*. Private facts
    belong in retrieval and memory, not in weights.
    """
    report = ValidationReport()
    flagged: list[str] = []
    for example in examples:
        for target in example.assistant_targets:
            if any(pattern.search(target) for pattern in _FACT_TEACHING_HINTS):
                flagged.append(example.id)
                break
    if flagged:
        report.add(
            Severity.WARNING,
            "possible_fact_memorization",
            f"{len(flagged)} example(s) phrase the answer as a specific fact about a "
            "person rather than a generalizable decision policy",
            example_ids=flagged,
            guidance=(
                "Rewrite so the assistant justifies the decision from the supplied "
                "evidence, not from a remembered private fact. See docs/data-contract.md."
            ),
        )
    return report


# ---------------------------------------------------------------------------
# Quality report (spec section 25)
# ---------------------------------------------------------------------------


@dataclass
class QualityReport:
    """Descriptive statistics for a dataset."""

    version: str
    counts: dict[str, int]
    total_examples: int
    task_distribution: dict[str, int]
    domain_distribution: dict[str, int]
    source_distribution: dict[str, int]
    quality_distribution: dict[str, int]
    token_stats: dict[str, float]
    message_stats: dict[str, float]
    missing_metadata: dict[str, int]
    axis_value_counts: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.version,
            "split_counts": self.counts,
            "total_examples": self.total_examples,
            "task_distribution": self.task_distribution,
            "domain_distribution": self.domain_distribution,
            "source_distribution": self.source_distribution,
            "quality_distribution": self.quality_distribution,
            "token_stats": self.token_stats,
            "message_stats": self.message_stats,
            "missing_metadata": self.missing_metadata,
            "axis_value_counts": self.axis_value_counts,
        }


def build_quality_report(
    examples: Sequence[TrainingExample],
    *,
    version: str = "unversioned",
    counts: dict[str, int] | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> QualityReport:
    """Compute descriptive dataset statistics."""
    count_tokens = token_counter or approximate_token_count
    token_lengths = [count_tokens(e.conversation_text()) for e in examples] or [0]
    message_counts = [len(e.messages) for e in examples] or [0]

    missing: dict[str, int] = {}
    for axis in VARIATION_AXES:
        absent = sum(1 for e in examples if e.variation_axes.as_dict().get(axis) is None)
        if absent:
            missing[axis] = absent
    if any(e.metadata.scenario_family is None for e in examples):
        missing["scenario_family"] = sum(1 for e in examples if e.metadata.scenario_family is None)

    axis_values: dict[str, dict[str, int]] = {}
    for axis in VARIATION_AXES:
        values = Counter(
            str(e.variation_axes.as_dict()[axis])
            for e in examples
            if e.variation_axes.as_dict().get(axis) is not None
        )
        if values:
            axis_values[axis] = dict(values.most_common())

    return QualityReport(
        version=version,
        counts=counts or {"all": len(examples)},
        total_examples=len(examples),
        task_distribution=dict(Counter(e.task for e in examples).most_common()),
        domain_distribution=dict(Counter(e.variation_axes.domain for e in examples).most_common()),
        source_distribution=dict(Counter(e.metadata.source for e in examples).most_common()),
        quality_distribution=dict(
            Counter(e.metadata.quality_status for e in examples).most_common()
        ),
        token_stats={
            "mean": round(statistics.fmean(token_lengths), 1),
            "median": float(statistics.median(token_lengths)),
            "min": float(min(token_lengths)),
            "max": float(max(token_lengths)),
            "p95": float(sorted(token_lengths)[int(len(token_lengths) * 0.95) - 1])
            if len(token_lengths) >= 20
            else float(max(token_lengths)),
        },
        message_stats={
            "mean": round(statistics.fmean(message_counts), 2),
            "min": float(min(message_counts)),
            "max": float(max(message_counts)),
        },
        missing_metadata=missing,
        axis_value_counts=axis_values,
    )


def render_quality_report(report: QualityReport) -> str:
    """Render the quality report as readable text."""
    lines = [
        "=" * 72,
        f"Dataset quality report — {report.version}",
        "=" * 72,
        "",
        f"Total examples : {report.total_examples}",
        f"Splits         : {', '.join(f'{k}={v}' for k, v in report.counts.items())}",
        "",
        "Tokens (approximate unless --tokenizer was supplied)",
        f"  mean={report.token_stats['mean']}  median={report.token_stats['median']}  "
        f"max={report.token_stats['max']}  p95={report.token_stats['p95']}",
        "",
        f"Messages per example: mean={report.message_stats['mean']} "
        f"min={report.message_stats['min']} max={report.message_stats['max']}",
        "",
    ]

    def section(title: str, distribution: dict[str, int]) -> None:
        lines.append(title)
        if not distribution:
            lines.append("  (none)")
        for key, value in distribution.items():
            share = 100.0 * value / max(report.total_examples, 1)
            lines.append(f"  {key:<34} {value:>6}  ({share:5.1f}%)")
        lines.append("")

    section("Task distribution", report.task_distribution)
    section("Domain distribution", report.domain_distribution)
    section("Source distribution", report.source_distribution)
    section("Quality status", report.quality_distribution)

    lines.append("Variation-axis coverage")
    if not report.axis_value_counts:
        lines.append("  (no axes populated)")
    for axis, values in report.axis_value_counts.items():
        rendered = ", ".join(f"{k}={v}" for k, v in list(values.items())[:6])
        suffix = f", +{len(values) - 6} more" if len(values) > 6 else ""
        lines.append(f"  {axis:<22} {len(values):>3} value(s): {rendered}{suffix}")
    lines.append("")

    if report.missing_metadata:
        lines.append("Missing metadata (examples lacking each field)")
        for key, value in sorted(report.missing_metadata.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {key:<34} {value:>6}")
        lines.append("")

    return "\n".join(lines)
