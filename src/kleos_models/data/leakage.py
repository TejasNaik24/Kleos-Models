"""Data leakage detection (spec section 11).

The failure this module exists to prevent: a model appears to generalize because
an evaluation example is a near-copy of something it trained on. That produces a
number that looks like a research result and is not one.

Detectors, cheapest first:

``exact``
    Byte-identical conversations.
``normalized``
    Identical after case-folding, whitespace collapse and punctuation stripping.
``near_duplicate``
    High character-n-gram Jaccard similarity, via MinHash + LSH banding so the
    cost stays near-linear instead of quadratic.
``id_collision``
    The same id in more than one split.
``scenario_repeat``
    The same scenario family on both sides of the split boundary.
``entity_leak``
    Entities held out of training that nevertheless appear in it.

Implemented in pure Python — no extra dependency for a check that must always be
runnable. Output is machine-readable JSON plus a Markdown summary.

This module imports no torch.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from kleos_models.data.schemas import EvaluationExample, TrainingExample
from kleos_models.errors import LeakageError
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)

#: Character n-gram width for near-duplicate shingling.
SHINGLE_SIZE = 5
#: Number of MinHash permutations. More = better recall, linearly more work.
MINHASH_PERMUTATIONS = 64
#: LSH bands. bands × rows must equal MINHASH_PERMUTATIONS.
LSH_BANDS = 16
#: Default Jaccard threshold above which two texts are "near duplicates".
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.85

_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^\w\s]")
_DIGITS = re.compile(r"\d+")

# 64-bit odd multipliers for the MinHash permutation family.
_MASK64 = (1 << 64) - 1
_PRIME = (1 << 61) - 1


class LeakageKind(str, Enum):
    """Categories of leakage finding."""

    EXACT_DUPLICATE = "exact_duplicate"
    NORMALIZED_DUPLICATE = "normalized_duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    ID_COLLISION = "id_collision"
    SCENARIO_REPEAT = "scenario_repeat"
    ENTITY_LEAK = "entity_leak"


#: Kinds that invalidate an evaluation outright, rather than merely warranting review.
FATAL_KINDS: frozenset[LeakageKind] = frozenset(
    {
        LeakageKind.EXACT_DUPLICATE,
        LeakageKind.NORMALIZED_DUPLICATE,
        LeakageKind.ID_COLLISION,
    }
)


class _HasText(Protocol):
    """Anything with an id and comparable conversation text."""

    id: str

    def conversation_text(self) -> str: ...


def normalize_text(text: str) -> str:
    """Aggressively normalize for duplicate detection.

    Case-folds, strips accents, removes punctuation, collapses whitespace and
    replaces digit runs with a placeholder — so "Deadline: Mar 3" and
    "deadline mar 7" normalize to the same string. That is intentional: changing
    only a number does not make a scenario new.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    lowered = stripped.casefold()
    no_digits = _DIGITS.sub("0", lowered)
    no_punct = _PUNCTUATION.sub(" ", no_digits)
    return _WHITESPACE.sub(" ", no_punct).strip()


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    """Character n-grams of normalized text."""
    normalized = normalize_text(text)
    if len(normalized) <= size:
        return {normalized} if normalized else set()
    return {normalized[i : i + size] for i in range(len(normalized) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    """Jaccard similarity of two shingle sets."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    return intersection / (len(left) + len(right) - intersection)


def _hash_shingle(shingle: str) -> int:
    """Stable 64-bit hash. Python's ``hash`` is salted per process, so not that."""
    import hashlib

    return int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")


def _permutations(count: int = MINHASH_PERMUTATIONS) -> list[tuple[int, int]]:
    """Deterministic (a, b) coefficients for the hash family h(x) = a*x + b."""
    coefficients: list[tuple[int, int]] = []
    state = 0x9E3779B97F4A7C15
    for _ in range(count):
        state = (state * 6364136223846793005 + 1442695040888963407) & _MASK64
        a = (state | 1) % _PRIME
        state = (state * 6364136223846793005 + 1442695040888963407) & _MASK64
        b = state % _PRIME
        coefficients.append((a, b))
    return coefficients


_PERMUTATION_COEFFICIENTS = _permutations()


def minhash_signature(shingle_set: set[str]) -> tuple[int, ...]:
    """MinHash signature approximating Jaccard similarity."""
    if not shingle_set:
        return tuple([0] * MINHASH_PERMUTATIONS)
    hashed = [_hash_shingle(s) % _PRIME for s in shingle_set]
    return tuple(
        min((a * value + b) % _PRIME for value in hashed) for a, b in _PERMUTATION_COEFFICIENTS
    )


def _lsh_buckets(signature: tuple[int, ...]) -> list[tuple[int, tuple[int, ...]]]:
    """Band a signature into LSH bucket keys."""
    rows = len(signature) // LSH_BANDS
    return [(band, signature[band * rows : (band + 1) * rows]) for band in range(LSH_BANDS)]


@dataclass
class LeakageFinding:
    """One detected leak."""

    kind: LeakageKind
    left_id: str
    right_id: str
    left_split: str
    right_split: str
    similarity: float = 1.0
    detail: str = ""

    @property
    def is_cross_split(self) -> bool:
        """Whether the pair straddles a split boundary. Those are the dangerous ones."""
        return self.left_split != self.right_split

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "left_id": self.left_id,
            "right_id": self.right_id,
            "left_split": self.left_split,
            "right_split": self.right_split,
            "similarity": round(self.similarity, 4),
            "cross_split": self.is_cross_split,
            "detail": self.detail,
        }


@dataclass
class LeakageReport:
    """Aggregated leakage findings."""

    findings: list[LeakageFinding] = field(default_factory=list)
    examined: dict[str, int] = field(default_factory=dict)
    threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def by_kind(self, kind: LeakageKind) -> list[LeakageFinding]:
        return [f for f in self.findings if f.kind is kind]

    @property
    def cross_split_findings(self) -> list[LeakageFinding]:
        return [f for f in self.findings if f.is_cross_split]

    @property
    def fatal_findings(self) -> list[LeakageFinding]:
        """Cross-split findings of a kind that invalidates the evaluation."""
        return [f for f in self.cross_split_findings if f.kind in FATAL_KINDS]

    @property
    def clean(self) -> bool:
        return not self.findings

    def counts(self) -> dict[str, int]:
        result = {kind.value: len(self.by_kind(kind)) for kind in LeakageKind}
        result["total"] = len(self.findings)
        result["cross_split"] = len(self.cross_split_findings)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "near_duplicate_threshold": self.threshold,
            "examined": self.examined,
            "counts": self.counts(),
            "clean": self.clean,
            "findings": [f.to_dict() for f in self.findings],
        }

    def render_markdown(self) -> str:
        """Human-readable summary for ``leakage_summary.md``."""
        counts = self.counts()
        lines = [
            "# Leakage report",
            "",
            f"Generated: {self.generated_at}",
            f"Near-duplicate threshold: {self.threshold}",
            "",
            "## Examined",
            "",
        ]
        lines.extend(f"- `{name}`: {count} example(s)" for name, count in self.examined.items())
        lines.extend(["", "## Summary", "", "| Kind | Count |", "| --- | ---: |"])
        for kind in LeakageKind:
            lines.append(f"| {kind.value} | {counts[kind.value]} |")
        lines.append(f"| **total** | **{counts['total']}** |")
        lines.append(f"| **cross-split** | **{counts['cross_split']}** |")
        lines.append("")

        if self.clean:
            lines.extend(["## Result", "", "No leakage detected.", ""])
            return "\n".join(lines)

        fatal = self.fatal_findings
        lines.extend(
            [
                "## Result",
                "",
                f"**{len(self.cross_split_findings)} cross-split finding(s)**, "
                f"of which **{len(fatal)}** are fatal "
                "(exact/normalized duplicates or id collisions across splits).",
                "",
                "A cross-split duplicate means an evaluation example is effectively",
                "present in training. Any generalization number computed against it",
                "is not measuring generalization.",
                "",
                "## Findings",
                "",
                "| Kind | Left | Right | Splits | Similarity |",
                "| --- | --- | --- | --- | ---: |",
            ]
        )
        for finding in sorted(self.findings, key=lambda f: (not f.is_cross_split, -f.similarity))[
            :200
        ]:
            lines.append(
                f"| {finding.kind.value} | `{finding.left_id}` | `{finding.right_id}` | "
                f"{finding.left_split} → {finding.right_split} | {finding.similarity:.3f} |"
            )
        if len(self.findings) > 200:
            lines.append(f"\n… {len(self.findings) - 200} further finding(s) in the JSON report.")
        lines.append("")
        return "\n".join(lines)

    def write(self, directory: Path | str) -> dict[str, Path]:
        """Write ``leakage_report.json`` and ``leakage_summary.md``."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        json_path = target / "leakage_report.json"
        md_path = target / "leakage_summary.md"
        json_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        md_path.write_text(self.render_markdown(), encoding="utf-8")
        logger.info("Wrote leakage report to %s and %s", json_path, md_path)
        return {"json": json_path, "markdown": md_path}


@dataclass
class _Item:
    """Internal record for one example under analysis."""

    id: str
    split: str
    text: str
    exact_hash: str
    normalized: str
    scenario_family: str | None = None
    entities: str | None = None
    _shingles: set[str] | None = None
    _signature: tuple[int, ...] | None = None

    @property
    def shingle_set(self) -> set[str]:
        if self._shingles is None:
            self._shingles = shingles(self.text)
        return self._shingles

    @property
    def signature(self) -> tuple[int, ...]:
        if self._signature is None:
            self._signature = minhash_signature(self.shingle_set)
        return self._signature


def _to_items(examples: Iterable[TrainingExample | EvaluationExample], split: str) -> list[_Item]:
    import hashlib

    items: list[_Item] = []
    for example in examples:
        text = example.conversation_text()
        items.append(
            _Item(
                id=example.id,
                split=split,
                text=text,
                exact_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                normalized=normalize_text(text),
                scenario_family=example.metadata.scenario_family,
                entities=example.variation_axes.as_dict().get("entities"),
            )
        )
    return items


def check_leakage(
    splits: dict[str, Sequence[TrainingExample | EvaluationExample]],
    *,
    threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    detect_near_duplicates: bool = True,
    within_split: bool = True,
) -> LeakageReport:
    """Run every leakage detector across the supplied splits.

    Args:
        splits: Mapping of split name to examples, e.g.
            ``{"train": [...], "test": [...]}``.
        threshold: Jaccard similarity at or above which a pair is a near duplicate.
        detect_near_duplicates: Run the MinHash stage. Disable for very large
            datasets where only exact checks are affordable.
        within_split: Also report duplicates inside a single split. Those are not
            leakage but they do inflate effective dataset size.

    Returns:
        A :class:`LeakageReport`.
    """
    items: list[_Item] = []
    examined: dict[str, int] = {}
    for split_name, examples in splits.items():
        split_items = _to_items(examples, split_name)
        items.extend(split_items)
        examined[split_name] = len(split_items)

    report = LeakageReport(examined=examined, threshold=threshold)
    if len(items) < 2:
        return report

    def record(
        kind: LeakageKind, left: _Item, right: _Item, similarity: float, detail: str = ""
    ) -> None:
        if not within_split and left.split == right.split:
            return
        report.findings.append(
            LeakageFinding(
                kind=kind,
                left_id=left.id,
                right_id=right.id,
                left_split=left.split,
                right_split=right.split,
                similarity=similarity,
                detail=detail,
            )
        )

    # --- id collisions ------------------------------------------------------
    by_id: dict[str, list[_Item]] = defaultdict(list)
    for item in items:
        by_id[item.id].append(item)
    for example_id, group in by_id.items():
        for index in range(1, len(group)):
            record(
                LeakageKind.ID_COLLISION,
                group[0],
                group[index],
                1.0,
                f"id {example_id!r} appears in multiple splits",
            )

    # --- exact duplicates ---------------------------------------------------
    by_hash: dict[str, list[_Item]] = defaultdict(list)
    for item in items:
        by_hash[item.exact_hash].append(item)
    exact_pairs: set[tuple[str, str]] = set()
    for group in by_hash.values():
        if len(group) < 2:
            continue
        for index in range(1, len(group)):
            record(LeakageKind.EXACT_DUPLICATE, group[0], group[index], 1.0)
            exact_pairs.add(tuple(sorted((group[0].id, group[index].id))))  # type: ignore[arg-type]

    # --- normalized duplicates ---------------------------------------------
    by_normalized: dict[str, list[_Item]] = defaultdict(list)
    for item in items:
        by_normalized[item.normalized].append(item)
    for group in by_normalized.values():
        if len(group) < 2:
            continue
        for index in range(1, len(group)):
            pair = tuple(sorted((group[0].id, group[index].id)))
            if pair in exact_pairs:
                continue  # already reported as an exact duplicate
            record(
                LeakageKind.NORMALIZED_DUPLICATE,
                group[0],
                group[index],
                1.0,
                "identical after case/punctuation/number normalization",
            )

    # --- near duplicates via MinHash + LSH ---------------------------------
    if detect_near_duplicates:
        buckets: dict[tuple[int, tuple[int, ...]], list[_Item]] = defaultdict(list)
        for item in items:
            for key in _lsh_buckets(item.signature):
                buckets[key].append(item)

        candidates: set[tuple[int, int]] = set()
        for bucket in buckets.values():
            if len(bucket) < 2 or len(bucket) > 200:
                # Huge buckets mean degenerate/empty text; comparing them all is
                # quadratic and uninformative.
                continue
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    candidates.add((id(bucket[i]), id(bucket[j])))

        by_identity = {id(item): item for item in items}
        reported: set[tuple[str, str]] = set(exact_pairs)
        for left_id, right_id in candidates:
            left, right = by_identity[left_id], by_identity[right_id]
            pair = tuple(sorted((left.id, right.id)))
            if pair in reported or left.id == right.id:
                continue
            similarity = jaccard(left.shingle_set, right.shingle_set)
            if similarity >= threshold:
                reported.add(pair)  # type: ignore[arg-type]
                record(
                    LeakageKind.NEAR_DUPLICATE,
                    left,
                    right,
                    similarity,
                    f"character-{SHINGLE_SIZE}-gram Jaccard >= {threshold}",
                )

    # --- scenario families straddling splits -------------------------------
    by_family: dict[str, dict[str, list[_Item]]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        if item.scenario_family:
            by_family[item.scenario_family][item.split].append(item)
    for family, per_split in by_family.items():
        if len(per_split) < 2:
            continue
        split_names = sorted(per_split)
        first = per_split[split_names[0]][0]
        for other in split_names[1:]:
            record(
                LeakageKind.SCENARIO_REPEAT,
                first,
                per_split[other][0],
                0.0,
                f"scenario_family {family!r} spans splits {split_names}",
            )

    # --- entities marked unseen that were actually trained on --------------
    train_entities = {
        item.entities
        for item in items
        if item.split == "train" and item.entities and item.entities != "unseen"
    }
    for item in items:
        if item.split == "train" or not item.entities:
            continue
        if item.entities == "unseen":
            continue
        if item.entities in train_entities:
            match = next(
                (t for t in items if t.split == "train" and t.entities == item.entities), None
            )
            if match is not None:
                record(
                    LeakageKind.ENTITY_LEAK,
                    match,
                    item,
                    0.0,
                    f"entity group {item.entities!r} occurs in both train and {item.split}",
                )

    logger.info("Leakage scan complete: %s", report.counts())
    return report


def enforce_leakage_policy(report: LeakageReport, *, fail_on: str = "fatal") -> None:
    """Raise when a report violates the configured policy.

    Args:
        report: The scan result.
        fail_on: ``"none"`` never raises, ``"fatal"`` raises on cross-split exact
            or normalized duplicates and id collisions, ``"any"`` raises on any
            cross-split finding at all.

    Raises:
        LeakageError: when the policy is violated.
    """
    if fail_on == "none":
        return

    offending = report.fatal_findings if fail_on == "fatal" else report.cross_split_findings
    if not offending:
        return

    preview = json.dumps([f.to_dict() for f in offending[:5]], indent=2)
    raise LeakageError(
        f"{len(offending)} cross-split leakage finding(s) violate the '{fail_on}' policy.",
        details={"counts": json.dumps(report.counts()), "first_findings": preview},
        suggestions=[
            "Deduplicate the dataset before splitting.",
            "Use a held-out split strategy (entity_holdout, scenario_family_holdout) "
            "so related examples cannot straddle the boundary.",
            "Inspect the full report: reports/leakage/leakage_summary.md",
            "Override deliberately with --leakage-policy none, and record that you did.",
        ],
    )
