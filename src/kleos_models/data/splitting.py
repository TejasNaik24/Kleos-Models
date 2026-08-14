"""Train/validation/test splitting (spec section 12).

Random splitting is the default only for development. A random split of a dataset
containing paraphrases and reordered variants of the same scenario puts near-copies
on both sides of the boundary, and the resulting "generalization" number measures
nothing.

The held-out strategies exist to make generalization claims checkable:

``random``
    Development only.
``group``
    Related examples stay together via an explicit grouping key.
``entity_holdout``
    Test contains entities never seen in training — does behaviour transfer?
``domain_holdout``
    Test contains unseen domains.
``format_holdout``
    Train on one presentation format, test on another.
``scenario_family_holdout``
    Logically related scenarios never straddle the split.

Every strategy is deterministic given a seed, and every result is checked for
overlap before being returned.

This module imports no torch.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from kleos_models.config import SplitConfig
from kleos_models.data.schemas import TrainingExample
from kleos_models.errors import DatasetIntegrityError
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SplitResult:
    """The outcome of a split, with enough detail to reproduce and audit it."""

    train: list[TrainingExample] = field(default_factory=list)
    validation: list[TrainingExample] = field(default_factory=list)
    test: list[TrainingExample] = field(default_factory=list)
    strategy: str = "random"
    seed: int = 42
    holdout_values: list[str] = field(default_factory=list)
    group_key: str | None = None
    notes: list[str] = field(default_factory=list)

    def split(self, name: str) -> list[TrainingExample]:
        return {"train": self.train, "validation": self.validation, "test": self.test}[name]

    @property
    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "counts": self.counts,
            "total": self.total,
            "holdout_values": self.holdout_values,
            "group_key": self.group_key,
            "notes": self.notes,
        }


def _stable_rank(key: str, seed: int) -> float:
    """Deterministic pseudo-random value in [0, 1) for a key.

    Hashing rather than shuffling means an example's assignment depends only on
    its key and the seed — adding new examples does not reshuffle existing ones.
    That property matters when a dataset grows between runs.
    """
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _assign_by_fraction(
    keys: Sequence[str],
    config: SplitConfig,
) -> dict[str, str]:
    """Map each key to a split name using stable hashing."""
    assignment: dict[str, str] = {}
    train_cut = config.train_fraction
    val_cut = train_cut + config.validation_fraction
    for key in keys:
        rank = _stable_rank(key, config.seed)
        if rank < train_cut:
            assignment[key] = "train"
        elif rank < val_cut:
            assignment[key] = "validation"
        else:
            assignment[key] = "test"
    return assignment


def _group_examples(
    examples: Sequence[TrainingExample], group_key: str | None
) -> dict[str, list[TrainingExample]]:
    """Bucket examples by their resolved group key."""
    groups: dict[str, list[TrainingExample]] = defaultdict(list)
    for example in examples:
        groups[example.group_key(group_key)].append(example)
    return dict(groups)


def _holdout_attribute(example: TrainingExample, attribute: str) -> str:
    """Read the attribute a *_holdout strategy partitions on."""
    if attribute == "domain":
        return example.variation_axes.domain
    if attribute == "entities":
        return str(example.variation_axes.entities or "<unspecified>")
    if attribute == "format":
        return str(example.variation_axes.format or "<unspecified>")
    value = example.variation_axes.as_dict().get(attribute)
    return str(value) if value is not None else "<unspecified>"


def _split_random(examples: Sequence[TrainingExample], config: SplitConfig) -> SplitResult:
    """Shuffle and slice. Development only."""
    ordered = sorted(examples, key=lambda e: e.id)
    rng = random.Random(config.seed)
    rng.shuffle(ordered)

    total = len(ordered)
    train_end = int(total * config.train_fraction)
    val_end = train_end + int(total * config.validation_fraction)

    result = SplitResult(
        train=ordered[:train_end],
        validation=ordered[train_end:val_end],
        test=ordered[val_end:],
        strategy="random",
        seed=config.seed,
    )
    result.notes.append(
        "Random split: development use only. Near-duplicate scenarios can land on "
        "both sides, which inflates apparent generalization."
    )
    return result


def _split_grouped(
    examples: Sequence[TrainingExample], config: SplitConfig, *, strategy: str
) -> SplitResult:
    """Assign whole groups to splits so related examples never straddle."""
    key = config.group_key
    if strategy == "scenario_family_holdout" and key is None:
        key = "scenario_family"

    groups = _group_examples(examples, key)
    assignment = _assign_by_fraction(sorted(groups), config)

    result = SplitResult(strategy=strategy, seed=config.seed, group_key=key)
    for group_name, members in sorted(groups.items()):
        target = assignment[group_name]
        result.split(target).extend(sorted(members, key=lambda e: e.id))

    singletons = sum(1 for members in groups.values() if len(members) == 1)
    result.notes.append(
        f"Grouped on {key or 'group_id|scenario_family|id'}: {len(groups)} group(s), "
        f"{singletons} of size 1."
    )
    if singletons == len(groups):
        result.notes.append(
            "WARNING: every group has exactly one example, so this behaves like a "
            "random split. Populate metadata.scenario_family or metadata.group_id "
            "for the grouping to have any effect."
        )
    return result


def _split_holdout(
    examples: Sequence[TrainingExample],
    config: SplitConfig,
    *,
    attribute: str,
    strategy: str,
) -> SplitResult:
    """Hold out whole attribute values for the test split.

    Validation is carved out of the *remaining* (seen-attribute) examples, so
    validation stays in-distribution while test is genuinely out-of-distribution.
    That asymmetry is intentional: early stopping on OOD data would leak the very
    thing we are trying to measure.
    """
    by_value: dict[str, list[TrainingExample]] = defaultdict(list)
    for example in examples:
        by_value[_holdout_attribute(example, attribute)].append(example)

    values = sorted(by_value)
    if len(values) < 2:
        raise DatasetIntegrityError(
            f"Cannot apply {strategy!r}: attribute {attribute!r} has only "
            f"{len(values)} distinct value(s).",
            details={"attribute": attribute, "values": values},
            suggestions=[
                f"A held-out split needs at least two distinct {attribute} values.",
                "Add examples covering another value, or use strategy='group'.",
                "Check coverage: python scripts/inspect_dataset.py --dataset <dir>",
            ],
        )

    if config.holdout_values:
        holdout = [v for v in config.holdout_values if v in by_value]
        unknown = sorted(set(config.holdout_values) - set(by_value))
        if unknown:
            raise DatasetIntegrityError(
                f"Configured holdout value(s) {unknown} do not occur in the dataset.",
                details={"attribute": attribute, "available_values": values},
                suggestions=["Fix split.holdout_values, or leave it empty to choose by seed."],
            )
    else:
        # Choose deterministically: take values in stable-hash order until the
        # test split is at least the configured fraction.
        ranked = sorted(values, key=lambda v: _stable_rank(v, config.seed))
        target = max(1, int(len(examples) * config.test_fraction))
        holdout = []
        accumulated = 0
        for value in ranked:
            if accumulated >= target and holdout:
                break
            holdout.append(value)
            accumulated += len(by_value[value])

    if len(holdout) >= len(values):
        raise DatasetIntegrityError(
            f"Holding out {holdout} would leave no training data for {attribute!r}.",
            details={"attribute": attribute, "all_values": values},
            suggestions=[
                "Reduce split.test_fraction, or hold out fewer values explicitly.",
            ],
        )

    test: list[TrainingExample] = []
    remaining: list[TrainingExample] = []
    for value, members in sorted(by_value.items()):
        (test if value in holdout else remaining).extend(members)

    # Split the remainder into train/validation, keeping groups intact.
    remainder_fraction = config.train_fraction + config.validation_fraction
    inner = SplitConfig(
        strategy="group",
        seed=config.seed,
        train_fraction=config.train_fraction / remainder_fraction,
        validation_fraction=config.validation_fraction / remainder_fraction,
        test_fraction=0.0,
        group_key=config.group_key,
    )
    groups = _group_examples(remaining, inner.group_key)
    assignment = _assign_by_fraction(sorted(groups), inner)

    result = SplitResult(
        strategy=strategy,
        seed=config.seed,
        holdout_values=sorted(holdout),
        group_key=config.group_key,
        test=sorted(test, key=lambda e: e.id),
    )
    for group_name, members in sorted(groups.items()):
        target_split = assignment[group_name]
        # test_fraction is 0 here, but stable hashing can still round into it.
        if target_split == "test":
            target_split = "train"
        result.split(target_split).extend(sorted(members, key=lambda e: e.id))

    result.notes.append(
        f"Held out {attribute} value(s) {sorted(holdout)} for the test split; "
        f"validation is drawn from seen values so early stopping does not leak OOD signal."
    )
    return result


#: Strategy name → implementation.
_STRATEGIES = {
    "random": lambda ex, cfg: _split_random(ex, cfg),
    "group": lambda ex, cfg: _split_grouped(ex, cfg, strategy="group"),
    "scenario_family_holdout": lambda ex, cfg: _split_grouped(
        ex, cfg, strategy="scenario_family_holdout"
    ),
    "entity_holdout": lambda ex, cfg: _split_holdout(
        ex, cfg, attribute="entities", strategy="entity_holdout"
    ),
    "domain_holdout": lambda ex, cfg: _split_holdout(
        ex, cfg, attribute="domain", strategy="domain_holdout"
    ),
    "format_holdout": lambda ex, cfg: _split_holdout(
        ex, cfg, attribute="format", strategy="format_holdout"
    ),
}


def split_examples(
    examples: Sequence[TrainingExample],
    config: SplitConfig,
    *,
    verify: bool = True,
) -> SplitResult:
    """Partition examples according to the configured strategy.

    Args:
        examples: Examples to split.
        config: Split strategy, seed and fractions.
        verify: Run the post-hoc overlap and determinism assertions.

    Raises:
        DatasetIntegrityError: when the split is impossible or produces overlap.
    """
    if not examples:
        raise DatasetIntegrityError(
            "Cannot split an empty dataset.",
            suggestions=["Check dataset filters; quality_status defaults to 'reviewed' only."],
        )

    implementation = _STRATEGIES.get(config.strategy)
    if implementation is None:  # pragma: no cover - guarded by config validation
        raise DatasetIntegrityError(
            f"Unknown split strategy {config.strategy!r}.",
            details={"valid": sorted(_STRATEGIES)},
        )

    result = implementation(list(examples), config)

    if verify:
        verify_split(result, expected_total=len(examples))

    logger.info(
        "Split %d example(s) with strategy=%s seed=%d → %s",
        len(examples),
        result.strategy,
        result.seed,
        result.counts,
    )
    for note in result.notes:
        logger.info("  note: %s", note)
    return result


def verify_split(result: SplitResult, *, expected_total: int | None = None) -> None:
    """Assert a split is well-formed.

    Checks that no example id appears in two splits, that nothing was lost, and
    that grouped strategies really kept groups together.

    Raises:
        DatasetIntegrityError: on any violation.
    """
    ids = {name: {e.id for e in result.split(name)} for name in ("train", "validation", "test")}

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = ids[left] & ids[right]
        if overlap:
            raise DatasetIntegrityError(
                f"Split overlap: {len(overlap)} example(s) appear in both {left} and {right}.",
                details={"overlapping_ids": sorted(overlap)[:20]},
                suggestions=[
                    "This is a bug in the split strategy, not a data problem.",
                    "Please report it with the dataset version and split config.",
                ],
            )

    if expected_total is not None and result.total != expected_total:
        raise DatasetIntegrityError(
            f"Split lost or duplicated examples: expected {expected_total}, got {result.total}.",
            details=dict(result.counts),
        )

    if result.group_key or result.strategy in ("group", "scenario_family_holdout"):
        placement: dict[str, str] = {}
        for split_name in ("train", "validation", "test"):
            for example in result.split(split_name):
                group = example.group_key(result.group_key)
                previous = placement.get(group)
                if previous is not None and previous != split_name:
                    raise DatasetIntegrityError(
                        f"Group {group!r} was split across {previous} and {split_name}.",
                        details={"group_key": result.group_key or "(default)"},
                        suggestions=["This is a bug in the grouped split implementation."],
                    )
                placement[group] = split_name

    if not result.train:
        raise DatasetIntegrityError(
            "Split produced an empty training set.",
            details=dict(result.counts),
            suggestions=[
                "Increase split.train_fraction, or hold out fewer values.",
            ],
        )
    if not result.test:
        logger.warning(
            "Split produced an empty test set (%s). No generalization claim can be "
            "made from this split.",
            result.counts,
        )
