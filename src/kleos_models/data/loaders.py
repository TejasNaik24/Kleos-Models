"""Dataset loading (spec sections 27 and 50).

The loader consumes a directory shaped like::

    dataset/
      manifest.json
      train.jsonl
      validation.jsonl
      test.jsonl

That directory may live anywhere on disk. This is the boundary with the private
``kleos-training-data`` repository: it produces the artifact, this repository
consumes it, and neither imports the other. Nothing here touches Supabase or any
production service.

This module imports no torch.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pydantic import ValidationError

from kleos_models.config import DatasetConfig
from kleos_models.constants import DATASET_MANIFEST_FILENAME
from kleos_models.data.schemas import (
    DatasetManifest,
    EvaluationExample,
    TrainingExample,
)
from kleos_models.errors import DatasetIntegrityError, DataValidationError
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)

ExampleT = TypeVar("ExampleT", TrainingExample, EvaluationExample)


@dataclass
class LoadReport:
    """What happened while reading a file."""

    path: Path
    total_lines: int = 0
    parsed: int = 0
    skipped_blank: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return len(self.errors)

    def summary(self) -> str:
        return (
            f"{self.path.name}: {self.parsed} parsed, {self.failed} invalid, "
            f"{self.skipped_blank} blank"
        )


def iter_jsonl(path: Path | str) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, record)`` from a JSONL file.

    Raises:
        DataValidationError: on malformed JSON, naming the exact line.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise DataValidationError(
            f"Dataset file not found: {file_path}",
            details={"path": str(file_path)},
            suggestions=[
                "Check the --dataset path points at a directory containing *.jsonl files.",
                "Generate development fixtures: ls data/examples",
            ],
        )
    with file_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DataValidationError(
                    f"Invalid JSON in {file_path.name} at line {line_number}.",
                    details={"path": str(file_path), "line": line_number, "error": str(exc)},
                    suggestions=[
                        "JSONL requires exactly one complete JSON object per line.",
                        "A pretty-printed JSON array is not valid JSONL.",
                    ],
                ) from exc
            if not isinstance(record, dict):
                raise DataValidationError(
                    f"Line {line_number} of {file_path.name} is not a JSON object.",
                    details={"found_type": type(record).__name__},
                )
            yield line_number, record


def load_examples(
    path: Path | str,
    *,
    model: type[ExampleT] = TrainingExample,  # type: ignore[assignment]
    strict: bool = True,
) -> tuple[list[ExampleT], LoadReport]:
    """Parse and validate a JSONL file into typed examples.

    Args:
        path: File to read.
        model: ``TrainingExample`` or ``EvaluationExample``.
        strict: Raise on the first invalid example. When false, invalid rows are
            collected in the report and skipped — useful for triaging a new
            dataset drop, never for a research run.

    Returns:
        ``(examples, report)``.
    """
    file_path = Path(path)
    report = LoadReport(path=file_path)
    examples: list[ExampleT] = []

    for line_number, record in iter_jsonl(file_path):
        report.total_lines += 1
        try:
            examples.append(model.model_validate(record))
            report.parsed += 1
        except ValidationError as exc:
            problem = {
                "line": line_number,
                "id": record.get("id", "<missing id>"),
                "errors": [
                    {
                        "field": ".".join(str(p) for p in err["loc"]),
                        "message": err["msg"],
                    }
                    for err in exc.errors()
                ],
            }
            report.errors.append(problem)
            if strict:
                raise DataValidationError(
                    f"Example on line {line_number} of {file_path.name} failed validation.",
                    details={
                        "example_id": problem["id"],
                        "problems": json.dumps(problem["errors"], indent=2),
                    },
                    suggestions=[
                        "Run: python scripts/validate_dataset.py --dataset <dir> --no-strict",
                        "That reports every invalid example at once instead of the first.",
                        "See docs/data-contract.md for the required fields.",
                    ],
                ) from exc

    logger.debug("%s", report.summary())
    return examples, report


def load_manifest(path: Path | str) -> DatasetManifest | None:
    """Read ``manifest.json`` from a dataset directory, if present."""
    directory = Path(path)
    manifest_path = directory if directory.is_file() else directory / DATASET_MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetIntegrityError(
            f"Dataset manifest {manifest_path} is not valid JSON.",
            details={"error": str(exc)},
        ) from exc
    try:
        return DatasetManifest.model_validate(payload)
    except ValidationError as exc:
        raise DatasetIntegrityError(
            f"Dataset manifest {manifest_path} failed validation.",
            details={"error": str(exc)},
            suggestions=["See data/schema/dataset_manifest.schema.json for the contract."],
        ) from exc


def file_sha256(path: Path | str) -> str:
    """Streaming sha256 of a file, used for dataset content hashes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class DatasetBundle:
    """A loaded dataset with its splits and provenance."""

    train: list[TrainingExample] = field(default_factory=list)
    validation: list[TrainingExample] = field(default_factory=list)
    test: list[TrainingExample] = field(default_factory=list)
    manifest: DatasetManifest | None = None
    reports: dict[str, LoadReport] = field(default_factory=dict)
    source_path: Path | None = None

    def split(self, name: str) -> list[TrainingExample]:
        """Examples in a named split."""
        return {"train": self.train, "validation": self.validation, "test": self.test}[name]

    def all_examples(self) -> list[TrainingExample]:
        """Every example across all splits."""
        return [*self.train, *self.validation, *self.test]

    @property
    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    @property
    def version(self) -> str:
        """Dataset version, from the manifest when available."""
        return self.manifest.version if self.manifest else "unversioned"

    def dataset_hash(self) -> str:
        """Content digest over every example, independent of file layout.

        Recorded in the experiment manifest so a run identifies the exact data it
        saw, even if the dataset was supplied as loose files without a manifest.
        """
        digest = hashlib.sha256()
        for split_name in ("train", "validation", "test"):
            digest.update(split_name.encode("utf-8"))
            for example in sorted(self.split(split_name), key=lambda e: e.id):
                digest.update(example.id.encode("utf-8"))
                digest.update(example.content_hash().encode("utf-8"))
        return digest.hexdigest()


def load_dataset_bundle(
    config: DatasetConfig,
    *,
    strict: bool = True,
    require_splits: Iterable[str] = ("train",),
) -> DatasetBundle:
    """Load every configured split, apply filters, and check integrity.

    Args:
        config: Dataset configuration, typically from the experiment config.
        strict: Fail on the first invalid example.
        require_splits: Splits that must be present and non-empty.

    Raises:
        DatasetIntegrityError: on duplicate ids across splits or a missing
            required split.
    """
    bundle = DatasetBundle(source_path=config.path)

    if config.path is not None:
        bundle.manifest = load_manifest(config.path)
        if bundle.manifest is None and config.require_manifest:
            raise DatasetIntegrityError(
                f"No {DATASET_MANIFEST_FILENAME} in {config.path} and require_manifest is set.",
                suggestions=[
                    "Generate one: python scripts/split_dataset.py --dataset <dir> ...",
                    "Or set dataset.require_manifest=false for exploratory work.",
                ],
            )

    if bundle.manifest is not None and bundle.manifest.contains_private_data:
        # Loud, but not fatal: the user may legitimately be training on a private
        # artifact stored outside this repository. What must never happen is that
        # artifact being committed here.
        logger.warning(
            "Dataset %s is flagged contains_private_data=true. This is expected for "
            "the private kleos-training-data artifact. Do NOT copy it into this "
            "public repository, and do not publish it with an adapter.",
            bundle.manifest.version,
        )

    for split_name in ("train", "validation", "test"):
        split_path = config.resolved_split_path(split_name)
        if split_path is None or not Path(split_path).exists():
            if split_name in require_splits:
                raise DatasetIntegrityError(
                    f"Required split {split_name!r} not found.",
                    details={
                        "looked_for": str(split_path) if split_path else "(not configured)",
                        "dataset_path": str(config.path),
                    },
                    suggestions=[
                        f"Create {split_name}.jsonl in the dataset directory.",
                        "Or run: python scripts/split_dataset.py --input <file> --output <dir>",
                    ],
                )
            continue

        examples, report = load_examples(split_path, model=TrainingExample, strict=strict)
        bundle.reports[split_name] = report
        setattr(bundle, split_name, examples)

    filtered = apply_filters(bundle, config)
    _check_id_uniqueness(filtered)

    for split_name in require_splits:
        if not filtered.split(split_name):
            raise DatasetIntegrityError(
                f"Split {split_name!r} is empty after filtering.",
                details={
                    "quality_statuses": config.filters.quality_statuses,
                    "tasks": config.filters.tasks or "(all)",
                    "domains": config.filters.domains or "(all)",
                },
                suggestions=[
                    "Filters default to quality_status='reviewed'; development "
                    "fixtures may need dataset.filters.quality_statuses widened.",
                    "Inspect the data: python scripts/inspect_dataset.py --dataset <dir>",
                ],
            )

    logger.info(
        "Loaded dataset %s: %s",
        filtered.version,
        ", ".join(f"{k}={v}" for k, v in filtered.counts.items()),
    )
    return filtered


def apply_filters(bundle: DatasetBundle, config: DatasetConfig) -> DatasetBundle:
    """Apply configured row filters to every split.

    ``max_examples`` truncates deterministically (first N after ordering) so a
    truncated run stays reproducible.
    """
    filters = config.filters
    allowed_status = set(filters.quality_statuses)
    if config.allow_unreviewed:
        allowed_status = set()  # empty means "accept anything"

    def keep(example: TrainingExample) -> bool:
        if filters.tasks and example.task not in filters.tasks:
            return False
        if filters.domains and example.variation_axes.domain not in filters.domains:
            return False
        if allowed_status and example.metadata.quality_status not in allowed_status:
            return False
        if example.metadata.source in filters.exclude_source_types:
            return False
        return len(example.messages) >= filters.min_messages

    result = DatasetBundle(
        manifest=bundle.manifest,
        reports=bundle.reports,
        source_path=bundle.source_path,
    )
    for split_name in ("train", "validation", "test"):
        kept = [e for e in bundle.split(split_name) if keep(e)]
        dropped = len(bundle.split(split_name)) - len(kept)
        if dropped:
            logger.info("Filter removed %d example(s) from %s", dropped, split_name)
        if filters.max_examples is not None and split_name == "train":
            kept = kept[: filters.max_examples]
        setattr(result, split_name, kept)
    return result


def _check_id_uniqueness(bundle: DatasetBundle) -> None:
    """Reject duplicate ids within or across splits.

    A duplicate id is not cosmetic: it makes per-example attribution ambiguous
    and can hide the same scenario appearing in both train and test.
    """
    seen: dict[str, str] = {}
    collisions: list[dict[str, str]] = []
    for split_name in ("train", "validation", "test"):
        for example in bundle.split(split_name):
            if example.id in seen:
                collisions.append(
                    {"id": example.id, "first_seen_in": seen[example.id], "also_in": split_name}
                )
            else:
                seen[example.id] = split_name

    if collisions:
        preview = json.dumps(collisions[:10], indent=2)
        raise DatasetIntegrityError(
            f"Found {len(collisions)} duplicate example id(s).",
            details={"collisions_preview": preview, "total": len(collisions)},
            suggestions=[
                "Example ids must be unique across the whole dataset version.",
                "Duplicates across train and test are also a leakage signal: run "
                "python scripts/validate_dataset.py --dataset <dir> --leakage-report reports/leakage",
            ],
        )


def load_evaluation_examples(
    path: Path | str, *, strict: bool = True, max_examples: int | None = None
) -> list[EvaluationExample]:
    """Load a benchmark file of evaluation examples."""
    examples, report = load_examples(path, model=EvaluationExample, strict=strict)
    if report.errors:
        logger.warning("%d invalid evaluation example(s) skipped", len(report.errors))
    if max_examples is not None:
        examples = examples[:max_examples]
    return examples


def write_jsonl(examples: Iterable[Any], path: Path | str) -> Path:
    """Write pydantic examples to JSONL, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for example in examples:
            payload = (
                example.model_dump(mode="json", exclude_none=False)
                if hasattr(example, "model_dump")
                else example
            )
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return target
