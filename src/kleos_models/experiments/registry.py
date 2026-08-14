"""Experiment registry (spec sections 36, 53).

Scans an outputs directory and indexes every run, including failed ones.

The research-integrity requirement this serves: failed and negative runs must stay
visible. A registry that only listed successful runs would make it trivially easy
to report a favourable subset without ever deciding to.

The registry also refuses to compare runs whose manifests say they are not
comparable — different dataset version, different model family, different config
hash — because pooling those silently is how a result stops meaning anything.

This module imports no torch.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kleos_models.constants import MANIFEST_FILENAME
from kleos_models.experiments.manifest import ExperimentManifest, RunStatus
from kleos_models.logging_utils import format_table, get_logger

logger = get_logger(__name__)


@dataclass
class RegistryEntry:
    """One indexed run."""

    manifest: ExperimentManifest
    directory: Path

    @property
    def experiment_id(self) -> str:
        return self.manifest.experiment_id

    def row(self) -> dict[str, Any]:
        """Compact record for tabular display."""
        manifest = self.manifest
        return {
            "experiment_id": manifest.experiment_id,
            "status": manifest.status.value,
            "kind": manifest.kind,
            "model": manifest.model.get("name", "-"),
            "family": manifest.model.get("family", "-"),
            "task": manifest.task or "-",
            "dataset": manifest.dataset_version,
            "seed": manifest.seed,
            "config": manifest.config_hash[:8],
            "created": manifest.created_at[:19],
        }


class ExperimentRegistry:
    """Index of experiment runs under an outputs root."""

    def __init__(self, root: Path | str = "outputs") -> None:
        self.root = Path(root)

    def scan(self) -> list[RegistryEntry]:
        """Find every run directory containing a manifest.

        Unreadable manifests are warned about, not skipped silently: a corrupt
        manifest is itself information about a run that went wrong.
        """
        entries: list[RegistryEntry] = []
        if not self.root.exists():
            return entries

        for manifest_path in sorted(self.root.glob(f"*/{MANIFEST_FILENAME}")):
            try:
                manifest = ExperimentManifest.load(manifest_path)
            except Exception as exc:
                logger.warning(
                    "Could not read manifest %s: %s. The run directory is kept; "
                    "inspect it manually.",
                    manifest_path,
                    exc,
                )
                continue
            entries.append(RegistryEntry(manifest=manifest, directory=manifest_path.parent))

        entries.sort(key=lambda e: e.manifest.created_at, reverse=True)
        return entries

    def __iter__(self) -> Iterator[RegistryEntry]:
        return iter(self.scan())

    def get(self, experiment_id: str) -> RegistryEntry | None:
        """Look up one run by id."""
        for entry in self.scan():
            if entry.experiment_id == experiment_id:
                return entry
        return None

    def filter(
        self,
        *,
        status: RunStatus | str | None = None,
        model_family: str | None = None,
        task: str | None = None,
        dataset_version: str | None = None,
        config_hash: str | None = None,
        kind: str | None = None,
    ) -> list[RegistryEntry]:
        """Filter runs by manifest fields."""
        wanted_status = RunStatus(status) if isinstance(status, str) else status
        results = []
        for entry in self.scan():
            manifest = entry.manifest
            if wanted_status is not None and manifest.status is not wanted_status:
                continue
            if model_family and manifest.model.get("family") != model_family:
                continue
            if task and manifest.task != task:
                continue
            if dataset_version and manifest.dataset_version != dataset_version:
                continue
            if config_hash and not manifest.config_hash.startswith(config_hash):
                continue
            if kind and manifest.kind != kind:
                continue
            results.append(entry)
        return results

    def summary(self) -> dict[str, Any]:
        """Aggregate counts, including failures."""
        entries = self.scan()
        by_status: dict[str, int] = {}
        by_family: dict[str, int] = {}
        for entry in entries:
            status = entry.manifest.status.value
            by_status[status] = by_status.get(status, 0) + 1
            family = entry.manifest.model.get("family", "unknown")
            by_family[family] = by_family.get(family, 0) + 1
        return {
            "root": str(self.root),
            "total_runs": len(entries),
            "by_status": by_status,
            "by_family": by_family,
        }

    def render(self, entries: list[RegistryEntry] | None = None) -> str:
        """Render the registry as a table."""
        rows = [entry.row() for entry in (entries if entries is not None else self.scan())]
        if not rows:
            return f"No experiment runs found under {self.root}."

        summary = self.summary()
        failed = summary["by_status"].get("failed", 0)
        header = [
            f"{summary['total_runs']} run(s) under {self.root}",
            "  " + ", ".join(f"{k}={v}" for k, v in sorted(summary["by_status"].items())),
        ]
        if failed:
            header.append(
                f"  {failed} failed run(s) are listed deliberately: failed and "
                "negative results are part of the record."
            )
        return "\n".join([*header, "", format_table(rows)])


def assert_comparable(
    left: ExperimentManifest,
    right: ExperimentManifest,
    *,
    require_same_dataset: bool = True,
    require_same_task: bool = True,
    allow_cross_family: bool = True,
) -> list[str]:
    """Check two runs can be meaningfully compared.

    Returns a list of warnings. Raises only when a difference makes the comparison
    invalid rather than merely interesting.

    Cross-family comparison is *allowed by default* — comparing Qwen against
    Mistral is the point of the research — but comparing across different dataset
    versions or different tasks is not, because then the number reflects the data,
    not the model.

    Raises:
        ValueError: when the runs are not comparable.
    """
    problems: list[str] = []
    warnings: list[str] = []

    if require_same_dataset and left.dataset_version != right.dataset_version:
        problems.append(
            f"different dataset versions: {left.dataset_version!r} vs {right.dataset_version!r}"
        )
    if left.dataset_hash and right.dataset_hash and left.dataset_hash != right.dataset_hash:
        problems.append("dataset content hashes differ, so the two runs did not see the same data")
    if require_same_task and left.task != right.task:
        problems.append(f"different tasks: {left.task!r} vs {right.task!r}")

    left_family = left.model.get("family")
    right_family = right.model.get("family")
    if left_family != right_family:
        message = f"different model families: {left_family!r} vs {right_family!r}"
        if allow_cross_family:
            warnings.append(
                message + " — valid as a family comparison, but scale and "
                "architecture are confounded unless the configs are scale-matched"
            )
        else:
            problems.append(message)

    if left.split_strategy != right.split_strategy:
        warnings.append(
            f"different split strategies ({left.split_strategy!r} vs "
            f"{right.split_strategy!r}); the test sets are not the same population"
        )
    if left.seed == right.seed and left.config_hash == right.config_hash:
        warnings.append(
            "identical config hash and seed: these runs should be identical, so any "
            "difference indicates nondeterminism"
        )
    if left.adjustments or right.adjustments:
        warnings.append(
            "at least one run had automatic configuration adjustments; check the "
            "manifest adjustments[] before treating the comparison as controlled"
        )

    if problems:
        raise ValueError(
            "These runs are not comparable:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n\nPooling them would produce a number that reflects the difference "
            "in setup rather than the difference in model."
        )

    for warning in warnings:
        logger.warning("Comparability: %s", warning)
    return warnings
