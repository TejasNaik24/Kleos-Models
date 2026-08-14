"""Experiment manifest, registry and checkpointing tests (spec §16, §33, §34, §36).

The properties under test are the ones that make a research record trustworthy:
a manifest identifies a run completely, failed runs stay recorded, adjustments are
never silent, and checkpoint retention cannot leave a Colab user with nothing to
resume from.
"""

from __future__ import annotations

import json

import pytest
from tests.conftest import CONFIGS_DIR

from kleos_models.config import load_config
from kleos_models.errors import CheckpointError
from kleos_models.experiments.environment import (
    capture_environment,
    capture_git_info,
    set_global_seed,
)
from kleos_models.experiments.manifest import (
    ExperimentManifest,
    RunStatus,
    build_manifest,
    generate_experiment_id,
)
from kleos_models.experiments.registry import ExperimentRegistry, assert_comparable
from kleos_models.training.checkpointing import (
    discover_checkpoints,
    find_latest_checkpoint,
    prune_checkpoints,
    resolve_resume_path,
    validate_checkpoint,
    write_checkpoint_metadata,
)


@pytest.fixture
def config():
    return load_config(CONFIGS_DIR / "training" / "debug.yaml")


def make_checkpoint(directory, step: int, *, complete: bool = True):
    """Create a checkpoint directory resembling a Trainer save."""
    path = directory / f"checkpoint-{step}"
    path.mkdir(parents=True, exist_ok=True)
    if complete:
        (path / "trainer_state.json").write_text(
            json.dumps({"global_step": step}), encoding="utf-8"
        )
        (path / "adapter_model.safetensors").write_bytes(b"weights")
    else:
        # A save that was interrupted part-way, as a Colab disconnect produces.
        (path / "adapter_model.safetensors").write_bytes(b"partial")
    return path


class TestExperimentId:
    def test_id_contains_the_model_name(self):
        assert "qwen3_8b" in generate_experiment_id(name="kleos", model_name="qwen3_8b")

    def test_id_contains_the_config_hash(self):
        assert "abcd1234" in generate_experiment_id(config_hash="abcd1234ef")

    def test_ids_are_unique_without_a_hash(self):
        assert generate_experiment_id() != generate_experiment_id()

    def test_slashes_are_sanitized(self):
        assert "/" not in generate_experiment_id(model_name="org/model")


class TestManifestLifecycle:
    def test_manifest_starts_in_started_state(self, config):
        assert build_manifest(config).status is RunStatus.STARTED

    def test_completion_records_metrics_and_duration(self, config):
        manifest = build_manifest(config)
        manifest.mark_started()
        manifest.mark_completed({"train_loss": 0.5})
        assert manifest.status is RunStatus.COMPLETED
        assert manifest.metrics["train_loss"] == 0.5
        assert manifest.duration_seconds is not None

    def test_failure_is_recorded_not_discarded(self, config):
        # Spec §36: failed runs must stay in the record.
        manifest = build_manifest(config)
        manifest.mark_started()
        manifest.mark_failed(ValueError("something went wrong"), stage="training")
        assert manifest.status is RunStatus.FAILED
        assert manifest.error["stage"] == "training"
        assert "something went wrong" in manifest.error["message"]

    def test_interruption_is_distinguished_from_failure(self, config):
        manifest = build_manifest(config)
        manifest.mark_started()
        manifest.mark_interrupted()
        assert manifest.status is RunStatus.INTERRUPTED


class TestManifestContent:
    def test_reproducibility_chain_is_recorded(self, config):
        manifest = build_manifest(
            config, dataset_version="v1", dataset_hash="abc123", dataset_counts={"train": 10}
        )
        assert manifest.config_hash == config.config_hash
        assert manifest.seed == config.seed
        assert manifest.dataset_version == "v1"
        assert manifest.dataset_hash == "abc123"
        assert "commit" in manifest.git

    def test_model_identity_is_recorded(self, config):
        manifest = build_manifest(config)
        assert manifest.model["base_model"] == config.model.base_model
        assert manifest.model["family"] == config.model.family
        assert "revision" in manifest.model

    def test_environment_is_captured(self, config):
        manifest = build_manifest(config)
        assert manifest.software
        assert "python" in manifest.environment
        assert "available" in manifest.hardware

    def test_missing_git_degrades_gracefully(self, config):
        # This repository may legitimately not be a git repo yet.
        manifest = build_manifest(config)
        assert manifest.git.get("commit") is not None
        if not manifest.git.get("available"):
            assert any("Git commit unavailable" in note for note in manifest.notes)

    def test_adjustments_are_recorded_with_a_reason(self, config):
        manifest = build_manifest(config)
        manifest.add_adjustment("model.max_seq_length", 2048, 1024, "did not fit the GPU")
        assert len(manifest.adjustments) == 1
        assert manifest.adjustments[0]["reason"] == "did not fit the GPU"
        assert manifest.adjustments[0]["original"] == 2048

    def test_checkpoints_are_tracked(self, config):
        manifest = build_manifest(config)
        manifest.add_checkpoint("/tmp/checkpoint-50", 50)
        assert manifest.checkpoints[0]["step"] == 50


class TestManifestPersistence:
    def test_manifest_round_trips(self, config, tmp_path):
        manifest = build_manifest(config, dataset_version="v1")
        manifest.mark_started()
        manifest.mark_completed({"loss": 0.1})
        path = manifest.save(tmp_path)

        restored = ExperimentManifest.load(path)
        assert restored.experiment_id == manifest.experiment_id
        assert restored.config_hash == manifest.config_hash
        assert restored.status is RunStatus.COMPLETED
        assert restored.metrics["loss"] == 0.1

    def test_manifest_loads_from_a_directory(self, config, tmp_path):
        build_manifest(config).save(tmp_path)
        assert ExperimentManifest.load(tmp_path).experiment_id

    def test_saving_is_atomic_enough_to_leave_no_temp_file(self, config, tmp_path):
        build_manifest(config).save(tmp_path)
        assert not list(tmp_path.glob("*.tmp"))

    def test_summary_renders(self, config):
        manifest = build_manifest(config)
        manifest.mark_started()
        manifest.mark_completed({"loss": 0.2})
        summary = manifest.summary()
        assert manifest.experiment_id in summary
        assert "config hash" in summary


class TestRegistry:
    def _write_run(self, root, config, experiment_id, status=RunStatus.COMPLETED, **kwargs):
        manifest = build_manifest(config, experiment_id=experiment_id, **kwargs)
        manifest.mark_started()
        if status is RunStatus.COMPLETED:
            manifest.mark_completed({"loss": 0.1})
        elif status is RunStatus.FAILED:
            manifest.mark_failed(RuntimeError("boom"), stage="training")
        directory = root / experiment_id
        manifest.save(directory)
        return manifest

    def test_runs_are_discovered(self, config, tmp_path):
        self._write_run(tmp_path, config, "run-a")
        self._write_run(tmp_path, config, "run-b")
        assert len(ExperimentRegistry(tmp_path).scan()) == 2

    def test_failed_runs_stay_visible(self, config, tmp_path):
        # Spec §36: a registry that hid failures would make cherry-picking easy.
        self._write_run(tmp_path, config, "run-ok")
        self._write_run(tmp_path, config, "run-bad", status=RunStatus.FAILED)
        registry = ExperimentRegistry(tmp_path)
        assert len(registry.scan()) == 2
        assert len(registry.filter(status=RunStatus.FAILED)) == 1
        assert registry.summary()["by_status"]["failed"] == 1

    def test_lookup_by_id(self, config, tmp_path):
        self._write_run(tmp_path, config, "run-a")
        assert ExperimentRegistry(tmp_path).get("run-a") is not None
        assert ExperimentRegistry(tmp_path).get("missing") is None

    def test_filtering_by_family(self, config, tmp_path):
        self._write_run(tmp_path, config, "run-a")
        registry = ExperimentRegistry(tmp_path)
        assert registry.filter(model_family="qwen")
        assert not registry.filter(model_family="nonexistent")

    def test_empty_registry_renders_a_message(self, tmp_path):
        assert "No experiment runs" in ExperimentRegistry(tmp_path).render()

    def test_render_notes_failed_runs(self, config, tmp_path):
        self._write_run(tmp_path, config, "run-bad", status=RunStatus.FAILED)
        assert "failed" in ExperimentRegistry(tmp_path).render()


class TestComparability:
    def _manifest(self, config, **overrides):
        manifest = build_manifest(config, dataset_version="v1")
        manifest.dataset_hash = "hash-1"
        manifest.task = "notification_prioritization"
        for key, value in overrides.items():
            setattr(manifest, key, value)
        return manifest

    def test_identical_setups_are_comparable(self, config):
        left = self._manifest(config)
        right = self._manifest(config)
        right.seed = 43
        assert_comparable(left, right)

    def test_different_dataset_versions_are_rejected(self, config):
        left = self._manifest(config)
        right = self._manifest(config, dataset_version="v2")
        with pytest.raises(ValueError, match="not comparable"):
            assert_comparable(left, right)

    def test_different_dataset_hashes_are_rejected(self, config):
        left = self._manifest(config)
        right = self._manifest(config)
        right.dataset_hash = "hash-2"
        with pytest.raises(ValueError, match="did not see the same data"):
            assert_comparable(left, right)

    def test_different_tasks_are_rejected(self, config):
        left = self._manifest(config)
        right = self._manifest(config, task="tool_routing")
        with pytest.raises(ValueError, match="different tasks"):
            assert_comparable(left, right)

    def test_cross_family_comparison_is_allowed_but_warned(self, config):
        left = self._manifest(config)
        right = self._manifest(config)
        right.model = {**right.model, "family": "mistral"}
        warnings = assert_comparable(left, right)
        assert any("model families" in w for w in warnings)

    def test_cross_family_can_be_forbidden(self, config):
        left = self._manifest(config)
        right = self._manifest(config)
        right.model = {**right.model, "family": "mistral"}
        with pytest.raises(ValueError, match="model families"):
            assert_comparable(left, right, allow_cross_family=False)

    def test_adjustments_produce_a_warning(self, config):
        left = self._manifest(config)
        right = self._manifest(config)
        right.seed = 43
        right.add_adjustment("model.max_seq_length", 2048, 1024, "memory")
        warnings = assert_comparable(left, right)
        assert any("adjustments" in w for w in warnings)


class TestCheckpointing:
    def test_complete_checkpoints_are_valid(self, tmp_path):
        path = make_checkpoint(tmp_path, 50)
        valid, reason = validate_checkpoint(path)
        assert valid, reason

    def test_partial_checkpoints_are_invalid(self, tmp_path):
        path = make_checkpoint(tmp_path, 50, complete=False)
        valid, reason = validate_checkpoint(path)
        assert not valid
        assert "interrupted" in reason or "missing" in reason

    def test_discovery_orders_newest_first(self, tmp_path):
        for step in (10, 50, 30):
            make_checkpoint(tmp_path, step)
        assert [c.step for c in discover_checkpoints(tmp_path)] == [50, 30, 10]

    def test_discovery_skips_incomplete_checkpoints(self, tmp_path):
        make_checkpoint(tmp_path, 10)
        make_checkpoint(tmp_path, 50, complete=False)
        assert [c.step for c in discover_checkpoints(tmp_path)] == [10]

    def test_latest_checkpoint_is_found(self, tmp_path):
        make_checkpoint(tmp_path, 10)
        make_checkpoint(tmp_path, 90)
        latest = find_latest_checkpoint(tmp_path)
        assert latest is not None and latest.step == 90

    def test_no_checkpoints_returns_none(self, tmp_path):
        assert find_latest_checkpoint(tmp_path) is None

    def test_auto_resume_finds_the_newest(self, tmp_path):
        make_checkpoint(tmp_path, 10)
        make_checkpoint(tmp_path, 60)
        _path, info = resolve_resume_path("auto", tmp_path)
        assert info is not None and info.step == 60

    def test_auto_resume_with_nothing_to_resume_starts_fresh(self, tmp_path):
        path, info = resolve_resume_path("auto", tmp_path)
        assert path is None and info is None

    def test_auto_resume_ignores_an_interrupted_save(self, tmp_path):
        make_checkpoint(tmp_path, 10)
        make_checkpoint(tmp_path, 60, complete=False)
        _, info = resolve_resume_path("auto", tmp_path)
        assert info is not None and info.step == 10

    def test_no_resume_requested_returns_nothing(self, tmp_path):
        assert resolve_resume_path(None, tmp_path) == (None, None)

    def test_missing_explicit_checkpoint_raises(self, tmp_path):
        with pytest.raises(CheckpointError, match="not found"):
            resolve_resume_path(str(tmp_path / "checkpoint-999"), tmp_path)

    def test_invalid_explicit_checkpoint_raises(self, tmp_path):
        path = make_checkpoint(tmp_path, 50, complete=False)
        with pytest.raises(CheckpointError, match="not resumable"):
            resolve_resume_path(str(path), tmp_path)

    def test_metadata_round_trips(self, tmp_path):
        from kleos_models.training.checkpointing import read_checkpoint_metadata

        path = make_checkpoint(tmp_path, 50)
        write_checkpoint_metadata(path, {"experiment_id": "exp-1", "dataset_version": "v1"})
        metadata = read_checkpoint_metadata(path)
        assert metadata is not None
        assert metadata["experiment_id"] == "exp-1"


class TestCheckpointRetention:
    def test_old_checkpoints_are_pruned(self, tmp_path):
        for step in (10, 20, 30, 40):
            make_checkpoint(tmp_path, step)
        prune_checkpoints(tmp_path, keep=2)
        assert [c.step for c in discover_checkpoints(tmp_path)] == [40, 30]

    def test_retention_never_empties_the_directory(self, tmp_path):
        # A Colab runtime can die at any moment; leaving zero checkpoints would
        # discard the whole session.
        for step in (10, 20):
            make_checkpoint(tmp_path, step)
        prune_checkpoints(tmp_path, keep=0)
        assert len(discover_checkpoints(tmp_path)) >= 1

    def test_incomplete_checkpoints_are_always_removed(self, tmp_path):
        make_checkpoint(tmp_path, 10)
        make_checkpoint(tmp_path, 20, complete=False)
        prune_checkpoints(tmp_path, keep=5)
        assert not (tmp_path / "checkpoint-20").exists()
        assert (tmp_path / "checkpoint-10").exists()

    def test_dry_run_deletes_nothing(self, tmp_path):
        for step in (10, 20, 30):
            make_checkpoint(tmp_path, step)
        removed = prune_checkpoints(tmp_path, keep=1, dry_run=True)
        assert removed
        assert len(discover_checkpoints(tmp_path)) == 3


class TestEnvironmentCapture:
    def test_environment_snapshot_has_the_required_fields(self):
        snapshot = capture_environment()
        assert snapshot.python_version
        assert "torch" in snapshot.libraries
        assert "available" in snapshot.gpu

    def test_git_capture_never_raises(self, tmp_path):
        info = capture_git_info(tmp_path)
        assert info.commit is not None
        if not info.available:
            assert info.reason

    def test_seeding_reports_what_was_seeded(self):
        record = set_global_seed(1234)
        assert record["seed"] == 1234
        assert record["python_random"] is True

    def test_seeding_is_reproducible(self):
        import random

        set_global_seed(99)
        first = [random.random() for _ in range(5)]
        set_global_seed(99)
        assert [random.random() for _ in range(5)] == first
