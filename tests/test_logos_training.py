"""Training-side changes made for KLEOS Logos, testable without torch.

* The pre-flight memory check is sized by the longest real training sequence,
  measured with the model's own tokenizer before any weights load.
* ``--feasibility record`` trains on without enforcing the estimate (the memory
  probe decides); ``enforce`` stays the default.
* The gradient check runs at the configured batch size.
* The smoke gate turns a smoke run's manifest into GO or NO-GO.

The real training loop, the memory probe and the text-only view on real weights
are covered by the torch tests (``test_training_tiny_model.py`` and
``test_ministral3_text_view.py``).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import pytest
from tests.conftest import CONFIGS_DIR, REPO_ROOT, FakeTokenizer, make_example

from kleos_models.config import load_config
from kleos_models.data.formatting import ConversationFormatter
from kleos_models.data.loaders import DatasetBundle
from kleos_models.data.schemas import TrainingExample
from kleos_models.errors import InsufficientMemoryError, MissingDependencyError
from kleos_models.training import trainer
from kleos_models.training.trainer import measure_sequence_lengths, padded_length


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", REPO_ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bundle(count: int = 4) -> DatasetBundle:
    examples = [
        TrainingExample.model_validate(
            make_example(
                f"train-{i:04d}",
                assistant="1. alpha\n2. beta "
                + "because " * (i * 3)
                + "\nWhat decided it: deadline",
            )
        )
        for i in range(count)
    ]
    return DatasetBundle(train=examples)


class TestSequenceMeasurement:
    @pytest.mark.parametrize(("raw", "padded"), [(1, 8), (8, 8), (9, 16), (492, 496), (440, 440)])
    def test_padding_follows_the_collator(self, raw, padded):
        assert padded_length(raw) == padded

    def test_the_longest_sequence_is_measured_as_the_trainer_formats_it(self):
        config = load_config(CONFIGS_DIR / "training" / "kleos_logos_v001.yaml")
        data = bundle()
        tokenizer = FakeTokenizer()
        lengths = measure_sequence_lengths(config, data, tokenizer=tokenizer)

        formatted, _ = ConversationFormatter(
            FakeTokenizer(), max_seq_length=config.model.max_seq_length
        ).format_dataset(data.train)
        expected = max(len(item.input_ids) for item in formatted)
        assert lengths["longest"] == expected
        assert lengths["longest_padded"] == padded_length(expected)
        assert lengths["examples_formatted"] == 4
        assert lengths["truncated"] == 0


class TestGradientCheckBatch:
    def test_the_check_uses_the_configured_batch_size(self):
        rows = [{"input_ids": [i]} for i in range(5)]
        seen: list[int] = []

        def collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
            seen.append(len(batch))
            return {}

        trainer._sample_batch(rows, collator, size=1)  # type: ignore[arg-type]
        trainer._sample_batch(rows, collator, size=3)  # type: ignore[arg-type]
        trainer._sample_batch(rows, collator, size=9)  # type: ignore[arg-type]
        assert seen == [1, 3, 5]


# ---------------------------------------------------------------------------
# scripts/train.py
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "dataset"
    directory.mkdir()
    with (directory / "train.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(4):
            handle.write(json.dumps(make_example(f"train-{i:04d}")) + "\n")
    return directory


@pytest.fixture
def train_script(monkeypatch) -> ModuleType:
    for name in ("KLEOS_BENCHMARK_PATH", "KLEOS_DATASET_PATH"):
        monkeypatch.delenv(name, raising=False)
    return load_script("train")


def no_gpu() -> Any:
    from kleos_models.models.feasibility import GPUInfo

    return GPUInfo(available=False, source="test: no GPU")


class TestTrainScript:
    def args(self, dataset_dir: Path, tmp_path: Path, *extra: str) -> list[str]:
        return [
            "--config",
            str(CONFIGS_DIR / "training" / "debug_logos.yaml"),
            "--dataset",
            str(dataset_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--experiment-id",
            "logos-test",
            # A scratch dataset, not a sealed release.
            "--set",
            "dataset.require_manifest=false",
            *extra,
        ]

    def test_a_dry_run_without_a_tokenizer_plans_for_the_worst_case(
        self, train_script, dataset_dir, tmp_path, monkeypatch
    ):
        def unavailable(*_args: Any, **_kwargs: Any) -> Any:
            raise MissingDependencyError("transformers", extra="train", purpose="load a tokenizer")

        monkeypatch.setattr(trainer, "measure_sequence_lengths", unavailable)
        assert train_script.main(self.args(dataset_dir, tmp_path, "--dry-run")) == 0
        manifest = json.loads((tmp_path / "out" / "logos-test" / "manifest.json").read_text())
        assert manifest["feasibility"]["policy"] == "enforce"
        assert "sequence_lengths" not in manifest["feasibility"]
        assert manifest["feasibility"]["estimate"]["sequence_length"] == 1024

    def test_a_real_run_needs_the_tokenizer(self, train_script, dataset_dir, tmp_path, monkeypatch):
        def unavailable(*_args: Any, **_kwargs: Any) -> Any:
            raise MissingDependencyError("transformers", extra="train", purpose="load a tokenizer")

        monkeypatch.setattr(trainer, "measure_sequence_lengths", unavailable)
        with pytest.raises(MissingDependencyError):
            train_script.main(self.args(dataset_dir, tmp_path))

    def test_the_estimate_is_sized_by_the_measured_sequence(
        self, train_script, dataset_dir, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            trainer,
            "measure_sequence_lengths",
            lambda *a, **k: {"longest": 492, "longest_padded": 496, "examples_formatted": 4},
        )
        assert train_script.main(self.args(dataset_dir, tmp_path, "--dry-run")) == 0
        manifest = json.loads((tmp_path / "out" / "logos-test" / "manifest.json").read_text())
        assert manifest["feasibility"]["sequence_lengths"]["longest_padded"] == 496
        assert manifest["feasibility"]["estimate"]["sequence_length"] == 496

    def _stub_training(self, monkeypatch) -> dict[str, Any]:
        calls: dict[str, Any] = {}
        monkeypatch.setattr(
            trainer,
            "measure_sequence_lengths",
            lambda *a, **k: {"longest": 492, "longest_padded": 496, "examples_formatted": 4},
        )
        from kleos_models.models import feasibility

        monkeypatch.setattr(feasibility, "probe_gpu", no_gpu)

        def fake_run_training(config: Any, bundle: Any, manifest: Any, **kwargs: Any) -> Any:
            calls.update(kwargs, manifest=manifest, config=config)
            return SimpleNamespace(render=lambda: "stub")

        monkeypatch.setattr(trainer, "run_training", fake_run_training)
        return calls

    def test_enforce_refuses_a_run_the_estimate_rejects(
        self, train_script, dataset_dir, tmp_path, monkeypatch
    ):
        calls = self._stub_training(monkeypatch)
        with pytest.raises(InsufficientMemoryError):
            train_script.main(self.args(dataset_dir, tmp_path))
        assert not calls

    def test_record_trains_on_with_nothing_adjusted(
        self, train_script, dataset_dir, tmp_path, monkeypatch
    ):
        calls = self._stub_training(monkeypatch)
        assert train_script.main(self.args(dataset_dir, tmp_path, "--feasibility", "record")) == 0
        manifest = calls["manifest"]
        assert manifest.feasibility["policy"] == "record"
        assert not manifest.adjustments
        assert any("--feasibility record" in note for note in manifest.notes)
        assert calls["longest_sequence"] == 496
        assert calls["probe_memory"] is True
        assert calls["config"].model.max_seq_length == 1024  # untouched

    def test_the_memory_probe_can_be_skipped(
        self, train_script, dataset_dir, tmp_path, monkeypatch
    ):
        calls = self._stub_training(monkeypatch)
        train_script.main(
            self.args(dataset_dir, tmp_path, "--feasibility", "record", "--skip-memory-probe")
        )
        assert calls["probe_memory"] is False


# ---------------------------------------------------------------------------
# scripts/check_smoke_gate.py
# ---------------------------------------------------------------------------


def smoke_manifest(**changes: Any) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "experiment_id": "kleos-logos-smoke-001",
        "status": "completed",
        "adjustments": [],
        "model": {
            "load": {
                "view": {
                    "kind": "text_only",
                    "loading_info": {
                        "missing_keys": 0,
                        "mismatched_keys": 0,
                        "unexpected_keys": 440,
                    },
                }
            }
        },
        "lora": {
            "trainable_parameters": 60_948_480,
            "target_modules": {"matched_module_count": 280},
        },
        "metrics": {
            "gradient_check": {"lora_tensors": 560, "lora_tensors_with_nonzero_grad": 280},
            "memory_probe": {
                "batch_size": 1,
                "sequence_length": 496,
                "peak_allocated_gb": 13.9,
                "peak_reserved_gb": 14.1,
                "total_gb": 14.56,
                "spare_after_optimizer_gb": 0.2,
            },
        },
    }
    for dotted, value in changes.items():
        cursor = manifest
        *parents, last = dotted.split("__")
        for part in parents:
            cursor = cursor[part]
        cursor[last] = value
    return manifest


class TestSmokeGate:
    EXPECT: ClassVar[dict[str, Any]] = {
        "expect_modules": 280,
        "expect_trainable": 60_948_480,
        "min_spare_gb": 0.15,
    }

    def failed(self, manifest: dict[str, Any]) -> list[str]:
        gate = load_script("check_smoke_gate")
        return [
            name
            for mark, name, _ in gate.evaluate_gate(manifest, **self.EXPECT)
            if mark == gate.FAIL
        ]

    def test_a_clean_smoke_run_is_go(self):
        assert self.failed(smoke_manifest()) == []

    @pytest.mark.parametrize(
        ("change", "failing"),
        [
            ({"status": "failed"}, "run completed"),
            ({"adjustments": [{"field": "model.max_seq_length"}]}, "nothing adjusted to fit"),
            (
                {"model__load__view__loading_info__missing_keys": 1},
                "text-only view loaded every weight",
            ),
            ({"lora__target_modules__matched_module_count": 252}, "LoRA modules"),
            ({"lora__trainable_parameters": 57_016_320}, "trainable parameters"),
            (
                {"metrics__gradient_check__lora_tensors_with_nonzero_grad": 0},
                "gradients reach the adapter",
            ),
            (
                {"metrics__memory_probe__spare_after_optimizer_gb": 0.05},
                "memory on the longest batch",
            ),
        ],
    )
    def test_each_failure_is_no_go(self, change, failing):
        assert self.failed(smoke_manifest(**change)) == [failing]

    def test_a_missing_probe_is_no_go(self):
        manifest = smoke_manifest()
        del manifest["metrics"]["memory_probe"]
        assert self.failed(manifest) == ["memory on the longest batch"]

    def test_exit_codes(self, tmp_path, capsys):
        gate = load_script("check_smoke_gate")
        path = tmp_path / "manifest.json"
        args = [
            "--manifest",
            str(path),
            "--expect-modules",
            "280",
            "--expect-trainable",
            "60948480",
        ]
        path.write_text(json.dumps(smoke_manifest()), encoding="utf-8")
        assert gate.main(args) == 0
        path.write_text(json.dumps(smoke_manifest(status="failed")), encoding="utf-8")
        assert gate.main(args) == 1
        path.write_text("{not json", encoding="utf-8")
        assert gate.main(args) == 2
        assert "NO-GO" in capsys.readouterr().out
