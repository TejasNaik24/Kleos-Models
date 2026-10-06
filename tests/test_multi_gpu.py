"""A model spread over two GPUs (Logos v0.0.2 on Kaggle's 2 x T4).

With ``device_map: auto`` and two visible GPUs, the layers of the quantized
model are split across both. Two things must then hold. The Trainer must run the
model in place (model parallel), not replicate it with ``DataParallel``, which
a 4-bit model cannot survive. And memory must be read on every GPU: the one
holding lm_head and the logits fills first, so a probe that read only GPU 0
would pass a run that dies at step 1. A single GPU must see exactly what it saw
before.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from kleos_models.errors import KleosError
from kleos_models.training.memory import (
    DeviceReading,
    model_devices,
    peak_allocated_all_gb,
    probe_result,
    reset_peak_memory,
)
from kleos_models.training.trainer import assert_model_parallel

GIB = 1024**3


def parameter(device_type: str, index: int | None) -> SimpleNamespace:
    return SimpleNamespace(device=SimpleNamespace(type=device_type, index=index))


def model_on(*devices: tuple[str, int | None]) -> SimpleNamespace:
    params = [parameter(kind, index) for kind, index in devices]
    return SimpleNamespace(parameters=lambda: iter(params))


def reading(index: int, *, free: float, paged: float = 0.06) -> DeviceReading:
    return DeviceReading(
        index=index,
        peak_allocated_gb=8.0,
        peak_reserved_gb=9.0,
        total_gb=14.56,
        free_at_peak_gb=free,
        paged_optimizer_gb=paged,
    )


SINGLE_GPU_KEYS = {
    "batch_size",
    "sequence_length",
    "precision",
    "peak_allocated_gb",
    "peak_reserved_gb",
    "total_gb",
    "free_at_peak_gb",
    "paged_optimizer_gb",
    "spare_after_optimizer_gb",
}


class TestModelDevices:
    def test_the_gpus_holding_parameters_in_order(self):
        model = model_on(("cuda", 1), ("cuda", 0), ("cuda", 1), ("cpu", None))
        assert model_devices(model) == [0, 1]

    def test_a_cpu_model_holds_none(self):
        assert model_devices(model_on(("cpu", None))) == []


class TestProbeResult:
    def test_one_gpu_reports_exactly_what_it_did_before(self):
        result = probe_result(
            [reading(0, free=1.2, paged=0.12)],
            batch_size=1,
            sequence_length=736,
            precision="fp16",
        )
        assert set(result) == SINGLE_GPU_KEYS
        assert result["spare_after_optimizer_gb"] == pytest.approx(1.08)

    def test_two_gpus_are_gated_on_the_worse_one(self):
        result = probe_result(
            [reading(0, free=6.0), reading(1, free=0.4)],
            batch_size=1,
            sequence_length=736,
            precision="fp16",
        )
        assert result["spare_after_optimizer_gb"] == pytest.approx(0.34)
        assert result["free_at_peak_gb"] == pytest.approx(0.4)
        assert result["gated_on_device"] == 1
        assert [d["index"] for d in result["devices"]] == [0, 1]
        assert result["devices"][0]["spare_after_optimizer_gb"] == pytest.approx(5.94)


def fake_torch(*, allocated_gib: list[float]) -> SimpleNamespace:
    resets: list[int] = []
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: len(allocated_gib),
        max_memory_allocated=lambda index: int(allocated_gib[index] * GIB),
        reset_peak_memory_stats=lambda index=None: resets.append(index),
    )
    return SimpleNamespace(cuda=cuda, resets=resets)


class TestPeakOverEveryGpu:
    def test_the_peak_is_the_fullest_gpu(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", fake_torch(allocated_gib=[6.1, 8.7]))
        assert peak_allocated_all_gb() == pytest.approx(8.7)

    def test_one_gpu_reports_its_own_peak(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", fake_torch(allocated_gib=[13.1]))
        assert peak_allocated_all_gb() == pytest.approx(13.1)

    def test_no_cuda_reports_nothing(self, monkeypatch):
        torch = fake_torch(allocated_gib=[1.0])
        torch.cuda.is_available = lambda: False
        monkeypatch.setitem(sys.modules, "torch", torch)
        assert peak_allocated_all_gb() is None

    def test_every_gpu_is_reset(self, monkeypatch):
        torch = fake_torch(allocated_gib=[1.0, 2.0])
        monkeypatch.setitem(sys.modules, "torch", torch)
        reset_peak_memory()
        assert torch.resets == [0, 1]


def trainer(*, model_parallel: bool, n_gpu: int) -> Any:
    return SimpleNamespace(is_model_parallel=model_parallel, args=SimpleNamespace(n_gpu=n_gpu))


class TestModelParallel:
    TWO = model_on(("cuda", 0), ("cuda", 1))

    def test_a_split_model_run_in_place_passes(self):
        assert_model_parallel(trainer(model_parallel=True, n_gpu=1), self.TWO)

    @pytest.mark.parametrize(("model_parallel", "n_gpu"), [(False, 2), (True, 2), (False, 1)])
    def test_a_split_model_that_would_be_replicated_is_refused(self, model_parallel, n_gpu):
        with pytest.raises(KleosError, match="model parallel"):
            assert_model_parallel(trainer(model_parallel=model_parallel, n_gpu=n_gpu), self.TWO)

    def test_one_gpu_is_not_checked(self):
        assert_model_parallel(trainer(model_parallel=False, n_gpu=1), model_on(("cuda", 0)))

    def test_a_cpu_run_is_not_checked(self):
        assert_model_parallel(trainer(model_parallel=False, n_gpu=0), model_on(("cpu", None)))


class TestDeviceMapRecord:
    def test_a_split_model_records_where_each_part_sits(self):
        from kleos_models.models.loading import spread_device_map

        model = SimpleNamespace(
            hf_device_map={"model.embed_tokens": 0, "model.layers.0": 0, "lm_head": 1}
        )
        assert spread_device_map(model) == {
            "model.embed_tokens": "0",
            "model.layers.0": "0",
            "lm_head": "1",
        }

    @pytest.mark.parametrize(
        "device_map", [{"": 0}, {"model": 0, "lm_head": "cpu"}, {"model": "cpu"}, None]
    )
    def test_one_gpu_records_nothing_new(self, device_map):
        from kleos_models.models.loading import spread_device_map

        assert spread_device_map(SimpleNamespace(hf_device_map=device_map)) is None


class TestEvaluationPeak:
    def test_evaluation_reports_the_fullest_gpu(self, monkeypatch):
        from scripts.evaluate import _peak_vram_gb

        monkeypatch.setitem(sys.modules, "torch", fake_torch(allocated_gib=[5.0, 9.25]))
        assert _peak_vram_gb() == 9.25
