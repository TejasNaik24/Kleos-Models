"""Feasibility and memory-estimation tests (spec §3, §13, §14).

These encode the rule that architectural support is not the same as trainability,
and that a configuration is never silently shrunk until the experiment stops being
comparable.

GPUs are simulated, so the whole matrix is testable without hardware.
"""

from __future__ import annotations

import pytest
from tests.conftest import CONFIGS_DIR

from kleos_models.config import (
    FeasibilityTier,
    TrainingConfig,
    load_config,
    load_model_config,
)
from kleos_models.errors import InsufficientMemoryError
from kleos_models.models.adapters import get_adapter
from kleos_models.models.feasibility import (
    GPUInfo,
    ModelShape,
    assess_feasibility,
    enforce_feasibility,
    estimate_lora_parameters,
    estimate_memory,
    render_feasibility_table,
)


def gpu(name: str, vram: float, capability: tuple[int, int]) -> GPUInfo:
    return GPUInfo(
        available=True,
        name=name,
        total_memory_gb=vram,
        free_memory_gb=vram * 0.95,
        compute_capability=capability,
        device_count=1,
        cuda_version="12.4",
        source="simulated",
    )


# Real runtimes this project targets.
T4 = gpu("Tesla T4", 15.8, (7, 5))  # Colab free tier
L4 = gpu("NVIDIA L4", 22.5, (8, 9))  # Colab Pro
A100_40 = gpu("NVIDIA A100-SXM4-40GB", 40.0, (8, 0))
NO_GPU = GPUInfo(available=False, source="simulated: no CUDA")


def model(name: str):
    return load_model_config(CONFIGS_DIR / "models" / f"{name}.yaml")


def assess(name: str, hardware: GPUInfo, training: TrainingConfig | None = None):
    config = model(name)
    adapter = get_adapter(config)
    return assess_feasibility(
        config,
        training or TrainingConfig(),
        gpu=hardware,
        target_modules=adapter.default_target_modules,
    )


class TestGPUInfo:
    def test_bf16_requires_compute_capability_8(self):
        # A Colab T4 is 7.5 and cannot do bfloat16. This is the single most
        # consequential hardware fact for the canonical training environment.
        assert not T4.bf16_supported
        assert L4.bf16_supported
        assert A100_40.bf16_supported

    def test_capability_string_is_rendered(self):
        assert T4.capability_string == "7.5"
        assert NO_GPU.capability_string == "unknown"

    def test_render_reports_bf16_support(self):
        assert "bf16 NO" in T4.render()
        assert "bf16 yes" in A100_40.render()

    def test_absent_gpu_renders_a_reason(self):
        assert "No GPU detected" in NO_GPU.render()

    def test_serialization_includes_bf16(self):
        assert T4.to_dict()["bf16_supported"] is False


class TestModelShape:
    @pytest.mark.parametrize(
        "name", ["qwen3_8b", "ministral_8b", "mistral_small_3_2", "qwen3_30b_a3b_thinking"]
    )
    def test_known_checkpoints_have_real_shapes(self, name):
        shape = ModelShape.from_config(model(name))
        assert shape.parameter_count > 0
        assert shape.num_layers > 0
        assert shape.hidden_size > 0

    def test_moe_shape_records_active_parameters(self):
        shape = ModelShape.from_config(model("qwen3_30b_a3b_thinking"))
        assert shape.is_moe
        assert shape.active_parameter_count is not None
        assert shape.active_parameter_count < shape.parameter_count


class TestLoRAParameterEstimate:
    def test_more_targets_means_more_parameters(self):
        shape = ModelShape.from_config(model("qwen3_8b"))
        attention_only = estimate_lora_parameters(
            shape, 16, ["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        with_mlp = estimate_lora_parameters(
            shape, 16, ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        assert with_mlp > attention_only

    def test_rank_scales_parameters_linearly(self):
        shape = ModelShape.from_config(model("qwen3_8b"))
        r16 = estimate_lora_parameters(shape, 16, ["q_proj"])
        r32 = estimate_lora_parameters(shape, 32, ["q_proj"])
        assert r32 == pytest.approx(r16 * 2)

    def test_adapter_is_a_tiny_fraction_of_the_model(self):
        shape = ModelShape.from_config(model("qwen3_8b"))
        params = estimate_lora_parameters(shape, 16, ["q_proj", "k_proj", "v_proj", "o_proj"])
        assert params / shape.parameter_count < 0.01


class TestMemoryEstimate:
    def test_quantization_reduces_weight_memory(self):
        config = model("qwen3_8b")
        quantized = estimate_memory(config, TrainingConfig())

        unquantized = config.model_copy(deep=True)
        unquantized.quantization.mode = unquantized.quantization.mode.__class__("none")
        full = estimate_memory(unquantized, TrainingConfig())

        assert quantized.base_weights_gb < full.base_weights_gb

    def test_longer_sequences_need_more_activation_memory(self):
        config = model("qwen3_8b")
        short = estimate_memory(config, TrainingConfig(), seq_length=512)
        long = estimate_memory(config, TrainingConfig(), seq_length=4096)
        assert long.activations_gb > short.activations_gb

    def test_gradient_checkpointing_reduces_activation_memory(self):
        config = model("qwen3_8b")
        with_ckpt = estimate_memory(config, TrainingConfig(gradient_checkpointing=True))
        without = estimate_memory(config, TrainingConfig(gradient_checkpointing=False))
        assert with_ckpt.activations_gb < without.activations_gb

    def test_paged_8bit_optimizer_uses_less_memory(self):
        config = model("qwen3_8b")
        paged = estimate_memory(config, TrainingConfig(optim="paged_adamw_8bit"))
        adamw = estimate_memory(config, TrainingConfig(optim="adamw_torch"))
        assert paged.optimizer_gb < adamw.optimizer_gb

    def test_bigger_models_need_more_memory(self):
        small = estimate_memory(model("qwen3_8b"), TrainingConfig())
        large = estimate_memory(model("mistral_small_3_2"), TrainingConfig())
        assert large.base_weights_gb > small.base_weights_gb

    def test_moe_memory_tracks_total_not_active_parameters(self):
        # All experts stay resident even though only a subset activates.
        estimate = estimate_memory(model("qwen3_30b_a3b_thinking"), TrainingConfig())
        assert estimate.base_weights_gb > 14
        assert any("MoE" in assumption for assumption in estimate.assumptions)

    def test_assumptions_are_stated(self):
        estimate = estimate_memory(model("qwen3_8b"), TrainingConfig())
        assert estimate.assumptions
        assert any("bytes/param" in a for a in estimate.assumptions)

    def test_estimate_renders_a_breakdown(self):
        rendered = estimate_memory(model("qwen3_8b"), TrainingConfig()).render()
        assert "base weights" in rendered
        assert "estimated peak" in rendered


class TestFeasibilityTiers:
    def test_8b_models_are_trainable_on_a_free_t4(self):
        for name in ("qwen3_8b", "ministral_8b"):
            report = assess(name, T4)
            assert report.fits, f"{name} should train on a T4 in 4-bit: {report.render()}"
            assert report.tier in (
                FeasibilityTier.ADAPTER_TRAIN,
                FeasibilityTier.FULL_RESEARCH,
            )

    def test_24b_is_not_trainable_on_a_free_t4(self):
        report = assess("mistral_small_3_2", T4)
        assert not report.fits
        assert report.blocking_reasons
        assert any("A100" in rec or "8B" in rec for rec in report.recommendations)

    def test_30b_moe_is_not_trainable_on_a_free_t4(self):
        report = assess("qwen3_30b_a3b_thinking", T4)
        assert not report.fits

    def test_larger_models_become_feasible_on_an_a100(self):
        report = assess("mistral_small_3_2", A100_40)
        assert report.fits, report.render()

    def test_no_gpu_is_infeasible_with_useful_advice(self):
        report = assess("qwen3_8b", NO_GPU)
        assert report.tier is FeasibilityTier.INFEASIBLE
        assert any("Colab" in rec for rec in report.recommendations)

    def test_report_serializes(self):
        payload = assess("qwen3_8b", T4).to_dict()
        assert "tier" in payload
        assert "estimate" in payload
        assert "gpu" in payload

    def test_table_renders_every_model(self):
        reports = [
            assess(name, T4)
            for name in ("qwen3_8b", "ministral_8b", "mistral_small_3_2", "qwen3_30b_a3b_thinking")
        ]
        table = render_feasibility_table(reports)
        assert "qwen3_8b" in table
        assert "infeasible" in table
        assert "Tesla T4" in table


class TestAdjustmentPolicy:
    def _tight_config(self) -> TrainingConfig:
        """A configuration that will not fit a T4 as requested."""
        return TrainingConfig(per_device_train_batch_size=8, gradient_checkpointing=False)

    def test_a_too_large_config_proposes_adjustments(self):
        config = model("qwen3_8b")
        config.max_seq_length = 8192
        adapter = get_adapter(config)
        report = assess_feasibility(
            config, self._tight_config(), gpu=T4, target_modules=adapter.default_target_modules
        )
        assert report.proposed_adjustments

    def test_adjustments_carry_a_reason(self):
        config = model("qwen3_8b")
        config.max_seq_length = 8192
        adapter = get_adapter(config)
        report = assess_feasibility(
            config, self._tight_config(), gpu=T4, target_modules=adapter.default_target_modules
        )
        for adjustment in report.proposed_adjustments:
            assert adjustment.reason
            assert adjustment.original != adjustment.adjusted

    def test_strict_config_refuses_to_adjust(self):
        # Spec §14: a research run must not be silently reshaped.
        config = model("qwen3_8b")
        config.max_seq_length = 8192
        training = self._tight_config()
        training.strict_config = True
        adapter = get_adapter(config)
        report = assess_feasibility(
            config, training, gpu=T4, target_modules=adapter.default_target_modules
        )
        if report.proposed_adjustments:
            with pytest.raises(InsufficientMemoryError, match="strict_config"):
                enforce_feasibility(report, training)

    def test_non_strict_config_returns_the_adjustments_to_record(self):
        config = model("qwen3_8b")
        config.max_seq_length = 8192
        training = self._tight_config()
        adapter = get_adapter(config)
        report = assess_feasibility(
            config, training, gpu=T4, target_modules=adapter.default_target_modules
        )
        adjustments = enforce_feasibility(report, training)
        assert isinstance(adjustments, list)

    def test_infeasible_runs_raise_rather_than_shrink(self):
        report = assess("qwen3_30b_a3b_thinking", T4)
        with pytest.raises(InsufficientMemoryError):
            enforce_feasibility(report, TrainingConfig())

    def test_a_fitting_run_needs_no_adjustment(self):
        report = assess("qwen3_8b", A100_40)
        assert enforce_feasibility(report, TrainingConfig()) == []


class TestQuantizationSupport:
    def test_support_probe_never_raises(self):
        from kleos_models.models.quantization import quantization_support

        support = quantization_support()
        assert isinstance(support.can_quantize, bool)
        assert isinstance(support.to_dict(), dict)

    def test_probe_explains_why_quantization_is_unavailable(self):
        from kleos_models.models.quantization import quantization_support

        support = quantization_support()
        if not support.can_quantize:
            assert support.reason


class TestPreFlightReport:
    def test_report_contains_everything_the_spec_requires(self):
        from kleos_models.training.memory import render_environment_report

        config = load_config(CONFIGS_DIR / "training" / "qlora_small.yaml")
        report = render_environment_report(config.model, config.training, gpu=T4)

        # Spec §13 requires each of these before an expensive run.
        for expected in (
            "GPU",
            "VRAM",
            "compute capability",
            "CUDA",
            "torch",
            "transformers",
            "peft",
            "bitsandbytes",
            "base model",
            "quantization",
            "LoRA",
            "Estimated memory",
            "seed",
        ):
            assert expected in report, f"pre-flight report is missing {expected!r}"

    def test_report_warns_when_bf16_is_unavailable(self):
        from kleos_models.training.memory import render_environment_report

        config = load_config(CONFIGS_DIR / "training" / "qlora_small.yaml")
        assert "float16 will be used" in render_environment_report(
            config.model, config.training, gpu=T4
        )


# ---------------------------------------------------------------------------
# The rebuilt estimator (finding H-F10) and KLEOS Logos
# ---------------------------------------------------------------------------

from kleos_models.models.feasibility import (  # noqa: E402
    EMPIRICAL_ANCHORS,
    GPU_PRESETS,
    NON_ALLOCATOR_GB,
    REFERENCE_PAGED_GB,
    memory_budget_gb,
    simulated_gpu,
)

SEVEN = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
T4_COLAB = GPU_PRESETS["t4-colab"]


def training_config(name: str):
    return load_config(CONFIGS_DIR / "training" / name)


class TestExactLoRACounts:
    """The counts PEFT reported on real runs, from the shapes alone."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("mistral_nemo_12b", 57_016_320),  # Hermes' manifest
            ("ministral_8b", 43_646_976),  # the Ministral-8B run's manifest
            ("ministral3_14b", 60_948_480),  # Logos, 280 modules
        ],
    )
    def test_seven_projection_counts(self, name, expected):
        shape = ModelShape.from_config(model(name))
        assert estimate_lora_parameters(shape, 16, SEVEN) == expected

    def test_logos_text_parameters_are_the_verified_count(self):
        shape = ModelShape.from_config(model("ministral3_14b"))
        assert shape.parameter_count == 13_506_073_600
        assert shape.vision_parameter_count == 438_958_080


class TestEmpiricalAnchors:
    @pytest.mark.parametrize("anchor", EMPIRICAL_ANCHORS, ids=lambda a: a.run)
    def test_the_estimate_reproduces_a_measured_peak(self, anchor):
        # Both measured on this project's T4 runs; a change that breaks either
        # is a change to the physics, and must be justified against them.
        assert abs(anchor.deviation()) < 0.02, (anchor.estimate_gb(), anchor.measured_peak_gb)


class TestLogosOnAFreeT4:
    def assess_at(self, name: str, seq: int | None):
        config = training_config(name)
        adapter = get_adapter(config.model)
        return assess_feasibility(
            config.model,
            config.training,
            gpu=T4_COLAB,
            target_modules=adapter.resolve_target_modules(config.model.lora),
            seq_length=seq,
        )

    def test_logos_at_its_longest_example_is_a_marginal_fit(self):
        report = self.assess_at("kleos_logos_v001.yaml", 496)
        assert 13.4 <= report.estimate.peak_allocated_gb <= 14.2
        assert report.tier is FeasibilityTier.ADAPTER_TRAIN
        assert report.fits and report.marginal
        assert not report.proposed_adjustments
        assert any("expandable_segments" in r for r in report.recommendations)

    def test_hermes_at_its_longest_example_fits_comfortably(self):
        report = self.assess_at("kleos_hermes_v006.yaml", 440)
        assert report.fits and not report.marginal
        assert report.headroom_gb > 0.5

    def test_the_worst_case_length_would_have_refused_both(self):
        # Why train.py measures the data first: at max_seq_length the estimate
        # would shrink Hermes' config and refuse Logos', though Hermes ran.
        for name in ("kleos_logos_v001.yaml", "kleos_hermes_v006.yaml"):
            assert not self.assess_at(name, None).fits

    def test_mistral_small_24b_does_not_fit_a_colab_t4(self):
        report = assess("mistral_small_3_2", T4_COLAB)
        assert not report.fits

    def test_the_vision_tower_counts_only_when_loaded(self):
        small = model("mistral_small_3_2")
        loaded = estimate_memory(small, TrainingConfig(), loads_vision_tower=True)
        skipped = estimate_memory(small, TrainingConfig(), loads_vision_tower=False)
        assert loaded.base_weights_gb > skipped.base_weights_gb
        logos = estimate_memory(model("ministral3_14b"), TrainingConfig())
        assert not any("vision tower" in a for a in logos.assumptions)

    def test_the_estimate_serializes_its_new_terms(self):
        payload = estimate_memory(
            model("ministral3_14b"), TrainingConfig(), seq_length=496
        ).to_dict()
        for key in ("peak_allocated_gb", "reserve_gb", "minimum_gb", "sequence_length"):
            assert key in payload
        assert payload["sequence_length"] == 496


class TestBudgets:
    def test_measured_free_memory_is_the_budget_less_paged_state(self):
        live = GPUInfo(available=True, total_memory_gb=15.0, free_memory_gb=14.0)
        assert memory_budget_gb(live, paged_gb=0.25) == pytest.approx(13.75)

    def test_without_a_measurement_the_non_allocator_use_is_deducted(self):
        preset = GPU_PRESETS["t4-colab"]
        assert memory_budget_gb(preset) == pytest.approx(14.56 - NON_ALLOCATOR_GB)
        excess = REFERENCE_PAGED_GB + 1.0
        assert memory_budget_gb(preset, paged_gb=excess) == pytest.approx(
            14.56 - NON_ALLOCATOR_GB - 1.0
        )

    def test_presets_hold_only_measured_hardware(self):
        # 14.56 GiB is what torch reported on Colab's T4 during Hermes' run.
        assert set(GPU_PRESETS) == {"t4-colab"}
        assert T4_COLAB.total_memory_gb == 14.56
        assert T4_COLAB.compute_capability == (7, 5)
        assert "Hermes run report" in T4_COLAB.source

    def test_a_custom_gpu_spec_is_parsed(self):
        gpu = simulated_gpu("L4:22.0:8.9")
        assert gpu.total_memory_gb == 22.0
        assert gpu.compute_capability == (8, 9)
        assert gpu.bf16_supported
        assert "not measured" in gpu.source

    @pytest.mark.parametrize("spec", ["a100", "L4:big:8.9", "L4:22:eight", "L4:-1:8.9"])
    def test_a_bad_gpu_spec_is_refused(self, spec):
        with pytest.raises(ValueError):
            simulated_gpu(spec)

    def test_the_smoke_tier_assumes_gradient_checkpointing(self):
        # A config without checkpointing that fits once it is switched on is
        # SMOKE (adjustable), not INFERENCE_ONLY.
        config = model("ministral_8b")
        report = assess_feasibility(
            config,
            TrainingConfig(gradient_checkpointing=False, per_device_train_batch_size=4),
            gpu=T4_COLAB,
            target_modules=SEVEN,
        )
        assert report.tier is FeasibilityTier.SMOKE
        assert any(
            a.field == "training.gradient_checkpointing" for a in report.proposed_adjustments
        )


class TestPlanRunScript:
    def run(self, capsys, *argv: str) -> tuple[int, str]:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "plan_run_under_test", CONFIGS_DIR.parent / "scripts" / "plan_run.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        code = module.main(list(argv))
        return code, capsys.readouterr().out

    def test_logos_plan_on_a_simulated_t4(self, capsys):
        code, out = self.run(
            capsys,
            "--config",
            str(CONFIGS_DIR / "training" / "kleos_logos_v001.yaml"),
            "--simulate-gpu",
            "t4-colab",
            "--seq-length",
            "496",
        )
        assert code == 0
        assert "MARGINAL" in out and "60,948,480" in out

    def test_set_model_plans_another_model_in_the_same_recipe(self, capsys):
        code, out = self.run(
            capsys,
            "--config",
            str(CONFIGS_DIR / "training" / "kleos_hermes_v006.yaml"),
            "--set-model",
            str(CONFIGS_DIR / "models" / "ministral3_14b.yaml"),
            "--simulate-gpu",
            "t4-colab",
            "--seq-length",
            "496",
        )
        assert code == 0
        assert "Ministral-3-14B" in out
        assert "in place of the config's own" in out

    def test_a_config_that_does_not_fit_exits_2(self, capsys):
        code, _ = self.run(
            capsys,
            "--config",
            str(CONFIGS_DIR / "training" / "kleos_logos_v001.yaml"),
            "--simulate-gpu",
            "t4-colab",
        )
        assert code == 2  # worst case: max_seq_length 1024

    @pytest.mark.parametrize(
        "argv",
        [
            ["--all-models", "--set-model", "x.yaml"],
            ["--config", "c.yaml", "--seq-length", "0"],
            ["--config", "c.yaml", "--simulate-gpu", "nonsense"],
        ],
    )
    def test_bad_arguments_are_refused(self, capsys, argv):
        with pytest.raises(SystemExit) as info:
            self.run(capsys, *argv)
        assert info.value.code == 2
