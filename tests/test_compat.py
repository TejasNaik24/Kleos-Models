"""transformers 4.56 ↔ 5.x compatibility tests.

The shim is tested by *injecting* each API generation's field set, so both paths
are covered without installing both versions of transformers. That matters: these
translations are the difference between a run that trains and a `TypeError` deep
inside TrainingArguments.
"""

from __future__ import annotations

from kleos_models.compat import (
    _parse_version,
    build_training_arguments_kwargs,
    check_transformers_version,
    dtype_kwarg,
    library_versions,
    package_version,
)

# Field sets as they exist in each generation.
V4_FIELDS = frozenset(
    {
        "output_dir",
        "num_train_epochs",
        "learning_rate",
        "warmup_ratio",
        "warmup_steps",
        "eval_strategy",
        "evaluation_strategy",
        "save_strategy",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "bf16",
        "fp16",
        "optim",
        "seed",
        "logging_steps",
        "overwrite_output_dir",
        "run_name",
    }
)

# v5 removed warmup_ratio, evaluation_strategy and overwrite_output_dir.
V5_FIELDS = frozenset(
    {
        "output_dir",
        "num_train_epochs",
        "learning_rate",
        "warmup_steps",
        "eval_strategy",
        "save_strategy",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "bf16",
        "fp16",
        "optim",
        "seed",
        "logging_steps",
        "run_name",
    }
)


class TestVersionParsing:
    def test_plain_versions(self):
        assert _parse_version("4.56.1") == (4, 56, 1)
        assert _parse_version("5.15.0") == (5, 15, 0)

    def test_prerelease_suffixes_are_dropped(self):
        assert _parse_version("5.1.0.dev0") == (5, 1, 0)
        assert _parse_version("4.56.0rc1") == (4, 56, 0)

    def test_local_version_is_dropped(self):
        assert _parse_version("2.4.0+cu121") == (2, 4, 0)

    def test_versions_compare_correctly(self):
        assert _parse_version("5.0.0") > _parse_version("4.99.9")
        assert _parse_version("4.56.0") >= (4, 56, 0)


class TestWarmupTranslation:
    def test_v4_keeps_warmup_ratio(self):
        kwargs, notes = build_training_arguments_kwargs({"warmup_ratio": 0.03}, supported=V4_FIELDS)
        assert kwargs["warmup_ratio"] == 0.03
        assert not notes  # nothing to translate on 4.x

    def test_v5_maps_warmup_ratio_to_warmup_steps(self):
        # v5 removed warmup_ratio; warmup_steps is a float where <1 is a ratio.
        kwargs, notes = build_training_arguments_kwargs({"warmup_ratio": 0.03}, supported=V5_FIELDS)
        assert "warmup_ratio" not in kwargs
        assert kwargs["warmup_steps"] == 0.03
        assert any("warmup_ratio" in note for note in notes)

    def test_explicit_warmup_steps_is_not_overwritten(self):
        kwargs, notes = build_training_arguments_kwargs(
            {"warmup_ratio": 0.03, "warmup_steps": 100}, supported=V5_FIELDS
        )
        assert kwargs["warmup_steps"] == 100
        assert any("dropped" in note for note in notes)


class TestEvalStrategyTranslation:
    def test_v5_keeps_eval_strategy(self):
        kwargs, _ = build_training_arguments_kwargs({"eval_strategy": "steps"}, supported=V5_FIELDS)
        assert kwargs["eval_strategy"] == "steps"

    def test_legacy_name_is_used_when_that_is_all_there_is(self):
        legacy = frozenset({"evaluation_strategy", "output_dir"})
        kwargs, _notes = build_training_arguments_kwargs(
            {"eval_strategy": "steps"}, supported=legacy
        )
        assert kwargs["evaluation_strategy"] == "steps"
        assert "eval_strategy" not in kwargs


class TestRemovedArguments:
    def test_overwrite_output_dir_is_dropped_on_v5(self):
        kwargs, notes = build_training_arguments_kwargs(
            {"output_dir": "out", "overwrite_output_dir": True}, supported=V5_FIELDS
        )
        assert "overwrite_output_dir" not in kwargs
        assert any("overwrite_output_dir" in note for note in notes)

    def test_overwrite_output_dir_is_kept_on_v4(self):
        kwargs, _ = build_training_arguments_kwargs(
            {"output_dir": "out", "overwrite_output_dir": True}, supported=V4_FIELDS
        )
        assert kwargs["overwrite_output_dir"] is True

    def test_completely_unknown_arguments_are_dropped_with_a_note(self):
        kwargs, notes = build_training_arguments_kwargs(
            {"output_dir": "out", "invented_flag": 1}, supported=V5_FIELDS
        )
        assert "invented_flag" not in kwargs
        assert any("invented_flag" in note for note in notes)

    def test_supported_arguments_pass_through_untouched(self):
        requested = {
            "output_dir": "out",
            "learning_rate": 2e-4,
            "per_device_train_batch_size": 1,
            "bf16": True,
            "optim": "paged_adamw_8bit",
            "seed": 42,
        }
        kwargs, notes = build_training_arguments_kwargs(requested, supported=V5_FIELDS)
        assert kwargs == requested
        assert not notes


class TestDtypeKwarg:
    def test_modern_signature_uses_dtype(self):
        assert dtype_kwarg("bfloat16", supported=frozenset({"dtype"})) == {"dtype": "bfloat16"}

    def test_legacy_signature_uses_torch_dtype(self):
        assert dtype_kwarg("bfloat16", supported=frozenset({"torch_dtype"})) == {
            "torch_dtype": "bfloat16"
        }

    def test_none_produces_no_kwarg(self):
        assert dtype_kwarg(None, supported=frozenset({"dtype"})) == {}

    def test_both_present_prefers_the_modern_name(self):
        result = dtype_kwarg("bfloat16", supported=frozenset({"dtype", "torch_dtype"}))
        assert result == {"dtype": "bfloat16"}


class TestVersionChecking:
    def test_no_transformers_installed_is_not_an_error(self):
        # This suite runs in the light environment, where transformers is absent.
        result = check_transformers_version()
        assert result is None or isinstance(result, str)

    def test_library_versions_reports_every_tracked_package(self):
        versions = library_versions()
        assert "torch" in versions
        assert "transformers" in versions
        assert "peft" in versions
        # pydantic is a core dependency, so it must be present.
        assert versions["pydantic"] is not None

    def test_package_version_returns_none_for_absent_packages(self):
        assert package_version("a-package-that-does-not-exist-xyz") is None
