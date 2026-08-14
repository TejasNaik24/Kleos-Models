"""Typed configuration for KLEOS experiments (spec section 6).

Design rules
------------
1. **No critical hyperparameter lives only in Python.** Everything that changes
   an experiment's outcome is a YAML field validated by a pydantic model.
2. **Configs compose.** A training config ``extends`` a base and ``includes``
   model/dataset/evaluation fragments, so the same pipeline consumes Qwen and
   Mistral configurations unchanged.
3. **Configs hash.** ``ExperimentConfig.config_hash`` is a stable digest of the
   fully resolved configuration and is recorded in every experiment manifest.
   Two runs with the same hash used the same knobs.

This module imports no torch and no transformers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kleos_models.constants import (
    QUALITY_STATUSES,
    RESEARCH_ARMS,
    SPLIT_STRATEGIES,
    SUPPORTED_TASKS,
)
from kleos_models.errors import ConfigError
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ReasoningCapability(str, Enum):
    """What a checkpoint can actually do with reasoning (spec section 39).

    This is a *model property*, not a prompting trick. It is declared by the
    model-family adapter and cross-checked against the config.
    """

    #: No reasoning mode. e.g. Mistral Small 3.2, Ministral 8B.
    UNSUPPORTED = "unsupported"
    #: Reasoning can be switched on or off. e.g. Qwen3-8B via `enable_thinking`.
    SWITCHABLE = "switchable"
    #: Always reasons; cannot be disabled. e.g. Qwen3-30B-A3B-Thinking-2507.
    ALWAYS_ON = "always_on"


class ReasoningMode(str, Enum):
    """Requested reasoning behaviour for a run."""

    STANDARD = "standard"
    NON_THINKING = "non_thinking"
    THINKING = "thinking"


class QuantizationMode(str, Enum):
    """Weight quantization for base-model loading."""

    NONE = "none"
    INT8 = "int8"
    NF4 = "nf4"
    FP4 = "fp4"

    @property
    def is_4bit(self) -> bool:
        return self in (QuantizationMode.NF4, QuantizationMode.FP4)

    @property
    def bits(self) -> int:
        if self.is_4bit:
            return 4
        if self is QuantizationMode.INT8:
            return 8
        return 16


class Precision(str, Enum):
    """Training compute precision. ``AUTO`` resolves from GPU capability."""

    AUTO = "auto"
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


class DType(str, Enum):
    """Torch dtype selector, resolved lazily so config stays torch-free."""

    AUTO = "auto"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


class FeasibilityTier(str, Enum):
    """How far a given (model, config, GPU) combination can go.

    Spec section 3: architectural support is not the same as trainability on a
    free GPU. These tiers keep that distinction explicit instead of letting a
    run quietly shrink until it is no longer comparable.
    """

    INFEASIBLE = "infeasible"
    INFERENCE_ONLY = "inference_only"
    SMOKE = "smoke"
    ADAPTER_TRAIN = "adapter_train"
    FULL_RESEARCH = "full_research"


# ---------------------------------------------------------------------------
# Base model with strict behaviour
# ---------------------------------------------------------------------------


class StrictModel(BaseModel):
    """Reject unknown keys so typos in YAML fail loudly rather than silently."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_enum_values=False)


# ---------------------------------------------------------------------------
# Model-side configuration
# ---------------------------------------------------------------------------


class ReasoningConfig(StrictModel):
    """Reasoning behaviour requested for a model."""

    supported: bool = Field(
        default=False,
        description="Whether the checkpoint supports a reasoning mode at all.",
    )
    default_mode: ReasoningMode = Field(
        default=ReasoningMode.STANDARD,
        description="Mode used when a run does not request one explicitly.",
    )
    capability: ReasoningCapability | None = Field(
        default=None,
        description=(
            "Declared capability. Left null in YAML, it is filled in from the "
            "model-family adapter and cross-checked."
        ),
    )
    strip_thinking_from_targets: bool = Field(
        default=True,
        description=(
            "Remove reasoning spans from assistant targets before computing loss. "
            "Keep this true: we train decision policy, not display chain-of-thought."
        ),
    )
    strip_thinking_from_history: bool = Field(
        default=True,
        description=(
            "Remove reasoning spans from prior assistant turns, per Qwen's "
            "multi-turn guidance for Thinking checkpoints."
        ),
    )

    @model_validator(mode="after")
    def _check_mode_matches_support(self) -> ReasoningConfig:
        if not self.supported and self.default_mode is ReasoningMode.THINKING:
            raise ValueError("reasoning.default_mode='thinking' requires reasoning.supported=true")
        return self


class QuantizationConfig(StrictModel):
    """bitsandbytes quantization settings (spec section 13)."""

    mode: QuantizationMode = QuantizationMode.NF4
    compute_dtype: DType = Field(
        default=DType.AUTO,
        description=(
            "Compute dtype for dequantized matmuls. 'auto' selects bfloat16 on "
            "compute capability >= 8.0 and float16 otherwise (a Colab T4 is 7.5 "
            "and cannot do bfloat16)."
        ),
    )
    double_quant: bool = Field(
        default=True, description="Nested quantization of the quantization constants."
    )
    quant_storage_dtype: DType = Field(
        default=DType.AUTO, description="Storage dtype for packed 4-bit weights."
    )
    skip_modules: list[str] = Field(
        default_factory=list,
        description=(
            "Modules kept in full precision. The family adapter adds architectural "
            "requirements (for example a vision tower) on top of this list."
        ),
    )

    @property
    def enabled(self) -> bool:
        return self.mode is not QuantizationMode.NONE


class LoRAConfig(StrictModel):
    """LoRA / QLoRA adapter configuration (spec section 6)."""

    r: int = Field(default=16, ge=1, le=512)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: list[str] | Literal["auto"] = Field(
        default="auto",
        description=(
            "Module suffixes to adapt. 'auto' asks the model-family adapter for "
            "architecture-appropriate defaults. Never copy these from a Llama "
            "tutorial: they are validated against the loaded model and a missing "
            "target is a hard error (spec section 46)."
        ),
    )
    exclude_modules: list[str] = Field(
        default_factory=list,
        description="Extra name fragments to exclude, merged with adapter exclusions.",
    )
    modules_to_save: list[str] = Field(
        default_factory=list,
        description="Non-LoRA modules trained in full and saved with the adapter.",
    )
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    use_rslora: bool = False
    init_lora_weights: bool | str = True

    @field_validator("target_modules")
    @classmethod
    def _no_empty_targets(cls, value: list[str] | str) -> list[str] | str:
        if isinstance(value, list) and not value:
            raise ValueError("target_modules must be 'auto' or a non-empty list")
        return value

    @property
    def scaling(self) -> float:
        """LoRA scaling factor alpha / r, reported in run summaries."""
        return self.alpha / self.r


class ModelConfig(StrictModel):
    """A base model plus everything family-specific about using it.

    Fields such as ``architecture``, ``parameter_count`` and
    ``active_parameter_count`` are recorded in the experiment manifest so results
    can never be accidentally pooled across incompatible checkpoints
    (spec sections 3 and 38).
    """

    name: str = Field(description="Short config identifier, e.g. 'qwen3_8b'.")
    family: str = Field(description="Model family: 'qwen', 'mistral', ...")
    base_model: str = Field(description="Hugging Face checkpoint id.")
    revision: str = Field(
        default="main",
        description="Git revision of the checkpoint. Pin a commit sha for real runs.",
    )
    tokenizer: str | None = Field(
        default=None, description="Tokenizer id when it differs from base_model."
    )
    trust_remote_code: bool = False

    # Architecture facts. Optional, but recorded when supplied and cross-checked
    # against the loaded model so a config cannot quietly describe a different
    # checkpoint than the one being trained.
    model_type: str | None = Field(
        default=None, description="HF config model_type, e.g. 'qwen3', 'mistral3'."
    )
    architecture: str | None = Field(default=None, description="Expected architecture class name.")
    parameter_count: float | None = Field(default=None, ge=0)
    active_parameter_count: float | None = Field(
        default=None,
        ge=0,
        description="Params active per token. MoE only; do not treat 30B MoE as dense 30B.",
    )
    is_moe: bool = False
    is_multimodal: bool = False
    context_limit: int | None = Field(default=None, ge=1)

    max_seq_length: int = Field(
        default=2048,
        ge=8,
        description="Training/eval sequence length. Must not exceed context_limit.",
    )
    dtype: DType = DType.AUTO
    attn_implementation: str | None = Field(
        default=None, description="e.g. 'sdpa', 'eager', 'flash_attention_2'."
    )
    device_map: str | None = Field(
        default="auto", description="accelerate device map; 'auto' for single-GPU 4-bit."
    )

    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)

    chat_template: str | None = Field(
        default=None,
        description="Override the tokenizer chat template. Leave null to use the official one.",
    )
    notes: str | None = None

    @model_validator(mode="after")
    def _check_lengths(self) -> ModelConfig:
        if self.context_limit is not None and self.max_seq_length > self.context_limit:
            raise ValueError(
                f"max_seq_length={self.max_seq_length} exceeds the model's "
                f"context_limit={self.context_limit}"
            )
        if self.is_moe and self.active_parameter_count is None:
            logger.warning(
                "Model %s is marked MoE but has no active_parameter_count; "
                "compute planning will assume dense behaviour.",
                self.name,
            )
        return self

    @property
    def tokenizer_id(self) -> str:
        """Tokenizer checkpoint, defaulting to the base model."""
        return self.tokenizer or self.base_model


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------


class DatasetFilters(StrictModel):
    """Row filters applied after loading and before splitting."""

    tasks: list[str] = Field(
        default_factory=list, description="Keep only these tasks. Empty means all."
    )
    domains: list[str] = Field(default_factory=list)
    quality_statuses: list[str] = Field(
        default_factory=lambda: ["reviewed"],
        description=(
            "Quality gate. Defaults to reviewed-only; widen it deliberately and "
            "record that you did."
        ),
    )
    exclude_source_types: list[str] = Field(default_factory=list)
    min_messages: int = Field(default=2, ge=1)
    max_examples: int | None = Field(default=None, ge=1)

    @field_validator("tasks")
    @classmethod
    def _known_tasks(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(SUPPORTED_TASKS))
        if unknown:
            raise ValueError(
                f"unknown task(s) {unknown}; registered tasks are {list(SUPPORTED_TASKS)}"
            )
        return value

    @field_validator("quality_statuses")
    @classmethod
    def _known_statuses(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(QUALITY_STATUSES))
        if unknown:
            raise ValueError(f"unknown quality status {unknown}; valid: {list(QUALITY_STATUSES)}")
        return value


class SplitConfig(StrictModel):
    """How to partition examples (spec section 12)."""

    strategy: str = Field(
        default="random",
        description=(
            "random is for development only. Generalization claims require a held-out strategy."
        ),
    )
    seed: int = 42
    train_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    validation_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    test_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    group_key: str | None = Field(
        default=None,
        description="Metadata key defining the group for group/scenario strategies.",
    )
    holdout_values: list[str] = Field(
        default_factory=list,
        description=(
            "Values routed to the test split for *_holdout strategies "
            "(entities, domains, formats). Empty means choose deterministically by seed."
        ),
    )

    @field_validator("strategy")
    @classmethod
    def _known_strategy(cls, value: str) -> str:
        if value not in SPLIT_STRATEGIES:
            raise ValueError(f"unknown split strategy {value!r}; valid: {list(SPLIT_STRATEGIES)}")
        return value

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> SplitConfig:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"split fractions must sum to 1.0, got {total:.6f} "
                f"(train={self.train_fraction}, validation={self.validation_fraction}, "
                f"test={self.test_fraction})"
            )
        return self


class DatasetConfig(StrictModel):
    """Where the data is and how it is prepared.

    ``path`` may point outside this repository. That is the supported route for
    consuming the private ``kleos-training-data`` artifact (spec section 27):
    this public repo never embeds private data and never reaches into Supabase.
    """

    version: str = Field(
        default="kleos-dev-fixtures-v0.0.0",
        description="Dataset version string, e.g. 'kleos-policy-v0.1.0'.",
    )
    path: Path | None = Field(
        default=None,
        description="Directory holding manifest.json and the split jsonl files.",
    )
    train_path: Path | None = None
    validation_path: Path | None = None
    test_path: Path | None = None

    filters: DatasetFilters = Field(default_factory=DatasetFilters)
    split: SplitConfig = Field(default_factory=SplitConfig)

    packing: bool = Field(
        default=False,
        description=(
            "Concatenate short examples up to max_seq_length. Off by default: "
            "packing changes the loss distribution and complicates per-example "
            "attribution."
        ),
    )
    shuffle: bool = True
    require_manifest: bool = Field(
        default=False,
        description="Refuse to train on a dataset directory with no manifest.json.",
    )
    allow_unreviewed: bool = Field(
        default=False,
        description="Permit non-reviewed examples. Real research runs keep this false.",
    )

    @model_validator(mode="after")
    def _need_some_source(self) -> DatasetConfig:
        if self.path is None and self.train_path is None:
            raise ValueError("dataset needs either 'path' or an explicit 'train_path'")
        return self

    def resolved_train_path(self) -> Path:
        """Path to the training jsonl."""
        if self.train_path is not None:
            return self.train_path
        assert self.path is not None  # guaranteed by validator
        return self.path / "train.jsonl"

    def resolved_split_path(self, split: str) -> Path | None:
        """Path to a named split, or ``None`` when it is not configured."""
        explicit = {
            "train": self.train_path,
            "validation": self.validation_path,
            "test": self.test_path,
        }[split]
        if explicit is not None:
            return explicit
        if self.path is None:
            return None
        candidate = self.path / f"{split}.jsonl"
        return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


class TrainingConfig(StrictModel):
    """Trainer hyperparameters and run policy.

    The defaults are conservative engineering defaults for a first end-to-end
    QLoRA run, **not** tuned research settings (spec sections 47 and 48).
    """

    output_dir: Path = Field(default=Path("outputs"))
    run_name: str | None = None

    num_train_epochs: float = Field(default=3.0, gt=0)
    max_steps: int = Field(
        default=-1, description="Overrides epochs when > 0. Used by the debug config."
    )
    learning_rate: float = Field(default=2e-4, gt=0, le=1.0)
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = Field(default=0.03, ge=0.0, lt=1.0)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    per_device_eval_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    optim: str = Field(
        default="paged_adamw_8bit",
        description=(
            "Optimizer. Paged 8-bit keeps optimizer state off the critical VRAM "
            "path; requires bitsandbytes. Falls back with a recorded adjustment "
            "when unavailable."
        ),
    )
    adam_beta1: float = Field(default=0.9, gt=0, lt=1)
    adam_beta2: float = Field(default=0.999, gt=0, lt=1)
    adam_epsilon: float = Field(default=1e-8, gt=0)

    gradient_checkpointing: bool = True
    precision: Precision = Precision.AUTO
    seed: int = 42
    data_seed: int | None = None

    logging_steps: int = Field(default=5, ge=1)
    save_strategy: Literal["no", "steps", "epoch"] = "steps"
    save_steps: int = Field(default=50, ge=1)
    save_total_limit: int = Field(
        default=3,
        ge=1,
        description=(
            "Checkpoints retained. Never set so low that a Colab disconnect leaves "
            "nothing to resume from; the checkpoint manager enforces a floor of 1."
        ),
    )
    eval_strategy: Literal["no", "steps", "epoch"] = "steps"
    eval_steps: int = Field(default=50, ge=1)
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    dataloader_num_workers: int = Field(default=0, ge=0)
    group_by_length: bool = False
    report_to: list[str] = Field(
        default_factory=list,
        description="Reporting integrations, e.g. ['wandb']. Empty disables all.",
    )

    resume_from_checkpoint: str | None = Field(
        default=None,
        description="Checkpoint path, or 'auto' to resume from the newest in output_dir.",
    )

    strict_config: bool = Field(
        default=False,
        description=(
            "When true, any automatic adjustment (sequence length, precision, "
            "optimizer, batch size) is a hard error instead of a recorded change. "
            "Turn this on for real research runs so a comparison cannot be "
            "silently invalidated by a fallback."
        ),
    )
    allow_full_finetune: bool = Field(
        default=False,
        description=(
            "Guard against accidentally training all parameters. QLoRA runs keep "
            "this false; the trainer refuses to proceed without an adapter."
        ),
    )

    @model_validator(mode="after")
    def _check_best_model_requires_eval(self) -> TrainingConfig:
        if self.load_best_model_at_end and self.eval_strategy == "no":
            raise ValueError("load_best_model_at_end=true requires eval_strategy != 'no'")
        if self.load_best_model_at_end and self.save_strategy != self.eval_strategy:
            raise ValueError(
                "load_best_model_at_end=true requires save_strategy == eval_strategy "
                f"(got {self.save_strategy!r} and {self.eval_strategy!r})"
            )
        return self

    @property
    def effective_batch_size(self) -> int:
        """Tokens-per-update batch, ignoring data parallelism."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------


class GenerationConfig(StrictModel):
    """Decoding settings held identical across arms for fair comparison."""

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    max_new_tokens: int = Field(default=512, ge=1)
    do_sample: bool = False
    repetition_penalty: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _sampling_consistency(self) -> GenerationConfig:
        if self.do_sample and self.temperature == 0.0:
            raise ValueError("do_sample=true with temperature=0.0 is contradictory")
        return self


class ConsistencyConfig(StrictModel):
    """Consistency testing settings (spec section 22)."""

    enabled: bool = True
    group_key: str = Field(
        default="scenario_family",
        description="Metadata key linking logically equivalent perturbations.",
    )
    perturbations: list[str] = Field(default_factory=list)
    min_group_size: int = Field(default=2, ge=2)


class OODConfig(StrictModel):
    """Out-of-distribution evaluation settings (spec section 23)."""

    enabled: bool = True
    benchmark_path: Path | None = None
    shift_kinds: list[str] = Field(default_factory=list)
    report_separately: bool = Field(
        default=True,
        description="Always report in-distribution, OOD and gap separately, never collapsed.",
    )


class CapabilityConfig(StrictModel):
    """General-capability regression suite (spec section 24)."""

    enabled: bool = False
    benchmark_path: Path | None = None
    description: str = "Fixed general-reasoning regression subset, held constant across runs."


class EvaluationConfig(StrictModel):
    """What to evaluate and how to grade it."""

    benchmark_path: Path | None = None
    arms: list[str] = Field(default_factory=lambda: ["arm0_base", "arm2_finetuned"])
    graders: list[str] = Field(default_factory=lambda: ["exact_match"])
    metrics: list[str] = Field(default_factory=lambda: ["accuracy"])
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    seeds: list[int] = Field(
        default_factory=lambda: [42],
        description="One evaluation pass per seed; results are reported with variance.",
    )
    batch_size: int = Field(default=1, ge=1)
    max_examples: int | None = Field(default=None, ge=1)

    consistency: ConsistencyConfig = Field(default_factory=ConsistencyConfig)
    ood: OODConfig = Field(default_factory=OODConfig)
    capability: CapabilityConfig = Field(default_factory=CapabilityConfig)

    adapter_path: Path | None = Field(default=None, description="LoRA adapter for fine-tuned arms.")
    orchestration_prompt_path: Path | None = Field(
        default=None, description="Scaffolding prompt used by orchestrated arms."
    )

    @field_validator("arms")
    @classmethod
    def _known_arms(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(RESEARCH_ARMS))
        if unknown:
            raise ValueError(f"unknown research arm(s) {unknown}; valid: {list(RESEARCH_ARMS)}")
        if not value:
            raise ValueError("at least one evaluation arm is required")
        return value

    @field_validator("seeds")
    @classmethod
    def _seeds_nonempty(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("at least one evaluation seed is required")
        if len(set(value)) != len(value):
            raise ValueError(f"duplicate evaluation seeds: {value}")
        return value


# ---------------------------------------------------------------------------
# Top-level experiment configuration
# ---------------------------------------------------------------------------


class ExperimentConfig(StrictModel):
    """A complete, self-describing experiment definition."""

    name: str = Field(default="kleos-experiment")
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    hypothesis: str | None = Field(
        default=None,
        description="Pre-registered hypothesis id from docs/experiments.md.",
    )
    task: str | None = Field(
        default=None, description="Primary KLEOS task under study, if single-task."
    )
    seed: int = 42

    model: ModelConfig
    dataset: DatasetConfig | None = None
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    #: Populated by the loader; not part of the hash input.
    source_files: list[str] = Field(default_factory=list, exclude=True)

    @field_validator("task")
    @classmethod
    def _known_task(cls, value: str | None) -> str | None:
        if value is not None and value not in SUPPORTED_TASKS:
            raise ValueError(f"unknown task {value!r}; registered: {list(SUPPORTED_TASKS)}")
        return value

    @model_validator(mode="after")
    def _propagate_seed(self) -> ExperimentConfig:
        # A single top-level seed keeps runs reproducible unless a component
        # deliberately overrides it in YAML.
        if "seed" not in self.training.model_fields_set:
            self.training.seed = self.seed
        return self

    def canonical_dict(self) -> dict[str, Any]:
        """Deterministic, JSON-safe view of the configuration."""
        return json.loads(self.model_dump_json(exclude={"source_files"}))

    @property
    def config_hash(self) -> str:
        """Stable sha256 over the resolved configuration.

        Recorded in every manifest. Identical hashes mean identical knobs, which
        is what makes "same config, different seed" a checkable claim.
        """
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def short_hash(self, length: int = 8) -> str:
        """Truncated config hash for directory and run names."""
        return self.config_hash[:length]

    def to_yaml(self) -> str:
        """Serialize back to YAML for the run's effective-config artifact."""
        return yaml.safe_dump(self.canonical_dict(), sort_keys=True, default_flow_style=False)

    def save(self, path: Path | str) -> Path:
        """Write the effective configuration next to the run outputs."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_yaml(), encoding="utf-8")
        return target


# ---------------------------------------------------------------------------
# YAML loading, composition and overrides
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """Expand ``${env:VAR}`` / ``${env:VAR:default}`` inside strings."""
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            resolved = os.environ.get(name)
            if resolved is None:
                if default is None:
                    raise ConfigError(
                        f"Config references environment variable {name!r}, which is not set.",
                        details={"variable": name},
                        suggestions=[
                            f"Export it: export {name}=...",
                            f"Or give it a default in YAML: ${{env:{name}:fallback}}",
                            "See .env.example for the variables this project understands.",
                        ],
                    )
                return default
            return resolved

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``.

    Dicts merge key-wise; every other type (including lists) is replaced
    outright. Replacing lists is deliberate — merging ``target_modules`` element
    by element would silently produce a target set nobody wrote down.
    """
    result: dict[str, Any] = dict(copy.deepcopy(dict(base)))
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(
            f"Config file not found: {path}",
            details={"resolved_path": str(path.resolve()) if path.parent.exists() else str(path)},
            suggestions=[
                "Check the path is relative to the repository root.",
                "List available configs: ls configs/training configs/models",
            ],
        )
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Config file {path} is not valid YAML.",
            details={"parser_error": str(exc)},
            suggestions=["Check indentation and quoting around ':' characters."],
        ) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(
            f"Config file {path} must contain a YAML mapping at the top level.",
            details={"found_type": type(loaded).__name__},
        )
    return loaded


def _resolve_layers(path: Path, seen: list[Path]) -> dict[str, Any]:
    """Load one YAML file with its ``extends`` and ``includes`` resolved.

    ``extends`` supplies defaults that the current file overrides. ``includes``
    maps a top-level section (``model``, ``dataset``, ``training``,
    ``evaluation``) to another YAML file, which is how one training config
    consumes an unmodified model config.

    Relative paths resolve against the file that declares them.
    """
    resolved = path.resolve()
    if resolved in seen:
        cycle = " -> ".join(p.name for p in [*seen, resolved])
        raise ConfigError(
            "Config inheritance cycle detected.",
            details={"cycle": cycle},
            suggestions=["Break the loop by removing one 'extends' entry."],
        )
    seen = [*seen, resolved]

    raw = _read_yaml(resolved)
    base_dir = resolved.parent

    merged: dict[str, Any] = {}

    extends = raw.pop("extends", None)
    if extends:
        parents = [extends] if isinstance(extends, str) else list(extends)
        for parent in parents:
            parent_path = (base_dir / str(parent)).resolve()
            merged = deep_merge(merged, _resolve_layers(parent_path, seen))

    includes = raw.pop("includes", None) or {}
    if not isinstance(includes, Mapping):
        raise ConfigError(
            f"'includes' in {path} must be a mapping of section -> file path.",
            details={"found_type": type(includes).__name__},
        )
    for section, include_path in includes.items():
        include_resolved = (base_dir / str(include_path)).resolve()
        fragment = _resolve_layers(include_resolved, seen)
        # A fragment may either be the section body directly, or wrap it under
        # the section name. Support both so configs/models/*.yaml can carry a
        # readable top-level `model:` key.
        body = fragment.get(section, fragment) if isinstance(fragment, dict) else fragment
        merged = deep_merge(merged, {section: body})

    return deep_merge(merged, raw)


#: YAML 1.1 only recognizes scientific notation with an explicit decimal point
#: and sign (``1.0e-4``), so a natural ``--set lr=1e-4`` would parse as a string.
#: This pattern catches the forms a user actually types.
_SCIENTIFIC_NOTATION = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


def parse_override(item: str) -> tuple[list[str], Any]:
    """Parse a ``--set a.b.c=value`` override into a key path and typed value.

    Values are parsed as YAML, so ``true``, ``3`` and ``[a,b]`` arrive with the
    right type. Scientific notation is handled explicitly because YAML 1.1 would
    otherwise turn ``1e-4`` into the string ``"1e-4"``.
    """
    if "=" not in item:
        raise ConfigError(
            f"Malformed override {item!r}; expected key.path=value.",
            suggestions=["Example: --set training.learning_rate=1e-4"],
        )
    key, _, raw_value = item.partition("=")
    key = key.strip()
    if not key:
        raise ConfigError(f"Malformed override {item!r}: empty key.")

    stripped = raw_value.strip()
    if _SCIENTIFIC_NOTATION.match(stripped):
        return key.split("."), float(stripped)

    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError:
        value = raw_value
    return key.split("."), value


def apply_overrides(data: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``key.path=value`` overrides to a raw config dict."""
    result = copy.deepcopy(data)
    for item in overrides:
        path, value = parse_override(item)
        cursor: dict[str, Any] = result
        for part in path[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[path[-1]] = value
        logger.debug("Applied override %s = %r", ".".join(path), value)
    return result


def load_raw_config(path: Path | str, overrides: Sequence[str] = ()) -> dict[str, Any]:
    """Resolve a config file to a plain dict without validating it."""
    resolved = _resolve_layers(Path(path), seen=[])
    resolved = apply_overrides(resolved, overrides)
    return _expand_env(resolved)


def load_config(
    path: Path | str,
    overrides: Sequence[str] = (),
    *,
    dataset_path: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> ExperimentConfig:
    """Load, compose, override and validate an experiment configuration.

    Args:
        path: Entry-point YAML file.
        overrides: ``key.path=value`` strings from ``--set``.
        dataset_path: CLI ``--dataset`` value. This is the supported way to point
            a run at the externally produced private dataset without this repo
            depending on the private one (spec section 27).
        output_dir: CLI ``--output-dir`` value.

    Raises:
        ConfigError: with the offending field path when validation fails.
    """
    raw = load_raw_config(path, overrides)

    if dataset_path is not None:
        raw.setdefault("dataset", {})
        raw["dataset"]["path"] = str(dataset_path)
    if output_dir is not None:
        raw.setdefault("training", {})
        raw["training"]["output_dir"] = str(output_dir)

    try:
        config = ExperimentConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigError(
            f"Configuration in {path} failed validation.",
            details={"validation_error": str(exc)},
            suggestions=[
                "Compare against the documented examples in configs/*/README.md.",
                "Unknown keys are rejected on purpose; check for typos.",
                "Run: python scripts/validate_configs.py",
            ],
        ) from exc

    config.source_files = [str(Path(path))]
    return config


def load_model_config(path: Path | str, overrides: Sequence[str] = ()) -> ModelConfig:
    """Load a standalone model config fragment (used by scripts/inspect_model.py)."""
    raw = load_raw_config(path, overrides)
    body = raw.get("model", raw)
    try:
        return ModelConfig.model_validate(body)
    except Exception as exc:
        raise ConfigError(
            f"Model configuration in {path} failed validation.",
            details={"validation_error": str(exc)},
        ) from exc
