# ---------------------------------------------------------------------------
# Loading a frozen deployment package, or refusing to.
#
# The research loader answers "can I run this experiment?". This one answers a
# narrower and stricter question: "is what I am about to serve the exact
# artifact that was measured?" Every check here fails closed, because the
# failure it guards against is silent — a LoRA adapter attached to the wrong
# base revision raises no error, it just gives quietly worse answers.
#
# Generation deliberately reuses HuggingFaceBackend, the same class the
# evaluation arms used. Serving through a second, similar-looking code path
# would mean the deployed model is not demonstrably the model that scored
# 0.8051 — and that demonstration is the whole point of this package.
#
# torch / transformers / peft are imported inside functions so that verifying a
# package, and importing this module, work without them.
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kleos_models.config import GenerationConfig, ModelConfig, load_model_config
from kleos_models.data.schemas import Message
from kleos_models.errors import ConfigError, ModelCompatibilityError
from kleos_models.logging_utils import get_logger
from kleos_models.publishing import is_pinned_revision
from kleos_models.serving.manifest import (
    ADAPTER_DIRNAME,
    PACKAGED_MODEL_CONFIG,
    TOKENIZER_DIRNAME,
    DeploymentManifest,
    verify_identity,
    verify_package,
)
from kleos_models.serving.status import finish_reason

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kleos_models.inference.backends import GenerationOutput, PreparedPrompt

logger = get_logger(__name__)


@dataclass
class LoadedDeployment:
    """A verified, loaded deployment ready to answer requests."""

    manifest: DeploymentManifest
    model_config: ModelConfig
    backend: Any
    package_dir: Path

    @property
    def generation_defaults(self) -> GenerationConfig:
        generation = self.manifest.generation
        return GenerationConfig(
            temperature=generation.temperature,
            top_p=generation.top_p,
            do_sample=generation.do_sample,
            max_new_tokens=generation.max_new_tokens,
            repetition_penalty=generation.repetition_penalty,
        )

    def generation_config(self, max_new_tokens: int | None = None) -> GenerationConfig:
        """The frozen decoding contract, with an optional *lower* token budget.

        ``max_new_tokens`` may be lowered by a caller but never raised above the
        manifest's limit: a request must not be able to widen the contract the
        model was measured under.
        """
        config = self.generation_defaults
        if max_new_tokens is not None:
            ceiling = self.manifest.limits.max_new_tokens
            config = config.model_copy(update={"max_new_tokens": min(max_new_tokens, ceiling)})
        return config

    def generate(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int | None = None,
    ) -> GenerationOutput:
        """Generate one response under the frozen decoding contract."""
        config = self.generation_config(max_new_tokens)
        output = self.backend.generate(messages, config)
        # The backend always reports "stop"; a truncated answer must say so.
        output.finish_reason = finish_reason(output.completion_tokens, config.max_new_tokens)
        return output

    # The three steps of generate(), exposed for hosts that bill GPU time
    # (ZeroGPU): only generate_ids needs the GPU. Same backend, same order.

    def prepare(self, messages: list[Message]) -> PreparedPrompt:
        return self.backend.prepare(messages)

    def generate_ids(self, prepared: PreparedPrompt, config: GenerationConfig) -> list[int]:
        return self.backend.generate_ids(prepared, config)

    def finish(self, prepared: PreparedPrompt, completion_ids: list[int]) -> GenerationOutput:
        return self.backend.finish(prepared, completion_ids)

    def describe(self) -> dict[str, Any]:
        """Identity of what is loaded. Safe to expose on a readiness endpoint."""
        return {
            "model_name": self.manifest.model_name,
            "model_version": self.manifest.model_version,
            "deployment_artifact_version": self.manifest.deployment_artifact_version,
            "base_model": self.manifest.base_model,
            "base_revision": self.manifest.base_revision,
            "adapter_sha256": self.manifest.adapter.weights_sha256,
            "experiment_id": self.manifest.adapter.experiment_id,
            "source_checkpoint": self.manifest.adapter.source_checkpoint,
            "dataset_version": self.manifest.dataset.version,
            "training_config_hash": self.manifest.training_config_hash,
            "tokenizer_fix_mistral_regex": self.manifest.tokenizer.fix_mistral_regex,
            "runtime": self.manifest.runtime.model_dump(mode="json"),
            "generation": self.manifest.generation.model_dump(mode="json"),
        }


def load_packaged_model_config(package_dir: Path | str) -> ModelConfig:
    """Read the model config that travels with the package."""
    path = Path(package_dir) / PACKAGED_MODEL_CONFIG
    if not path.exists():
        raise ConfigError(
            f"No packaged model config at {path}.",
            suggestions=[
                "Rebuild the package with scripts/build_deployment_package.py.",
                "A deployment must not depend on a config that lives elsewhere.",
            ],
        )
    return load_model_config(path)


def check_config_matches_manifest(config: ModelConfig, manifest: DeploymentManifest) -> None:
    """Refuse a package whose two records disagree about the base weights."""
    problems: list[str] = []
    if config.base_model != manifest.base_model:
        problems.append(
            f"packaged model config base_model is {config.base_model!r}, "
            f"manifest says {manifest.base_model!r}"
        )
    if config.revision != manifest.base_revision:
        problems.append(
            f"packaged model config revision is {config.revision!r}, "
            f"manifest says {manifest.base_revision!r}"
        )
    runtime = manifest.runtime
    loaded = {
        "quantization_mode": config.quantization.mode.value,
        "double_quant": config.quantization.double_quant,
        "compute_dtype": config.quantization.compute_dtype.value,
        "attn_implementation": config.attn_implementation,
        "max_seq_length": config.max_seq_length,
    }
    for key, want in runtime.model_dump().items():
        if loaded[key] != want:
            problems.append(
                f"packaged model config {key} is {loaded[key]!r}, manifest says {want!r}"
            )
    if problems:
        raise ConfigError(
            "The deployment package contradicts itself about which base weights to load.",
            details={"problems": problems},
            suggestions=["Rebuild the package; do not serve it in this state."],
        )


def verify_base_revision(base_model: str, revision: str, *, require_remote: bool = False) -> str:
    """Check the base revision is a real, pinned commit before downloading 24 GB.

    Passing a 40-character sha to ``from_pretrained`` already resolves that exact
    commit or fails, so the pin is enforced regardless. This adds an early, cheap
    failure and a positive confirmation from the Hub when it is reachable.

    Args:
        require_remote: Fail when the Hub cannot be reached, instead of relying
            on the load-time pin alone. Use in environments that must never
            serve from a stale cache.
    """
    if not is_pinned_revision(revision):
        raise ConfigError(
            f"Refusing to serve {base_model!r} at revision {revision!r}.",
            details={"revision": revision},
            suggestions=[
                "A deployment must pin a 40-character commit sha.",
                "A moving pointer lets upstream replace the weights this adapter "
                "was trained against, with no error raised (finding H-F1).",
            ],
        )

    try:
        from huggingface_hub import HfApi
    except ImportError:
        if require_remote:
            raise ConfigError(
                "huggingface_hub is not installed, so the base revision cannot be "
                "confirmed remotely.",
                suggestions=['pip install -e ".[train]"', "Or set require_remote=False."],
            ) from None
        logger.info("huggingface_hub unavailable; relying on the load-time revision pin.")
        return revision

    try:
        info = HfApi().model_info(base_model, revision=revision)
    except Exception as exc:
        if require_remote:
            raise ConfigError(
                f"Could not confirm {base_model}@{revision[:12]} with the Hub.",
                details={"error": str(exc)[:400]},
                suggestions=[
                    "Check network access and HF_TOKEN.",
                    "Do not serve an artifact whose base cannot be identified.",
                ],
            ) from exc
        logger.warning(
            "Could not reach the Hub to confirm %s@%s (%s). The load-time pin still "
            "applies: a wrong revision will fail to resolve.",
            base_model,
            revision[:12],
            type(exc).__name__,
        )
        return revision

    resolved = getattr(info, "sha", None)
    if resolved and resolved != revision:
        raise ConfigError(
            f"{base_model}@{revision[:12]} resolved to a different commit ({resolved[:12]}).",
            suggestions=["The pin is not what it claims to be; do not serve this package."],
        )
    logger.info("Confirmed base revision %s@%s with the Hub.", base_model, revision[:12])
    return revision


def load_deployment(
    package_dir: Path | str,
    *,
    verify: bool = True,
    require_remote_revision: bool = False,
    device_map: str | None = None,
    expected_identity: dict[str, Any] | None = None,
) -> LoadedDeployment:
    """Verify a deployment package and load the model it describes.

    Args:
        package_dir: Root of the package (holds ``manifest.json``).
        verify: Re-hash every artifact first. Leave on: the hashes are the only
            thing standing between a corrupted adapter and a served one.
        require_remote_revision: Confirm the base commit with the Hub rather
            than relying on the load-time pin alone.
        device_map: Override the packaged device map (e.g. ``"cuda:0"``).
        expected_identity: The identity this deployment was built to serve (see
            :func:`kleos_models.serving.manifest.load_expected_identity`). When
            given, a package that is intact but *different* is refused. Checked
            before anything is downloaded.

    Raises:
        ConfigError: the package does not match its manifest, contradicts
            itself, or is not the expected artifact.
        ModelCompatibilityError: the artifacts do not load.
    """
    root = Path(package_dir)
    manifest = verify_package(root) if verify else DeploymentManifest.load(root)
    if expected_identity is not None:
        verify_identity(manifest, expected_identity)

    config = load_packaged_model_config(root)
    check_config_matches_manifest(config, manifest)
    verify_base_revision(
        manifest.base_model, manifest.base_revision, require_remote=require_remote_revision
    )

    # The tokenizer is the frozen copy inside the package, not the Hub's. Set at
    # runtime rather than baked into the packaged YAML so the package stays
    # relocatable.
    config.tokenizer = str((root / TOKENIZER_DIRNAME).resolve())
    if device_map:
        config.device_map = device_map

    from kleos_models.inference.backends import HuggingFaceBackend

    adapter_dir = root / ADAPTER_DIRNAME
    logger.info(
        "Loading %s %s: %s@%s + adapter %s…",
        manifest.model_name,
        manifest.model_version,
        manifest.base_model,
        manifest.base_revision[:12],
        manifest.adapter.weights_sha256[:16],
    )
    backend = HuggingFaceBackend(
        config,
        adapter_path=adapter_dir,
        name=f"{manifest.model_name}-{manifest.model_version}",
        fix_mistral_regex=manifest.tokenizer.fix_mistral_regex,
    )

    check_loaded_tokenizer(backend.loaded.tokenizer, manifest)
    check_loaded_adapter(backend.loaded.model, manifest)
    logger.info("Deployment ready: %s", manifest.model_name)
    return LoadedDeployment(
        manifest=manifest, model_config=config, backend=backend, package_dir=root
    )


def check_loaded_tokenizer(tokenizer: Any, manifest: DeploymentManifest) -> None:
    """Confirm the live tokenizer matches the contract the manifest states."""
    contract = manifest.tokenizer
    problems: list[str] = []

    padding_side = getattr(tokenizer, "padding_side", None)
    if padding_side != contract.padding_side:
        problems.append(
            f"padding_side is {padding_side!r}, contract says {contract.padding_side!r}"
        )

    if contract.pad_token is not None:
        pad_token = getattr(tokenizer, "pad_token", None)
        if pad_token != contract.pad_token:
            problems.append(f"pad_token is {pad_token!r}, contract says {contract.pad_token!r}")

    if getattr(tokenizer, "chat_template", None) is None:
        problems.append("the tokenizer has no chat template, so prompts cannot be rendered")

    if problems:
        raise ModelCompatibilityError(
            "The loaded tokenizer does not match the frozen serving contract.",
            details={"problems": problems},
            suggestions=[
                "Tokenization that differs from training is train/serve skew and "
                "fails silently. Do not serve.",
                "Rebuild the package from the frozen run.",
            ],
        )


def check_loaded_adapter(model: Any, manifest: DeploymentManifest) -> None:
    """Confirm the attached adapter is the one the manifest describes."""
    peft_config = None
    configs = getattr(model, "peft_config", None)
    if isinstance(configs, dict) and configs:
        peft_config = next(iter(configs.values()))
    if peft_config is None:
        raise ModelCompatibilityError(
            "No PEFT adapter is attached to the loaded model.",
            suggestions=["The base model alone is not Hermes; do not serve it as Hermes."],
        )

    problems: list[str] = []
    for attribute, expected in (
        ("r", manifest.peft.r),
        ("lora_alpha", manifest.peft.lora_alpha),
        ("task_type", manifest.peft.task_type),
    ):
        actual = getattr(peft_config, attribute, None)
        actual = getattr(actual, "value", actual)  # task_type is an enum
        if actual != expected:
            problems.append(f"{attribute} is {actual!r}, manifest says {expected!r}")

    base = getattr(peft_config, "base_model_name_or_path", None)
    if base is not None and base != manifest.base_model:
        problems.append(
            f"base_model_name_or_path is {base!r}, manifest says {manifest.base_model!r}"
        )

    if problems:
        raise ModelCompatibilityError(
            "The attached adapter does not match the deployment manifest.",
            details={"problems": problems},
            suggestions=["Do not serve; rebuild the package from the frozen run."],
        )
