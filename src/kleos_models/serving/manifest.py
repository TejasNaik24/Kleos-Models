# ---------------------------------------------------------------------------
# The deployment manifest: what is actually running, provable at startup.
#
# A research manifest (kleos_models.experiments.manifest) records what happened
# during a run. This records what a *server* must load, and carries the hashes
# needed to prove it loaded exactly that.
#
# The question it exists to answer, six months from now, without guessing:
#
#     "Which base revision, tokenizer, adapter, dataset release and generation
#      settings produced this deployed Hermes instance?"
#
# Deliberately NOT in here: filesystem paths outside the package, Drive URLs,
# credentials, endpoint hostnames. A manifest travels with the artifact and may
# be read by anyone who can reach the service, so it carries identity, not
# secrets. `scripts/check_no_private_data.py` scans it like any other file.
#
# This module imports no torch and no transformers: verifying an artifact must
# work on a laptop, in CI, and at container start before any GPU is touched.
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from kleos_models.config import StrictModel
from kleos_models.data.loaders import file_sha256
from kleos_models.errors import ConfigError
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)

#: Bumped when the manifest *schema* changes in a way a loader must notice.
#: Distinct from the model version: the same frozen adapter can be repackaged.
DEPLOYMENT_ARTIFACT_VERSION = 1

#: Layout every deployment package must use. The loader resolves nothing outside
#: the package except the base model, which comes from the registry by revision.
MANIFEST_FILENAME = "manifest.json"
ADAPTER_DIRNAME = "adapter"
TOKENIZER_DIRNAME = "tokenizer"
DEPLOYMENT_DIRNAME = "deployment"

#: Files the adapter directory must contain for the package to be loadable.
REQUIRED_ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")

#: Files the tokenizer directory must contain. All-or-nothing on purpose: a
#: tokenizer_config.json without tokenizer.json looks loadable and is not.
REQUIRED_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$")


class FileRecord(StrictModel):
    """One file inside the package, identified by content rather than by name."""

    path: str = Field(description="Path relative to the package root, POSIX separators.")
    size_bytes: int = Field(ge=0)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def _digest_is_sha256(cls, value: str) -> str:
        if not _SHA256.match(value):
            raise ValueError(f"not a lowercase 64-character sha256: {value!r}")
        return value

    @field_validator("path")
    @classmethod
    def _path_stays_inside_the_package(cls, value: str) -> str:
        # A manifest is data, and data that names ".." or "/etc/passwd" must not
        # be able to steer a verifier outside the package it describes.
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"path must be relative and stay inside the package: {value!r}")
        return candidate.as_posix()


class TokenizerContract(StrictModel):
    """Exactly how the serving tokenizer must be built.

    Every field here changes tokenization, and tokenization that differs from
    training is train/serve skew: no error, just quietly worse answers. The
    contract is explicit so a library default can never move it silently.
    """

    source: str = Field(
        default="frozen_package",
        description=(
            "'frozen_package' loads the pinned files in tokenizer/. That is the "
            "only supported value; loading from the Hub would reintroduce drift."
        ),
    )
    fix_mistral_regex: bool = Field(
        description=(
            "transformers>=5 can patch the Mistral pre-tokenizer Split regex. The "
            "flag defaults to False, which is what v0.0.6 trained and evaluated "
            "with, so serving pins False explicitly. See docs/deployment.md."
        )
    )
    padding_side: str = Field(default="right")
    pad_token: str | None = Field(default=None)
    files: list[FileRecord] = Field(default_factory=list)

    @field_validator("source")
    @classmethod
    def _only_frozen_files(cls, value: str) -> str:
        if value != "frozen_package":
            raise ValueError(
                f"tokenizer.source must be 'frozen_package', got {value!r}. "
                "Resolving the tokenizer anywhere else defeats the pin."
            )
        return value

    @field_validator("padding_side")
    @classmethod
    def _padding_side_is_valid(cls, value: str) -> str:
        if value not in {"left", "right"}:
            raise ValueError(f"padding_side must be 'left' or 'right', got {value!r}")
        return value


class AdapterRecord(StrictModel):
    """Which trained weights these are, and where they came from."""

    #: The run that produced them. Not a path — an identity.
    experiment_id: str
    #: The checkpoint best-model selection chose, e.g. "checkpoint-200".
    source_checkpoint: str
    #: sha256 of adapter_model.safetensors. The single most important field here.
    weights_sha256: str
    trainable_parameters: int = Field(gt=0)
    files: list[FileRecord] = Field(default_factory=list)
    #: Validation loss of the selected checkpoint, for the record.
    selection_metric: str = "eval_loss"
    selection_value: float | None = None

    @field_validator("weights_sha256")
    @classmethod
    def _digest_is_sha256(cls, value: str) -> str:
        if not _SHA256.match(value):
            raise ValueError(f"not a lowercase 64-character sha256: {value!r}")
        return value


class PeftRecord(StrictModel):
    """The LoRA shape, mirrored from adapter_config.json so drift is visible."""

    peft_type: str = "LORA"
    task_type: str = "CAUSAL_LM"
    r: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    lora_dropout: float = Field(ge=0.0, le=1.0)
    bias: str = "none"
    target_modules: list[str] = Field(default_factory=list)
    exclude_modules: list[str] = Field(default_factory=list)
    peft_version: str | None = None


class DatasetRecord(StrictModel):
    """The sealed release the behaviour came from."""

    version: str
    sha256: str
    split_strategy: str | None = None

    @field_validator("sha256")
    @classmethod
    def _digest_is_sha256(cls, value: str) -> str:
        if not _SHA256.match(value):
            raise ValueError(f"not a lowercase 64-character sha256: {value!r}")
        return value


class GenerationDefaults(StrictModel):
    """Decoding the reported numbers were produced under.

    Changing any of these makes the served model something other than the model
    that was measured, so they are recorded rather than left to the caller.
    """

    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    do_sample: bool = False
    max_new_tokens: int = Field(default=512, gt=0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)


class ServingLimits(StrictModel):
    """Bounds the service enforces. Present in the manifest so they are reviewable."""

    max_new_tokens: int = Field(default=512, gt=0)
    max_input_chars: int = Field(default=24_000, gt=0)
    max_messages: int = Field(default=64, gt=0)
    request_timeout_seconds: float = Field(default=120.0, gt=0)


class DeploymentManifest(StrictModel):
    """The complete, verifiable description of a deployable KLEOS model."""

    deployment_artifact_version: int = DEPLOYMENT_ARTIFACT_VERSION
    model_name: str
    model_version: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    base_model: str
    base_revision: str

    adapter: AdapterRecord
    peft: PeftRecord
    tokenizer: TokenizerContract
    dataset: DatasetRecord
    generation: GenerationDefaults = Field(default_factory=GenerationDefaults)
    limits: ServingLimits = Field(default_factory=ServingLimits)

    #: config_hash of the training run. Ties this package to a reproducible run.
    training_config_hash: str
    #: Where the evidence lives in the repository, as a relative path.
    research_report: str | None = None
    known_limitations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("base_revision")
    @classmethod
    def _revision_must_be_pinned(cls, value: str) -> str:
        # Fail closed. A moving pointer ("main", a tag) silently pairs the
        # adapter with weights it never saw, and nothing raises.
        if not _PINNED_REVISION.match(value):
            raise ValueError(
                f"base_revision must be a 40-character commit sha, got {value!r}. "
                "Serving an unpinned base is the failure this manifest exists to "
                "prevent."
            )
        return value

    @field_validator("deployment_artifact_version")
    @classmethod
    def _version_is_known(cls, value: int) -> int:
        if value != DEPLOYMENT_ARTIFACT_VERSION:
            raise ValueError(
                f"manifest schema version {value} is not {DEPLOYMENT_ARTIFACT_VERSION}; "
                "this loader cannot guarantee it understands the package"
            )
        return value

    # -- io ------------------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=False)

    def save(self, package_dir: Path | str) -> Path:
        path = Path(package_dir) / MANIFEST_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, package_dir: Path | str) -> DeploymentManifest:
        directory = Path(package_dir)
        path = directory / MANIFEST_FILENAME if directory.is_dir() else directory
        if not path.exists():
            raise ConfigError(
                f"No deployment manifest at {path}.",
                suggestions=[
                    "Build one with scripts/build_deployment_package.py.",
                    "A package without a manifest cannot be verified and must not be served.",
                ],
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"Deployment manifest at {path} is not valid JSON: {exc}",
                suggestions=["The package is corrupt; rebuild it."],
            ) from exc
        try:
            return cls.model_validate(payload)
        except Exception as exc:
            raise ConfigError(
                f"Deployment manifest at {path} does not match the expected schema.",
                details={"error": str(exc)[:600]},
                suggestions=["Rebuild the package with scripts/build_deployment_package.py."],
            ) from exc

    # -- verification --------------------------------------------------------

    def all_files(self) -> list[FileRecord]:
        return [*self.adapter.files, *self.tokenizer.files]

    def check_files(self, package_dir: Path | str) -> list[str]:
        """Re-hash every recorded file. Returns human-readable problems."""
        root = Path(package_dir)
        problems: list[str] = []

        for record in self.all_files():
            path = root / record.path
            if not path.exists():
                problems.append(f"missing file: {record.path}")
                continue
            size = path.stat().st_size
            if size != record.size_bytes:
                problems.append(
                    f"{record.path}: size {size} != {record.size_bytes} recorded in the manifest"
                )
                continue
            digest = file_sha256(path)
            if digest != record.sha256:
                problems.append(
                    f"{record.path}: sha256 {digest[:16]}… != {record.sha256[:16]}… recorded"
                )

        recorded = {r.path for r in self.all_files()}
        for required in REQUIRED_ADAPTER_FILES:
            if f"{ADAPTER_DIRNAME}/{required}" not in recorded:
                problems.append(f"manifest does not record {ADAPTER_DIRNAME}/{required}")
        for required in REQUIRED_TOKENIZER_FILES:
            if f"{TOKENIZER_DIRNAME}/{required}" not in recorded:
                problems.append(f"manifest does not record {TOKENIZER_DIRNAME}/{required}")

        weights = f"{ADAPTER_DIRNAME}/adapter_model.safetensors"
        for record in self.adapter.files:
            if record.path == weights and record.sha256 != self.adapter.weights_sha256:
                problems.append(
                    "adapter.weights_sha256 disagrees with the file record for "
                    f"{weights}; the manifest contradicts itself"
                )
        return problems

    def check_adapter_config(self, package_dir: Path | str) -> list[str]:
        """Verify adapter_config.json agrees with the manifest.

        This is the H-F1 check. PEFT writes ``base_model_name_or_path`` and a
        ``revision`` field into the adapter config; the research artifact left
        the revision null. A deployment package must carry the pin, and it must
        be the *same* pin the manifest names, or the two records disagree about
        which weights these deltas belong to.
        """
        path = Path(package_dir) / ADAPTER_DIRNAME / "adapter_config.json"
        if not path.exists():
            return [f"missing {ADAPTER_DIRNAME}/adapter_config.json"]
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return [f"adapter_config.json is not valid JSON: {exc}"]

        problems: list[str] = []
        base = config.get("base_model_name_or_path")
        if base != self.base_model:
            problems.append(
                f"adapter_config.json base_model_name_or_path is {base!r}, "
                f"manifest says {self.base_model!r}"
            )
        revision = config.get("revision")
        if revision is None:
            problems.append(
                "adapter_config.json records revision=null. The deployment package "
                "must pin the base revision (finding H-F1); serving an unpinned "
                "base pairs the adapter with weights it may never have seen."
            )
        elif revision != self.base_revision:
            problems.append(
                f"adapter_config.json revision is {revision!r}, "
                f"manifest says {self.base_revision!r}"
            )

        for field_name, expected in (
            ("r", self.peft.r),
            ("lora_alpha", self.peft.lora_alpha),
            ("task_type", self.peft.task_type),
            ("peft_type", self.peft.peft_type),
        ):
            actual = config.get(field_name)
            if actual != expected:
                problems.append(
                    f"adapter_config.json {field_name} is {actual!r}, manifest says {expected!r}"
                )
        return problems

    def problems(self, package_dir: Path | str) -> list[str]:
        """Every reason this package must not be served."""
        return [*self.check_files(package_dir), *self.check_adapter_config(package_dir)]


def verify_package(package_dir: Path | str, *, strict: bool = True) -> DeploymentManifest:
    """Load and fully verify a deployment package.

    Args:
        package_dir: Directory holding ``manifest.json``, ``adapter/`` and ``tokenizer/``.
        strict: Raise on any problem. ``False`` logs instead, for inspection tools.

    Returns:
        The validated manifest.

    Raises:
        ConfigError: when the package does not match its manifest.
    """
    manifest = DeploymentManifest.load(package_dir)
    problems = manifest.problems(package_dir)
    if problems:
        message = (
            f"Deployment package at {package_dir} does not match its manifest "
            f"({len(problems)} problem(s))."
        )
        if strict:
            raise ConfigError(
                message,
                details={"problems": problems},
                suggestions=[
                    "Do not serve this package.",
                    "Rebuild it from the frozen run with scripts/build_deployment_package.py.",
                    "If a hash changed, the artifact was modified — find out why before rebuilding.",
                ],
            )
        for problem in problems:
            logger.warning("Package problem: %s", problem)
    logger.info(
        "Verified deployment package %s %s (adapter %s…, base %s@%s)",
        manifest.model_name,
        manifest.model_version,
        manifest.adapter.weights_sha256[:16],
        manifest.base_model,
        manifest.base_revision[:12],
    )
    return manifest


def build_file_record(package_dir: Path | str, relative_path: str) -> FileRecord:
    """Hash one file inside the package and describe it."""
    path = Path(package_dir) / relative_path
    if not path.exists():
        raise ConfigError(f"Cannot record a file that does not exist: {path}")
    return FileRecord(
        path=Path(relative_path).as_posix(),
        size_bytes=path.stat().st_size,
        sha256=file_sha256(path),
    )


def build_deployment_manifest(
    package_dir: Path | str,
    *,
    model_name: str,
    model_version: str,
    base_model: str,
    base_revision: str,
    experiment_id: str,
    source_checkpoint: str,
    trainable_parameters: int,
    dataset_version: str,
    dataset_sha256: str,
    training_config_hash: str,
    adapter_config: dict[str, Any],
    fix_mistral_regex: bool,
    tokenizer_padding_side: str = "right",
    tokenizer_pad_token: str | None = None,
    split_strategy: str | None = None,
    selection_value: float | None = None,
    generation: GenerationDefaults | None = None,
    limits: ServingLimits | None = None,
    research_report: str | None = None,
    known_limitations: list[str] | None = None,
    notes: list[str] | None = None,
) -> DeploymentManifest:
    """Hash a prepared package directory and describe it.

    The package must already contain ``adapter/`` and ``tokenizer/``; this
    function records what is there rather than copying anything, so the hashes
    describe the bytes that will actually be served.
    """
    root = Path(package_dir)

    adapter_files = [
        build_file_record(root, f"{ADAPTER_DIRNAME}/{path.name}")
        for path in sorted((root / ADAPTER_DIRNAME).iterdir())
        if path.is_file()
    ]
    tokenizer_files = [
        build_file_record(root, f"{TOKENIZER_DIRNAME}/{path.name}")
        for path in sorted((root / TOKENIZER_DIRNAME).iterdir())
        if path.is_file()
    ]

    weights = next((r for r in adapter_files if r.path.endswith("adapter_model.safetensors")), None)
    if weights is None:
        raise ConfigError(
            f"No adapter_model.safetensors in {root / ADAPTER_DIRNAME}.",
            suggestions=["A deployment package without weights cannot be served."],
        )

    return DeploymentManifest(
        model_name=model_name,
        model_version=model_version,
        base_model=base_model,
        base_revision=base_revision,
        adapter=AdapterRecord(
            experiment_id=experiment_id,
            source_checkpoint=source_checkpoint,
            weights_sha256=weights.sha256,
            trainable_parameters=trainable_parameters,
            files=adapter_files,
            selection_value=selection_value,
        ),
        peft=PeftRecord(
            peft_type=adapter_config.get("peft_type", "LORA"),
            task_type=adapter_config.get("task_type", "CAUSAL_LM"),
            r=int(adapter_config["r"]),
            lora_alpha=int(adapter_config["lora_alpha"]),
            lora_dropout=float(adapter_config.get("lora_dropout", 0.0)),
            bias=adapter_config.get("bias", "none"),
            target_modules=sorted(adapter_config.get("target_modules", []) or []),
            exclude_modules=sorted(adapter_config.get("exclude_modules", []) or []),
            peft_version=adapter_config.get("peft_version"),
        ),
        tokenizer=TokenizerContract(
            fix_mistral_regex=fix_mistral_regex,
            padding_side=tokenizer_padding_side,
            pad_token=tokenizer_pad_token,
            files=tokenizer_files,
        ),
        dataset=DatasetRecord(
            version=dataset_version,
            sha256=dataset_sha256,
            split_strategy=split_strategy,
        ),
        generation=generation or GenerationDefaults(),
        limits=limits or ServingLimits(),
        training_config_hash=training_config_hash,
        research_report=research_report,
        known_limitations=known_limitations or [],
        notes=notes or [],
    )
