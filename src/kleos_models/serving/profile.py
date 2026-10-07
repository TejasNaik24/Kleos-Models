"""Per-model serving names and options, read from a deployment record.

Hermes was the only model served, so what makes a Space "the Hermes Space" was
spelled as constants: the ``HERMES_*`` secrets, the ``x-hermes-key`` header,
the Space folder and record file name, the base checkpoint files baked into the
image, the status copy. A second model (KLEOS Logos v0.0.2) needs its own.

A deployment record (``configs/deployment/<model>.yaml``) may carry an optional
``serving`` block. Without one, every value here is today's Hermes constant, so
the Hermes record needs no edit and Hermes serves exactly as before. The block
also turns on what a thinking model needs: the trace in the reply (contract
version 2) and the rule that a request starts with a system message.

Unknown keys are refused rather than ignored: a misspelt option that silently
falls back to a Hermes default would serve the wrong contract.

This module imports no torch.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kleos_models.errors import ConfigError

#: The base checkpoint files baked into the Hermes Space image: the Hugging
#: Face shards of Mistral-Nemo and their configs, at the pinned revision.
HERMES_BASE_FILES: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    *(f"model-{i:05d}-of-00005.safetensors" for i in range(1, 6)),
)

_PREFIX = re.compile(r"^[A-Z][A-Z0-9]*$")
_HEADER = re.compile(r"^x-[a-z0-9-]+$")
_SERVING_KEYS = frozenset(
    {
        "short_name",
        "display_name",
        "env_prefix",
        "key_header",
        "space_dir",
        "record_file",
        "research_report",
        "base_files",
        "space_only",
        "reply",
        "gpu",
    }
)
_REPLY_KEYS = frozenset({"reasoning", "require_system_message"})
_GPU_KEYS = frozenset({"base_seconds", "tokens_per_second", "min_seconds", "max_seconds"})


@dataclass(frozen=True)
class GpuDefaults:
    """Defaults for the GPU-duration rule; ``<PREFIX>_GPU_*`` secrets still override."""

    base_seconds: float = 10.0
    tokens_per_second: float = 12.0
    min_seconds: float = 15.0
    max_seconds: float = 60.0


@dataclass(frozen=True)
class ServingProfile:
    """What makes one model's Space its own. The defaults are Hermes'."""

    #: "Hermes", "Logos": status copy and log lines.
    short_name: str
    #: "Hermes v0.0.6": the ready message, Space title, script headers.
    display_name: str
    env_prefix: str = "HERMES"
    key_header: str = "x-hermes-key"
    space_dir: str = "deploy/zerogpu-space"
    record_file: str = "hermes_record.yaml"
    research_report: str = "docs/experiments/kleos-v006-mistralnemo12b-run1-report.md"
    base_files: tuple[str, ...] = HERMES_BASE_FILES
    #: Return the thinking trace beside the answer (contract version 2).
    reasoning: bool = False
    #: Refuse a request whose first message is not a system message.
    require_system_message: bool = False
    #: Served from a ZeroGPU Space only; there is no Docker service for it.
    space_only: bool = False
    gpu: GpuDefaults = field(default_factory=GpuDefaults)

    def env(self, suffix: str) -> str:
        """The name of one of this model's settings, e.g. ``LOGOS_API_KEY``."""
        return f"{self.env_prefix}_{suffix}"

    @property
    def contract_version(self) -> int:
        """1 for a plain reply; 2 when the reply carries ``reasoning``."""
        return 2 if self.reasoning else 1


def _short_name(name: Any) -> str:
    text = str(name or "").strip()
    if text.startswith("kleos-"):
        text = text[len("kleos-") :]
    if not text:
        raise ConfigError("The deployment record has no `name` to derive a short name from.")
    return text[:1].upper() + text[1:]


def _unknown(keys: Any, allowed: frozenset[str], where: str) -> None:
    extra = sorted(set(keys) - allowed)
    if extra:
        raise ConfigError(
            f"Unknown key(s) in {where}: {', '.join(extra)}.",
            suggestions=[f"Allowed: {', '.join(sorted(allowed))}."],
        )


def _flag(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"serving.reply.{key} must be true or false, got {value!r}.")
    return value


def profile_from_record(spec: Mapping[str, Any]) -> ServingProfile:
    """The serving profile of a record's ``deployment:`` mapping.

    Raises:
        ConfigError: the ``serving`` block has an unknown key or a malformed value.
    """
    short = _short_name(spec.get("name"))
    version = str(spec.get("version") or "").strip()
    display = f"{short} {version}".strip()
    block = spec.get("serving")
    if block is None:
        return ServingProfile(short_name=short, display_name=display)
    if not isinstance(block, Mapping):
        raise ConfigError("serving must be a mapping.")
    _unknown(block, _SERVING_KEYS, "serving")

    values: dict[str, Any] = {
        "short_name": str(block.get("short_name") or short),
        "display_name": str(block.get("display_name") or display),
    }
    if "env_prefix" in block:
        prefix = str(block["env_prefix"])
        if not _PREFIX.match(prefix):
            raise ConfigError(f"serving.env_prefix must be upper case, e.g. LOGOS; got {prefix!r}.")
        values["env_prefix"] = prefix
    if "key_header" in block:
        header = str(block["key_header"])
        if not _HEADER.match(header):
            raise ConfigError(f"serving.key_header must be a lower-case x- header, got {header!r}.")
        values["key_header"] = header
    if "record_file" in block:
        record_file = str(block["record_file"])
        if not record_file.endswith(".yaml") or "/" in record_file or "\\" in record_file:
            raise ConfigError(
                f"serving.record_file must be a bare .yaml file name, got {record_file!r}."
            )
        values["record_file"] = record_file
    for key in ("space_dir", "research_report"):
        if key in block:
            values[key] = str(block[key])
    if "base_files" in block:
        files = block["base_files"]
        if (
            not isinstance(files, list)
            or not files
            or any(not isinstance(f, str) or not f or "/" in f for f in files)
        ):
            raise ConfigError("serving.base_files must be a non-empty list of bare file names.")
        values["base_files"] = tuple(files)
    if "space_only" in block:
        values["space_only"] = _flag(block["space_only"], "space_only")

    reply = block.get("reply") or {}
    if not isinstance(reply, Mapping):
        raise ConfigError("serving.reply must be a mapping.")
    _unknown(reply, _REPLY_KEYS, "serving.reply")
    for key in _REPLY_KEYS:
        if key in reply:
            values[key] = _flag(reply[key], key)

    gpu = block.get("gpu") or {}
    if not isinstance(gpu, Mapping):
        raise ConfigError("serving.gpu must be a mapping.")
    _unknown(gpu, _GPU_KEYS, "serving.gpu")
    if gpu:
        values["gpu"] = GpuDefaults(**{key: float(value) for key, value in gpu.items()})
    return ServingProfile(**values)


def load_profile(record_path: Path | str) -> ServingProfile:
    """The serving profile of the record at ``record_path``."""
    source = Path(record_path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    spec = raw.get("deployment") if isinstance(raw, dict) else None
    if not isinstance(spec, dict):
        raise ConfigError(f"{source} has no top-level 'deployment:' mapping.")
    return profile_from_record(spec)


#: The profile of a record with no ``serving`` block: today's Hermes.
HERMES_PROFILE = profile_from_record({"name": "kleos-hermes", "version": "v0.0.6"})
