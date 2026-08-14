"""Compatibility shim across transformers 4.56 → 5.x.

Why this module exists
----------------------
transformers v5 renamed or removed several APIs this pipeline depends on:

===========================  ==============================  =======================
Concern                      transformers 4.x                transformers 5.x
===========================  ==============================  =======================
dtype at load time           ``torch_dtype=``                ``dtype=``
LR warmup                    ``warmup_ratio=0.03``           ``warmup_steps=0.03``
Trainer tokenizer argument   ``tokenizer=``                  ``processing_class=``
Eval cadence                 ``eval_strategy=``              ``eval_strategy=``
Output dir clobbering        ``overwrite_output_dir=``       removed
Serialization                ``safe_serialization=``         removed (always safetensors)
4-bit shorthand              ``load_in_4bit=True``           removed (use quantization_config)
===========================  ==============================  =======================

Rather than branching on a version number alone — which breaks on backports and
release candidates — the helpers below **introspect the installed classes** and
adapt to whatever is actually present. Version numbers are used only for
reporting and for guarding known-bad ranges.

Everything here degrades gracefully when transformers is not installed, so the
module stays importable in the light (no-torch) environment.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.metadata
import importlib.util
import inspect
from functools import lru_cache
from typing import Any

from kleos_models.errors import ConfigError
from kleos_models.logging_utils import get_logger

logger = get_logger(__name__)

#: Lowest transformers version this pipeline is tested against.
MIN_TRANSFORMERS = (4, 56, 0)
#: First major version known to be incompatible (none yet; v6 is the guard).
MAX_TRANSFORMERS_EXCLUSIVE = (6, 0, 0)


# ---------------------------------------------------------------------------
# Version discovery
# ---------------------------------------------------------------------------


def _parse_version(raw: str) -> tuple[int, ...]:
    """Parse a version string into a comparable integer tuple.

    Non-numeric suffixes (``rc1``, ``dev0``, ``+cu121``) are dropped so that
    ``"5.1.0.dev0"`` compares as ``(5, 1, 0)``.
    """
    parts: list[int] = []
    for chunk in raw.split("+")[0].split("."):
        digits = ""
        for char in chunk:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else (0,)


def package_version(name: str) -> str | None:
    """Return an installed package version, or ``None`` if absent."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover - defensive
        return None


def is_available(module: str) -> bool:
    """Return whether a module can be imported without importing it eagerly."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - odd import states
        return False


@lru_cache(maxsize=1)
def transformers_version() -> tuple[int, ...] | None:
    """Installed transformers version as a tuple, or ``None`` if absent."""
    raw = package_version("transformers")
    return _parse_version(raw) if raw else None


def require_transformers() -> Any:
    """Import and return the transformers module, or raise an actionable error."""
    try:
        return importlib.import_module("transformers")
    except ImportError as exc:
        from kleos_models.errors import MissingDependencyError

        raise MissingDependencyError(
            "transformers", extra="train", purpose="load models and tokenizers"
        ) from exc


def check_transformers_version(*, strict: bool = False) -> str | None:
    """Validate the installed transformers version against the tested range.

    Args:
        strict: Raise instead of warning when the version is out of range.

    Returns:
        A warning message when out of range, else ``None``.
    """
    version = transformers_version()
    if version is None:
        return None
    raw = package_version("transformers") or "unknown"
    problem: str | None = None
    if version < MIN_TRANSFORMERS:
        problem = (
            f"transformers {raw} is older than the tested minimum "
            f"{'.'.join(map(str, MIN_TRANSFORMERS))}."
        )
    elif version >= MAX_TRANSFORMERS_EXCLUSIVE:
        problem = (
            f"transformers {raw} is newer than the tested range "
            f"(<{'.'.join(map(str, MAX_TRANSFORMERS_EXCLUSIVE))}); "
            "APIs may have changed again."
        )
    if problem is None:
        return None
    if strict:
        raise ConfigError(
            problem,
            details={"installed": raw},
            suggestions=[
                'Install a tested version: pip install "transformers>=4.56,<6"',
                "Or re-run with strict version checking disabled to proceed anyway.",
            ],
        )
    logger.warning("%s Proceeding, but behaviour is unverified.", problem)
    return problem


# ---------------------------------------------------------------------------
# Field introspection
# ---------------------------------------------------------------------------


def _dataclass_field_names(cls: type) -> frozenset[str]:
    """Return dataclass field names, falling back to ``__init__`` parameters."""
    if dataclasses.is_dataclass(cls):
        return frozenset(f.name for f in dataclasses.fields(cls))
    try:
        initializer = getattr(cls, "__init__", None)
        if initializer is None:  # pragma: no cover - defensive
            return frozenset()
        return frozenset(inspect.signature(initializer).parameters) - {"self"}
    except (TypeError, ValueError):  # pragma: no cover - exotic classes
        return frozenset()


@lru_cache(maxsize=1)
def training_argument_names() -> frozenset[str]:
    """Field names accepted by the installed ``TrainingArguments``."""
    transformers = require_transformers()
    return _dataclass_field_names(transformers.TrainingArguments)


@lru_cache(maxsize=1)
def trainer_parameter_names() -> frozenset[str]:
    """Parameter names accepted by the installed ``Trainer.__init__``."""
    transformers = require_transformers()
    try:
        initializer = transformers.Trainer.__init__
        return frozenset(inspect.signature(initializer).parameters) - {"self"}
    except (TypeError, ValueError):  # pragma: no cover
        return frozenset()


# ---------------------------------------------------------------------------
# Argument translation
# ---------------------------------------------------------------------------


def build_training_arguments_kwargs(
    requested: dict[str, Any],
    *,
    supported: frozenset[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Translate KLEOS training settings into installed-version kwargs.

    KLEOS configs keep the names the specification uses (notably
    ``warmup_ratio``). This function maps them onto whatever the installed
    ``TrainingArguments`` actually accepts.

    Args:
        requested: Desired settings using KLEOS/4.x names.
        supported: Field names the target class accepts. Defaults to
            introspecting the installed transformers. Injectable so the
            translation can be unit-tested against both API generations without
            installing either.

    Returns:
        ``(kwargs, notes)`` where ``notes`` describes every translation and every
        dropped key, so the effective configuration can be recorded honestly.
    """
    if supported is None:
        supported = training_argument_names()

    kwargs = dict(requested)
    notes: list[str] = []

    # --- warmup ------------------------------------------------------------
    # v5 removed `warmup_ratio`; `warmup_steps` is a float where values < 1 are
    # interpreted as a fraction of total training steps.
    if "warmup_ratio" in kwargs and "warmup_ratio" not in supported:
        ratio = kwargs.pop("warmup_ratio")
        if "warmup_steps" in supported:
            existing = kwargs.get("warmup_steps")
            if existing in (None, 0, 0.0):
                kwargs["warmup_steps"] = ratio
                notes.append(
                    f"warmup_ratio={ratio} mapped to warmup_steps={ratio} "
                    "(transformers>=5 treats values <1 as a ratio)"
                )
            else:
                notes.append(
                    f"warmup_ratio={ratio} dropped because an explicit "
                    f"warmup_steps={existing} was also provided"
                )
        else:  # pragma: no cover - no known version lacks both
            notes.append(f"warmup_ratio={ratio} dropped: no warmup field available")

    # --- eval cadence ------------------------------------------------------
    # `evaluation_strategy` was renamed to `eval_strategy` in 4.41 and the old
    # alias was removed in v5. Emit whichever the installed version accepts.
    if "eval_strategy" in kwargs and "eval_strategy" not in supported:
        value = kwargs.pop("eval_strategy")
        if "evaluation_strategy" in supported:
            kwargs["evaluation_strategy"] = value
            notes.append("eval_strategy mapped to legacy evaluation_strategy")

    # --- removed knobs -----------------------------------------------------
    for removed in ("overwrite_output_dir", "safe_serialization", "logging_dir", "jit_mode_eval"):
        if removed in kwargs and removed not in supported:
            kwargs.pop(removed)
            notes.append(f"{removed} dropped: removed in the installed transformers")

    # --- anything else the installed version does not know about -----------
    for key in list(kwargs):
        if key not in supported:
            value = kwargs.pop(key)
            notes.append(f"{key}={value!r} dropped: not accepted by installed TrainingArguments")

    for note in notes:
        logger.debug("TrainingArguments compat: %s", note)
    return kwargs, notes


def dtype_kwarg(dtype: Any, *, supported: frozenset[str] | None = None) -> dict[str, Any]:
    """Return the correct ``from_pretrained`` dtype keyword for this version.

    transformers v5 renamed ``torch_dtype`` to ``dtype``. Passing the wrong one
    is either a hard ``TypeError`` or, worse, a silently ignored argument that
    loads the model in the wrong precision.
    """
    if dtype is None:
        return {}
    if supported is None:
        transformers = require_transformers()
        try:
            supported = frozenset(
                inspect.signature(transformers.PreTrainedModel.from_pretrained).parameters
            )
        except (AttributeError, TypeError, ValueError):  # pragma: no cover
            supported = frozenset()
    # Prefer the modern name when the signature exposes it explicitly.
    if "dtype" in supported:
        return {"dtype": dtype}
    if "torch_dtype" in supported:
        return {"torch_dtype": dtype}
    # Most from_pretrained signatures end in **kwargs, so neither name appears.
    # Fall back on the version number.
    version = transformers_version()
    if version is not None and version >= (5, 0, 0):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def trainer_tokenizer_kwarg(tokenizer: Any) -> dict[str, Any]:
    """Return the Trainer keyword carrying the tokenizer for this version.

    ``tokenizer=`` was deprecated in 4.46 in favour of ``processing_class=`` and
    removed in v5.
    """
    params = trainer_parameter_names()
    if "processing_class" in params:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def supports_field(name: str) -> bool:
    """Whether the installed ``TrainingArguments`` accepts ``name``."""
    return name in training_argument_names()


# ---------------------------------------------------------------------------
# Environment summary
# ---------------------------------------------------------------------------

#: Packages whose versions are recorded in every experiment manifest.
TRACKED_PACKAGES: tuple[str, ...] = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "datasets",
    "bitsandbytes",
    "safetensors",
    "huggingface-hub",
    "trl",
    "numpy",
    "pydantic",
)


def library_versions() -> dict[str, str | None]:
    """Version of every tracked package (``None`` when not installed)."""
    return {name: package_version(name) for name in TRACKED_PACKAGES}
