"""Actionable exception types.

Spec section 32: an error must tell the user what happened *and* what to do next.
Every exception here carries an optional list of suggested remedies which the
CLI renders as a numbered list.
"""

from __future__ import annotations

from collections.abc import Sequence


class KleosError(Exception):
    """Base class for all KLEOS errors.

    Args:
        message: What went wrong, in one sentence.
        details: Optional key/value diagnostics printed verbatim.
        suggestions: Concrete next actions, most likely fix first.
    """

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, object] | None = None,
        suggestions: Sequence[str] | None = None,
    ) -> None:
        self.message = message
        self.details = dict(details or {})
        self.suggestions = list(suggestions or [])
        super().__init__(self.render())

    def render(self) -> str:
        """Format the error as a multi-line, human-readable diagnostic."""
        parts = [self.message]
        if self.details:
            parts.append("")
            width = max(len(str(k)) for k in self.details)
            for key, value in self.details.items():
                parts.append(f"  {str(key).ljust(width)} : {value}")
        if self.suggestions:
            parts.append("")
            parts.append("Suggested actions:")
            parts.extend(f"  {i}. {s}" for i, s in enumerate(self.suggestions, start=1))
        return "\n".join(parts)


class ConfigError(KleosError):
    """Configuration is missing, malformed, or internally inconsistent."""


class DataValidationError(KleosError):
    """A dataset or example failed schema/semantic validation."""


class DatasetIntegrityError(KleosError):
    """Dataset-level integrity problem: duplicate ids, split overlap, bad hashes."""


class LeakageError(KleosError):
    """Train/eval leakage was detected at a level configured to be fatal."""


class ModelCompatibilityError(KleosError):
    """The requested model cannot be used the way the config asks.

    Raised for genuine architectural mismatches: an auto-class that cannot load
    the checkpoint, LoRA target modules absent from the model, or a reasoning
    mode the checkpoint does not support.
    """


class MissingDependencyError(KleosError):
    """An optional dependency is required for this code path but not installed."""

    def __init__(self, package: str, *, extra: str, purpose: str) -> None:
        super().__init__(
            f"{package!r} is required to {purpose} but is not installed.",
            details={"missing_package": package, "install_extra": extra},
            suggestions=[
                f'Install the extra: pip install -e ".[{extra}]"',
                "On Google Colab run: python scripts/colab_setup.py",
                "See docs/troubleshooting.md for environment-specific notes.",
            ],
        )
        self.package = package
        self.extra = extra


class InsufficientMemoryError(KleosError):
    """The configured run does not fit in available accelerator memory."""


class CheckpointError(KleosError):
    """A checkpoint could not be discovered, read, or resumed."""


class EvaluationError(KleosError):
    """Evaluation could not be run or produced structurally invalid results."""


class PublishingError(KleosError):
    """Publishing to the Hugging Face Hub was refused or failed."""


class PrivacyViolationError(KleosError):
    """Content that looks private/secret was found on a path into the public repo."""
