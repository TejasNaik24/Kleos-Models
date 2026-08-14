"""Enforce the dependency-isolation rule.

The data, config, experiment and evaluation-scoring layers must be importable
with **no torch and no transformers installed**. That property is what makes
dataset work fast, CI cheap, and re-scoring possible on any machine.

It is easy to break by accident: one convenience import at module scope in a
utility file pulls torch into the whole light layer. These tests make that
failure loud.

The technique is to poison ``sys.modules`` so any attempt to import torch raises,
then import each light module in a subprocess-clean namespace.
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest

#: Modules that must never import torch or transformers at module scope.
LIGHT_MODULES = [
    "kleos_models",
    "kleos_models.config",
    "kleos_models.compat",
    "kleos_models.constants",
    "kleos_models.errors",
    "kleos_models.logging_utils",
    "kleos_models.publishing",
    "kleos_models.data",
    "kleos_models.data.schemas",
    "kleos_models.data.loaders",
    "kleos_models.data.validation",
    "kleos_models.data.formatting",
    "kleos_models.data.splitting",
    "kleos_models.data.leakage",
    "kleos_models.data.coverage",
    "kleos_models.evaluation",
    "kleos_models.evaluation.metrics",
    "kleos_models.evaluation.graders",
    "kleos_models.evaluation.consistency",
    "kleos_models.evaluation.ood",
    "kleos_models.evaluation.faithfulness",
    "kleos_models.evaluation.capability",
    "kleos_models.evaluation.reports",
    "kleos_models.evaluation.runner",
    "kleos_models.experiments",
    "kleos_models.experiments.manifest",
    "kleos_models.experiments.registry",
    "kleos_models.experiments.environment",
]

#: The heavy layer may not import torch at module scope either — it imports
#: inside functions — so that inspection and planning work without it.
LAZY_HEAVY_MODULES = [
    "kleos_models.models",
    "kleos_models.models.adapters",
    "kleos_models.models.loading",
    "kleos_models.models.quantization",
    "kleos_models.models.peft_setup",
    "kleos_models.models.feasibility",
    "kleos_models.training",
    "kleos_models.training.trainer",
    "kleos_models.training.arguments",
    "kleos_models.training.callbacks",
    "kleos_models.training.checkpointing",
    "kleos_models.training.memory",
    "kleos_models.inference",
    "kleos_models.inference.backends",
    "kleos_models.inference.generate",
]

BLOCKED = ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets")


class _BlockingFinder:
    """A meta-path finder that refuses to locate the blocked modules.

    Patching ``builtins.__import__`` alone is not enough:
    ``importlib.import_module`` goes through ``_bootstrap._gcd_import`` and never
    consults it. Installing a finder catches both routes, which matters because
    the heavy layer imports torch via ``importlib``.
    """

    def find_module(self, fullname, path=None):  # pragma: no cover - legacy API
        return None

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ImportError(f"{fullname} is blocked by the import-isolation test")
        return None


@pytest.fixture
def block_heavy_imports(monkeypatch):
    """Make importing torch and friends raise ImportError, by either route."""
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root in BLOCKED:
            raise ImportError(f"{root} is blocked by the import-isolation test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(sys, "meta_path", [_BlockingFinder(), *sys.meta_path])

    # Drop already-imported copies so the guard is actually exercised.
    for name in list(sys.modules):
        if name.split(".")[0] in BLOCKED:
            monkeypatch.delitem(sys.modules, name, raising=False)
    return guarded


def _reimport(module_name: str, monkeypatch):
    """Force a fresh import of a module."""
    for name in list(sys.modules):
        if name.startswith("kleos_models"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", LIGHT_MODULES)
def test_light_module_imports_without_torch(module_name, block_heavy_imports, monkeypatch):
    """Every light module imports with torch unavailable."""
    module = _reimport(module_name, monkeypatch)
    assert module is not None


@pytest.mark.parametrize("module_name", LAZY_HEAVY_MODULES)
def test_heavy_module_imports_lazily(module_name, block_heavy_imports, monkeypatch):
    """Model/training modules import without torch; they import it inside functions.

    This is what lets `scripts/inspect_model.py` and `scripts/plan_run.py` report
    architecture and feasibility before anything heavy is installed.
    """
    module = _reimport(module_name, monkeypatch)
    assert module is not None


def test_importing_the_package_does_not_pull_in_torch(monkeypatch):
    """A plain `import kleos_models` must not load torch even when it exists."""
    for name in list(sys.modules):
        if name.startswith("kleos_models"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    for name in list(sys.modules):
        if name.split(".")[0] in ("torch", "transformers"):
            monkeypatch.delitem(sys.modules, name, raising=False)

    importlib.import_module("kleos_models")
    importlib.import_module("kleos_models.data")
    importlib.import_module("kleos_models.evaluation")

    leaked = [name for name in ("torch", "transformers") if name in sys.modules]
    assert not leaked, (
        f"importing the light layer pulled in {leaked}. Move the import inside a "
        "function: a module-scope import here makes every dataset operation "
        "require a multi-gigabyte install."
    )


def test_missing_dependency_error_is_actionable(block_heavy_imports, monkeypatch):
    """Calling into the heavy layer without torch gives an install hint."""
    compat = _reimport("kleos_models.compat", monkeypatch)
    # Resolve the exception class *after* the re-import: _reimport rebuilds the
    # kleos_models modules, so a class bound beforehand is a different object.
    errors = importlib.import_module("kleos_models.errors")

    with pytest.raises(errors.MissingDependencyError) as info:
        compat.require_transformers()

    message = str(info.value)
    assert "transformers" in message
    assert "pip install" in message
    assert "colab_setup" in message
