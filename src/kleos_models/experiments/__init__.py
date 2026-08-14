"""Experiment infrastructure: manifests, registry, environment capture.

Imports no torch, so run history can be inspected anywhere.
"""

from kleos_models.experiments.environment import (
    EnvironmentSnapshot,
    GitInfo,
    capture_environment,
    capture_git_info,
    detect_colab,
    set_global_seed,
)
from kleos_models.experiments.manifest import (
    ExperimentManifest,
    RunStatus,
    build_manifest,
    generate_experiment_id,
)
from kleos_models.experiments.registry import (
    ExperimentRegistry,
    RegistryEntry,
    assert_comparable,
)

__all__ = [
    "EnvironmentSnapshot",
    "ExperimentManifest",
    "ExperimentRegistry",
    "GitInfo",
    "RegistryEntry",
    "RunStatus",
    "assert_comparable",
    "build_manifest",
    "capture_environment",
    "capture_git_info",
    "detect_colab",
    "generate_experiment_id",
    "set_global_seed",
]
