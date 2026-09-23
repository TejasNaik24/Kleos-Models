"""Serving-side code for frozen KLEOS adapters.

This package is deliberately separate from ``training`` and ``evaluation``. A
research run answers "does this work?"; a deployment answers "is the thing
running right now the exact artifact we measured?". They have different failure
modes, so they get different code.

Nothing here retrains, re-evaluates or mutates a research artifact. The one job
is to take a frozen adapter and prove, at load time, that it is paired with the
base weights, tokenizer and configuration it was measured against — and to
refuse to serve when it cannot.

``manifest`` is importable without torch (it only hashes files and validates
schemas). ``loader`` and ``app`` import torch / transformers / fastapi lazily
inside functions, so planning and verification work on a laptop.
"""

from __future__ import annotations

from kleos_models.serving.manifest import (
    DEPLOYMENT_ARTIFACT_VERSION,
    AdapterRecord,
    DatasetRecord,
    DeploymentManifest,
    FileRecord,
    GenerationDefaults,
    PeftRecord,
    ServingLimits,
    TokenizerContract,
    build_deployment_manifest,
    verify_package,
)

__all__ = [
    "DEPLOYMENT_ARTIFACT_VERSION",
    "AdapterRecord",
    "DatasetRecord",
    "DeploymentManifest",
    "FileRecord",
    "GenerationDefaults",
    "PeftRecord",
    "ServingLimits",
    "TokenizerContract",
    "build_deployment_manifest",
    "verify_package",
]
