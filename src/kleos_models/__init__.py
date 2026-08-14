"""KLEOS Models: research engineering for KLEOS behavioral fine-tuning.

This package is deliberately split into a *light* layer and a *heavy* layer.

Light layer (no torch / transformers required)
    ``kleos_models.config``       typed configuration
    ``kleos_models.data``         schemas, loading, validation, splitting, leakage
    ``kleos_models.experiments``  manifests, registry, environment capture
    ``kleos_models.evaluation``   metrics, graders, consistency, OOD, reporting

Heavy layer (requires the ``[train]`` extra)
    ``kleos_models.models``       model loading, family adapters, quantization, PEFT
    ``kleos_models.training``     QLoRA trainer, callbacks, checkpointing
    ``kleos_models.inference``    generation backends

Importing this top-level package never imports torch. That property is enforced
by ``tests/test_import_isolation.py`` and is what keeps dataset work and CI fast.
"""

from kleos_models.constants import (
    DATASET_SCHEMA_VERSION,
    SUPPORTED_TASKS,
    VARIATION_AXES,
)

__version__ = "0.1.0"

__all__ = [
    "DATASET_SCHEMA_VERSION",
    "SUPPORTED_TASKS",
    "VARIATION_AXES",
    "__version__",
]
