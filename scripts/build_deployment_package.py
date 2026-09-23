#!/usr/bin/env python3
"""Build a verifiable deployment package from a frozen training run.

The research artifact is READ-ONLY here. Nothing in the run directory is
modified, and the package is a separate copy with its own manifest.

The one substantive difference between the two is the base revision. PEFT wrote
``revision: null`` into the research ``adapter_config.json`` (finding H-F1), and
that file stays exactly as it is, as historical evidence of what the run
produced. The package gets its own ``adapter_config.json`` with the revision
pinned, and the manifest records both the pin and the fact that the original
lacked it. Nothing is rewritten to look like it was always correct.

Usage::

    python scripts/build_deployment_package.py \\
        --run /path/to/outputs/kleos-v006-mistralnemo12b-run1 \\
        --deployment-config configs/deployment/kleos_hermes_v006.yaml \\
        --output /path/to/packages/hermes-v0.0.6

Produces::

    hermes-v0.0.6/
    ├── manifest.json
    ├── adapter/{adapter_model.safetensors,adapter_config.json,README.md}
    ├── tokenizer/{tokenizer.json,tokenizer_config.json,chat_template.jinja}
    └── deployment/{README.md,model_config.yaml}

Never writes model weights into this repository: point --output outside it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import add_common_arguments, print_header, run, setup_logging
from kleos_models.config import load_model_config
from kleos_models.data.loaders import file_sha256
from kleos_models.errors import KleosError
from kleos_models.serving.manifest import (
    ADAPTER_DIRNAME,
    DEPLOYMENT_DIRNAME,
    REQUIRED_ADAPTER_FILES,
    REQUIRED_TOKENIZER_FILES,
    TOKENIZER_DIRNAME,
    GenerationDefaults,
    ServingLimits,
    build_deployment_manifest,
    verify_package,
)

#: Copied into the package. Anything else in the run directory stays behind:
#: checkpoints, logs, events and the dataset are not deployment artifacts.
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json", "README.md")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")

DEPLOYMENT_README = """\
# {model_name} {model_version} — deployment package

Built from the frozen research run `{experiment_id}`.

This directory is a **deployment artifact**, not the research artifact. It is a
copy, with one deliberate difference recorded in `manifest.json`:
`adapter/adapter_config.json` pins the base revision, where the research copy
records `revision: null` (finding H-F1). The research artifact is unchanged.

## What this is

| | |
| --- | --- |
| Base | `{base_model}` |
| Base revision | `{base_revision}` |
| Adapter | `{adapter_sha256}` |
| From checkpoint | `{source_checkpoint}` ({selection_metric} {selection_value}) |
| Dataset | `{dataset_version}` |
| Training config hash | `{training_config_hash}` |

## Verify before serving

```bash
python scripts/verify_deployment_package.py --package .
```

Re-hashes every file against the manifest, checks the adapter config pins the
recorded base revision, and refuses the package if anything disagrees.

## Serve

```bash
export HERMES_PACKAGE_DIR=$(pwd)
export HERMES_API_KEY=...          # shared secret; never commit one
python scripts/serve_hermes.py --host 127.0.0.1 --port 8000
```

The base weights are **not** in this package. They are downloaded from the model
registry at the pinned revision on first start and verified at load time.

## Contract

Greedy decoding, matching the conditions the reported scores were measured
under. The tokenizer is the frozen copy in `tokenizer/`, loaded with
`fix_mistral_regex={fix_mistral_regex}` stated explicitly so a library default
cannot move tokenization under the adapter.

Known limitations are in `manifest.json` under `known_limitations`, with the
evidence in `docs/experiments/kleos-v006-mistralnemo12b-run1-report.md`.
"""


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise KleosError(
            f"{what} not found: {path}",
            suggestions=["Point --run at a completed training run directory."],
        )
    return path


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_packaged_adapter_config(source: Path, destination: Path, revision: str) -> dict[str, Any]:
    """Copy adapter_config.json with the base revision pinned.

    Returns the packaged config. Only ``revision`` is changed; every other field
    is carried across untouched so the LoRA shape cannot drift between the
    research artifact and the deployed one.
    """
    config = json.loads(source.read_text(encoding="utf-8"))
    original_revision = config.get("revision")
    config["revision"] = revision
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"config": config, "original_revision": original_revision}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True, help="Frozen training run directory.")
    parser.add_argument(
        "--deployment-config",
        type=Path,
        default=Path("configs/deployment/kleos_hermes_v006.yaml"),
        help="Declarative serving record to build from.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/models/mistral_nemo_12b.yaml"),
        help="Model config packaged for the loader.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Package destination.")
    parser.add_argument("--force", action="store_true", help="Overwrite a non-empty destination.")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("Build deployment package")

    spec = yaml.safe_load(args.deployment_config.read_text(encoding="utf-8"))["deployment"]
    model_config = load_model_config(args.model_config)

    # The two records must already agree; tests assert this too, but a build is
    # the last moment it can be caught before an artifact exists.
    if model_config.revision != spec["revision"]:
        raise KleosError(
            "The model config and the deployment config disagree about the base revision.",
            details={"model_config": model_config.revision, "deployment": spec["revision"]},
            suggestions=["Reconcile them before building; do not pick one here."],
        )
    if model_config.base_model != spec["base_model"]:
        raise KleosError(
            "The model config and the deployment config disagree about the base model.",
            details={"model_config": model_config.base_model, "deployment": spec["base_model"]},
        )

    run_dir = _require(args.run, "Run directory")
    adapter_src = _require(run_dir / "adapter", "adapter/ directory")
    tokenizer_src = _require(run_dir / "tokenizer", "tokenizer/ directory")

    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        raise KleosError(
            f"{args.output} already exists and is not empty.",
            suggestions=["Choose another --output, or pass --force."],
        )

    # Fail closed before copying: if these are not the frozen weights, nothing
    # downstream is the model that was measured.
    weights = _require(adapter_src / "adapter_model.safetensors", "adapter weights")
    actual_sha = file_sha256(weights)
    expected_sha = spec.get("adapter_sha256")
    if expected_sha and actual_sha != expected_sha:
        raise KleosError(
            "The adapter in this run is not the frozen Hermes adapter.",
            details={"expected": expected_sha, "actual": actual_sha},
            suggestions=[
                "Point --run at the run that produced the frozen artifact.",
                "If the file really changed, stop and find out why before packaging.",
            ],
        )
    print(f"  adapter sha256 : {actual_sha}")
    print(f"  matches record : {'yes' if expected_sha == actual_sha else 'NOT CHECKED'}")

    package = args.output
    package.mkdir(parents=True, exist_ok=True)

    for name in ADAPTER_FILES:
        source = adapter_src / name
        if not source.exists():
            if name in REQUIRED_ADAPTER_FILES:
                raise KleosError(f"Required adapter file missing: {source}")
            print(f"    - adapter/{name} (absent in the run)")
            continue
        if name == "adapter_config.json":
            continue  # written below, with the revision pinned
        _copy(source, package / ADAPTER_DIRNAME / name)
        print(f"    + {ADAPTER_DIRNAME}/{name}")

    pinned = build_packaged_adapter_config(
        adapter_src / "adapter_config.json",
        package / ADAPTER_DIRNAME / "adapter_config.json",
        spec["revision"],
    )
    print(
        f"    + {ADAPTER_DIRNAME}/adapter_config.json "
        f"(revision {pinned['original_revision']!r} → {spec['revision'][:12]}…)"
    )

    for name in TOKENIZER_FILES:
        source = tokenizer_src / name
        if not source.exists():
            if name in REQUIRED_TOKENIZER_FILES:
                raise KleosError(f"Required tokenizer file missing: {source}")
            print(f"    - tokenizer/{name} (absent in the run)")
            continue
        _copy(source, package / TOKENIZER_DIRNAME / name)
        print(f"    + {TOKENIZER_DIRNAME}/{name}")

    # The model config travels with the package so a deployment never depends on
    # this repository being checked out beside it. The tokenizer is resolved to
    # the package's own directory at load time, so it is left unset here and the
    # package stays relocatable.
    packaged_model = model_config.model_dump(mode="json")
    packaged_model["tokenizer"] = None
    quantization = spec.get("quantization") or {}
    packaged_model["quantization"].update(
        {k: v for k, v in quantization.items() if k in packaged_model["quantization"]}
    )
    model_config_path = package / DEPLOYMENT_DIRNAME / "model_config.yaml"
    model_config_path.parent.mkdir(parents=True, exist_ok=True)
    model_config_path.write_text(
        yaml.safe_dump({"model": packaged_model}, sort_keys=True, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"    + {DEPLOYMENT_DIRNAME}/model_config.yaml")

    tokenizer_spec = spec.get("tokenizer") or {}
    generation = GenerationDefaults(**(spec.get("generation") or {}))
    limits = ServingLimits(**(spec.get("limits") or {}))

    manifest = build_deployment_manifest(
        package,
        model_name=spec["name"],
        model_version=spec["version"],
        base_model=spec["base_model"],
        base_revision=spec["revision"],
        experiment_id=spec["experiment_id"],
        source_checkpoint=spec.get("source_checkpoint", "unknown"),
        trainable_parameters=int(spec["trainable_parameters"]),
        dataset_version=spec["dataset_version"],
        dataset_sha256=spec["dataset_sha256"],
        training_config_hash=spec["training_config_hash"],
        adapter_config=pinned["config"],
        fix_mistral_regex=bool(tokenizer_spec.get("fix_mistral_regex", False)),
        tokenizer_padding_side=tokenizer_spec.get("padding_side", "right"),
        tokenizer_pad_token=tokenizer_spec.get("pad_token"),
        split_strategy=spec.get("split_strategy"),
        selection_value=spec.get("selection_value"),
        generation=generation,
        limits=limits,
        research_report="docs/experiments/kleos-v006-mistralnemo12b-run1-report.md",
        known_limitations=list(spec.get("notes") or []),
        notes=[
            "Deployment artifact. The research artifact is unchanged and remains "
            "the historical record.",
            f"adapter_config.json revision was {pinned['original_revision']!r} in the "
            "research artifact and is pinned here (finding H-F1).",
            "Base weights are not packaged; they are resolved from the model "
            "registry at the pinned revision and verified at load time.",
        ],
    )
    manifest_path = manifest.save(package)
    print(f"    + {manifest_path.name}")

    (package / DEPLOYMENT_DIRNAME / "README.md").write_text(
        DEPLOYMENT_README.format(
            model_name=manifest.model_name,
            model_version=manifest.model_version,
            experiment_id=manifest.adapter.experiment_id,
            base_model=manifest.base_model,
            base_revision=manifest.base_revision,
            adapter_sha256=manifest.adapter.weights_sha256,
            source_checkpoint=manifest.adapter.source_checkpoint,
            selection_metric=manifest.adapter.selection_metric,
            selection_value=manifest.adapter.selection_value,
            dataset_version=manifest.dataset.version,
            training_config_hash=manifest.training_config_hash,
            fix_mistral_regex=manifest.tokenizer.fix_mistral_regex,
        ),
        encoding="utf-8",
    )
    print(f"    + {DEPLOYMENT_DIRNAME}/README.md")

    verify_package(package)
    print(f"\n✓ Package built and verified: {package}")
    print(f"  {manifest.model_name} {manifest.model_version}")
    print(f"  base    : {manifest.base_model}@{manifest.base_revision[:12]}…")
    print(f"  adapter : {manifest.adapter.weights_sha256[:16]}…")
    print("\n  The research run was not modified.")
    print(f"\n  Next: python scripts/hermes_smoke.py --package {package}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
