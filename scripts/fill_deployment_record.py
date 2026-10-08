#!/usr/bin/env python3
"""Complete a deployment record from its frozen training run.

A deployment record (``configs/deployment/<model>.yaml``) is the identity the
package and the Space must match. Some of it exists only in the training run's
output: the adapter's sha256, the selection value of the best checkpoint, the
tokenizer's pad token and the sha256 of the tokenizer files the run saved
(``save_pretrained`` may rewrite them, so the Hub's hashes would be wrong).
Those lines are left as ``null`` in the record until this script reads them.

It checks the run is the one the record describes first: the best checkpoint
must be the record's ``source_checkpoint``, and the run's ``config_hash``,
``dataset_hash`` and ``experiment_id`` must equal the record's. It then fills
only the ``null`` lines, line by line so the record's comments survive, and
refuses to replace a value that is already there and different.

The run directory is READ-ONLY here.

Usage::

    python scripts/fill_deployment_record.py \\
        --run ~/kleos-private/logos-v002/outputs/kleos-v007-ministral314breasoning-run1 \\
        --record configs/deployment/kleos_logos_v002.yaml            # prints the values
    python scripts/fill_deployment_record.py --run ... --record ... --write
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import add_common_arguments, print_header, run, setup_logging
from kleos_models.data.loaders import file_sha256
from kleos_models.errors import ConfigError
from kleos_models.serving.manifest import load_expected_identity

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def _step(path: Path) -> int:
    return int(path.parent.name.rsplit("-", 1)[1])


def read_run(run_dir: Path) -> dict[str, Any]:
    """The values a record takes from its training run."""
    adapter = run_dir / "adapter" / "adapter_model.safetensors"
    tokenizer = run_dir / "tokenizer"
    manifest_path = run_dir / "manifest.json"
    required = [adapter, *(tokenizer / name for name in TOKENIZER_FILES), manifest_path]
    missing = [str(path.relative_to(run_dir)) for path in required if not path.is_file()]
    states = sorted(run_dir.glob("checkpoint-*/trainer_state.json"), key=_step)
    if not states:
        missing.append("checkpoint-*/trainer_state.json")
    if missing:
        raise ConfigError(
            f"The run directory is missing: {', '.join(missing)}.",
            suggestions=["Point --run at the run folder inside the downloaded training output."],
        )

    state = json.loads(states[-1].read_text(encoding="utf-8"))
    best = state.get("best_model_checkpoint")
    metric = state.get("best_metric")
    if not best or metric is None:
        raise ConfigError(
            f"{states[-1].relative_to(run_dir)} names no best checkpoint.",
            suggestions=["The run must have been trained with load_best_model_at_end."],
        )
    pad = json.loads((tokenizer / "tokenizer_config.json").read_text(encoding="utf-8")).get(
        "pad_token"
    )
    if isinstance(pad, dict):
        pad = pad.get("content")
    if not isinstance(pad, str) or not pad:
        raise ConfigError("tokenizer/tokenizer_config.json states no pad_token.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "adapter_sha256": file_sha256(adapter),
        "selection_value": float(metric),
        "best_checkpoint": Path(str(best)).name,
        "pad_token": pad,
        "files_sha256": {name: file_sha256(tokenizer / name) for name in TOKENIZER_FILES},
        "config_hash": manifest.get("config_hash"),
        "dataset_hash": manifest.get("dataset_hash"),
        "experiment_id": manifest.get("experiment_id"),
    }


def check_run(values: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """Reasons the run is not the one the record describes."""
    problems = []
    if values["best_checkpoint"] != spec.get("source_checkpoint"):
        problems.append(
            f"the run's best checkpoint is {values['best_checkpoint']}, the record names "
            f"{spec.get('source_checkpoint')}"
        )
    for run_key, record_key in (
        ("config_hash", "training_config_hash"),
        ("dataset_hash", "dataset_sha256"),
        ("experiment_id", "experiment_id"),
    ):
        if values[run_key] != spec.get(record_key):
            problems.append(
                f"the run's {run_key} is {values[run_key]!r}, the record's {record_key} is "
                f"{spec.get(record_key)!r}"
            )
    return problems


def _targets(values: dict[str, Any]) -> list[tuple[str, Any]]:
    return [
        ("adapter_sha256", values["adapter_sha256"]),
        ("selection_value", values["selection_value"]),
        ("pad_token", values["pad_token"]),
        *((name, digest) for name, digest in values["files_sha256"].items()),
    ]


def _render(value: Any) -> str:
    if isinstance(value, float):
        return repr(value)
    return json.dumps(value) if not re.fullmatch(r"[0-9a-f]{64}", str(value)) else str(value)


def fill_text(text: str, values: dict[str, Any]) -> tuple[str, list[str]]:
    """The record with its ``null`` lines filled, and what changed."""
    lines = text.splitlines(keepends=True)
    changed = []
    for key, value in _targets(values):
        pattern = re.compile(rf"^(\s*){re.escape(key)}:[ \t]*(.*?)[ \t]*$")
        hits = [i for i, line in enumerate(lines) if pattern.match(line.rstrip("\n"))]
        if len(hits) != 1:
            raise ConfigError(f"The record must have exactly one `{key}:` line, found {len(hits)}.")
        index = hits[0]
        match = pattern.match(lines[index].rstrip("\n"))
        assert match is not None
        indent, current = match.group(1), match.group(2)
        present = yaml.safe_load(current) if current else None
        if present is None:
            ending = "\n" if lines[index].endswith("\n") else ""
            lines[index] = f"{indent}{key}: {_render(value)}{ending}"
            changed.append(key)
        elif present != value:
            raise ConfigError(
                f"`{key}` already holds {present!r}; the run has {value!r}.",
                suggestions=[
                    "A filled record is an identity. If the run really changed, this is a "
                    "different artifact: stop and investigate rather than overwrite it."
                ],
            )
    return "".join(lines), changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True, help="Frozen training run directory.")
    parser.add_argument("--record", type=Path, required=True, help="Deployment record to fill.")
    parser.add_argument("--write", action="store_true", help="Write the values into the record.")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("Complete a deployment record from its training run")
    text = args.record.read_text(encoding="utf-8")
    spec = (yaml.safe_load(text) or {}).get("deployment") or {}
    values = read_run(args.run)
    problems = check_run(values, spec)
    if problems:
        raise ConfigError(
            "The run is not the one the record describes: " + "; ".join(problems) + ".",
            suggestions=["Check you downloaded the right training notebook version's output."],
        )

    print(f"  run            : {args.run}")
    print(f"  record         : {args.record}")
    print(f"  best checkpoint: {values['best_checkpoint']} (eval_loss {values['selection_value']})")
    for key, value in _targets(values):
        print(f"    {key:<22} {value}")

    filled, changed = fill_text(text, values)
    if not args.write:
        print("\n✓ The run matches the record. Re-run with --write to fill the record.\n")
        return 0
    if changed:
        args.record.write_text(filled, encoding="utf-8")
    load_expected_identity(args.record)
    print(
        f"\n✓ Filled {len(changed)} value(s); the record is complete."
        if changed
        else "\n✓ The record already holds these values; nothing changed."
    )
    print("  Next: commit and push the record, then build the package.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
