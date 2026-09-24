#!/usr/bin/env python3
"""Decide GO / NO-GO for a full training run from a smoke run's manifest.

A smoke run (``configs/training/debug_logos.yaml``) exercises the real model
settings for ten steps. This reads what it recorded and checks, one line each:

* the run completed, and nothing was adjusted to make it fit;
* a text-only checkpoint view loaded every weight (no missing or mismatched key);
* LoRA attached where expected (module and trainable-parameter counts);
* gradients reached the adapter;
* the memory probe on the longest batch left enough spare memory once the
  optimizer state exists.

    python scripts/check_smoke_gate.py \\
        --manifest /content/logos-smoke/kleos-logos-smoke-001/manifest.json \\
        --expect-modules 280 --expect-trainable 60948480 --min-spare-gb 0.15

Exit codes: 0 GO, 1 NO-GO, 2 the manifest could not be read. Imports no torch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

#: A failed check, a passed one, and one the manifest cannot answer.
FAIL, PASS, SKIP = "✗", "✓", "-"


def evaluate_gate(
    manifest: dict[str, Any],
    *,
    expect_modules: int | None = None,
    expect_trainable: int | None = None,
    min_spare_gb: float = 0.15,
) -> list[tuple[str, str, str]]:
    """Every check as ``(mark, name, detail)``."""
    checks: list[tuple[str, str, str]] = []

    def add(ok: bool | None, name: str, detail: str) -> None:
        checks.append((SKIP if ok is None else PASS if ok else FAIL, name, detail))

    status = manifest.get("status")
    add(status == "completed", "run completed", f"status {status!r}")

    adjustments = manifest.get("adjustments") or []
    add(
        not adjustments,
        "nothing adjusted to fit",
        "none" if not adjustments else "; ".join(str(a.get("field")) for a in adjustments),
    )

    load = (manifest.get("model") or {}).get("load") or {}
    view = load.get("view")
    if view:
        info = view.get("loading_info") or {}
        missing, mismatched = info.get("missing_keys"), info.get("mismatched_keys")
        add(
            missing == 0 and mismatched == 0,
            "text-only view loaded every weight",
            f"missing {missing}, mismatched {mismatched}, "
            f"ignored {info.get('unexpected_keys')} vision/projector key(s)",
        )
    else:
        add(None, "text-only view loaded every weight", "no view in this run")

    lora = manifest.get("lora") or {}
    modules = (lora.get("target_modules") or {}).get("matched_module_count")
    if expect_modules is None:
        add(None, "LoRA modules", f"{modules} (no expectation given)")
    else:
        add(modules == expect_modules, "LoRA modules", f"{modules}, expected {expect_modules}")
    trainable = lora.get("trainable_parameters")
    if expect_trainable is None:
        add(None, "trainable parameters", f"{trainable} (no expectation given)")
    else:
        add(
            trainable == expect_trainable,
            "trainable parameters",
            f"{trainable:,}, expected {expect_trainable:,}"
            if isinstance(trainable, int)
            else f"{trainable}, expected {expect_trainable:,}",
        )

    metrics = manifest.get("metrics") or {}
    gradients = metrics.get("gradient_check") or {}
    nonzero = gradients.get("lora_tensors_with_nonzero_grad")
    add(
        bool(nonzero) if gradients else None,
        "gradients reach the adapter",
        f"{nonzero} of {gradients.get('lora_tensors')} LoRA tensors"
        if gradients
        else "no gradient check recorded",
    )

    probe = metrics.get("memory_probe") or {}
    if probe:
        spare = probe.get("spare_after_optimizer_gb")
        add(
            spare is not None and spare >= min_spare_gb,
            "memory on the longest batch",
            f"peak {probe.get('peak_allocated_gb')} GB allocated, "
            f"{probe.get('peak_reserved_gb')} GB reserved of {probe.get('total_gb')} GB "
            f"(batch {probe.get('batch_size')} x {probe.get('sequence_length')} tokens); "
            f"{spare} GB spare after the optimizer state, need >= {min_spare_gb}",
        )
    else:
        add(False, "memory on the longest batch", "no memory probe recorded")
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, required=True, help="The smoke run's manifest.")
    parser.add_argument("--expect-modules", type=int, help="Expected LoRA module count.")
    parser.add_argument("--expect-trainable", type=int, help="Expected trainable parameters.")
    parser.add_argument(
        "--min-spare-gb",
        type=float,
        default=0.15,
        help="Spare GPU memory required at the longest batch (default 0.15).",
    )
    args = parser.parse_args(argv)

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"✗ Cannot read {args.manifest}: {exc}", file=sys.stderr)
        return 2

    checks = evaluate_gate(
        manifest,
        expect_modules=args.expect_modules,
        expect_trainable=args.expect_trainable,
        min_spare_gb=args.min_spare_gb,
    )
    print(f"Smoke gate for {manifest.get('experiment_id', args.manifest.parent.name)}")
    for mark, name, detail in checks:
        print(f"  {mark} {name:<38} {detail}")
    failed = [name for mark, name, _ in checks if mark == FAIL]
    if failed:
        print(f"\nNO-GO: {', '.join(failed)}.")
        return 1
    print("\nGO: the full run may start.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
