#!/usr/bin/env python3
"""Install KLEOS dependencies on Google Colab without breaking the runtime.

The critical rule: **do not reinstall torch on Colab.** Colab ships a torch build
compiled against its specific CUDA driver. `pip install torch` (or any package
that pulls torch as a dependency without constraint) replaces it with a generic
wheel, and CUDA silently stops working or the runtime starts crashing. Every
install here uses `--no-deps` for packages that would otherwise drag torch along,
and torch itself is installed only when it is genuinely absent.

Usage, from a notebook cell::

    !python scripts/colab_setup.py

    # Then verify:
    !python scripts/smoke_test.py
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Packages that pull torch transitively. Installed with --no-deps so pip cannot
#: swap out Colab's torch build.
TORCH_DEPENDENT = (
    ("transformers", ">=4.56,<6"),
    ("peft", ">=0.12"),
    ("accelerate", ">=0.34"),
    ("bitsandbytes", ">=0.43"),
    ("trl", None),
)

#: Safe to install normally: no torch dependency.
INDEPENDENT = (
    ("datasets", ">=2.20"),
    ("safetensors", ">=0.4.3"),
    ("huggingface_hub", ">=0.25"),
    ("pydantic", ">=2.7,<3"),
    ("PyYAML", ">=6.0"),
    ("jsonschema", ">=4.21"),
    ("sentencepiece", ">=0.2"),
)

#: Dependencies of the --no-deps packages that Colab may not already have.
NO_DEPS_REQUIREMENTS = (
    "tokenizers",
    "regex",
    "requests",
    "tqdm",
    "filelock",
    "packaging",
    "psutil",
)


def _pip(*args: str, quiet: bool = True) -> int:
    """Run pip in this interpreter."""
    command = [sys.executable, "-m", "pip", "install", *args]
    if quiet:
        command.insert(4, "--quiet")
    print(f"  $ pip install {' '.join(args)}")
    return subprocess.run(command, check=False).returncode


def _installed(module: str) -> str | None:
    """Return an installed package's version, or None."""
    try:
        import importlib.metadata as metadata

        return metadata.version(module)
    except Exception:
        return None


def detect_environment() -> dict[str, object]:
    """Report what the runtime already provides."""
    info: dict[str, object] = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "in_colab": "google.colab" in sys.modules or Path("/content").exists(),
    }
    torch_version = _installed("torch")
    info["torch"] = torch_version
    if torch_version:
        try:
            torch = importlib.import_module("torch")
            info["cuda_available"] = bool(torch.cuda.is_available())
            info["cuda_version"] = torch.version.cuda
            if torch.cuda.is_available():
                properties = torch.cuda.get_device_properties(0)
                info["gpu"] = properties.name
                info["vram_gb"] = round(properties.total_memory / 1024**3, 1)
                info["compute_capability"] = f"{properties.major}.{properties.minor}"
                info["bf16"] = (properties.major, properties.minor) >= (8, 0)
        except Exception as exc:  # pragma: no cover
            info["torch_error"] = str(exc)
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--no-quant", action="store_true", help="Skip bitsandbytes (no 4-bit training)."
    )
    parser.add_argument(
        "--upgrade", action="store_true", help="Upgrade packages that are already present."
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Report the environment and exit."
    )
    args = parser.parse_args(argv)

    print("=" * 72)
    print("KLEOS Colab setup")
    print("=" * 72)

    info = detect_environment()
    print("\nRuntime")
    for key, value in info.items():
        print(f"  {key:<20}: {value}")

    if not info.get("torch"):
        print("\n  ! torch is not installed. On Colab it normally is - if you are")
        print("    seeing this on Colab, the runtime may not be a GPU runtime.")
        print("    Runtime → Change runtime type → GPU.")
    elif not info.get("cuda_available"):
        print("\n  ! torch is installed but reports no CUDA device.")
        print("    Runtime → Change runtime type → GPU, then run this again.")
    elif not info.get("bf16"):
        print(
            f"\n  Note: {info.get('gpu')} is compute capability "
            f"{info.get('compute_capability')}, which does not support bfloat16."
        )
        print("  KLEOS configs use compute_dtype: auto, which selects float16 here.")

    if args.check_only:
        return 0

    print("\n── installing (torch is left alone) " + "─" * 28)

    failures: list[str] = []

    # Prerequisites of the --no-deps installs.
    missing_requirements = [p for p in NO_DEPS_REQUIREMENTS if not _installed(p)]
    if missing_requirements and _pip(*missing_requirements) != 0:
        failures.extend(missing_requirements)

    for name, constraint in INDEPENDENT:
        if _installed(name) and not args.upgrade:
            print(f"  · {name} already present ({_installed(name)})")
            continue
        spec = f"{name}{constraint}" if constraint else name
        if _pip(spec) != 0:
            failures.append(name)

    for name, constraint in TORCH_DEPENDENT:
        if name == "bitsandbytes" and args.no_quant:
            print("  · skipping bitsandbytes (--no-quant)")
            continue
        if name == "trl":
            continue  # not used by the canonical training path
        if _installed(name) and not args.upgrade:
            print(f"  · {name} already present ({_installed(name)})")
            continue
        spec = f"{name}{constraint}" if constraint else name
        # --no-deps is the whole point: it stops pip replacing Colab's torch.
        if _pip(spec, "--no-deps") != 0:
            failures.append(name)

    # Install the repository itself without dependencies, for the same reason.
    print("\n── installing kleos-models " + "─" * 37)
    if _pip("-e", str(REPO_ROOT), "--no-deps") != 0:
        failures.append("kleos-models")

    print("\n── verification " + "─" * 48)
    ok = True
    for module in ("torch", "transformers", "peft", "kleos_models"):
        version = _installed(module) or _installed(module.replace("_", "-"))
        if version:
            print(f"  ✓ {module:<16} {version}")
        else:
            print(f"  ✗ {module:<16} not importable")
            ok = False

    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA still works after install: {torch.cuda.get_device_name(0)}")
        else:
            print("  ! CUDA is not available after install.")
            print("    If it worked before, restart the runtime (Runtime → Restart).")
    except Exception as exc:
        print(f"  ✗ torch import failed after install: {exc}")
        ok = False

    print()
    if failures or not ok:
        print("✗ Setup incomplete.", file=sys.stderr)
        if failures:
            print(f"  Failed: {', '.join(sorted(set(failures)))}", file=sys.stderr)
        print("\n  Try restarting the runtime and running this again.", file=sys.stderr)
        print("  See docs/colab.md and docs/troubleshooting.md.\n", file=sys.stderr)
        return 1

    print("✓ Setup complete.")
    print("\n  Next:")
    print("    !python scripts/smoke_test.py")
    print("    !python scripts/plan_run.py --all-models")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
