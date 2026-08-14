"""Shared plumbing for the CLI scripts.

Keeps every script thin: argument parsing plus a call into the library. No
pipeline logic lives in ``scripts/`` — that would make it untestable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parent.parent

# Allow running the scripts directly from a checkout without installing.
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from kleos_models.errors import KleosError
from kleos_models.logging_utils import configure_logging, get_logger, resolve_level


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add flags every script accepts."""
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Warnings and errors only.")
    return parser


def add_config_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the config-loading flags shared by train/evaluate/plan."""
    parser.add_argument("--config", type=Path, required=True, help="Experiment config YAML.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set training.learning_rate=1e-4",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help=(
            "Dataset directory, overriding the config. This is how a run consumes "
            "the externally produced private dataset artifact."
        ),
    )
    parser.add_argument("--output-dir", type=Path, help="Output root, overriding the config.")
    return parser


def setup_logging(args: argparse.Namespace, *, log_file: Path | None = None) -> None:
    """Configure logging from parsed arguments."""
    configure_logging(
        level=resolve_level(
            verbose=getattr(args, "verbose", False), quiet=getattr(args, "quiet", False)
        ),
        log_file=log_file,
    )


def fail(error: BaseException, *, exit_code: int = 1) -> NoReturn:
    """Print an actionable error and exit.

    KleosError already renders its own diagnostics and suggestions, so it is shown
    verbatim rather than wrapped in a traceback the user cannot act on.
    """
    logger = get_logger("cli")
    if isinstance(error, KleosError):
        print(f"\n✗ {error.render()}\n", file=sys.stderr)
    else:
        logger.exception("Unexpected error")
        print(f"\n✗ {type(error).__name__}: {error}\n", file=sys.stderr)
        print("This looks like a bug. See docs/troubleshooting.md.\n", file=sys.stderr)
    raise SystemExit(exit_code)


def run(main_fn, argv: list[str] | None = None) -> int:
    """Invoke a script entry point with uniform error handling."""
    try:
        return main_fn(argv)
    except KeyboardInterrupt:
        print("\n✗ Interrupted.\n", file=sys.stderr)
        return 130
    except KleosError as exc:
        fail(exc)
    except Exception as exc:
        fail(exc)


def print_header(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
