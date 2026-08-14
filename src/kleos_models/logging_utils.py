"""Structured logging (spec section 31).

Two sinks:

* a human-readable console/file stream, and
* a machine-readable JSONL event stream used by report generation.

**Privacy rule:** this module never logs raw example content. Helpers that
accept free text redact it to a length and a short digest. If you need to debug
actual example text, do it in a notebook against the local fixtures, not through
the logger.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Names of the loggers this package configures, so repeated setup is idempotent.
_ROOT_LOGGER_NAME = "kleos_models"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a package logger.

    Args:
        name: Dotted suffix, typically ``__name__``. ``None`` returns the root
            package logger.
    """
    if name is None or name == _ROOT_LOGGER_NAME:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    if name.startswith(_ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def configure_logging(
    *,
    level: int | str = logging.INFO,
    log_file: Path | str | None = None,
    quiet_libraries: bool = True,
) -> logging.Logger:
    """Configure the package logger. Safe to call more than once.

    Args:
        level: Threshold for the console handler.
        log_file: Optional path receiving the same records at DEBUG level.
        quiet_libraries: Suppress noisy third-party INFO logs.

    Returns:
        The configured root package logger.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Remove handlers we installed previously so re-configuration does not
    # duplicate every line.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(level if isinstance(level, int) else str(level).upper())
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if quiet_libraries:
        for noisy in ("transformers", "datasets", "accelerate", "peft", "urllib3", "filelock"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger


def resolve_level(verbose: bool = False, quiet: bool = False) -> int:
    """Map CLI verbosity flags (and ``KLEOS_LOG_LEVEL``) to a logging level."""
    env = os.environ.get("KLEOS_LOG_LEVEL")
    if env:
        resolved = logging.getLevelName(env.upper())
        if isinstance(resolved, int):
            return resolved
    if quiet:
        return logging.WARNING
    if verbose:
        return logging.DEBUG
    return logging.INFO


# ---------------------------------------------------------------------------
# Privacy-preserving helpers
# ---------------------------------------------------------------------------


def redact(text: str, *, keep: int = 0) -> str:
    """Render text as a length + digest instead of its content.

    Args:
        text: Content that may be private. Never logged verbatim.
        keep: Number of leading characters to retain. Keep this at 0 for
            anything sourced from real user data.

    Returns:
        A string such as ``<redacted len=412 sha256=1a2b3c4d>``.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    prefix = ""
    if keep > 0:
        head = text[:keep].replace("\n", " ")
        prefix = f"{head!r}… "
    return f"{prefix}<redacted len={len(text)} sha256={digest}>"


# ---------------------------------------------------------------------------
# Structured JSONL event stream
# ---------------------------------------------------------------------------


class EventLogger:
    """Append-only JSONL event writer for machine-readable run history.

    Each record carries ``ts``, ``experiment_id``, ``stage`` and ``event`` plus
    whatever structured fields the caller supplies. Values are passed through
    ``json.dumps`` with ``default=str`` so unusual types degrade to their repr
    rather than raising mid-training.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        experiment_id: str,
        stage: str = "init",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.stage = stage
        self._logger = get_logger("events")

    def set_stage(self, stage: str) -> None:
        """Set the stage label attached to subsequent events."""
        self.stage = stage

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        """Write one event record.

        Never raises: logging failures must not abort a training run.
        """
        record: dict[str, Any] = {
            "ts": time.time(),
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "experiment_id": self.experiment_id,
            "stage": self.stage,
            "event": event,
        }
        record.update(fields)
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:  # pragma: no cover - filesystem dependent
            self._logger.warning("Could not write event log to %s: %s", self.path, exc)
        return record

    def read_all(self) -> list[dict[str, Any]]:
        """Read every event back. Malformed lines are skipped, not fatal."""
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events


@contextmanager
def log_stage(logger: logging.Logger, stage: str) -> Iterator[None]:
    """Log entry/exit of a pipeline stage with elapsed wall time."""
    logger.info("▶ %s", stage)
    started = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - started
        logger.error("✗ %s failed after %.1fs", stage, elapsed)
        raise
    else:
        elapsed = time.perf_counter() - started
        logger.info("✓ %s (%.1fs)", stage, elapsed)


def format_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    """Render rows as a fixed-width text table for console reports.

    Args:
        rows: Records to display.
        columns: Column order. Defaults to the keys of the first row.
    """
    if not rows:
        return "(no rows)"
    cols = columns or list(rows[0].keys())
    widths = {c: len(c) for c in cols}
    rendered: list[dict[str, str]] = []
    for row in rows:
        cells = {}
        for col in cols:
            value = row.get(col, "")
            text = f"{value:.4f}" if isinstance(value, float) else str(value)
            cells[col] = text
            widths[col] = max(widths[col], len(text))
        rendered.append(cells)

    header = "  ".join(c.ljust(widths[c]) for c in cols)
    divider = "  ".join("-" * widths[c] for c in cols)
    body = "\n".join("  ".join(cells[c].ljust(widths[c]) for c in cols) for cells in rendered)
    return f"{header}\n{divider}\n{body}"
