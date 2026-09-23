#!/usr/bin/env python3
"""Run the Hermes inference service.

Configuration comes from the environment, never from this repository::

    HERMES_PACKAGE_DIR              deployment package built by build_deployment_package.py
    HERMES_API_KEY                  shared secret(s), comma-separated; required
    HERMES_DEVICE_MAP               optional, e.g. "cuda:0"
    HERMES_REQUIRE_REMOTE_REVISION  1 to confirm the base commit with the Hub at startup
    HERMES_REQUEST_TIMEOUT_SECONDS  optional, may only tighten the manifest limit
    HERMES_MAX_INPUT_CHARS          optional, may only tighten the manifest limit

Usage::

    export HERMES_PACKAGE_DIR=/srv/hermes-v0.0.6
    export HERMES_API_KEY=...
    python scripts/serve_hermes.py --host 127.0.0.1 --port 8000

The default bind address is loopback on purpose. Exposing this process directly
to the internet is not the intended deployment: put it behind a TLS terminator
that the KLEOS backend talks to, and keep the shared secret out of logs and out
of version control.

The model is loaded and verified at startup. If verification fails the process
stays up and reports the reason on /health, but /v1/generate returns 503 — an
artifact that failed verification is never served.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import add_common_arguments, print_header, run, setup_logging
from kleos_models.serving.app import ServingSettings, create_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: loopback).")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--package",
        type=Path,
        help="Deployment package, overriding HERMES_PACKAGE_DIR.",
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("KLEOS Hermes inference service")

    settings = ServingSettings.from_env()
    if args.package:
        settings.package_dir = args.package

    print(f"  package : {settings.package_dir}")
    print(f"  bind    : {args.host}:{args.port}")
    print(f"  auth    : {'enabled' if settings.api_keys else 'DISABLED (local only)'}")
    print("\n  Loading and verifying the model at startup…\n")

    app = create_app(settings)

    try:
        import uvicorn
    except ImportError as exc:
        from kleos_models.errors import MissingDependencyError

        raise MissingDependencyError(
            "uvicorn", extra="serve", purpose="run the Hermes inference service"
        ) from exc

    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
