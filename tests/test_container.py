"""The Hermes serving container, checked without Docker or a GPU.

These are static checks on docker/hermes.Dockerfile, .dockerignore,
docker/compose.yaml, docker/hermes.env.example and the pinned requirements.
They exist because a container definition fails in the least convenient place —
on a rented GPU host — and because two of its failure modes are silent:

* a build context that sweeps in `.env` or the private dataset, and
* an image whose pins drift from the versions the frozen model was measured
  under.

The startup sequence itself is tested in tests/test_startup.py.
"""

from __future__ import annotations

import json
import re
import shlex

import pytest
import yaml
from tests.conftest import REPO_ROOT

from kleos_models.data.validation import scan_sensitive_content

DOCKERFILE = REPO_ROOT / "docker" / "hermes.Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
COMPOSE = REPO_ROOT / "docker" / "compose.yaml"
ENV_EXAMPLE = REPO_ROOT / "docker" / "hermes.env.example"
REQUIREMENTS = REPO_ROOT / "docker" / "requirements-hermes.txt"
SERVING_RECORD = REPO_ROOT / "configs" / "deployment" / "kleos_hermes_v006.yaml"

#: Versions recorded in the kleos-v006-mistralnemo12b-run1 manifest. The served
#: model is only the measured model if these match.
RESEARCH_VERSIONS = {
    "transformers": "5.16.1",
    "peft": "0.20.0",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.50.2",
}
RESEARCH_TORCH = "2.11.0+cu128"

SECRET_NAMES = ("HERMES_API_KEY", "HF_TOKEN", "HERMES_API_KEY_FILE", "HF_TOKEN_FILE")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def instructions() -> list[tuple[str, str]]:
    """(KEYWORD, arguments) for every Dockerfile instruction, continuations joined."""
    joined: list[str] = []
    buffer = ""
    for raw in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not buffer and (not line.strip() or line.lstrip().startswith("#")):
            continue
        if line.lstrip().startswith("#"):
            continue  # a comment inside a continued instruction
        if line.endswith("\\"):
            buffer += line[:-1] + " "
            continue
        joined.append(buffer + line)
        buffer = ""
    result = []
    for line in joined:
        keyword, _, rest = line.strip().partition(" ")
        result.append((keyword.upper(), rest.strip()))
    return result


def of(keyword: str) -> list[str]:
    return [args for kw, args in instructions() if kw == keyword]


def env_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for args in of("ENV"):
        for token in shlex.split(args):
            key, _, value = token.partition("=")
            values[key] = value
    return values


def copy_instructions() -> list[tuple[list[str], str]]:
    """(sources, destination) for every COPY."""
    result = []
    for args in of("COPY"):
        parts = [p for p in shlex.split(args) if not p.startswith("--")]
        result.append((parts[:-1], parts[-1]))
    return result


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Docker's .dockerignore glob, as moby/patternmatcher implements it."""
    out = ""
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out += "(.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile(f"^{out}$")


def dockerignore_patterns() -> list[tuple[bool, re.Pattern[str]]]:
    patterns = []
    for raw in DOCKERIGNORE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        exclusion = line.startswith("!")
        body = line[1:] if exclusion else line
        patterns.append((exclusion, _glob_to_regex(body.strip().lstrip("/"))))
    return patterns


def excluded_from_context(relative: str) -> bool:
    """Whether Docker would leave this path out of the build context.

    Last matching pattern wins, and a pattern matching any parent directory
    matches the file — the MatchesOrParentMatches rule Docker applies.
    """
    parts = relative.split("/")
    parents = ["/".join(parts[: i + 1]) for i in range(len(parts) - 1)]
    excluded = False
    for exception, regex in dockerignore_patterns():
        if regex.match(relative) or any(regex.match(parent) for parent in parents):
            excluded = not exception
    return excluded


# ---------------------------------------------------------------------------
# Build context
# ---------------------------------------------------------------------------


class TestBuildContextIsAnAllowlist:
    @pytest.mark.parametrize(
        "relative",
        [
            ".env",  # holds a real HF token on development machines
            ".env.local",
            "data/processed/kleos-policy-v0.0.6/train.jsonl",
            "data/processed/kleos-policy-v0.0.6-benchmark/benchmark.jsonl",
            "outputs/kleos-v006-mistralnemo12b-run1/adapter/adapter_model.safetensors",
            "packages/hermes-v0.0.6/adapter/adapter_model.safetensors",
            ".git/config",
            ".venv/bin/python",
            ".claude/settings.local.json",
            "notebooks/02_train_qlora.ipynb",
            "tests/test_container.py",
            "docs/deployment.md",
            "docker/hermes.env",
            "docker/secrets/hermes_api_key",
            "scripts/train.py",
            "configs/training/kleos_hermes_v006.yaml",
            "src/kleos_models/__pycache__/config.cpython-312.pyc",
            "src/kleos_models/stray_weights.safetensors",
        ],
    )
    def test_private_and_unneeded_paths_stay_out(self, relative):
        assert excluded_from_context(relative), f"{relative} would be sent to the Docker daemon"

    @pytest.mark.parametrize(
        "relative",
        [
            "pyproject.toml",
            "README.md",  # hatchling reads it when the package is built
            "LICENSE",  # likewise
            "src/kleos_models/serving/startup.py",
            "src/kleos_models/serving/manifest.py",
            "scripts/serve_hermes.py",
            "scripts/hermes_smoke.py",
            "configs/deployment/kleos_hermes_v006.yaml",
            "docker/requirements-hermes.txt",
        ],
    )
    def test_what_the_image_needs_is_included(self, relative):
        assert not excluded_from_context(relative), f"{relative} is needed but excluded"

    def test_every_copy_source_exists_and_reaches_the_build(self):
        # A COPY whose source is excluded fails the build — on the GPU host.
        for sources, _ in copy_instructions():
            for source in sources:
                path = REPO_ROOT / source
                assert path.exists(), f"COPY source {source} does not exist"
                files = (
                    [p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
                    if path.is_dir()
                    else [path]
                )
                assert files, f"COPY source {source} is empty"
                for file in files:
                    relative = file.relative_to(REPO_ROOT).as_posix()
                    assert not excluded_from_context(relative), (
                        f"COPY {source} needs {relative}, but .dockerignore excludes it"
                    )

    def test_the_dockerignore_starts_by_excluding_everything(self):
        first = next(p for p in dockerignore_patterns())
        assert first[0] is False and first[1].pattern == "^[^/]*$", (
            "Keep the allowlist shape: '*' first, then explicit '!' re-includes."
        )


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------


class TestDockerfile:
    def test_base_image_is_pinned_by_digest(self):
        base = next(a for a in of("ARG") if a.startswith("BASE_IMAGE="))
        image = base.split("=", 1)[1]
        assert re.search(r"@sha256:[0-9a-f]{64}$", image), image
        assert ":latest" not in image
        assert of("FROM") == ["${BASE_IMAGE}"]

    def test_base_image_is_cuda_12_8_to_match_the_torch_build(self):
        base = next(a for a in of("ARG") if a.startswith("BASE_IMAGE="))
        assert "nvidia/cuda:12.8" in base

    def test_torch_is_the_research_build_from_the_cuda_index(self):
        text = DOCKERFILE.read_text(encoding="utf-8")
        assert f'"torch=={RESEARCH_TORCH}"' in text
        assert "download.pytorch.org/whl/cu128" in text

    def test_runs_as_a_non_root_user(self):
        users = of("USER")
        assert users and users[-1] not in {"root", "0"}
        assert "--uid 10001" in DOCKERFILE.read_text(encoding="utf-8")

    def test_the_entrypoint_is_the_verifying_startup_module(self):
        assert [json.loads(a) for a in of("ENTRYPOINT")] == [
            ["python", "-m", "kleos_models.serving.startup"]
        ]

    def test_exposes_the_service_port(self):
        assert of("EXPOSE") == ["8000"]
        assert env_values()["HERMES_PORT"] == "8000"
        assert env_values()["HERMES_HOST"] == "0.0.0.0"

    def test_the_healthcheck_waits_for_the_first_download(self):
        (check,) = of("HEALTHCHECK")
        assert "--healthcheck" in check
        start = re.search(r"--start-period=(\d+)m", check)
        assert start and int(start.group(1)) >= 30, (
            "The first start downloads ~24.5 GB; a short start period marks a "
            "healthy container unhealthy before the model can load."
        )

    def test_the_model_cache_is_a_dedicated_directory(self):
        assert env_values()["HF_HOME"] == "/cache/huggingface"

    def test_load_failures_exit_rather_than_idle(self):
        assert env_values()["HERMES_EXIT_ON_LOAD_FAILURE"] == "1"

    def test_the_expected_identity_record_is_copied_into_the_image(self):
        expected = env_values()["HERMES_EXPECTED_DEPLOYMENT_CONFIG"]
        destinations = {dest for _, dest in copy_instructions()}
        assert expected in destinations, (
            "The identity the image serves must be baked in; otherwise any intact "
            "package mounted at run time would be accepted as Hermes."
        )

    def test_the_package_is_mounted_not_baked_in(self):
        assert env_values()["HERMES_PACKAGE_DIR"] == "/models/hermes-v0.0.6"
        for sources, _ in copy_instructions():
            for source in sources:
                assert not source.startswith(("outputs", "packages", "data")), source

    def test_no_secret_is_set_in_the_image(self):
        defined = set(env_values()) | {a.split("=", 1)[0] for a in of("ARG")}
        leaked = defined & set(SECRET_NAMES)
        assert not leaked, f"{leaked} must arrive at run time, never be baked in"
        assert scan_sensitive_content(DOCKERFILE.read_text(encoding="utf-8")) == []

    def test_labels_agree_with_the_serving_record(self):
        record = yaml.safe_load(SERVING_RECORD.read_text(encoding="utf-8"))["deployment"]
        labels: dict[str, str] = {}
        for args in of("LABEL"):
            for token in shlex.split(args):
                key, _, value = token.partition("=")
                labels[key] = value
        assert labels["ai.kleos.adapter.sha256"] == record["adapter_sha256"]
        assert labels["ai.kleos.base.revision"] == record["revision"]
        assert labels["ai.kleos.base.model"] == record["base_model"]
        assert labels["org.opencontainers.image.version"] == record["version"]


# ---------------------------------------------------------------------------
# Pinned requirements
# ---------------------------------------------------------------------------


def requirement_lines() -> list[str]:
    return [
        line.strip()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


class TestPinnedRequirements:
    def test_every_requirement_is_an_exact_pin(self):
        for line in requirement_lines():
            assert re.fullmatch(r"[A-Za-z0-9_.\-]+==[A-Za-z0-9_.+\-]+", line), (
                f"{line!r} is not an exact pin; a rebuild could install something else"
            )

    def test_the_model_runtime_matches_the_research_record(self):
        pins = {normalize(n): v for n, v in (line.split("==") for line in requirement_lines())}
        for name, version in RESEARCH_VERSIONS.items():
            assert pins.get(name) == version, (
                f"{name} is pinned to {pins.get(name)}, but the frozen model was "
                f"measured under {version}"
            )

    def test_torch_is_left_to_its_own_layer(self):
        names = {normalize(line.split("==")[0]) for line in requirement_lines()}
        assert "torch" not in names, (
            "torch comes from the CUDA 12.8 index in the Dockerfile; listing it here "
            "would let PyPI's default CUDA build replace it"
        )

    def test_no_package_is_pinned_twice(self):
        names = [normalize(line.split("==")[0]) for line in requirement_lines()]
        assert len(names) == len(set(names))

    def test_the_serving_layer_is_present(self):
        names = {normalize(line.split("==")[0]) for line in requirement_lines()}
        assert {"fastapi", "uvicorn", "anyio", "tokenizers", "jinja2"} <= names


# ---------------------------------------------------------------------------
# Compose example
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def service() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]["hermes"]


class TestComposeExample:
    def test_reserves_an_nvidia_gpu(self, service):
        (device,) = service["deploy"]["resources"]["reservations"]["devices"]
        assert device["driver"] == "nvidia"
        assert "gpu" in device["capabilities"]

    def test_publishes_on_loopback_only(self, service):
        for port in service["ports"]:
            assert str(port).startswith("127.0.0.1:"), port

    def test_the_model_cache_is_a_named_volume(self, service):
        document = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
        cache = next(v for v in service["volumes"] if v.endswith(":/cache/huggingface"))
        assert cache.split(":")[0] in document["volumes"]

    def test_the_package_is_mounted_read_only(self, service):
        assert any(v.endswith(":/models/hermes-v0.0.6:ro") for v in service["volumes"])

    def test_secrets_arrive_as_files_not_values(self, service):
        environment = service["environment"]
        assert environment["HERMES_API_KEY_FILE"].startswith("/run/secrets/")
        assert environment["HF_TOKEN_FILE"].startswith("/run/secrets/")
        assert "HERMES_API_KEY" not in environment and "HF_TOKEN" not in environment

    def test_builds_from_the_repository_dockerfile(self, service):
        context = (COMPOSE.parent / service["build"]["context"]).resolve()
        assert (context / service["build"]["dockerfile"]).exists()

    def test_contains_no_secret_values(self):
        assert scan_sensitive_content(COMPOSE.read_text(encoding="utf-8")) == []


# ---------------------------------------------------------------------------
# Environment template
# ---------------------------------------------------------------------------


def template_variables() -> dict[str, str]:
    """Every NAME=value in the template, including commented-out ones."""
    variables: dict[str, str] = {}
    for raw in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw.strip().lstrip("#").strip()
        match = re.fullmatch(r"([A-Z][A-Z0-9_]*)=(.*)", line)
        if match:
            variables[match.group(1)] = match.group(2)
    return variables


class TestEnvironmentTemplate:
    def test_secret_values_are_empty(self):
        variables = template_variables()
        for name in ("HERMES_API_KEY", "HF_TOKEN"):
            assert variables[name] == "", (
                f"{name} must be empty in the template: an unreplaced placeholder "
                "would become a real, guessable credential"
            )

    def test_contains_nothing_that_looks_like_a_credential(self):
        assert scan_sensitive_content(ENV_EXAMPLE.read_text(encoding="utf-8")) == []

    def test_every_documented_variable_is_one_the_code_reads(self):
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (REPO_ROOT / "src" / "kleos_models" / "serving").glob("*.py")
        )
        known_elsewhere = {"HF_TOKEN", "HF_HUB_OFFLINE", "HF_HOME"}  # read by huggingface_hub
        for name in template_variables():
            if name in known_elsewhere:
                continue
            base = name.removesuffix("_FILE")
            assert name in source or f'"{base}"' in source or base in source, (
                f"{name} is documented in the template but nothing reads it"
            )
