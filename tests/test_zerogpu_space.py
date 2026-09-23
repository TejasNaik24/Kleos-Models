"""What goes to Hugging Face, and what must never go there.

Two artifacts leave this repository for the Hub: the deployment package (to a
private model repository) and the ZeroGPU Space. These tests pin what each may
contain, and that the Space runs the exact runtime Hermes was measured under:

* the Space holds four small files — no weights, no datasets, no secrets;
* its base weights are the pinned revision's shards, preloaded, never
  ``consolidated.safetensors`` and never the Hub's tokenizer;
* its model runtime pins equal the Docker image's, which equal the research
  record's;
* the package upload refuses anything the manifest does not account for, and
  proves the remote bytes afterwards.
"""

from __future__ import annotations

import ast
import re
import shutil

import pytest
import yaml
from tests.conftest import CONFIGS_DIR, REPO_ROOT
from tests.test_deployment_manifest import build_package

from kleos_models.data.validation import scan_sensitive_content
from kleos_models.errors import ConfigError
from kleos_models.serving.space import (
    BASE_PRELOAD_FILES,
    COMMIT_PLACEHOLDER,
    STAGED_FILES,
    RemoteFile,
    check_space_readme,
    compare_remote_files,
    git_blob_sha1,
    package_upload_files,
    parse_front_matter,
    render_requirements,
    stage_space,
)

SPACE_DIR = REPO_ROOT / "deploy" / "zerogpu-space"
RECORD = CONFIGS_DIR / "deployment" / "kleos_hermes_v006.yaml"
DOCKER_LOCK = REPO_ROOT / "docker" / "requirements-hermes.txt"
DOCKERFILE = REPO_ROOT / "docker" / "hermes.Dockerfile"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
PINNED = "04d8a90549d23fc6bd7f642064003592df51e9b3"

#: torch versions the ZeroGPU documentation lists as supported (checked
#: 2026-09-23). A version outside this list may not be patched for emulation.
ZEROGPU_TORCH = {"2.8.0", "2.9.1", "2.10.0", "2.11.0", "2.12.1", "2.13.0"}

#: The only pins allowed to differ from the Docker image, each because the
#: Space platform forces it, and none of them part of the model runtime.
#: pydantic: the platform installs gradio[oauth,mcp]==6.28.0, whose `mcp` extra
#: requires pydantic<=2.12.5 (first Space build failed on 2.13.5, 2026-09-23).
PLATFORM_CONSTRAINED = {"pydantic": "2.12.5"}
MODEL_RUNTIME = {
    "transformers",
    "peft",
    "accelerate",
    "bitsandbytes",
    "tokenizers",
    "jinja2",
    "markupsafe",
    "safetensors",
}


def record() -> dict:
    return yaml.safe_load(RECORD.read_text(encoding="utf-8"))["deployment"]


def pins(text: str) -> dict[str, str]:
    """name -> version for every `name==version` line."""
    found = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", line.strip())
        if match:
            found[match.group(1).lower().replace("_", "-")] = match.group(2)
    return found


class TestSpaceRepository:
    def test_the_space_source_is_three_small_text_files(self):
        names = sorted(p.name for p in SPACE_DIR.iterdir())
        assert names == ["README.md", "app.py", "requirements.txt.template"]
        for path in SPACE_DIR.iterdir():
            assert path.stat().st_size < 20_000, f"{path.name} is not a small text file"

    def test_no_secret_or_personal_data_in_the_space(self):
        for path in SPACE_DIR.iterdir():
            assert scan_sensitive_content(path.read_text(encoding="utf-8")) == [], path.name


class TestSpaceConfiguration:
    def test_the_readme_agrees_with_the_serving_record(self):
        readme = (SPACE_DIR / "README.md").read_text(encoding="utf-8")
        assert check_space_readme(readme, record()) == []

    def test_only_the_pinned_shards_and_configs_are_preloaded(self):
        config = parse_front_matter((SPACE_DIR / "README.md").read_text(encoding="utf-8"))
        (entry,) = config["preload_from_hub"]
        repo, files, commit = entry.split()
        assert repo == "mistralai/Mistral-Nemo-Instruct-2407"
        assert commit == PINNED
        listed = set(files.split(","))
        assert listed == BASE_PRELOAD_FILES
        assert "consolidated.safetensors" not in listed  # a second 24.5 GB copy
        assert not any("tokenizer" in name for name in listed)  # Hermes brings its own

    def test_startup_has_room_to_load_and_quantize(self):
        config = parse_front_matter((SPACE_DIR / "README.md").read_text(encoding="utf-8"))
        assert config["startup_duration_timeout"] == "1h"

    def test_a_drifted_revision_is_caught(self):
        readme = (SPACE_DIR / "README.md").read_text(encoding="utf-8")
        drifted = readme.replace(PINNED, "a" * 40)
        assert any("preload commit" in p for p in check_space_readme(drifted, record()))

    def test_a_consolidated_checkpoint_in_the_preload_is_caught(self):
        readme = (SPACE_DIR / "README.md").read_text(encoding="utf-8")
        heavy = readme.replace("config.json,", "config.json,consolidated.safetensors,", 1)
        assert any("preload files" in p for p in check_space_readme(heavy, record()))


@pytest.fixture(scope="module")
def template() -> str:
    return (SPACE_DIR / "requirements.txt.template").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def source() -> str:
    return (SPACE_DIR / "app.py").read_text(encoding="utf-8")


class TestSpaceRuntimePins:
    def test_every_pin_matches_the_docker_image(self, template):
        docker = pins(DOCKER_LOCK.read_text(encoding="utf-8"))
        space = pins(template)
        assert space, "no pins found"
        for name, version in space.items():
            expected = PLATFORM_CONSTRAINED.get(name, docker.get(name))
            assert version == expected, f"{name}: Space {version}, expected {expected}"

    def test_platform_exceptions_never_touch_the_model_runtime(self):
        assert not set(PLATFORM_CONSTRAINED) & MODEL_RUNTIME

    def test_the_output_deciding_packages_are_all_pinned(self, template):
        space = pins(template)
        for name in (
            "transformers",
            "peft",
            "accelerate",
            "bitsandbytes",
            "tokenizers",
            "jinja2",
            "safetensors",
        ):
            assert name in space, f"{name} is not pinned"

    def test_torch_is_the_exact_cuda_12_8_wheel_docker_uses(self, template):
        (line,) = [ln for ln in template.splitlines() if ln.startswith("torch")]
        assert line.startswith("torch @ https://download.pytorch.org/whl/cu128/")
        assert "torch-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl" in line
        assert re.search(r"#sha256=[0-9a-f]{64}$", line)
        assert "torch==2.11.0" in DOCKERFILE.read_text(encoding="utf-8")
        assert "2.11.0" in ZEROGPU_TORCH

    def test_bitsandbytes_needs_no_zerogpu_patch(self, template):
        major, minor, *_ = (int(p) for p in pins(template)["bitsandbytes"].split("."))
        # spaces patches only bitsandbytes < 0.46; newer releases run as-is.
        assert (major, minor) >= (0, 46)

    def test_gradio_and_spaces_are_left_to_the_platform(self, template):
        space = pins(template)
        assert "gradio" not in space and "spaces" not in space
        assert not re.search(r"^(gradio|spaces)\b", template, re.MULTILINE)

    def test_kleos_models_is_pinned_by_commit(self, template):
        assert template.count(COMMIT_PLACEHOLDER) == 1
        assert f".git@{COMMIT_PLACEHOLDER}" in template


class TestSpaceApp:
    def test_spaces_is_imported_before_anything_else(self, source):
        imports = [
            node for node in ast.parse(source).body if isinstance(node, ast.Import | ast.ImportFrom)
        ]
        first = imports[0]
        assert isinstance(first, ast.Import) and first.names[0].name == "spaces"

    def test_type_hints_are_real_for_gr_api(self, source):
        futures = [
            alias.name
            for node in ast.parse(source).body
            if isinstance(node, ast.ImportFrom) and node.module == "__future__"
            for alias in node.names
        ]
        assert "annotations" not in futures

    def test_only_the_generate_step_is_on_the_gpu(self, source):
        assert source.count("@spaces.GPU") == 1
        assert "@spaces.GPU(duration=gpu_duration)" in source
        assert "return run_gpu_step(DEPLOYMENT, prepared, config)" in source

    def test_it_is_an_api_not_a_chat_demo(self, source):
        for widget in ("gr.Chatbot", "gr.ChatInterface", "gr.Interface(", "gr.Textbox"):
            assert widget not in source
        assert 'gr.api(generate, api_name="generate")' in source
        assert 'gr.api(status, api_name="status"' in source

    def test_errors_and_sharing_stay_closed(self, source):
        assert "show_error=False" in source
        assert "share=True" not in source
        assert "demo.queue(max_size=" in source

    def test_logging_is_on_before_the_model_loads(self, source):
        # Without it the startup report (load time, memory, versions) is
        # dropped: the first deployment's log had no `Hermes ready:` line.
        assert 0 < source.index("configure_logging()") < source.index("load_space_deployment(Path")

    def test_it_reads_no_secret_itself(self, source):
        # Secrets reach the shared, tested code through the environment.
        assert "os.environ" not in source and "getenv" not in source
        assert 'with_name("hermes_record.yaml")' in source


class TestStaging:
    def test_staging_renders_exactly_the_space_files(self, tmp_path):
        out = tmp_path / "space"
        staged = stage_space(SPACE_DIR, RECORD, out, commit=COMMIT)
        assert sorted(p.name for p in staged) == sorted(STAGED_FILES)
        requirements = (out / "requirements.txt").read_text(encoding="utf-8")
        assert f"Kleos-Models.git@{COMMIT}" in requirements
        assert "{{" not in requirements
        assert (out / "hermes_record.yaml").read_bytes() == RECORD.read_bytes()
        assert (out / "app.py").read_bytes() == (SPACE_DIR / "app.py").read_bytes()

    def test_a_non_empty_destination_is_refused(self, tmp_path):
        out = tmp_path / "space"
        out.mkdir()
        (out / "leftover.safetensors").write_bytes(b"x")
        with pytest.raises(ConfigError, match="not empty"):
            stage_space(SPACE_DIR, RECORD, out, commit=COMMIT)

    @pytest.mark.parametrize("commit", ["main", COMMIT[:12], COMMIT.upper(), ""])
    def test_only_a_full_commit_sha_is_installed(self, tmp_path, commit):
        with pytest.raises(ConfigError, match="40-character"):
            stage_space(SPACE_DIR, RECORD, tmp_path / "space", commit=commit)

    def test_a_readme_that_drifted_from_the_record_is_refused(self, tmp_path):
        source = tmp_path / "source"
        shutil.copytree(SPACE_DIR, source)
        readme = source / "README.md"
        readme.write_text(readme.read_text(encoding="utf-8").replace(PINNED, "b" * 40))
        with pytest.raises(ConfigError, match="disagrees"):
            stage_space(source, RECORD, tmp_path / "space", commit=COMMIT)

    def test_a_secret_in_a_staged_file_is_refused(self, tmp_path):
        source = tmp_path / "source"
        shutil.copytree(SPACE_DIR, source)
        planted = "hf_" + "A" * 34
        app = source / "app.py"
        app.write_text(app.read_text(encoding="utf-8") + f"\n# {planted}\n")
        with pytest.raises(ConfigError, match="hf_token"):
            stage_space(source, RECORD, tmp_path / "space", commit=COMMIT)

    def test_a_template_without_the_placeholder_is_refused(self):
        with pytest.raises(ConfigError, match="once"):
            render_requirements("torch==2.11.0\n", COMMIT)


class TestPackageUpload:
    def test_a_built_package_uploads_exactly_its_recorded_files(self, tmp_path):
        package = build_package(tmp_path)
        (package / "deployment" / "README.md").write_text("# Hermes package\n")
        _, files = package_upload_files(package)
        assert files == [
            "adapter/adapter_config.json",
            "adapter/adapter_model.safetensors",
            "deployment/README.md",
            "deployment/model_config.yaml",
            "manifest.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        ]

    @pytest.mark.parametrize(
        "stray",
        [
            "adapter/optimizer.pt",
            "train.jsonl",
            ".DS_Store",
            "checkpoint-200/adapter_model.safetensors",
            ".env",
        ],
    )
    def test_a_file_the_manifest_does_not_account_for_is_refused(self, tmp_path, stray):
        package = build_package(tmp_path)
        (package / stray).parent.mkdir(parents=True, exist_ok=True)
        (package / stray).write_bytes(b"not part of the package")
        with pytest.raises(ConfigError, match="Refusing to upload") as error:
            package_upload_files(package)
        assert any(stray in p for p in error.value.details["problems"])

    def test_a_forbidden_name_is_refused_even_when_recorded(self, tmp_path):
        package = build_package(tmp_path, extra_files={"adapter/events.jsonl": b"{}\n"})
        with pytest.raises(ConfigError) as error:
            package_upload_files(package)
        assert any("forbidden name" in p for p in error.value.details["problems"])

    def test_personal_data_in_a_text_file_is_refused(self, tmp_path):
        package = build_package(tmp_path)
        (package / "deployment" / "README.md").write_text("Contact: someone@example.org\n")
        with pytest.raises(ConfigError) as error:
            package_upload_files(package)
        assert any("email_address" in p for p in error.value.details["problems"])

    def test_a_corrupted_package_is_refused_before_anything_else(self, tmp_path):
        package = build_package(tmp_path)
        (package / "adapter" / "adapter_model.safetensors").write_bytes(b"tampered")
        with pytest.raises(ConfigError):
            package_upload_files(package)


class TestRemoteVerification:
    @pytest.fixture
    def uploaded(self, tmp_path):
        package = build_package(tmp_path)
        _, files = package_upload_files(package)
        remote = {}
        for relative in files:
            path = package / relative
            if relative.endswith(".safetensors"):
                from kleos_models.data.loaders import file_sha256

                remote[relative] = RemoteFile(
                    relative, path.stat().st_size, lfs_sha256=file_sha256(path)
                )
            else:
                remote[relative] = RemoteFile(
                    relative, path.stat().st_size, blob_id=git_blob_sha1(path)
                )
        remote[".gitattributes"] = RemoteFile(".gitattributes", 10, blob_id="0" * 40)
        return package, files, remote

    def test_identical_remote_bytes_pass(self, uploaded):
        package, files, remote = uploaded
        assert compare_remote_files(package, files, remote) == []

    def test_a_remote_weights_file_with_other_bytes_is_caught(self, uploaded):
        package, files, remote = uploaded
        name = "adapter/adapter_model.safetensors"
        entry = remote[name]
        remote[name] = RemoteFile(name, entry.size, lfs_sha256="f" * 64)
        assert compare_remote_files(package, files, remote) == [f"{name}: remote sha256 differs"]

    def test_missing_resized_and_unexpected_files_are_caught(self, uploaded):
        package, files, remote = uploaded
        del remote["manifest.json"]
        entry = remote["tokenizer/tokenizer.json"]
        remote["tokenizer/tokenizer.json"] = RemoteFile(entry.path, entry.size + 1, entry.blob_id)
        remote["notes.txt"] = RemoteFile("notes.txt", 3, blob_id="1" * 40)
        problems = compare_remote_files(package, files, remote)
        assert "manifest.json: missing from the repository" in problems
        assert any(p.startswith("tokenizer/tokenizer.json:") for p in problems)
        assert "notes.txt: in the repository but not in the verified package" in problems

    def test_git_blob_ids_match_git(self, tmp_path):
        path = tmp_path / "hello.txt"
        path.write_bytes(b"hello\n")
        # `printf 'hello\n' | git hash-object --stdin`
        assert git_blob_sha1(path) == "ce013625030ba8dba906f756967f9e9ca394464a"
