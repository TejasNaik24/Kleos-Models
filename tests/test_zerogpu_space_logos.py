"""The Logos v0.0.2 Space: its own folder, staged from its own record.

The Hermes Space folder and its tests (tests/test_zerogpu_space.py) are not
touched. What is pinned here: the Logos Space stages exactly its four files,
bakes in the Ministral 3 Reasoning shards at the pinned revision (and refuses
Hermes' Nemo shards), reads `logos_record.yaml`, and wires the profile into the
service, the settings and the GPU duration.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from kleos_models.errors import ConfigError
from kleos_models.serving.profile import load_profile
from kleos_models.serving.space import check_space_readme, parse_front_matter, stage_space

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "configs" / "deployment" / "kleos_logos_v002.yaml"
LOGOS_DIR = ROOT / "deploy" / "zerogpu-space-logos"
HERMES_DIR = ROOT / "deploy" / "zerogpu-space"
COMMIT = "a" * 40


def spec() -> dict:
    return yaml.safe_load(RECORD.read_text(encoding="utf-8"))["deployment"]


class TestTheLogosSpaceFolder:
    def test_the_profile_points_at_this_folder(self):
        assert ROOT / load_profile(RECORD).space_dir == LOGOS_DIR
        assert sorted(p.name for p in LOGOS_DIR.iterdir()) == [
            "README.md",
            "app.py",
            "requirements.txt.template",
        ]

    def test_it_preloads_the_reasoning_shards_at_the_pinned_revision(self):
        record = spec()
        (entry,) = parse_front_matter((LOGOS_DIR / "README.md").read_text())["preload_from_hub"]
        repo, files, commit = entry.split()
        assert repo == record["base_model"] and commit == record["revision"]
        assert files.split(",") == record["serving"]["base_files"]
        assert check_space_readme((LOGOS_DIR / "README.md").read_text(), record) == []

    def test_hermes_shards_are_refused_for_logos(self):
        hermes_readme = (HERMES_DIR / "README.md").read_text()
        problems = check_space_readme(hermes_readme, spec())
        assert any("preload repo" in p for p in problems)
        assert any("preload files" in p for p in problems)

    def test_the_app_reads_its_record_and_wires_the_profile(self):
        source = (LOGOS_DIR / "app.py").read_text()
        tree = ast.parse(source)
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
        assert all(node.module != "__future__" for node in imports)
        assert source.lstrip().startswith("#") and "import spaces" in source
        assert source.index("import spaces") < source.index("import gradio")
        assert 'with_name("logos_record.yaml")' in source
        assert "@spaces.GPU(duration=gpu_duration_for(PROFILE))" in source
        assert "ZeroGPUSettings.from_env(profile=PROFILE)" in source
        assert "profile=PROFILE)" in source

    def test_the_requirements_template_is_hermes_byte_for_byte(self):
        assert (LOGOS_DIR / "requirements.txt.template").read_bytes() == (
            HERMES_DIR / "requirements.txt.template"
        ).read_bytes()


class TestStagingLogos:
    def test_exactly_four_files_are_staged_with_the_record_under_its_name(self, tmp_path):
        staged = stage_space(LOGOS_DIR, RECORD, tmp_path / "space", commit=COMMIT)
        assert sorted(p.name for p in staged) == [
            "README.md",
            "app.py",
            "logos_record.yaml",
            "requirements.txt",
        ]
        assert (tmp_path / "space" / "logos_record.yaml").read_text() == RECORD.read_text()
        assert COMMIT in (tmp_path / "space" / "requirements.txt").read_text()

    def test_staging_logos_from_the_hermes_folder_is_refused(self, tmp_path):
        with pytest.raises(ConfigError, match="disagrees"):
            stage_space(HERMES_DIR, RECORD, tmp_path / "space", commit=COMMIT)
