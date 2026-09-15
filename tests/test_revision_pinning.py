"""Base-model revision pinning (audit finding F3).

A LoRA adapter is deltas against *specific* base weights. If the base revision
is a moving pointer, an upstream re-upload silently pairs the adapter with
weights it never saw — no error is raised, the judgment just degrades. These
tests exist so that cannot happen by accident, and so the training and serving
pins cannot drift apart.

They also pin the honesty requirement: a model card generated from an unpinned
run must say so rather than imply reproducibility it cannot offer.
"""

from __future__ import annotations

import yaml
from tests.conftest import CONFIGS_DIR

from kleos_models.config import load_model_config
from kleos_models.experiments.manifest import ExperimentManifest
from kleos_models.publishing import build_model_card, is_pinned_revision

#: The base revision KLEOS trains and serves against, verified 2026-09-15.
PINNED_SHA = "2f494a194c5b980dfb9772cb92d26cbb671fce5a"

MINISTRAL_CONFIG = CONFIGS_DIR / "models" / "ministral_8b.yaml"
DEPLOYMENT_CONFIG = CONFIGS_DIR / "deployment" / "kleos_v006_ministral8b.yaml"


def _deployment() -> dict:
    return yaml.safe_load(DEPLOYMENT_CONFIG.read_text(encoding="utf-8"))["deployment"]


class TestIsPinnedRevision:
    def test_a_full_sha_is_pinned(self):
        assert is_pinned_revision(PINNED_SHA)

    def test_moving_pointers_are_not_pinned(self):
        for revision in ("main", "master", "v1.0", "refs/heads/main", "", None):
            assert not is_pinned_revision(revision), revision

    def test_a_short_sha_is_not_accepted(self):
        # Short hashes are ambiguous and can collide as a repository grows.
        assert not is_pinned_revision(PINNED_SHA[:12])

    def test_uppercase_is_not_accepted(self):
        # The Hub emits lowercase; accepting both would let two spellings of the
        # same commit compare unequal in the drift check below.
        assert not is_pinned_revision(PINNED_SHA.upper())


class TestMinistralIsPinned:
    """The config KLEOS actually trains with must never revert to a pointer."""

    def test_ministral_revision_is_a_pinned_sha(self):
        config = load_model_config(MINISTRAL_CONFIG)
        assert is_pinned_revision(config.revision), (
            f"ministral_8b.yaml has revision={config.revision!r}. It must be a "
            "40-character commit sha; 'main' lets upstream move under a trained "
            "adapter without any error being raised."
        )

    def test_ministral_revision_is_the_verified_sha(self):
        assert load_model_config(MINISTRAL_CONFIG).revision == PINNED_SHA

    def test_ministral_base_model_is_unchanged(self):
        assert load_model_config(MINISTRAL_CONFIG).base_model == (
            "mistralai/Ministral-8B-Instruct-2410"
        )


class TestDeploymentMatchesTraining:
    """Serving a different revision than was trained is a silent failure."""

    def test_deployment_config_exists(self):
        assert DEPLOYMENT_CONFIG.exists()

    def test_deployment_revision_matches_the_training_config(self):
        training = load_model_config(MINISTRAL_CONFIG)
        assert _deployment()["revision"] == training.revision, (
            "The serving pin and the training pin have drifted apart. An adapter "
            "served against different base weights than it was trained on "
            "degrades silently."
        )

    def test_deployment_base_model_matches_the_training_config(self):
        training = load_model_config(MINISTRAL_CONFIG)
        assert _deployment()["base_model"] == training.base_model

    def test_deployment_revision_is_pinned(self):
        assert is_pinned_revision(_deployment()["revision"])

    def test_deployment_ships_no_adapter_weights_path_by_default(self):
        # The adapter lives on private storage; a committed path would leak where.
        assert _deployment()["adapter_path"].startswith("${env:")


class TestOtherConfigsDidNotInheritTheSha:
    """The verified sha belongs to ONE checkpoint and must not be copy-pasted."""

    def test_no_other_model_config_uses_the_ministral_sha(self):
        for path in sorted((CONFIGS_DIR / "models").glob("*.yaml")):
            if path.name == "ministral_8b.yaml":
                continue
            revision = load_model_config(path).revision
            assert revision != PINNED_SHA, (
                f"{path.name} carries Ministral-8B's commit sha. That sha "
                "identifies a different checkpoint and would load the wrong "
                "weights or fail outright."
            )


class TestModelCardStatesRevisionHonestly:
    """A card must not imply reproducibility the release cannot deliver."""

    @staticmethod
    def _card(revision: str) -> str:
        manifest = ExperimentManifest(experiment_id="test-run")
        manifest.model = {
            "base_model": "mistralai/Ministral-8B-Instruct-2410",
            "revision": revision,
        }
        return build_model_card(repo_id="user/kleos-test", manifest=manifest)

    def test_a_pinned_card_records_the_sha_in_the_load_snippet(self):
        card = self._card(PINNED_SHA)
        assert "revision=REVISION" in card
        assert PINNED_SHA in card
        assert "moving pointer" not in card

    def test_an_unpinned_card_warns_that_weights_are_unidentifiable(self):
        # This is the case the v0.0.6 run falls into: it was trained with 'main'
        # and the resolved commit is not recoverable from the artifacts.
        card = self._card("main")
        assert "moving pointer" in card
        assert "not recoverable" in card

    def test_the_tokenizer_is_loaded_from_the_base_at_the_same_revision(self):
        card = self._card(PINNED_SHA)
        assert "AutoTokenizer.from_pretrained(BASE, revision=REVISION)" in card
