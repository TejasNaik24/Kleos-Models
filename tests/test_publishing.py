"""The publishing boundary: what may leave the machine, and what the card claims.

This is the last gate before an artifact becomes public, and it had no test
coverage at all until audit finding F2. These tests pin two separate things:

* **Security** — the forbidden patterns, the allowlist, and the content scanner.
  Those controls are deliberately unchanged by F2 and are pinned here so a later
  change cannot quietly widen them.
* **Packaging** — tokenizer artifacts are excluded on purpose (F2), and the model
  card must tell a reader where the tokenizer actually comes from.

The scanner tests assemble their fake credentials at runtime rather than
embedding literals, so this module does not need a ``SELF_EXEMPT`` entry in
``scripts/check_no_private_data.py``. Exempting a file turns the scanner off for
it entirely, which is a poor trade for test-fixture convenience.
"""

from __future__ import annotations

import json

import pytest

from kleos_models.experiments.manifest import ExperimentManifest
from kleos_models.publishing import (
    ALLOWED_UPLOAD_NAMES,
    TOKENIZER_ARTIFACTS,
    build_model_card,
    collect_upload_files,
)

PINNED_SHA = "2f494a194c5b980dfb9772cb92d26cbb671fce5a"
BASE_MODEL = "mistralai/Ministral-8B-Instruct-2410"


@pytest.fixture
def run_dir(tmp_path):
    """A realistic finished run: adapter/, tokenizer/, and the run metadata."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00" * 64)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    (adapter / "README.md").write_text("# model card\n")

    (tmp_path / "config.yaml").write_text("model:\n  name: ministral_8b\n")
    (tmp_path / "manifest.json").write_text(json.dumps({"experiment_id": "run-1"}))
    (tmp_path / "metrics.json").write_text(json.dumps({"eval_loss": 0.04}))

    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {}}}))
    (tokenizer / "tokenizer_config.json").write_text(json.dumps({"pad_token": "</s>"}))
    (tokenizer / "chat_template.jinja").write_text("{% for m in messages %}{% endfor %}")
    (tokenizer / "special_tokens_map.json").write_text(json.dumps({"eos_token": "</s>"}))
    return tmp_path


def _names(run_dir):
    allowed, _ = collect_upload_files(run_dir / "adapter", run_dir)
    return {p.name for p in allowed}


class TestTokenizerIsNotPublished:
    """F2: the tokenizer comes from the pinned base model, not the adapter repo."""

    def test_tokenizer_json_is_not_uploaded(self, run_dir):
        assert "tokenizer.json" not in _names(run_dir)

    def test_no_tokenizer_artifact_is_uploaded(self, run_dir):
        assert not (_names(run_dir) & TOKENIZER_ARTIFACTS)

    def test_no_partial_tokenizer_bundle_is_emitted(self, run_dir):
        # The bug this replaces: tokenizer_config.json shipped while the ~17MB
        # vocabulary was refused by the scan cap, producing a repository that
        # looks like it has a tokenizer but cannot build one.
        uploaded = _names(run_dir)
        assert not (uploaded & TOKENIZER_ARTIFACTS), (
            "A partial tokenizer bundle is worse than none: without a vocabulary "
            "source, AutoTokenizer.from_pretrained on this repo fails."
        )

    def test_the_allowlist_itself_names_no_tokenizer_file(self):
        # Defence in depth: even if one appeared in the adapter directory.
        assert not (ALLOWED_UPLOAD_NAMES & TOKENIZER_ARTIFACTS)

    def test_a_tokenizer_file_inside_the_adapter_dir_is_refused_by_name(self, run_dir):
        (run_dir / "adapter" / "tokenizer.json").write_text("{}")
        _, rejected = collect_upload_files(run_dir / "adapter", run_dir)
        reasons = {p.name: r for p, r in rejected}
        assert "tokenizer.json" in reasons
        assert "tokenizer artifact" in reasons["tokenizer.json"]

    def test_the_adapter_itself_is_still_uploaded(self, run_dir):
        uploaded = _names(run_dir)
        assert "adapter_model.safetensors" in uploaded
        assert "adapter_config.json" in uploaded

    def test_run_metadata_is_still_uploaded(self, run_dir):
        assert {"config.yaml", "manifest.json", "metrics.json"} <= _names(run_dir)


class TestForbiddenFilesAreRefused:
    """Security controls, unchanged by F2 and pinned so they stay that way."""

    @pytest.mark.parametrize(
        "filename",
        [
            ".env",
            ".env.local",
            "train.jsonl",
            "benchmark.jsonl",
            "events.jsonl",
            "training.log",
            "private.key",
            "cert.pem",
        ],
    )
    def test_forbidden_names_never_upload(self, run_dir, filename):
        (run_dir / "adapter" / filename).write_text("payload")
        assert filename not in _names(run_dir)

    def test_checkpoint_directories_are_refused(self, run_dir):
        (run_dir / "adapter" / "checkpoint-200").write_text("x")
        assert "checkpoint-200" not in _names(run_dir)

    def test_an_unrecognised_file_is_refused_by_the_allowlist(self, run_dir):
        (run_dir / "adapter" / "notes.txt").write_text("hello")
        _, rejected = collect_upload_files(run_dir / "adapter", run_dir)
        assert any(p.name == "notes.txt" and "allowlist" in r for p, r in rejected)

    def test_the_allowlist_is_closed_not_open(self, run_dir):
        # Anything new must be added deliberately; nothing leaks by default.
        (run_dir / "adapter" / "surprise.bin").write_bytes(b"\x00")
        assert "surprise.bin" not in _names(run_dir)


class TestContentScanning:
    """The scanner must still reject secrets in otherwise-allowed files."""

    @pytest.mark.parametrize(
        "prefix,body", [("sk-", "a" * 32), ("AKIA", "B" * 16), ("hf_", "c" * 36)]
    )
    def test_a_secret_in_an_allowed_file_is_refused(self, run_dir, prefix, body):
        # Assembled at runtime rather than written as literals, so this file does
        # not itself trip the repository's private-data scanner and does not need
        # a SELF_EXEMPT entry. Exempting a file turns the scanner off for it.
        secret = prefix + body
        (run_dir / "manifest.json").write_text(json.dumps({"note": f"key {secret}"}))
        _, rejected = collect_upload_files(run_dir / "adapter", run_dir)
        assert any(p.name == "manifest.json" for p, _ in rejected)
        assert "manifest.json" not in _names(run_dir)

    def test_a_clean_allowed_file_passes(self, run_dir):
        assert "manifest.json" in _names(run_dir)

    def test_the_five_megabyte_cap_still_refuses_large_scanned_text(self, run_dir):
        # F2 did NOT widen this. A >5MB json is refused rather than skipped.
        (run_dir / "manifest.json").write_text("{" + '"pad":"' + "x" * (6 * 1024 * 1024) + '"}')
        _, rejected = collect_upload_files(run_dir / "adapter", run_dir)
        reasons = {p.name: r for p, r in rejected}
        assert "manifest.json" in reasons
        assert "was not scanned" in reasons["manifest.json"]

    def test_binary_weights_are_not_scanned_as_text(self, run_dir):
        # .safetensors is outside _SCANNED_SUFFIXES; it must still upload.
        assert "adapter_model.safetensors" in _names(run_dir)


class TestModelCardTokenizerGuidance:
    """The card must say where the tokenizer comes from, and name the revision."""

    @staticmethod
    def _card(revision: str = PINNED_SHA) -> str:
        manifest = ExperimentManifest(experiment_id="run-1")
        manifest.model = {"base_model": BASE_MODEL, "revision": revision}
        return build_model_card(repo_id="user/kleos-adapter", manifest=manifest)

    def test_the_pinned_revision_appears_in_the_card(self):
        assert PINNED_SHA in self._card()

    def test_the_tokenizer_is_loaded_from_the_base_model_not_the_adapter_repo(self):
        card = self._card()
        assert "AutoTokenizer.from_pretrained(BASE, revision=REVISION)" in card
        assert 'AutoTokenizer.from_pretrained("user/kleos-adapter")' not in card

    def test_the_card_states_that_no_tokenizer_is_shipped(self):
        card = self._card()
        assert "does **not** ship a tokenizer" in card
        assert BASE_MODEL in card

    def test_the_card_explains_why_rather_than_just_asserting_it(self):
        card = self._card()
        assert "embed_tokens" in card

    def test_both_base_and_tokenizer_use_the_pinned_revision(self):
        card = self._card()
        assert "AutoModelForCausalLM.from_pretrained(BASE, revision=REVISION)" in card
        assert "AutoTokenizer.from_pretrained(BASE, revision=REVISION)" in card

    def test_an_unpinned_run_is_still_flagged(self):
        assert "moving pointer" in self._card("main")

    def test_no_results_means_no_performance_claim(self):
        card = build_model_card(repo_id="user/kleos-adapter")
        assert "No evaluation results" in card


class TestServingRevisionIsDistinctFromTrainingRevision:
    """A run made against a moving pointer still needs an actionable base.

    v0.0.6 was trained with ``revision: main`` and the commit it resolved to is
    not recoverable. Reporting only that leaves a consumer with a warning and no
    action; reporting only the serving pin would imply the adapter was trained
    against it. The card states both, and says which is which.
    """

    @staticmethod
    def _card(*, revision: str = "main", serving: str | None = None) -> str:
        manifest = ExperimentManifest(experiment_id="kleos-v006-ministral8b-run1")
        manifest.model = {"base_model": BASE_MODEL, "revision": revision}
        return build_model_card(
            repo_id="tejas/kleos-policy-v006-ministral8b",
            manifest=manifest,
            serving_revision=serving,
        )

    def test_the_card_distinguishes_the_two_revisions(self):
        card = self._card(serving=PINNED_SHA)
        assert "| Training revision | `main` |" in card
        assert f"| Recommended serving revision | `{PINNED_SHA}` |" in card

    def test_the_serving_revision_is_the_pinned_sha(self):
        card = self._card(serving=PINNED_SHA)
        assert f'REVISION = "{PINNED_SHA}"' in card

    def test_the_historical_training_revision_is_preserved_as_main(self):
        card = self._card(serving=PINNED_SHA)
        assert "trained against `main`" in card
        assert "not recoverable" in card

    def test_the_card_does_not_claim_the_run_used_the_serving_sha(self):
        # The whole point of F3: we pin going forward without rewriting history.
        card = self._card(serving=PINNED_SHA)
        assert "not** a claim that the original run used that commit" in card

    def test_the_load_snippet_uses_the_serving_revision_for_base_and_tokenizer(self):
        card = self._card(serving=PINNED_SHA)
        assert "AutoModelForCausalLM.from_pretrained(BASE, revision=REVISION)" in card
        assert "AutoTokenizer.from_pretrained(BASE, revision=REVISION)" in card
        assert f"at revision `{PINNED_SHA}`" in card

    def test_omitting_the_serving_revision_preserves_the_previous_behaviour(self):
        card = self._card()
        assert "| Revision | `main` |" in card
        assert "Recommended serving revision" not in card
        assert "moving pointer" in card

    def test_a_serving_revision_equal_to_the_training_revision_adds_nothing(self):
        # No spurious second row when there is only one fact to report.
        card = self._card(revision=PINNED_SHA, serving=PINNED_SHA)
        assert "Recommended serving revision" not in card
        assert f"| Revision | `{PINNED_SHA}` |" in card

    def test_an_already_pinned_run_still_reads_correctly(self):
        card = self._card(revision=PINNED_SHA)
        assert "pinned to" in card
        assert "moving pointer" not in card

    def test_no_tokenizer_artifact_is_referenced_as_shipped(self):
        card = self._card(serving=PINNED_SHA)
        assert "does **not** ship a tokenizer" in card
        for artifact in TOKENIZER_ARTIFACTS:
            assert f"`{artifact}`" not in card, f"card implies {artifact} is included"
