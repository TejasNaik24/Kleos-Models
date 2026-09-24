# Test fixtures

Most tests build their data in-memory through the factories in `conftest.py`
(`make_example`, `make_eval_example`), which keeps each test's inputs visible
beside its assertions.

This directory exists for fixtures that must be files on disk — malformed JSONL
for loader error paths, for instance.

Fixtures shared with the pipeline itself live in `data/examples/`.

## `ministral3_chat_template.jinja`

The chat template of `mistralai/Ministral-3-14B-Instruct-2512-BF16` at revision
`3cea74c1ebaf5ce5f5a2553de470e2ceab825142`, copied verbatim (Apache-2.0,
Mistral AI). sha256 `2f545122222db8bb43ca0ea0c49e9185320a8670f7d35575b0da0eb48b1e8970`,
asserted by `tests/test_revision_pinning.py`. Vendored so the tests of KLEOS
Logos' prompt rendering and loss masking run without network access.
