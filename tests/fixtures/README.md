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

## `ministral3_reasoning_chat_template.jinja`

The chat template of `mistralai/Ministral-3-14B-Reasoning-2512` at revision
`51f9210f3cd20f3452a80d5819d15dc61cc50630`, copied verbatim (Apache-2.0,
Mistral AI). sha256 `6b5044075f09f4daa57beebe2d989d9fbe67dea351f220e1a84e19ddc893f2c2`,
asserted by `tests/test_revision_pinning.py`. Vendored so the tests of KLEOS
Logos v0.0.2's thinking span (`[THINK]trace[/THINK]answer`) and its loss masking
run without network access.
