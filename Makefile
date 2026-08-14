.DEFAULT_GOAL := help
PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin

.PHONY: help
help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --- setup ----------------------------------------------------------------

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --quiet --upgrade pip

.PHONY: install
install: $(BIN)/python ## Install the light (no-torch) dev environment
	$(BIN)/pip install -e ".[dev]"

.PHONY: install-train
install-train: $(BIN)/python ## Install training deps (torch, transformers, peft)
	$(BIN)/pip install -e ".[train,dev]"

.PHONY: install-cuda
install-cuda: $(BIN)/python ## Install training deps incl. bitsandbytes (CUDA only)
	$(BIN)/pip install -e ".[train,quant,dev]"

# --- quality --------------------------------------------------------------

.PHONY: lint
lint: ## Run ruff lint checks
	$(BIN)/ruff check .

.PHONY: format
format: ## Auto-format with ruff
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

.PHONY: format-check
format-check: ## Verify formatting without writing
	$(BIN)/ruff format --check .

.PHONY: typecheck
typecheck: ## Run mypy
	$(BIN)/mypy src/kleos_models

.PHONY: test
test: ## Run the test suite
	$(BIN)/pytest

.PHONY: test-fast
test-fast: ## Run tests excluding tiny-model training tests
	$(BIN)/pytest -m "not slow"

.PHONY: check
check: lint format-check typecheck test ## Run every quality gate

# --- pipeline shortcuts ---------------------------------------------------

.PHONY: smoke
smoke: ## Run the pre-flight smoke test
	$(BIN)/python scripts/smoke_test.py

.PHONY: validate
validate: ## Validate the bundled synthetic development fixtures
	$(BIN)/python scripts/validate_dataset.py --dataset data/examples

.PHONY: inspect
inspect: ## Print a dataset quality report for the fixtures
	$(BIN)/python scripts/inspect_dataset.py --dataset data/examples

.PHONY: leakage
leakage: ## Run the leakage checker over the fixtures
	$(BIN)/python scripts/validate_dataset.py --dataset data/examples --leakage-report reports/leakage

.PHONY: privacy
privacy: ## Scan the repository for secrets / private data
	$(BIN)/python scripts/check_no_private_data.py .

.PHONY: configs
configs: ## Validate every YAML config resolves and type-checks
	$(BIN)/python scripts/validate_configs.py

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
