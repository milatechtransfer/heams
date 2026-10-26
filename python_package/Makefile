.PHONY: help setup ollama-install check test test-ci run run-overwrite run-smoke run-smoke-benchmark run-smoke-memory-system browse launch-campaign memory-list memory-test docs docs-serve docs-strict

PYTHON_VERSION ?= 3.12

CAMPAIGN ?= smoke_campaign
CAMPAIGN_OUTPUT_DIR ?= results/campaigns
SMOKE_BENCHMARK_OUTPUT_DIR ?= results/smoke_test_benchmark
SMOKE_MEMORY_SYSTEM_OUTPUT_DIR ?= results/smoke_test_memory_system
DOCS_PORT ?= 8000
DOCS_ADDR := 127.0.0.1:$(DOCS_PORT)

help: ## Show this help message
	@awk -F':.*?## ' '/^[a-zA-Z0-9_-]+:.*## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Set up the development environment
	uv sync --dev --extra all-memory-plugins --python $(PYTHON_VERSION)

ollama-install: ## Ensure Ollama is installed locally
	@if command -v ollama >/dev/null 2>&1; then \
		echo "Ollama is already installed."; \
	elif command -v brew >/dev/null 2>&1; then \
		brew install ollama; \
	else \
		echo "Ollama is not installed and Homebrew is unavailable."; \
		echo "Install it manually from https://ollama.com/download"; \
		exit 1; \
	fi

check: ## Run pre-commit across the whole repo
	$(MAKE) -C .. check

test: ## Run the full Python test suite
	uv run --python $(PYTHON_VERSION) pytest -s tests/

test-ci: ## Run tests excluding integration and Ollama tests
	uv run --python $(PYTHON_VERSION) pytest -s -m "not integration and not requires_ollama" tests/

run: ## Execute one run (CONFIG=... OUTPUT_DIR=...).
	$(if $(strip $(CONFIG)),,$(error CONFIG is required. Use 'make run CONFIG=<run-config-name>'.))
	$(if $(strip $(OUTPUT_DIR)),,$(error OUTPUT_DIR is required. Use 'make run CONFIG=<run-config-name> OUTPUT_DIR=<directory>'.))
	uv run --python $(PYTHON_VERSION) heams run --config configs/runs/$(CONFIG).yaml --output-dir "$(OUTPUT_DIR)"

run-overwrite: ## Execute one exact run directory, overwriting existing output (CONFIG=... OUTPUT_DIR=...).
	$(if $(strip $(CONFIG)),,$(error CONFIG is required. Use 'make run-overwrite CONFIG=<run-config-name>'.))
	$(if $(strip $(OUTPUT_DIR)),,$(error OUTPUT_DIR is required. Use 'make run-overwrite CONFIG=<run-config-name> OUTPUT_DIR=<directory>'.))
	uv run --python $(PYTHON_VERSION) heams run --config configs/runs/$(CONFIG).yaml --output-dir "$(OUTPUT_DIR)" --overwrite

run-smoke: run-smoke-benchmark run-smoke-memory-system ## Run the benchmark and memory-system smoke tests and generate their reports.

run-smoke-benchmark: ## Run a smoke test for every benchmark except Retention policy with in-context memory and generate reports.
	$(MAKE) run CONFIG=smoke_test_benchmark OUTPUT_DIR="$(SMOKE_BENCHMARK_OUTPUT_DIR)"
	uv run --python $(PYTHON_VERSION) heams generate-report --results-dir "$(SMOKE_BENCHMARK_OUTPUT_DIR)"

run-smoke-memory-system: ## Run every memory system except Letta on one LoCoMo scenario and generate reports.
	$(MAKE) run CONFIG=smoke_test_memory_system OUTPUT_DIR="$(SMOKE_MEMORY_SYSTEM_OUTPUT_DIR)"
	uv run --python $(PYTHON_VERSION) heams generate-report --results-dir "$(SMOKE_MEMORY_SYSTEM_OUTPUT_DIR)"

browse: ## Browse results in the local web GUI (RESULTS_DIR=...).
	$(if $(strip $(RESULTS_DIR)),,$(error RESULTS_DIR is required. Use 'make browse RESULTS_DIR=<directory>'.))
	uv run --python $(PYTHON_VERSION) heams browse --results-dir "$(RESULTS_DIR)"

launch-campaign: ## Submit a Slurm campaign (CAMPAIGN=... CAMPAIGN_OUTPUT_DIR=...; PARTITION and SLURM_ACCOUNT from the environment).
	cluster/submit_campaign.sh cluster/campaigns/$(CAMPAIGN).sh "$(CAMPAIGN_OUTPUT_DIR)"

memory-list: ## List available memory plugins
	uv run --python $(PYTHON_VERSION) heams memory list

memory-test: ## Test memory plugins
	uv run --python $(PYTHON_VERSION) heams memory test

docs: ## Build documentation site (Zensical → site/)
	uv run zensical build -f ../zensical.yml

docs-serve: ## Serve documentation locally (DOCS_PORT=8000 default)
	uv run zensical serve -f ../zensical.yml -a $(DOCS_ADDR)

docs-strict: ## Build documentation site (fail on warnings)
	uv run zensical build -f ../zensical.yml --strict
