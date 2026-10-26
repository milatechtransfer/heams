.PHONY: help setup setup-pre-commit check test test-ci docs docs-serve docs-strict ollama-install

PYTHON_VERSION ?= 3.12
# Plugin tests run in the python_package venv, where `make setup` installs every plugin.
PLUGIN_UV_RUN := uv run --project $(CURDIR)/python_package --no-sync

help: ## Show this help message
	@awk -F':.*?## ' '/^[a-zA-Z0-9_-]+:.*## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install Python dev dependencies via uv
	$(MAKE) -C python_package setup

setup-pre-commit: ## Install the pre-commit git hook locally
	uvx pre-commit install

check: ## Run pre-commit across the whole repo
	uvx pre-commit run --all-files --show-diff-on-failure

test: ## Run the full Python test suite, including integration and Ollama tests
	$(MAKE) -C python_package test
	$(MAKE) -C memory_plugins/heams_memory_mem0 test
	$(MAKE) -C memory_plugins/heams_memory_hipporag test

test-ci: ## Run CI-safe tests, excluding integration and Ollama tests
	$(MAKE) -C python_package test-ci
	$(MAKE) -C memory_plugins/heams_memory_milamem test-ci UV_RUN="$(PLUGIN_UV_RUN)"
	$(MAKE) -C memory_plugins/heams_memory_mem0 test-ci UV_RUN="$(PLUGIN_UV_RUN)"
	$(MAKE) -C memory_plugins/heams_memory_hipporag test-ci
	$(MAKE) -C memory_plugins/heams_memory_letta test-ci UV_RUN="$(PLUGIN_UV_RUN)"
	$(MAKE) -C memory_plugins/heams_memory_privacy_wrapper test-ci UV_RUN="$(PLUGIN_UV_RUN)"

ollama-install: ## Ensure Ollama is installed locally
	$(MAKE) -C python_package ollama-install

docs: ## Build documentation site
	$(MAKE) -C python_package docs

docs-serve: ## Serve documentation locally
	$(MAKE) -C python_package docs-serve

docs-strict: ## Build documentation site (fail on warnings)
	$(MAKE) -C python_package docs-strict
