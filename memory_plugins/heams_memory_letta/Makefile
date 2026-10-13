.PHONY: setup test test-ci test-live

PYTHON_VERSION ?= 3.12.6
UV_RUN ?= uv run --python $(PYTHON_VERSION)

setup:
	uv sync --dev --python $(PYTHON_VERSION)

test:
	$(UV_RUN) pytest tests

test-ci:
	$(UV_RUN) pytest tests -m "not integration"

test-live:
	$(UV_RUN) pytest tests -m integration -rs
