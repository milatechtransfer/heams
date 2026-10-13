.PHONY: test test-ci

PYTHON_VERSION ?= 3.12

# Runs in python_package's environment: HippoRAG only resolves there, through the
# dependency overrides in python_package/pyproject.toml.
test:
	cd ../../python_package && uv run --python $(PYTHON_VERSION) pytest $(CURDIR)/tests

test-ci:
	cd ../../python_package && uv run --python $(PYTHON_VERSION) pytest $(CURDIR)/tests -m "not integration"
