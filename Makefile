# Developer entry points for ai-calls-router. All targets run against the
# project-local virtual environment in .venv (created by `make install`).

VENV := .venv
PY := $(VENV)/bin/python
PIP := uv pip install --python $(PY)

.PHONY: help install test lint coverage build run clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-12s %s\n", $$1, $$2}'

install: ## Create venv and install package with dev dependencies
	uv venv $(VENV) --python 3.13
	$(PIP) -e ".[dev]"

test: ## Run the test suite
	$(PY) -m pytest -q

lint: ## Run ruff checks
	$(PY) -m ruff check src tests

coverage: ## Run tests with coverage report (fails under 98%)
	$(PY) -m pytest -q --cov --cov-report=term-missing --cov-fail-under=98

build: ## Build sdist and wheel into dist/
	$(PY) -m build

run: ## Run the proxy server in the foreground
	$(PY) -m ai_calls_router

clean: ## Remove build artifacts and caches
	rm -rf dist build *.egg-info .pytest_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
