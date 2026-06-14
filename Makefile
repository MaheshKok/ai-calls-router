# Developer entry points for ai-calls-router. All targets run against the
# project-local virtual environment in .venv (created by `make install`).

VENV := .venv
PY := $(VENV)/bin/python
PIP := uv pip install --python $(PY)

.PHONY: help install test lint format coverage build run clean
.PHONY: type check-package check-security check-deps check-complexity qa

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-16s %s\n", $$1, $$2}'

install: ## Create or reuse venv and install package with dev dependencies
	@if [ ! -x "$(PY)" ]; then \
		uv venv $(VENV) --python 3.13; \
	else \
		echo "Using existing virtual environment at $(VENV)"; \
	fi
	$(PIP) -e ".[dev]"

test: ## Run the test suite
	$(PY) -m pytest -q

lint: ## Run ruff checks
	$(PY) -m ruff check src tests

format: ## Format source and tests with ruff
	$(PY) -m ruff format src tests

type: ## Run mypy static type checking
	$(PY) -m mypy src/ai_calls_router

coverage: ## Run tests with coverage report (fails under 98%)
	$(PY) -m pytest -q --cov --cov-report=term-missing --cov-fail-under=98

build: ## Build sdist and wheel into dist/
	$(PY) -m build

check-package: build ## Validate built package
	$(PY) -m twine check dist/*
	$(PY) -m check_wheel_contents dist/*.whl

check-security: ## Run security audit on dependencies
	$(PY) -m pip_audit

check-deps: ## Check for unused, missing, and transitive dependencies
	$(PY) -m deptry src

check-complexity: ## Measure code complexity
	$(PY) -m radon cc src tests -s -a --total-average
	$(PY) -m radon mi src -s
	$(PY) -m xenon --max-absolute B --max-modules A --max-average A src

vulture: ## Report potentially dead code (advisory only)
	$(PY) -m vulture src --min-confidence 80 || true

refurb: ## Suggest modern Python idioms (advisory only)
	$(PY) -m refurb src || true

bandit: ## Run security lint on source
	$(PY) -m bandit -r src -c pyproject.toml

interrogate: ## Report docstring coverage (advisory)
	$(PY) -m interrogate src || true

qa: lint type test coverage check-deps check-security check-package check-complexity ## Run all quality gates

run: ## Run the proxy server in the foreground
	$(PY) -m ai_calls_router

clean: ## Remove build artifacts and caches
	rm -rf dist build *.egg-info .pytest_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
