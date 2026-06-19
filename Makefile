# Developer entry points for ai-calls-router. All targets run against the
# project-local virtual environment in .venv (created by `make install`).

VENV := .venv
PY := $(VENV)/bin/python
PIP := uv pip install --python $(PY)

.PHONY: help install test lint format coverage build run clean
.PHONY: type check-package check-security check-deps check-complexity qa
.PHONY: vulture refurb bandit interrogate semgrep mutmut

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-18s %s\n", $$1, $$2}'

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
	$(PY) -m ruff check ai_calls_router tests

format: ## Format source and tests with ruff
	$(PY) -m ruff format ai_calls_router tests

type: ## Run pyrefly static type checking
	$(PY) -m pyrefly check ai_calls_router

coverage: ## Run tests with coverage report (fails under 95%)
	$(PY) -m pytest -q --cov --cov-report=term-missing --cov-fail-under=95

build: ## Build sdist and wheel into dist/
	$(PY) -m build

check-package: build ## Validate built package
	$(PY) -m twine check dist/*
	$(PY) -m check_wheel_contents dist/*.whl

check-security: ## Run security audit on dependencies
	$(PY) -m pip_audit

check-deps: ## Check for unused, missing, and transitive dependencies
	$(PY) -m deptry ai_calls_router

check-complexity: ## CI-gated complexity check (xenon on radon + lizard CCN)
	$(PY) -m radon cc ai_calls_router tests -s -a --total-average
	$(PY) -m radon mi ai_calls_router -s
	$(PY) -m xenon --max-absolute D --max-modules C --max-average A ai_calls_router
	$(PY) -m lizard --languages python -C 20 -w ai_calls_router

vulture: ## Report potentially dead code (advisory only)
	$(PY) -m vulture ai_calls_router --min-confidence 80 || true

refurb: ## Suggest modern Python idioms (advisory only)
	$(PY) -m refurb ai_calls_router || true

bandit: ## Run security lint on source
	$(PY) -m bandit -r ai_calls_router -c pyproject.toml

interrogate: ## Report docstring coverage (advisory)
	$(PY) -m interrogate ai_calls_router || true

semgrep: ## Run semgrep pattern-based security/code analysis (advisory)
	$(PY) -m semgrep --config auto ai_calls_router || true

mutmut: ## Run mutation testing on critical modules (advisory, slow)
	$(PY) -m mutmut run --paths-to-mutate ai_calls_router/routing || true

check-cognitive: ## Cognitive complexity check (≤15 per function, matches PyCharm)
	$(PY) scripts/check-cognitive-complexity.py

qa: lint type test coverage check-deps check-security check-package check-complexity check-cognitive ## Run all blocking quality gates

qa-full: qa bandit semgrep vulture refurb interrogate ## Run all gates including advisory

run: ## Run the proxy server in the foreground
	$(PY) -m ai_calls_router serve

clean: ## Remove build artifacts and caches
	rm -rf dist build *.egg-info .pytest_cache .coverage .mutmut-cache
	find . -type d -name __pycache__ -exec rm -rf {} +
