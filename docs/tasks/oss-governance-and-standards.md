# Task: OSS governance and standards files

This document is a self-contained implementation spec. An engineer or LLM with
no prior context should be able to complete the task from this file alone.

## Goal

Bring `ai-calls-router` up to the standards expected of a public, pip-installable
open-source Python package: legal, contributor governance, automated quality
gates, typing distribution markers, and a presentable README. The package itself
already works and is fully tested; this task is about the repository envelope, not
runtime behavior.

## Hard constraints (must follow)

These come from the project's coding rules and override any default behavior:

- Markdown documentation must contain no emojis or emoticons.
- Python: target 3.11-3.13, full type hints (prefer built-in generics), Google-style
  docstrings on every module/class/function, a 2-4 sentence module header comment.
- Tooling: `ruff` is the single source of truth for lint and format. Use Makefile
  targets where they exist (`make help`).
- Tests: TDD, spec-derived (not implementation-mirroring), adversarial/boundary
  cases. Coverage gate is 98% (the suite currently holds 100%).
- Git: Conventional Commits. Do not add Claude as a co-author. Do not hand-edit the
  version or `CHANGELOG.md` release sections (release-please owns them).
- Never fabricate; verify against the actual files and cite paths.

## Repository facts the implementer needs

- Package source: `src/ai_calls_router/` (layered: `_lib/`, `accounting/`,
  `routing/`, `proxy/`, `ops/`, plus `cli.py`, `__main__.py`, `__init__.py`).
- Version is single-sourced: `src/ai_calls_router/__init__.py` holds
  `__version__ = "0.1.0"  # x-release-please-version`; `pyproject.toml` uses
  `dynamic = ["version"]` with `[tool.hatch.version] path = "src/ai_calls_router/__init__.py"`.
- Build backend: hatchling. Wheel packages `src/ai_calls_router`.
- CI lives in `.github/workflows/`: `ci.yml` (lint + test matrix 3.11/3.12/3.13 +
  build), `release-please.yml`, `publish.yml` (PyPI Trusted Publishing on release).
- Release automation: `release-please-config.json` (`release-type: simple`,
  `extra-files: ["src/ai_calls_router/__init__.py"]`) and
  `.release-please-manifest.json` (`{ ".": "0.1.0" }`).
- Repo URL used throughout: `https://github.com/maheshkokare/ai-calls-router`.
- Author / security contact: Mahesh Kokare, maheshkokare100@gmail.com.
- Makefile targets today: `help install test lint coverage build run clean`.
  Local toolchain is `uv` (`uv venv .venv --python 3.13`, `uv pip install`).

## Status

Already done and committed (commit `chore: add OSS governance files ... and adopt ruff format`):

- `LICENSE` (MIT, 2026, Mahesh Kokare).
- `CHANGELOG.md` (Keep a Changelog format, seeded with the 0.1.0 entry).
- `CONTRIBUTING.md` (dev workflow, TDD, Conventional Commits, release pipeline).
- `SECURITY.md` (private reporting + security model).
- `ruff format` adopted across all source and test files (25 files reformatted).

Remaining deliverables are listed below in recommended order.

---

## Deliverable 1 - Typing marker (`py.typed`)

The package is fully type-hinted; ship the PEP 561 marker so downstream type
checkers use the inline types.

1. Create an empty file `src/ai_calls_router/py.typed` (zero bytes is correct).
2. In `pyproject.toml`, add `"Typing :: Typed"` to the `classifiers` list.
3. Confirm hatchling includes it in the wheel. Because the wheel target is the
   whole `src/ai_calls_router` package, `py.typed` is included automatically.
   Verify after building: `unzip -l dist/*.whl | grep py.typed`.

Acceptance: `python -m build` produces a wheel containing
`ai_calls_router/py.typed`; `pip show` of an installed wheel lists the
`Typing :: Typed` classifier.

---

## Deliverable 2 - `make format` target and CI format gate

`ruff format` is now the formatter; make it a first-class, enforced gate.

1. In the `Makefile`:
   - Add `format` to the `.PHONY` line.
   - Add a target:
     ```make
     format:
     	$(PY) -m ruff format src tests
     ```
     (Indent with a tab. `$(PY)` is the venv python already defined in the Makefile.)
   - Add a one-line description to the `help` target output, matching the existing
     style of the other entries.
2. In `.github/workflows/ci.yml`, in the `lint` job, add a step after the existing
   `ruff check src tests` step:
   ```yaml
   - run: ruff format --check src tests
   ```

Acceptance: `make format` reformats in place; `ruff format --check src tests`
exits 0 on a clean tree; CI fails if someone commits unformatted code.

---

## Deliverable 3 - `CODE_OF_CONDUCT.md`

Use the Contributor Covenant version 2.1 verbatim, with the enforcement contact
set to `maheshkokare100@gmail.com`. Source text:
https://www.contributor-covenant.org/version/2/1/code_of_conduct/

Requirements:
- Keep the standard headings (Our Pledge, Our Standards, Enforcement
  Responsibilities, Scope, Enforcement, Enforcement Guidelines, Attribution).
- Replace the `[INSERT CONTACT METHOD]` placeholder with the email above.
- No emojis (the canonical text has none; keep it that way).

Acceptance: file exists, is the unmodified 2.1 text apart from the contact, and
`CONTRIBUTING.md` / GitHub community profile recognize it.

---

## Deliverable 4 - Issue and pull request templates

Create under `.github/`:

### `.github/ISSUE_TEMPLATE/bug_report.md`
```markdown
---
name: Bug report
about: Report incorrect behavior in the proxy or CLI
title: ""
labels: bug
assignees: ""
---

## Summary

A clear, concise description of the bug.

## Reproduction

Steps to reproduce, including the exact `acr` command and any relevant config
(redact API keys).

1.
2.
3.

## Expected behavior

What you expected to happen.

## Actual behavior

What actually happened. Include the relevant lines from `~/.ai-calls-router/acr.log`
(redact API keys and tokens).

## Environment

- ai-calls-router version (`acr version`):
- Python version (`python --version`):
- OS:
- Provider / tier model in use:
```

### `.github/ISSUE_TEMPLATE/feature_request.md`
```markdown
---
name: Feature request
about: Suggest an enhancement
title: ""
labels: enhancement
assignees: ""
---

## Problem

The problem or limitation you are hitting.

## Proposed solution

What you would like to happen.

## Alternatives considered

Other approaches you thought about.

## Additional context

Anything else relevant (links, config, prior art).
```

### `.github/ISSUE_TEMPLATE/config.yml`
```yaml
blank_issues_enabled: false
contact_links:
  - name: Security vulnerability
    url: https://github.com/maheshkokare/ai-calls-router/security/advisories/new
    about: Report security issues privately, not as a public issue.
```

### `.github/PULL_REQUEST_TEMPLATE.md`
```markdown
## Summary

What this change does and why.

## Related issues

Closes #

## Test plan

- [ ] `make coverage` passes (>= 98%, suite green)
- [ ] `make lint` reports no issues
- [ ] `make format` produces no diff
- [ ] New behavior is covered by spec-derived, adversarial tests

## Checklist

- [ ] Commits follow Conventional Commits
- [ ] No secrets, tokens, or keys in the diff or tests
- [ ] Docs updated if behavior or config changed
```

Acceptance: opening a new issue on GitHub offers the two templates and hides blank
issues; opening a PR pre-fills the template.

---

## Deliverable 5 - `.github/dependabot.yml`

Keep dependencies and Actions current.

```yaml
version: 2
updates:
  - package-ecosystem: pip
    directory: "/"
    schedule:
      interval: weekly
    commit-message:
      prefix: chore
    groups:
      python-dependencies:
        patterns:
          - "*"
  - package-ecosystem: github-actions
    directory: "/"
    schedule:
      interval: weekly
    commit-message:
      prefix: ci
```

Notes:
- The `pip` ecosystem reads `pyproject.toml`.
- `commit-message.prefix` keeps Dependabot PRs Conventional-Commit-clean so they
  do not trigger spurious release-please version bumps (`chore`/`ci` do not bump).

Acceptance: Dependabot config validates (GitHub Insights > Dependency graph >
Dependabot shows the two ecosystems).

---

## Deliverable 6 - `.pre-commit-config.yaml`

Mirror the CI gates locally. Keep the ruff version aligned with `ruff>=0.6` from
the dev extras; Dependabot/`pre-commit autoupdate` will bump it later.

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-merge-conflict
      - id: check-added-large-files
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.6
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

Then:
1. Add `pre-commit>=3.5` to the `dev` extra in `pyproject.toml`.
2. Document `pre-commit install` in `CONTRIBUTING.md` (already mentioned there;
   verify the path is correct for the venv).
3. Run `pre-commit run --all-files` once and confirm it is clean (the tree is
   already ruff-clean, so only the hygiene hooks might touch trailing whitespace /
   final newlines; commit any such fixes).

Acceptance: `pre-commit run --all-files` exits 0 on a clean checkout.

---

## Deliverable 7 - README polish

Fix one correctness bug and add the standard badge row. File: `README.md`.

1. Bug fix: the Install section says "Requires Python 3.13 or newer." The package
   now supports `>=3.11`. Change it to "Requires Python 3.11 or newer." (Verify
   against `requires-python` in `pyproject.toml` before editing.)
2. Add a badge block immediately under the top-level `# ai-calls-router` heading.
   Use shields.io. No emojis.
   ```markdown
   [![CI](https://github.com/maheshkokare/ai-calls-router/actions/workflows/ci.yml/badge.svg)](https://github.com/maheshkokare/ai-calls-router/actions/workflows/ci.yml)
   [![PyPI](https://img.shields.io/pypi/v/ai-calls-router.svg)](https://pypi.org/project/ai-calls-router/)
   [![Python versions](https://img.shields.io/pypi/pyversions/ai-calls-router.svg)](https://pypi.org/project/ai-calls-router/)
   [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
   [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen)](https://github.com/pre-commit/pre-commit)
   ```
3. Add a short "Contributing" section near the end linking `CONTRIBUTING.md`,
   `CODE_OF_CONDUCT.md`, and `SECURITY.md`.

Acceptance: README renders with a badge row; the Python version statement matches
`pyproject.toml`; no emojis anywhere.

---

## Verification (run all before committing)

```bash
make format          # no diff after
make lint            # clean
ruff format --check src tests   # exits 0
make coverage        # 404+ tests pass, >= 98% (currently 100%)
python -m build      # builds sdist + wheel
unzip -l dist/*.whl | grep -E 'py.typed'   # marker present
```

Then commit the remaining files. Suggested commit (single, since these are all
repo-envelope additions):

```
chore: add code of conduct, issue/PR templates, dependabot, pre-commit; README badges
```

## Manual step that cannot be automated here

PyPI Trusted Publishing requires a one-time "pending publisher" entry on the PyPI
dashboard for project `ai-calls-router`, pointing at the
`maheshkokare/ai-calls-router` repo, workflow `publish.yml`, environment `pypi`.
Document this in `CONTRIBUTING.md` (release section) so the first publish works.

## Recommended study/build order

`py.typed` + classifier -> Makefile/CI format gate -> CODE_OF_CONDUCT ->
issue/PR templates -> dependabot -> pre-commit -> README badges -> verify -> commit.
