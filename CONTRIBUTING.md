# Contributing

Thanks for your interest in improving ai-calls-router. This guide covers the
local workflow, the conventions the project enforces, and how releases are cut.

## Development setup

The project targets Python 3.11, 3.12, and 3.13 and uses [uv](https://docs.astral.sh/uv/)
through a Makefile.

```bash
make install   # create .venv and install the package with dev extras
make test      # run the test suite
make lint      # ruff check
make format    # ruff format
make coverage  # suite with coverage, fails under 98%
```

Run `make help` to list every target.

### Pre-commit hooks

Install the hooks once after cloning so formatting and linting run on every
commit:

```bash
.venv/bin/pre-commit install
```

## Tests

Tests are mandatory for every change and are written from the specification, not
by mirroring the implementation. The suite is organised by layer under `tests/`
(`unit/`, `integration/`, `e2e/`) and currently holds full coverage. New code
must keep coverage at or above the 98% gate, and the integration suite must keep
one explicit test per routing invariant.

Prefer adversarial and boundary cases over happy-path-only coverage. A test that
cannot fail when the implementation is wrong does not belong in the suite.

## Coding conventions

- Full type hints on every function and method; prefer built-in generics.
- Google-style docstrings on every module, class, and function.
- `ruff` is the single source of truth for linting and formatting.
- Keep modules focused; one significant public class per file.

## Commit messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/).
The commit type drives the next version bump, so use them accurately:

- `feat:` a new feature (minor bump)
- `fix:` a bug fix (patch bump)
- `feat!:` or a `BREAKING CHANGE:` footer (major bump once past 1.0)
- `docs:`, `test:`, `refactor:`, `chore:`, `ci:`, `perf:` for everything else

Example:

```
feat: add openrouter provider preset to the init wizard
```

## Releases

Releases are automated. Merging Conventional Commits into `main` causes
[release-please](https://github.com/googleapis/release-please) to open or update
a release pull request that bumps the version, updates `CHANGELOG.md`, and tags
the release on merge. Publishing to PyPI then runs automatically through GitHub
Actions using Trusted Publishing. Do not bump the version or edit the changelog
by hand.

## Pull requests

1. Branch from `main`.
2. Keep the change focused and the suite green (`make coverage`).
3. Ensure `make lint` and `make format` report no changes.
4. Fill out the pull request template, including the test plan.

By contributing you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
