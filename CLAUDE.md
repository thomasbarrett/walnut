# walnut

An inference engine, built on PyTorch, exposing an OpenAI-compatible API.

## Commands

- Lint / format:  `uv run ruff check` (`--fix`) / `uv run ruff format` (`--check`)
- Type check:     `uv run ty check`
- Test:           `uv run pytest`
- All checks:     `prek run --all-files`  (ruff, ty, hadolint, helm lint)
- Deps:           `uv sync --extra cpu --dev` / `uv add [--dev] <pkg>`
- Chart lint:     `helm lint charts/walnut --strict --values charts/walnut/ci/lint-values.yaml`
- Docs:           `uv run mkdocs serve` (preview) / `uv run mkdocs build --strict` (CI gate)

`ruff` and `ty` are CI gates; run them (or `prek`) before declaring work done.
`.github/workflows/ci.yml` also runs `helm-lint` + `dockerfile-lint`.

## Conventions

- **uv only**: prefix every Python command with `uv run`; never `pip` or a
  global `python`. Lint/format with **ruff** (not black/flake8/isort).
- `pyproject.toml` is the source of truth for deps and ruff/ty/pytest
  settings — check there, don't restate settings elsewhere.
- Tests live in `tests/`, named `test_*.py`.

## Docs

MkDocs Material under `docs/` (deps in the `docs` group).

- Prefer generated over hand-written: CLI via `mkdocs-typer2`, API reference
  via `mkdocstrings`. Don't hand-maintain what these generate.
- `docs/http-api.md` stays a thin narrative over the server's `/docs`; document
  only walnut-specific behavior.
- Changing the CLI, HTTP API, or `Engine` interface? Update the matching
  `docs/` page.
