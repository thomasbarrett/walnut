# walnut

An inference engine, built on PyTorch, exposing an OpenAI-compatible API.

## Commands

- Lint / format:  `uv run ruff check` (`--fix`) / `uv run ruff format` (`--check`)
- Type check:     `uv run ty check`
- Test:           `uv run pytest`
- All checks:     `uvx prek run --all-files --stage pre-push`  (mirrors CI)
- Fast subset:    `uvx prek run --all-files`  (drops pytest, docs, helm template)
- Deps:           `uv sync --extra cpu --dev` / `uv add [--dev] <pkg>`
- Chart lint:     `helm lint charts/walnut --strict --values charts/walnut/ci/lint-values.yaml`
- Docs:           `uv run mkdocs serve` (preview) / `uv run mkdocs build --strict` (CI gate)

`.pre-commit-config.yaml` mirrors `.github/workflows/ci.yml` job for job and is
the single definition of the checks. `uv run pytest` alone passes while `ruff`
and `ty` fail CI.

`uvx` fetches `prek`, so there is nothing to install first. Nothing runs on its
own, though, until `prek install --hook-type pre-commit --hook-type pre-push` —
use a real `prek` for that, since the hook it writes calls `prek` by name.

`hadolint` and `helm` come from the `Brewfile`; their hooks fail rather than
skip when the binary is missing. Without them: `uvx prek run --all-files
--stage pre-push --skip hadolint --skip helm-lint --skip helm-template`. CI
covers all three.

## Conventions

- **uv only**: prefix every Python command with `uv run`; never `pip` or a
  global `python`. Lint/format with **ruff** (not black/flake8/isort).
- `pyproject.toml` is the source of truth for deps and ruff/ty/pytest
  settings — check there, don't restate settings elsewhere.
- Tests live in `tests/`, named `test_*.py`.

## Profiles

Three skills in `.claude/skills/` cover performance work, and they compose:

- **`analyze-trace`** — why it is slow. Wraps Perfetto's `trace_processor` for
  traces from `walnut profile` (or `/stop_profile`). Don't read a
  `.trace.json.gz` by hand.
- **`benchmark`** — how fast it is. TTFT, TPOT, ITL percentiles, and a `compare`
  that diffs two runs. Numbers quoted to a human come from here, not
  from a trace.
- **`optimize`** — the loop that uses both: baseline, profile, diagnose,
  prototype, implement, re-measure, then a PR from
  `.github/PULL_REQUEST_TEMPLATE/optimize.md` or an honest abandon.

## Docs

MkDocs Material under `docs/` (deps in the `docs` group).

- Prefer generated over hand-written: CLI via `mkdocs-typer2`, API reference
  via `mkdocstrings`. Don't hand-maintain what these generate.
- `docs/http-api.md` stays a thin narrative over the server's `/docs`; document
  only walnut-specific behavior.
- Changing the CLI, HTTP API, or `Engine` interface? Update the matching
  `docs/` page.
