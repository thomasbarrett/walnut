# Development

## Setup

```bash
uv sync --extra cpu --dev
```

Optional developer tooling (Helm, hadolint, prek, gh) is captured in the
`Brewfile`:

```bash
brew bundle
```

## Checks

```bash
uv run ruff check      # lint
uv run ruff format     # format
uv run ty check        # type check
uv run pytest          # run tests
```

Or everything CI runs, in one command — the same hooks as [Git
hooks](#git-hooks), so there is one definition and not a list to keep in step:

```bash
uvx prek run --all-files --stage pre-push
```

`pyproject.toml` is the source of truth for dependencies and the ruff/ty/pytest
settings.

## Analyzing profiles

`walnut profile` writes a Chrome trace that opens at
<https://ui.perfetto.dev/>. To query one instead:

```bash
uv run python .claude/skills/analyze-trace/scripts/analyze_trace.py \
    overview profiles/walnut-20260815-205304-497286.trace.json.gz
```

The commands are `overview`, `top-ops`, `device`, `launch`, and `sql`; pass
`--help` for their options, or `--json` for machine-readable output. It needs
`perfetto` from the `dev` group, and downloads `trace_processor_shell` on
first use.

`.claude/skills/analyze-trace/` documents which command answers which
question, and the trace schema for writing your own queries.

## Git hooks

Local hooks are managed with [prek](https://prek.j178.dev) (a drop-in
`pre-commit` replacement) and mirror the CI checks:

```bash
prek install --hook-type pre-commit --hook-type pre-push   # once per clone
prek run --all-files                  # the fast hooks: ruff, ty, hadolint, helm lint
prek run --all-files --stage pre-push # those plus pytest, docs, helm template
```

The whole-repo hooks (`pytest`, `mkdocs build --strict`, `helm template`) are
tagged `pre-push` so committing stays instant. `--stage pre-push` runs every
hook, not just those three.

`uvx prek run …` fetches `prek` on demand, so nothing needs installing — except
for `install` itself, whose git hook calls `prek` by name. The `hadolint`,
`helm-lint` and `helm-template` hooks shell out to system binaries either way,
and fail rather than skip when those are missing:

```bash
uvx prek run --all-files --stage pre-push \
    --skip hadolint --skip helm-lint --skip helm-template
```

## Docs

The docs are built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
and API reference is generated from docstrings with
[mkdocstrings](https://mkdocstrings.github.io/). Docs dependencies live in the
`docs` dependency group:

```bash
uv sync --group docs        # install docs deps
uv run mkdocs serve         # live preview at http://127.0.0.1:8000
uv run mkdocs build         # build static site into ./site
```

## CI

`.github/workflows/ci.yml` runs on every push to `main` and all pull requests:

| Job | What it runs |
| --- | --- |
| `lint` / `test` | `ruff check`, `ruff format --check`, `ty check`, `pytest` |
| `docs` | `mkdocs build --strict` |
| `helm-lint` | `helm lint` + `helm template` |
| `dockerfile-lint` | `hadolint` |

Docs are published to GitHub Pages on merge to `main` (`docs.yml`).

## Releases

Releases are automated with
[release-please](https://github.com/googleapis/release-please) using
Conventional Commits. Merging the release PR bumps the version in
`pyproject.toml`, `walnut/__init__.py`, and `charts/walnut/Chart.yaml`, tags
the release, and publishes a container image to
`ghcr.io/thomasbarrett/walnut` (see `.github/workflows/release.yml`).
