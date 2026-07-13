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

`ruff` and `ty` are CI gates — run them (or `prek`) before declaring work
done. `pyproject.toml` is the source of truth for dependencies and the
ruff/ty/pytest settings.

## Git hooks

Local hooks are managed with [prek](https://prek.j178.dev) (a drop-in
`pre-commit` replacement) and mirror the CI checks:

```bash
prek install           # enable the hook (once per clone)
prek run --all-files   # ruff, ty, hadolint, and helm lint
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
