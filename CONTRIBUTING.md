# Contributing to walnut

Thanks for contributing! For setup and the day-to-day commands, see the
[Development section of the README](README.md#development).

- **Before a PR:** run `prek run --all-files` and `uv run pytest`; CI must pass.
- **Commit messages:** use [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:` → minor, `fix:` → patch, `feat!:`/`BREAKING CHANGE:` → major). This
  drives [release-please](https://github.com/googleapis/release-please), so
  don't bump the version by hand — merging the release PR does that.
- **For anything larger than a small fix,** open an issue to discuss first.
