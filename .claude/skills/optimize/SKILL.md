---
name: optimize
description: >-
  Run the performance-optimization workflow on walnut — benchmark, trace, form
  a hypothesis, change the code, re-benchmark, open a PR. Use when asked to make
  something faster, find and fix a bottleneck, improve tok/s or latency, act on
  a profile's findings, or "optimize decode", "speed this up", "what should we
  optimize next".
---

# Optimizing walnut

[`benchmark`](../benchmark/SKILL.md) measures,
[`analyze-trace`](../analyze-trace/SKILL.md) diagnoses. This is the order.

## 1. Benchmark

```bash
BENCH=.claude/skills/benchmark/scripts/bench.py
uv run python $BENCH run <model> --label baseline -o /tmp/before.json
```

Before editing anything — a baseline reconstructed later is worth less and
costs more.

## 2. Trace

Follow `analyze-trace`: capture, preflight (§2.3), triage (§3.3). You want a
ranked table of where the token goes, and the ceiling it could reach.

## 3. Form a hypothesis

A `file:line`, its cost in µs/token and as a share of the phase, and what you
expect to change — "fusing this drops ~550 kernels/token and ~0.6 ms of TPOT".
Write the expected number down before you test it.

## 4. Edit the code

Prototype in the scratchpad first — monkeypatch or wrap before touching library
code; it's the cheapest way to kill a bad idea. Then implement following
`CLAUDE.md`, putting anything with a real cost behind a flag threaded the way
`--cuda-graph` is.

## 5. Re-benchmark

```bash
uv run ruff format --check && uv run ruff check && uv run ty check && uv run pytest
uv run python $BENCH run <model> --label <change> -o /tmp/after.json
uv run python $BENCH compare /tmp/before.json /tmp/after.json
```

Read the `output:` line first: sampling is greedy, so a changed hash means
changed behavior. Then capture a second trace and re-run the query that produced
the diagnosis — the mechanism you named should be the one that moved.

If it's inside the noise or costs more than it's worth, revert and say what you
tried and what it measured.

## 6. Open a PR

Fill in `.github/PULL_REQUEST_TEMPLATE/optimize.md`, write it to a scratch file,
and open with it — `--body-file` bypasses template selection, so pass the
filled-in copy:

```bash
gh pr create --title "perf: ..." --body-file <scratch>/pr.md
```
