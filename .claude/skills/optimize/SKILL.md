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
costs more. Run it twice on the same build and `compare` the two: that delta is
the noise floor, and anything smaller later is not a result.

## 2. Trace

Follow `analyze-trace`: capture, preflight (§2.3 of
`references/chapter-2-capture-and-view-layer.md`), triage (§3.3 of
`chapter-3-performance-model-and-triage.md`). You want a ranked table of where
the token goes, and the ceiling it could reach.

Profile the workload you benchmarked — pass the same `--temperature` and
`--max-tokens` as step 1. Different values measure different code: at
`--temperature 1.0` the sampler's softmax is the largest non-graph kernel in the
trace and a greedy benchmark never runs it.

## 3. Form a hypothesis

A `file:line`, its cost in µs/token and as a share of the phase, and what you
expect to change — "fusing this drops ~550 kernels/token and ~0.6 ms of TPOT".
Write the expected number down before you test it.

Getting to `file:line` means reading source: the trace gives you kernels, and
under a replayed CUDA graph it does not even give you operator names. Expect to
read the layer and model files to map shapes onto modules. State which
denominator "share of the phase" uses — decode wall, GPU busy and graph replay
differ by up to 29%.

Check the ceiling per candidate, not just in aggregate. The biggest kernel is
often already at the roofline (walnut's `lm_head` runs at 91% of peak
bandwidth); the winnable ones are the small kernels stuck at the ~1.9 µs
kernel-duration floor.

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

`compare` exits non-zero if the two runs measured different things. Read the
`output:` line first: sampling is greedy, so a changed hash means changed
numerics. One prompt at one temperature is a weak correctness check, so re-run
with a second `--prompt` and a longer `--tokens` before believing it.

Then capture a second trace and re-run the query that produced the diagnosis —
the mechanism you named should be the one that moved.

If it's inside the noise or costs more than it's worth, revert and say what you
tried and what it measured.

## 6. Open a PR

Fill in `.github/PULL_REQUEST_TEMPLATE/optimize.md` and write it to a scratch
file. Then branch, commit and push before opening — `--body-file` bypasses
template selection, so pass the filled-in copy:

```bash
git checkout -b perf-<change> && git add -A && git commit -m "perf: ..."
git push -u origin perf-<change>
gh pr create --title "perf: ..." --body-file <scratch>/pr.md
```

Paste the `compare` table and the before/after trace query output into the body;
traces are gitignored, so a path helps nobody.
