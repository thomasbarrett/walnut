## Motivation

<!-- Why this change, and why this target. Not "decode was slow" — the
measurement that made this the right thing to fix:

  - How the token divided: wall = GPU busy + idle, or the equivalent budget.
  - The ranked cost table, with the hot item in us/token AND as a share.
  - The source file and line it lives in.
  - The ceiling (bandwidth roofline, or the GPU-side floor) and how far off you
    were, as a ratio.

Traces are gitignored, so paste the query output you relied on rather than
pointing at a file nobody else has. Link any issue this resolves. -->

## Modifications

<!-- One bullet per touched file: what it now does. Behavior, not diff — the
diff is already on the page. -->

## Design decisions

<!-- The heart of the review. One bold-led paragraph per decision, each
answering: what you chose, what you rejected, and the measurement or constraint
that decided it. A decision with no alternative wasn't a decision — leave it
out. A rejected alternative with a number attached is worth more than three
paragraphs of reasoning, so paste the number.

Cover at minimum:
  - Why this intervention and not the others the profile offered.
  - Any narrowing of scope, and what it costs.
  - Anything that looks like a workaround, and why the direct fix doesn't work.
  - The default: is it on? why? -->

## Accuracy

<!-- Does the model still produce the same thing? `bench.py compare` prints an
`output:` line comparing a hash of greedy output. Paste it.

If the output changed, that is not automatically a failure — but it has to be
explained here, with evidence that the new output is correct, before any
speedup below is worth reading. -->

## Benchmark results

<!-- `bench.py compare` output, verbatim. It already carries the configuration
line (model, GPU, dtype, prompt/output tokens, repeats), so don't restate it.

Include every metric it prints, not the flattering subset. If TTFT regressed
while TPOT improved, that is the interesting part of the PR. -->

```
BENCH=.claude/skills/benchmark/scripts/bench.py
uv run python $BENCH run <model> --label before -o before.json   # on main
uv run python $BENCH run <model> --label after  -o after.json    # on this branch
uv run python $BENCH compare before.json after.json
```

<!-- paste the compare table here -->

## Profiling results

<!-- The same trace query before and after, showing that the mechanism you
predicted in Motivation is the one that moved. A wall-clock win alone does not
confirm the diagnosis — if the claim was "elementwise kernels dominate", show
elementwise kernels falling.

Give the command that captured each trace. -->

| | before | after |
|---|---|---|
| kernels / token | | |
| GPU busy (µs/token) | | |
| <the hot item> | | |

## Regressions and trade-offs

<!-- Anything this makes worse: first-request or startup cost, memory, TTFT,
build time, a new dependency, a code path that is now harder to read. If you
found none, say "none found" and say what you checked — an empty section reads
as an omission. -->

## What's next

<!-- The bottleneck this exposes. One or two sentences and a number: what
dominates now, and how far it sits from the ceiling. This is how the next PR
gets started. -->

## Checklist

- [ ] Checks pass locally (`prek run --all-files`, `uv run pytest`)
- [ ] Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
- [ ] Docs updated if the CLI, HTTP API, or `Engine` interface changed
- [ ] Before and after benchmarked on the same quiet machine, same model,
      same device, same prompt and token count
- [ ] Greedy output is unchanged, or the change is explained under Accuracy
- [ ] A second trace confirms the predicted mechanism, not just the wall clock
