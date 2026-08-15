# Evals for the analyze-trace skill

`../evals.json` describes what a model using the skill should do. There is no
built-in runner: give a fresh session the `query` (and `files`, if any) with
the skill available, then check its answer against `expected_behavior`.

## Running them

```bash
uv sync --extra cpu --dev
```

The first run downloads `trace_processor_shell` (~14 MB) to
`~/.local/share/perfetto/`, so it needs network once.

The first four cases are deterministic: their traces are committed here, so a
fresh clone gets the same numbers on any machine.

The last case is marked `"optional": true` because it is not. It profiles
`Qwen/Qwen3.5-0.8B` end to end, which needs a ~1.7 GB model that is not in the
repo, takes minutes, and produces host-dependent numbers. It earns its place
anyway: it is the only case that exercises running the profiler and picking up
the path it prints. Skip it when offline.

## The trace fixtures

`cpu-real.json` is a real torch capture. The other three are hand-written
Chrome JSON, because each isolates a single condition that a real run does not
hand you on demand — a driver launch nested inside a runtime launch needs
driver tracing enabled, and two shapes sharing a dimension has to be
constructed. torch exports this same format, so all four take the same import
path; timestamps are microseconds, as torch writes them.

All four are committed uncompressed, and deliberately. `walnut profile` gzips
what it writes, but a `.gz` in git cannot be delta-compressed against its
previous version, so every edit stores a whole new blob — this fixture is
3,485 bytes gzipped against 4,990 stored as plain JSON, and the plain one is
reviewable in a diff. Compress or fetch out-of-band only if a fixture ever
needs to be megabytes; keep them small instead.

| File | Shape | Why it exists |
| --- | --- | --- |
| `gpu-bound.json` | 1% launch cost, 55% busy, 5 ms in `cudaDeviceSynchronize` | Summing the `cuda_runtime` category instead of matching launch names reports ~101% here and inverts the verdict |
| `launch-bound.json` | 400% launch cost, 8% busy, 19 idle gaps | The opposite verdict from the same commands, separated by `synchronize_ms` |
| `cpu-shapes.json` | Two `aten::mm` shapes sharing a dimension | Grouping by a single `args.Input Dims` key merges them and reports one shape where there are two |
| `cpu-real.json` | Real torch capture, 56 KB | Genuine operator nesting: `aten::addmm` really sits inside `aten::linear`, so ranking by inclusive time visibly puts the wrapper on top |

`tests/test_trace.py` asserts these still produce those numbers, so a fixture
that stops discriminating fails the suite rather than quietly passing evals.
