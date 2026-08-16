# Analyzing PyTorch Inference Traces with PerfettoSQL

A textbook on finding and quantifying bottlenecks in PyTorch inference by
querying Kineto traces as SQL. Chapters 1 and 2 cover what a trace is and how
to capture and load one. Chapter 3 identifies which bottleneck you have.
Chapters 4 and 5 are the host-side and device-side branches it sends you down,
read on demand rather than front to back. Chapter 6 is practice.

Sections are cited throughout as `§N.M`; the leading digit is the chapter.

## [Chapter 1 — The Trace as a Database](chapter-1-trace-as-a-database.md)

What a Kineto trace contains and how Perfetto's trace processor turns it into
tables — the Chrome JSON event format at the byte level, the ingestion rules
that decide which events become slices and which are dropped, and the tools
that query the result.

- 1.1 Why SQL
- 1.2 The Kineto trace format
- 1.3 Ingestion semantics: JSON → relational
- 1.4 Tooling

## [Chapter 2 — Capture and the Analysis Kernel](chapter-2-capture-and-view-layer.md)

Producing a trace worth trusting — warm-up, flushing before the window closes,
and what each capture flag costs — then the five-view SQL layer that every
later query is written against. That layer ships as
[`scripts/prelude.sql`](../scripts/prelude.sql); run it against the trace
first, or the queries fail with `no such table`. The preflight in §2.3 catches
dropped events and truncated timelines before they turn into confident, wrong
answers.

- 2.1 Capturing an inference trace
- 2.2 The view layer
- 2.3 Preflight: validate before you conclude

## [Chapter 3 — The Performance Model and Bottleneck Triage](chapter-3-performance-model-and-triage.md)

The diagnostic core. Prefill and decode as separate workloads with separate
limits, per-token latency budgets, and a three-test procedure in §3.3 that
tells you which bottleneck you have before you try to fix one. §3.4 attributes
each GPU idle gap to the host work that caused it.

- 3.1 The inference performance model
- 3.2 Phase segmentation and the per-token budget
- 3.3 Bottleneck triage: the decision procedure
- 3.4 Gap analysis with blame attribution

## [Chapter 4 — Host-Side Bottlenecks](chapter-4-host-side-bottlenecks.md)

The branch to follow when the GPU is starved: ATen dispatch and Python
interpreter cost, an audit of the synchronization points that block the host
on the device, and the two interventions that remove them — CUDA graphs and
`torch.compile`.

- 4.1 Host-side cost: dispatch and Python
- 4.2 The synchronization audit
- 4.3 CUDA graphs
- 4.4 torch.compile and Inductor kernels

## [Chapter 5 — Device-Side Bottlenecks](chapter-5-device-side-bottlenecks.md)

The branch to follow when the GPU is genuinely the constraint: kernel inventory
and rollups by operator and family, the gemv trap that batch-1 decode falls
into, memory traffic and the KV cache, stream overlap and collectives, and
occupancy and wave quantization.

- 5.1 Kernel inventory
- 5.2 Shapes, dtypes, and the gemv trap
- 5.3 Memory traffic and the KV cache
- 5.4 Streams, overlap, and collectives
- 5.5 Occupancy and wave quantization

## [Chapter 6 — Practice](chapter-6-practice.md)

Putting it together: a diagnosis carried end to end, inter-token latency
percentiles and what explains the outliers, comparing two traces, wiring the
queries into CI, and the twelve mistakes in §6.5 that invalidate a result.

- 6.1 Worked case study: 485 → 2457 tok/s
- 6.2 Tail latency and outlier tokens
- 6.3 A/B trace comparison
- 6.4 Automation and CI
- 6.5 Twelve traps

## Sources

- [Perfetto: SQL tables reference](https://perfetto.dev/docs/analysis/sql-tables)
- [Perfetto: Getting started with PerfettoSQL](https://perfetto.dev/docs/analysis/perfetto-sql-getting-started)
- [Perfetto: PerfettoSQL standard library](https://perfetto.dev/docs/analysis/stdlib-docs)
- [Perfetto: Trace Processor (C++/CLI)](https://perfetto.dev/docs/analysis/trace-processor)
- [Perfetto: Trace Processor (Python)](https://perfetto.dev/docs/analysis/trace-processor-python)
- [Perfetto: Visualizing external trace formats](https://perfetto.dev/docs/getting-started/other-formats)
- [PyTorch: torch.profiler API](https://docs.pytorch.org/docs/main/profiler.html)
- [PyTorch: Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [PyTorch: Profiling to understand torch.compile performance](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html)
- [Kineto: `output_json.cpp` (trace emitter)](https://github.com/pytorch/kineto/blob/main/libkineto/src/output_json.cpp)
- [vLLM: Profiling guide](https://docs.vllm.ai/en/stable/contributing/profiling/)
