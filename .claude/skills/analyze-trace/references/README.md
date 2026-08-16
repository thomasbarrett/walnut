# Analyzing PyTorch Inference Traces with PerfettoSQL

### A practitioner's textbook

---

**Scope.** This book teaches you to find and quantify bottlenecks in PyTorch *inference* workloads by writing SQL against Kineto traces loaded into Perfetto's trace processor. It covers the trace format at the byte level, the ingestion semantics of trace processor, a reusable SQL view layer, and 43 diagnostic queries organized around the inference performance model (TTFT, ITL, prefill, decode, batching, CUDA graphs, collectives).

**Method.** Every query in this book was executed against real traces captured on an RTX 5090 (170 SMs, CUDA 13.0, driver 610.43.02) with PyTorch 2.13 and Perfetto trace processor `becb22d3`. Numbers shown in output blocks are real measurements, not illustrations. The running example is an 8-layer fp16 decoder-only transformer (d_model 1024, 16 heads, vocab 32000) generating 32 tokens after a 512-token prefill at batch size 1 — a latency-regime workload chosen because it exposes every host-side pathology at once.

**Prerequisites.** SQL (window functions, CTEs, correlated subqueries). Familiarity with CUDA's asynchronous execution model. No prior Perfetto knowledge assumed.

**Structure.** Six chapters, in the order you would actually work. **Chapter 1** establishes what a trace *is* — the Kineto event format and how trace processor turns it into tables. **Chapter 2** covers producing a trustworthy trace and building the SQL view layer that everything afterwards is written against. **Chapter 3** is the diagnostic core: the prefill/decode performance model and a three-test triage procedure that tells you which bottleneck you have. **Chapters 4 and 5** are the two branches that triage sends you down — host-side and device-side — and are meant to be read on demand rather than front to back. **Chapter 6** puts it together: a full worked diagnosis, tail-latency analysis, A/B comparison, and CI automation.

If you are debugging something right now, read §3.3 in
[Chapter 3](chapter-3-performance-model-and-triage.md), run its three queries,
and follow the branch it sends you down.

---

## Reading order

**[Chapter 1 — The Trace as a Database](chapter-1-trace-as-a-database.md)** · What a Kineto trace contains, how Perfetto's trace processor turns it into tables, and the tools that query them.

&nbsp;&nbsp;&nbsp;&nbsp;1.1 Why SQL
&nbsp;&nbsp;&nbsp;&nbsp;1.2 The Kineto trace format
&nbsp;&nbsp;&nbsp;&nbsp;1.3 Ingestion semantics: JSON → relational
&nbsp;&nbsp;&nbsp;&nbsp;1.4 Tooling

**[Chapter 2 — Capture and the Analysis Kernel](chapter-2-capture-and-view-layer.md)** · Producing a trace you can trust, and the reusable SQL view layer that every later query is written against.

&nbsp;&nbsp;&nbsp;&nbsp;2.1 Capturing an inference trace
&nbsp;&nbsp;&nbsp;&nbsp;2.2 The view layer
&nbsp;&nbsp;&nbsp;&nbsp;2.3 Preflight: validate before you conclude

**[Chapter 3 — The Performance Model and Bottleneck Triage](chapter-3-performance-model-and-triage.md)** · The prefill/decode model, per-token accounting, and a three-test procedure that identifies which bottleneck you have before you try to fix one.

&nbsp;&nbsp;&nbsp;&nbsp;3.1 The inference performance model
&nbsp;&nbsp;&nbsp;&nbsp;3.2 Phase segmentation and the per-token budget
&nbsp;&nbsp;&nbsp;&nbsp;3.3 Bottleneck triage: the decision procedure
&nbsp;&nbsp;&nbsp;&nbsp;3.4 Gap analysis with blame attribution

**[Chapter 4 — Host-Side Bottlenecks](chapter-4-host-side-bottlenecks.md)** · Dispatch overhead, synchronization stalls, and the two interventions that remove them: CUDA graphs and torch.compile.

&nbsp;&nbsp;&nbsp;&nbsp;4.1 Host-side cost: dispatch and Python
&nbsp;&nbsp;&nbsp;&nbsp;4.2 The synchronization audit
&nbsp;&nbsp;&nbsp;&nbsp;4.3 CUDA graphs
&nbsp;&nbsp;&nbsp;&nbsp;4.4 torch.compile and Inductor kernels

**[Chapter 5 — Device-Side Bottlenecks](chapter-5-device-side-bottlenecks.md)** · What to do once the GPU is actually the constraint: kernels, shapes, memory traffic, concurrency, and occupancy.

&nbsp;&nbsp;&nbsp;&nbsp;5.1 Kernel inventory
&nbsp;&nbsp;&nbsp;&nbsp;5.2 Shapes, dtypes, and the gemv trap
&nbsp;&nbsp;&nbsp;&nbsp;5.3 Memory traffic and the KV cache
&nbsp;&nbsp;&nbsp;&nbsp;5.4 Streams, overlap, and collectives
&nbsp;&nbsp;&nbsp;&nbsp;5.5 Occupancy and wave quantization

**[Chapter 6 — Practice](chapter-6-practice.md)** · An end-to-end diagnosis, tail-latency work, regression testing, and the mistakes that invalidate results.

&nbsp;&nbsp;&nbsp;&nbsp;6.1 Worked case study: 485 → 2457 tok/s
&nbsp;&nbsp;&nbsp;&nbsp;6.2 Tail latency and outlier tokens
&nbsp;&nbsp;&nbsp;&nbsp;6.3 A/B trace comparison
&nbsp;&nbsp;&nbsp;&nbsp;6.4 Automation and CI
&nbsp;&nbsp;&nbsp;&nbsp;6.5 Twelve traps

**Appendices** · [A. Category and argument reference](appendix-a-category-and-arg-reference.md) · [B. The complete prelude](appendix-b-prelude.md) · [C. Query index](appendix-c-query-index.md)

---

## Two things to know before you start

**Cross-references.** The text cites sections as `§N.M`. The leading digit is the
chapter, so `§4.3` is section 4.3 of
[Chapter 4](chapter-4-host-side-bottlenecks.md). Query numbers in
[Appendix C](appendix-c-query-index.md) map the other way — from a question to
the section that answers it.

**The prelude.** Nearly every query from Chapter 3 onward is written against the
five views defined in §2.2 and collected in
[Appendix B](appendix-b-prelude.md). Run that prelude against the trace first,
or the queries will fail with `no such table`.

---

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
