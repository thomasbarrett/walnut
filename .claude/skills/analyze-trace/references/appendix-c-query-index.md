# Appendix C. Query index

| # | Question | Section |
|---|---|---|
| 1 | Did the parser drop events? | §2.3 |
| 2 | Which categories exist and how much time? | §2.3 |
| 3 | Are all device ops linked to launches? | §2.3 |
| 4 | Are phase annotations intact? | §2.3 |
| 5 | Is the GPU timeline truncated? | §2.3 |
| 6 | Prefill vs decode kernel profile | §3.1 |
| 7 | **Per-token budget (master query)** | §3.2.1 |
| 8 | Throughput and ITL, warm-up excluded | §3.2.2 |
| 9 | TTFT | §3.2.3 |
| 10 | GPU-side phase spans | §3.2.4 |
| 11 | **GPU utilization over a window** | §3.3.1 |
| 12 | **Queue latency (starved vs backlogged)** | §3.3.2 |
| 13 | **Host time decomposition** | §3.3.3 |
| 14 | Total idle and gap count | §3.4.1 |
| 15 | Blame attribution per gap | §3.4.2 |
| 16 | Gap histogram | §3.4.3 |
| 17 | ATen self time | §4.1.1 |
| 18 | Dispatch-overhead ratio | §4.1.2 |
| 19 | Python source attribution | §4.1.3 |
| 20 | **Synchronization audit** | §4.2 |
| 21 | Sync bubble cost | §4.2 |
| 22 | Verify CUDA graph capture | §4.3.1 |
| 23 | Launch fan-out | §4.3.2 |
| 24 | Work outside the graph | §4.3.4 |
| 25 | Triton vs cuBLAS vs ATen mix | §4.4 |
| 26 | Find recompiles / host stalls | §4.4 |
| 27 | Top kernels by device time | §5.1.1 |
| 28 | Rollup by ATen operator | §5.1.2 |
| 29 | Kernel family classification | §5.1.3 |
| 30 | Tiny-kernel census | §5.1.4 |
| 31 | GEMM/GEMV shape census | §5.2 |
| 32 | Dtype audit | §5.2 |
| 33 | Memcpy by direction and bandwidth | §5.3 |
| 34 | Peak allocated/reserved | §5.3.1 |
| 35 | Per-token allocator churn | §5.3.1 |
| 36 | Stream census | §5.4.1 |
| 37 | Cross-stream overlap | §5.4.2 |
| 38 | Collective inventory and share | §5.4.3 |
| 39 | Occupancy and grid geometry | §5.5 |
| 40 | Wave quantization waste | §5.5 |
| 41 | Occupancy limiters by weighted time | §5.5 |
| 42 | ITL percentiles | §6.2 |
| 43 | Explain outlier tokens | §6.2 |

---

[Index](README.md) · [← Appendix B](appendix-b-prelude.md)
