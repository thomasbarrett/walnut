# Appendix A. Category and argument reference

### Categories

| `cat` | Track | `dur` | Key args |
|---|---|---|---|
| `cpu_op` | host thread | yes | `External id`, `Sequence number`, `Fwd thread id`, `Record function id`, `Input Dims[i][j]`, `Input Strides[i][j]`, `Input type[i]`, `Concrete Inputs[i]`, `Flops` |
| `user_annotation` | host thread | yes | `External id`, `Record function id` |
| `cuda_runtime` | host thread | yes | `External id`, `cbid`, `correlation` |
| `cuda_driver` | host thread | yes | `External id`, `cbid`, `correlation` |
| `kernel` | device stream | yes | `device`, `context`, `stream`, `correlation`, `queued`, `registers per thread`, `shared memory`, `blocks per SM`, `warps per SM`, `grid[0..2]`, `block[0..2]`, `est. achieved occupancy %`, `occupancy.*`, `graph id`, `graph node id`, `channel`, `channel_type` |
| `gpu_memcpy` | device stream | yes | as kernel, plus `bytes`, `memory bandwidth (GB/s)` |
| `gpu_memset` | device stream | yes | as kernel, plus `bytes`, `memory bandwidth (GB/s)` |
| `gpu_user_annotation` | device stream | yes | `External id` |
| `cpu_instant_event` | host thread | 0 | `Total Reserved`, `Total Allocated`, `Bytes`, `Addr`, `Device Id`, `Device Type` |
| `python_function` | host thread | yes | `Python id`, `Python parent id` |
| `ac2g` / `fwdbwd` | — | — | become rows in `flow` |
| `overhead` | host thread | yes | profiler self-cost |
| `Trace` | pseudo-process | yes | `Op count` |

### NCCL kernel args
`Collective name`, `dtype`, `In msg nelems`, `Out msg nelems`, `Group size`, `In split size`, `Out split size`, `Process Group Name`, `Process Group Description`, `Process Group Ranks`, `Rank`, `Src Rank`, `Dst Rank`, `Seq`, `Comms Id`.

### Occupancy sub-object
`args.occupancy.activeBlocksPerMultiprocessor`, `.limitingFactors` (e.g. `"WARPS|REGS"`), `.blockLimitRegs`, `.blockLimitSharedMem`, `.blockLimitWarps`, `.blockLimitBlocks`, `.blockLimitBarriers`, `.allocatedRegistersPerBlock`, `.allocatedSharedMemPerBlock`.

### Unit conventions
| Quantity | Table unit | Divide by |
|---|---|---|
| `slice.ts`, `slice.dur` | nanoseconds | `1e3` → µs, `1e6` → ms |
| `args.bytes` | bytes | `1e6` → MB |
| `args.memory bandwidth (GB/s)` | GB/s | — |
| `args.est. achieved occupancy %` | percent | — |


---

[Index](README.md) · [← Chapter 6](chapter-6-practice.md) · [Appendix B →](appendix-b-prelude.md)
