# c05 — Pipelined double-buffered matmul

**Goal:** `C = A @ B` for square `N x N x N` fp32 matrices — the **same contract and
oracle as c02**. You are not re-deriving matmul here; you are changing only the
**K-loop's load/compute structure** so that the load of the next K-tile overlaps the
compute on the current one. This is the CUDA expression of Triton's `num_stages`.

Unlocked by: `3f_async_pipelining`.

## The spec
- Inputs: `A`, `B` row-major `N x N` float32. Output: `A @ B`. Timed at `N=768`
  (a multiple of tiles 16/32/64; `768 = 256*3`).
- A small ragged size runs first (untimed) to catch a missing edge guard.
- Metric: **TFLOP/s** = `2*N^3 / time`. The win over a plain tiled c02 may be modest on
  a compute-light fp32 kernel — the lesson here is the **structure**, plus what the
  profiler shows (see *Going for performance*).

## What to write (`kernel.cu`)
- `pipelined_matmul_kernel` + `solve(...)`. Each block computes one output tile by
  walking across K, but with **double-buffered shared tiles** filled by **async copies**.
- The harness owns all `cudaMalloc` / `cudaMemcpy` / timing — `solve()` only sets the
  launch config and fires the kernel. While `solve()` returns the sentinel `77`, the
  harness prints `[TODO]` and checks nothing.

## Run it
```bash
python -m harness.runner c05 --watch
```

## Hints — one at a time
1. **Start from a WORKING tiled matmul (your c02).** Correctness is already settled
   (lecture 3f) — keep the same tile shape, accumulator, and ragged-N guard. Only the
   inner load/compute schedule changes.
2. **What async buys you:** a synchronous load routes global → register → shared and
   *stalls* the thread on the load. An async copy moves global → shared directly, with
   no register stop and without blocking — the bytes fly in the background while the
   thread does other work.
3. **Double-buffering arranges the overlap that the async copy enables:** two shared
   tiles. One is being *read* by compute while the other is being *filled* by an async
   copy. They swap roles every K-step.
4. **The loop order is the whole trick:** issue-next-load → wait-for-current → compute.
   Because *next* was issued before the *wait*, it runs concurrently with the compute on
   the *current* buffer. Prime the pipe with a **prologue** load before the loop.
5. **`wait_prior`'s depth is a knob.** For a depth-2 (double) buffer you want exactly
   *one* copy still outstanding when you start computing. Reason out what argument that
   is — and remember `__syncthreads()` belongs after the wait, before you read the tile,
   and again before you overwrite a buffer you might still be reading.
6. **Headers matter here:** the `__pipeline_memcpy_async` / `__pipeline_commit` /
   `__pipeline_wait_prior` intrinsics live in `<cuda_pipeline.h>` — **not** `<cuda/pipeline>`
   (that's the heavier libcu++ `cuda::pipeline` C++ API, a different interface). The
   runner already builds with `-std=c++17`, which the pipeline headers require.

## Validate & benchmark it yourself
Timing is the same device-event pattern as `c01` (warm up, bracket many `solve()` launches in
a `cudaEventRecord` pair, then `cudaEventElapsedTime / iters`). What changes per kernel is the
**reference**, the **tolerance**, and the **throughput formula**:

- **Correctness:** same as `c02` — a triple-loop host reference
  (`C[i*N+j] += A[i*N+k] * B[k*N+j]`), compare with `atol=1e-1, rtol=1e-2` (matmul reorders adds).
- **Throughput:** compute-bound, so report TFLOP/s:
  ```cpp
  double tflops = 2.0 * N * N * N / (ms * 1e-3) / 1e12;   // 2·N³ flops
  ```
  The pipelining (cp.async double-buffering) should push you toward the same roof `c02` aimed
  at by hiding global-load latency behind compute — so compare your number to `c02`'s and to cuBLAS.

The full method (median over samples, choosing tolerances, the L2 trap) is the reference card, `7b`.

## Going for performance
- **More stages hide more latency only up to a knee** — once compute fully covers the
  load, extra buffers just spend shared memory and cost you occupancy. This is exactly
  what Triton's `num_stages` trades off. Two buffers is the natural starting point.
- **Don't expect a guaranteed speedup.** On this fp32 kernel the arithmetic is light
  enough that you may already be near compute-bound; the structural lesson stands either
  way. Measure before and after rather than assuming.
- **Read it in the profiler.** Run `ncu` and watch the **Long Scoreboard** stall reason:
  the pipelined version should show the warps spending *less* time stalled waiting on
  global-memory loads, because the copies were issued ahead of the compute that needs
  them. That shift — not necessarily the wall-clock number — is the thing to see.
