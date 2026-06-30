# c03 — Warp-shuffle reduction (array sum)

**Goal:** sum a large 1-D array down to a single scalar — `d_out[0] = sum(d_in)`.
Your first **warp primitive**: fold 32 lanes with register-to-register shuffles
instead of shared memory.

Unlocked by: `3c_warp_primitives`.

## The spec
- Input: `d_in`, `n = 1<<24` floats (64 MB), filled with a deterministic small-range
  LCG so the true sum stays O(n) and fp32 round-off is bounded. Output: `d_out[0]`.
- A tiny `n = 100` case is checked **first** (untimed) — it catches tail off-by-one
  bugs before the big run.
- Reference: a host **double-precision** sum (a fair oracle for a parallel tree, which
  reorders the additions). Compared with **relative** tolerance `rtol=1e-3, atol=1e-2`.
- Metric: **bandwidth** = `n * sizeof(float) / time`. You read all `n` inputs once;
  the single output write is negligible.

## What to write (`kernel.cu`)
- A `__global__` reduction kernel + `extern "C" int solve(const float* d_in, float* d_out, int n)`.
- `solve()` chooses the launch config, launches, and returns `0`. The stub returns the
  sentinel `77` so the harness prints `[TODO]`.
- **The harness does NOT pre-zero `d_out`.** If you finalize with `atomicAdd`, you must
  `cudaMemset(d_out, 0, sizeof(float))` inside `solve()` first.

Run it (re-runs on every save):

    python -m harness.runner c03 --watch

## Hints — one at a time
1. Within a warp the 32 lanes already run in lockstep — no `__syncthreads()` needed.
   Exchange registers directly with the shuffle intrinsic (lecture `3c`).
2. The fold is a `log2(32) = 5`-step tree: each step halves the active span and adds in
   the value from the lane `delta` above. Work out the delta sequence yourself.
3. Every lane named in the participation mask must execute the shuffle — keep the warp
   converged through the whole fold (the warp-level cousin of "all threads hit the
   barrier").
4. Split the reduction at the warp boundary: *within* a warp use shuffles; *across*
   warps in a block, go through a tiny `__shared__` array + one `__syncthreads()`, then
   let one warp finish the per-warp partials.
5. Crossing the whole array needs more than one block. Either finalize with `atomicAdd`
   into a single output, or write one partial per block and reduce those in a second
   pass — decide and be consistent (zero the output first if you use atomics).
6. A grid-stride loop lets a fixed grid sum any `n`. Size the grid for occupancy, not to
   cover `n` one-element-per-thread.

## Validate & benchmark it yourself
Timing is the same device-event pattern as `c01` (warm up, bracket many `solve()` launches in
a `cudaEventRecord` pair, then `cudaEventElapsedTime / iters`). What changes per kernel is the
**reference**, the **tolerance**, and the **throughput formula**:

- **Correctness:** compute the host reference sum by accumulating every input in **double** (so
  the reference itself isn't the thing that's wrong), copy your scalar back, and compare with
  `atol=1e-2, rtol=1e-3`. A tree / warp reduction sums in a different order than a serial loop,
  so allow slack.
- **Throughput:** a reduction is **memory-bound** (you read `n` floats and write one), so report GB/s:
  ```cpp
  double gbps = (double)n * sizeof(float) / (ms * 1e-3) / 1e9;   // input-dominated
  ```
  Judge it against the ~896 GB/s roof.

The full method (median over samples, choosing tolerances, the L2 trap) is the reference card, `7b`.

## Going for performance
This is **memory-bound**: you touch every element once, so the ceiling is your `e03`
bandwidth. To hit it the per-thread input reads must be **coalesced** — adjacent lanes
should read adjacent addresses *within a grid-stride step*, so a warp's 32 reads land on
one contiguous 128-byte line. The shuffle fold and the atomic finalize are off the
critical path; the loads are what saturate HBM.
