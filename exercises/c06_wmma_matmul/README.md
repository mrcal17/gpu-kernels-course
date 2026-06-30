# c06 — WMMA tensor-core matmul (fp16)

**Goal:** `C = A @ B` for square `N×N×N`, with `A`/`B` in fp16 (`__half`) and
`C` in fp32. The capstone of the CUDA-C path: this is where you stop doing
scalar FMAs and let a whole **warp** issue a `16×16×16` matrix-multiply-
accumulate as a single hardware op. It's the c02 tiled-matmul shape with the
inner product replaced by a sequence of tile-MMAs — the hand-built core of what
CUTLASS generates (lecture `3g_tensor_cores`).

## The spec
- Inputs: `A (512,512)`, `B (512,512)` in `__half`. Output: `A @ B` in fp32.
  `N=512` is a clean multiple of the `16×16×16` tile (`512 = 32·16`), so you can
  start **without** a K-tail special case.
- The harness owns the fp32→fp16 conversion and keeps the fp32 originals for its
  reference, so your `solve()` only ever sees `__half` device pointers.
- Metric: **FLOP/s** = `2·N³ / time` (reported as TFLOP/s). Expect it to clearly
  exceed the fp32 CUDA matmuls — that's the tensor cores earning their keep.
- **Loose tolerance by design** (`atol=1`, `rtol=2e-2`): fp16 inputs round off
  ~3 mantissa bits and that error grows across K. The answer is *close, not
  exact*. That's the precision-for-throughput trade, not a bug.

## What to write (`kernel.cu`)
- The `__global__ wmma_matmul` kernel + `extern "C" int solve(...)`. One warp
  computes one `16×16` tile of `C`.

Run it:

    python -m harness.runner c06 --watch

## Hints — one at a time
1. **A tensor-core op is warp-collective.** All 32 lanes of a warp issue it
   *together*, over fragments that are distributed across the warp's registers.
   So map **warps** (not individual threads) to `16×16` output tiles. A warp is a
   fixed number of lanes; from `threadIdx` work out which warp within the block
   you are, then map that warp to a `(tileRow, tileCol)` in `C`.
2. **A fragment is opaque.** You never index into it — you only `fill` it, `load`
   it, `mma` it, and `store` it. Every WMMA call ends in `_sync` precisely
   because all 32 lanes must reach it together with **converged** control flow
   (no lane peeling off in an `if`).
3. **It's the c02 structure with the inner product swapped out.** Mirror that
   shape: allocate an fp32 accumulator fragment, zero it, then walk `K` one tile
   at a time — load an A-tile and a B-tile fragment and accumulate one tile-MMA
   per step. Keep the accumulator in fp32 throughout for precision.
4. **Layouts and leading dimensions are the subtle part.** Decide which layout
   each operand fragment should declare, and what stride the load needs — relate
   the leading dimension to how a row-major `N×N` buffer is laid out in memory,
   and reason about what declaring one operand `col_major` over a row-major
   buffer actually implies. Then work out the base pointer of this warp's tile
   `(tileRow, tileCol)` at K-step `k`.
5. **Get ONE tile correct first.** Before worrying about covering the whole
   matrix, make a single `16×16×16` tile match the reference. Remember the
   tolerance is loose — close-but-not-exact is the expected fp16 behavior, so
   don't chase the last few ulps.

## Validate & benchmark it yourself
Timing is the same device-event pattern as `c01` (warm up, bracket many `solve()` launches in
a `cudaEventRecord` pair, then `cudaEventElapsedTime / iters`). What changes per kernel is the
**reference**, the **tolerance**, and the **throughput formula**:

- **Correctness:** a triple-loop host reference over the original fp32 values, but compare with
  a **very loose** `atol=1.0, rtol=2e-2` — the tensor cores consume fp16 inputs, so
  half-precision rounding plus accumulation order makes large absolute differences expected.
  (Too tight here and a *correct* WMMA kernel fails.)
- **Throughput:** compute-bound, so report TFLOP/s:
  ```cpp
  double tflops = 2.0 * N * N * N / (ms * 1e-3) / 1e12;   // 2·N³ flops
  ```
  This is where you compare against the **tensor-core** peak (much higher than the fp32 number),
  not the CUDA-core roof.

The full method (median over samples, choosing tolerances, the L2 trap) is the reference card, `7b`.

## Going for performance
- This is your first taste of the tensor cores from CUDA C directly (in Triton,
  `tl.dot` hid them). The fp32→fp16 trade is exactly why the TFLOP/s jumps.
- The accumulator staying fp32 is what keeps the loose-tolerance answer usable —
  fp16 *accumulate* would drift much further.
- Once it's correct, the natural next step is feeding the MMA loop with the
  `cp.async` double-buffered pipeline from c05: prefetch the next K-tile into
  shared memory while the current `mma_sync` runs. That overlap — load latency
  hidden behind tensor-core math — is the heart of what CUTLASS/cuBLAS do.
- A real library also masks ragged `N` (tiles that fall off the edge when `N`
  isn't a multiple of 16). Not needed here (`N=512` is clean), but it's the
  obvious generalization to add later.
