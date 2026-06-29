# c04 — Conflict-free transpose

**Goal:** out-of-place transpose, `d_out (cols × rows) = d_in (rows × cols).T`,
both row-major. This is the CUDA counterpart of the Triton transpose (`e06`) — but
here you stage the tile through **shared memory** by hand, and you pay for (or fix)
a bank conflict you can't see in Triton.

Unlocked by: `3d_memory_banks`.

## The spec
- Inputs: `d_in`, `(rows, cols)` float32, filled with an **index encoding**
  (`in[r*cols+c] = r*cols+c`) so a wrong-index bug is obvious and the check is
  **exact** (atol = 0). Output: the transpose, `(cols, rows)`.
- Two sizes: a non-square `1000 × 1500` (correctness only, checked **first** —
  it forces your edge guards) and the timed `4096 × 4096`.
- Metric: **bandwidth**, `GB/s = 2 * rows * cols * 4 / time` (read every element
  once, write every element once — same accounting as `e06`).

## What to write (`kernel.cu`)
- `transpose_kernel` — a 2-D-block kernel that stages a tile through `__shared__`.
- `solve(d_in, d_out, rows, cols)` — pick the block/grid and launch; return 0.

Run it:

    python -m harness.runner c04 --watch

## Hints — one at a time
1. A naive transpose coalesces the read but **scatters the write** (or vice versa).
   Route the tile through a `__shared__` tile so BOTH the global read and the
   global write are coalesced — this is the lesson from `3d`.
2. Reading the staged tile back **transposed** means a **column** access of shared
   memory. On an unpadded `TILE × TILE` tile whose width is a multiple of 32, a
   whole column collapses onto a single bank — the worst-case 32-way conflict.
3. The fix is **one character** of the declaration: pad the inner dimension by 1
   (`TILE × (TILE+1)`) so column neighbors land in consecutive banks. Work out on
   paper *why* a width of `TILE+1` shears the bank mapping.
4. **Two index maps, not one.** The `(row, col)` you READ from `d_in` is not the
   `(row, col)` you WRITE to `d_out`. Derive each separately so that within a warp
   each global access walks **contiguous** addresses (that's what keeps both
   coalesced — the swap happens in shared memory, not in global addressing).
5. Guard **both** the load and the store for the `1000 × 1500` case (sizes that
   aren't multiples of `TILE`). And make sure no `__syncthreads()` sits behind a
   branch that only some threads in the block take — that hangs the block.

## Going for performance
- With the padding in, you should land a large fraction of your `c03` copy
  bandwidth; without it you'll see the conflict drag you down — flip the `+1`
  on and off and watch the GB/s move. That A/B is the whole point of the lecture.
- Next step toward the roofline: a `float4` vectorized copy moves 16 bytes per
  lane, cutting instruction overhead once the conflict is gone. (Mind that the
  edge sizes are no longer guaranteed to be a multiple of 4 — guard accordingly.)
