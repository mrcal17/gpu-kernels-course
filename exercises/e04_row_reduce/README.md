# e04 — Row reduction

**Goal:** sum each row of an `(M, N)` tensor → an `(M,)` vector. Your first
**reduction** and your first 2-D indexing.

Unlocked by: `1c_reductions`.

## The spec
- Input: `x`, shape `(4096, 4096)` float32. Output: `x.sum(dim=1)`, shape `(4096,)`.
- Metric: **bandwidth** (you read all `M*N` elements once).

## What to write (`kernel.py`)
- `row_sum_kernel` + `row_sum(x)`. One program per row.

## Hints — one at a time
1. **Decomposition:** launch `M` programs, program `i` owns row `i`. The grid is 1-D
   with `M` entries.
2. **Indexing a row:** you need the row stride and column stride of `x` to compute the
   address of element `(i, j)`. Pass them in (`x.stride(0)`, `x.stride(1)`).
3. **If `N > BLOCK_SIZE`:** you can't load the whole row at once. Loop over the row in
   chunks of `BLOCK_SIZE`, masking the final partial chunk, accumulating a running
   partial.
4. **Block → scalar:** Triton reduces a vector to a scalar with a reduction op along an
   axis (look for `tl.sum`). Add each chunk's partial into your accumulator.
5. **Store** one value per program.

## Going for performance
This is memory-bound: you touch every element once, so the ceiling is your `e03`
bandwidth. Compare against `torch`'s own `sum(dim=1)` (the runner prints it). If you're
much slower, your row accesses may not be coalesced — adjacent programs should read
adjacent columns *within a step* so the warp's lanes hit contiguous addresses.
