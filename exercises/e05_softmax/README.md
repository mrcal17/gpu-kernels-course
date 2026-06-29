# e05 — Softmax (numerically stable)

**Goal:** row-wise softmax of an `(M, N)` tensor. Combines a reduction (`max`, then
`sum`) with a map (`exp`, divide) — and forces you to get numerical stability right.

Unlocked by: `1d_softmax`.

## The spec
- Input: `x`, shape `(4096, 2048)` float32, scaled up so the *naive* version overflows.
- Output: `torch.softmax(x, dim=1)`.
- Metric: **bandwidth** (read the matrix, write the matrix).

## What to write (`kernel.py`)
- `softmax_kernel` + `softmax(x)`. One program per row. For your first version assume a
  row fits in one `BLOCK_SIZE` (pick a power of two ≥ `N`).

## Hints — one at a time
1. Per row: load it, compute `m = max(row)`, compute `e = exp(row - m)`, compute
   `s = sum(e)`, write `e / s`.
2. **Why subtract the max?** `exp(x)` overflows fp32 around `x ≈ 88`. Subtracting the
   row max makes the largest exponent exactly `exp(0) = 1`, and it's *mathematically
   identical*: `softmax(x) == softmax(x - c)` for any constant `c`. Prove that to
   yourself — it's the reason this exercise uses a ×10 scale.
3. Use masked loads/stores so lanes past `N` don't corrupt the max/sum (mask them to
   `-inf` for the max, `0` for the sum).
4. `tl.max` and `tl.sum` reduce along the block axis; `tl.exp` is your map.

## Going for performance
Memory-bound: ceiling is your `e03` bandwidth. Compare to `torch.softmax`. The single-
pass *online* softmax (compute max and sum in one sweep) is the bridge to flash
attention in `e11` — try it once the two-pass version passes.
