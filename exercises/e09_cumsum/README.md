# e09 — Cumsum / scan (advanced)

**Goal:** row-wise inclusive prefix sum: `out[i,j] = sum(x[i, 0..j])`. Scan is the
hardest of the parallel patterns — every output depends on a running total, so it
doesn't fall to the same trick as a reduction.

Unlocked by: `1g_scan` (optional/advanced).

## The spec
- Input: `x (4096, 2048)` float32. Output: `torch.cumsum(x, dim=1)`.
- Metric: **bandwidth** (read the matrix, write the matrix). A row fits one block.

## What to write (`kernel.py`)
- `cumsum_kernel` + `cumsum(x)`. One program per row.

## Hints — one at a time
1. **Mask matters here:** masked-out tail lanes must load as `0`, or they'd shift the
   running sum.
2. **The primitive:** within a block, Triton can do an inclusive scan for you — look for
   `tl.cumsum` (or the more general `tl.associative_scan`). Using it is fair; the point
   of this exercise is understanding *why scan needs special handling*, not hand-rolling
   Hillis-Steele in Triton.
3. Store the scanned row with the same offsets/mask you loaded with.

## Going further
- **Cross-block scan:** if a row were larger than one block, a single `tl.cumsum`
  wouldn't suffice — you'd scan each block, then add each block's total as an offset to
  all later blocks (a scan of the block sums). That two-level structure is the general
  parallel-scan algorithm from `1g`. Try it with a deliberately small `BLOCK_SIZE`.
- Memory-bound: compare to `torch.cumsum`.
