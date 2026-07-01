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
2. **The primitive:** within a block, Triton can perform an inclusive scan for you, so
   you don't need to hand-roll Hillis-Steele here. Search the Triton language module for
   a cumulative-sum / associative-scan primitive — the point of this exercise is
   understanding *why scan needs special handling*, not reimplementing the scan itself.
3. Store the scanned row with the same offsets/mask you loaded with.

## Validate & benchmark it yourself
The runner's `[PASS]` / `[PERF]` / `[REF]` lines are just the `1a` correctness-and-speed
loop. Here it is for this kernel, to run yourself in a scratch script:

```python
import torch, triton

torch.manual_seed(0); x = torch.randn(4096, 2048, device="cuda", dtype=torch.float32)   # as spec.py builds it

ref = torch.cumsum(x, dim=1)                # reference FIRST (torch)
out = cumsum(x)                             # your kernel
torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-3)   # a scan accumulates along the row -> reordered adds

ms     = triton.testing.do_bench(lambda: cumsum(x), warmup=25, rep=100, return_mode="median")
gbps   = 2 * x.numel() * x.element_size() / (ms * 1e-3) / 1e9   # read x, write the scan
ref_ms = triton.testing.do_bench(lambda: torch.cumsum(x, dim=1), warmup=25, rep=100, return_mode="median")
print(f"{gbps:.0f} GB/s   ({ref_ms/ms:.2f}x torch)")
```

Bandwidth-bound (read once, write once), so GB/s against the ~896 roof. The scan reorders the
running sum versus torch, so bit-equality would be the wrong test — small slack. Full table
and timing traps: `7b`.

## Going further
- **Cross-block scan:** if a row were larger than one block, a single in-block scan
  wouldn't suffice — you'd scan each block, then add each block's total as an offset to
  all later blocks (a scan of the block sums). That two-level structure is the general
  parallel-scan algorithm from `1g`. Try it with a deliberately small `BLOCK_SIZE`.
- Memory-bound: compare to `torch.cumsum`.
