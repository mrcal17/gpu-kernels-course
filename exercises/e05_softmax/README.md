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
1. Per row, the stable recipe is: find the row's maximum, subtract it before
   exponentiating, sum the exponentials, then normalize each entry by that sum.
   Translate each of those four steps into code yourself.
2. **Why subtract the max?** `exp(x)` overflows fp32 around `x ≈ 88`. Subtracting the
   row max makes the largest exponent exactly `exp(0) = 1`, and it's *mathematically
   identical*: `softmax(x) == softmax(x - c)` for any constant `c`. Prove that to
   yourself — it's the reason this exercise uses a ×10 scale.
3. Use masked loads/stores so lanes past `N` don't corrupt the max/sum. Masked-out
   tail lanes must not pollute the reduction — give them whatever neutral value leaves
   a max unchanged, and a different neutral value that leaves a sum unchanged. Work out
   what each identity is.
4. There are block-level reduction primitives that collapse your loaded row to a single
   max and a single sum, and an elementwise exponential that maps the whole vector at
   once — find them in the Triton language reference.

## Validate & benchmark it yourself
The runner's `[PASS]` / `[PERF]` / `[REF]` lines are just the `1a` correctness-and-speed
loop. Here it is for this kernel, to run yourself in a scratch script:

```python
import torch, triton

ref = torch.softmax(x, dim=1)               # reference FIRST (torch)
out = softmax(x)                            # your kernel
torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)   # exp + normalize over a row; tight but nonzero

ms     = triton.testing.do_bench(lambda: softmax(x), warmup=25, rep=100, return_mode="median")
gbps   = 2 * x.numel() * x.element_size() / (ms * 1e-3) / 1e9   # read the matrix, write the matrix
ref_ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=1), warmup=25, rep=100, return_mode="median")
print(f"{gbps:.0f} GB/s   ({ref_ms/ms:.2f}x torch)")
```

Bandwidth-bound (you stream the matrix in and back out once), so judge GB/s against the ~896
roof. The tolerance is tight because the online-max trick keeps softmax numerically stable —
but still nonzero, since `exp`/sum reorder. Full table and timing traps: `7b`.

## Going for performance
Memory-bound: ceiling is your `e03` bandwidth. Compare to `torch.softmax`. The single-
pass *online* softmax (compute max and sum in one sweep) is the bridge to flash
attention in `e11` — try it once the two-pass version passes.
