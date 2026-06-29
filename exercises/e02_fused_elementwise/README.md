# e02 — Fused elementwise (SiLU)

**Goal:** `out = x * sigmoid(x)` in one kernel. The lesson is **fusion**: several
arithmetic ops per element, but still exactly one global read and one global write.

Unlocked by: `1a_triton_model`. Builds directly on `e01`.

## The spec
- Input: `x`, `N = 2**24` float32 on CUDA.
- Output: `x * sigmoid(x)`, elementwise.
- Metric: **bandwidth** (1 read + 1 write).

## What to write (`kernel.py`)
- `silu_kernel` (`@triton.jit`) and the `silu(x)` launch wrapper.

## Run
```bash
python -m harness.runner e02 --watch
```

## Hints — one at a time
1. Start from your `e01` skeleton: same program-id → offsets → mask → load pattern.
2. The whole point is to compute `sigmoid(x)` **in registers** after the load. What is
   `sigmoid` in terms of `exp`? Triton's language module has the exponential you need.
3. Don't allocate a temporary for `sigmoid(x)` — just write the expression; the
   compiler keeps it in registers.
4. One masked store at the end.

## Going for performance
Same ceiling as `e01` (memory-bound): the extra math is free because it overlaps with
memory. If your GB/s matches `e01`, you've proven that fusing arithmetic costs nothing
when you're bandwidth-bound — the core reason fused kernels win.
