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

## Validate & benchmark it yourself
The runner's `[PASS]` / `[PERF]` / `[REF]` lines are just the `1a` correctness-and-speed
loop. Here it is for this kernel, to run yourself in a scratch script:

```python
import torch, triton

ref = x * torch.sigmoid(x)                  # reference FIRST (torch)
out = silu(x)                               # your kernel
torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)   # a couple of ops -> mild rounding

ms     = triton.testing.do_bench(lambda: silu(x), warmup=25, rep=100, return_mode="median")
gbps   = 2 * x.numel() * x.element_size() / (ms * 1e-3) / 1e9   # read x, write out
ref_ms = triton.testing.do_bench(lambda: x * torch.sigmoid(x), warmup=25, rep=100, return_mode="median")
print(f"{gbps:.0f} GB/s   ({ref_ms/ms:.2f}x torch)")
```

Bandwidth-bound, so judge GB/s against the ~896 roof — the whole point of fusing is to do all
the math in one pass over the data instead of one pass per op. The tolerance is loose-ish
because `sigmoid` rounds; an exact copy would be `0/0`. Full table and timing traps: `7b`.

## Going for performance
Same ceiling as `e01` (memory-bound): the extra math is free because it overlaps with
memory. If your GB/s matches `e01`, you've proven that fusing arithmetic costs nothing
when you're bandwidth-bound — the core reason fused kernels win.
