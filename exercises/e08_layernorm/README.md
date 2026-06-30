# e08 — LayerNorm

**Goal:** row-wise LayerNorm with affine: normalize each row of `(M, N)` to zero
mean / unit variance, then scale and shift by per-column `weight`/`bias`. The lesson is
**fusion** — one pass instead of separate mean, var, normalize, and affine kernels.

Unlocked by: `1f_fused_norms`.

## The spec
- Inputs: `x (4096,2048)`, `weight (2048,)`, `bias (2048,)` float32. `eps = 1e-5`.
- Output: `F.layer_norm(x, (N,), weight, bias, eps)`.
- Metric: **bandwidth**.

## What to write (`kernel.py`)
- `layernorm_kernel` + `layernorm(x, weight, bias)`. One program per row (assume a row
  fits one `BLOCK_SIZE` for v1).

## Hints — one at a time
1. Per row, compute the mean, then the average squared deviation from it — derive what
   each is divided by.
2. Normalize by centering and dividing by the (eps-stabilized) standard deviation —
   Triton has a reciprocal-sqrt helper if you look for it. Think about whether eps
   belongs inside or outside the root.
3. Affine: apply the per-column scale and shift — figure out which axis `weight`/`bias`
   are indexed along, and which offsets that reuses.
4. Mask the tail; weight/bias loads use the same mask.

## Validate & benchmark it yourself
The runner's `[PASS]` / `[PERF]` / `[REF]` lines are just the `1a` correctness-and-speed
loop. Here it is for this kernel, to run yourself in a scratch script:

```python
import torch, triton
import torch.nn.functional as F

ref = F.layer_norm(x, (x.shape[1],), weight, bias, eps=1e-5)   # reference FIRST (torch)
out = layernorm(x, weight, bias)                               # your kernel
torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-3)   # per-row mean/var reorder sums

ms     = triton.testing.do_bench(lambda: layernorm(x, weight, bias), warmup=25, rep=100, return_mode="median")
gbps   = 2 * x.numel() * x.element_size() / (ms * 1e-3) / 1e9   # read x, write out (weight/bias negligible)
ref_ms = triton.testing.do_bench(lambda: F.layer_norm(x, (x.shape[1],), weight, bias, eps=1e-5), warmup=25, rep=100, return_mode="median")
print(f"{gbps:.0f} GB/s   ({ref_ms/ms:.2f}x torch)")
```

Bandwidth-bound — the win is fusing mean, variance, normalize, and the affine into one pass
over `x` instead of several. The tolerance loosens because the per-row reductions reorder
their sums. Full tolerance table and timing traps: `7b`.

## Going for performance
Memory-bound → ceiling is your `e03` bandwidth. The whole reason to write this by hand
(vs three torch ops) is to read `x` **once**: separate mean/var/normalize kernels would
read it 2–3×. Compare your GB/s to `torch`'s fused `layer_norm`. A single-pass variance
(Welford, or the `E[x²]-E[x]²` trick) avoids a second sweep — try it.
