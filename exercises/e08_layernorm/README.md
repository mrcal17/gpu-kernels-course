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
1. Per row: `mean = sum(row)/N`, then `var = sum((row-mean)**2)/N`.
2. Normalize: `(row - mean) * rsqrt(var + eps)`. Triton has `tl.rsqrt` (or use
   `1/tl.sqrt`).
3. Affine: multiply by `weight` and add `bias`, both indexed by **column** (the same
   offsets you used to load the row).
4. Mask the tail; weight/bias loads use the same mask.

## Going for performance
Memory-bound → ceiling is your `e03` bandwidth. The whole reason to write this by hand
(vs three torch ops) is to read `x` **once**: separate mean/var/normalize kernels would
read it 2–3×. Compare your GB/s to `torch`'s fused `layer_norm`. A single-pass variance
(Welford, or the `E[x²]-E[x]²` trick) avoids a second sweep — try it.
