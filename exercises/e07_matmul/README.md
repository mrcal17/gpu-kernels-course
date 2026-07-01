# e07 — Tiled matmul (GEMM)

**Goal:** `C = A @ B` for `(M,K) @ (K,N)`. The centerpiece of the whole course. Tiling
is what raises arithmetic intensity enough to become compute-bound instead of
bandwidth-bound — connect this back to the roofline in `0d`.

Unlocked by: `1e_tiling_matmul`.

## The spec
- Inputs: `A (1024,1024)`, `B (1024,1024)` float32. Output: `A @ B`.
- Metric: **FLOP/s** = `2*M*N*K / time`. Compare to torch (cuBLAS) — you won't beat it,
  but see how close you get.

## What to write (`kernel.py`)
- `matmul_kernel` + `matmul(a, b)`. 2-D grid; each program computes one
  `BLOCK_M × BLOCK_N` tile of `C`.

## Hints — one at a time
1. **The tile a program owns:** each program computes one output tile of `C`; the two
   program ids pick which tile.
2. **Accumulator:** allocate a register accumulator sized to the output tile this
   program owns, initialized to 0, in an accuracy-preserving dtype. You add into it
   across the K loop, then write once at the end.
3. **The K loop:** walk `K` one tile-width at a time. Each iteration: load the slice of
   `A` and the slice of `B` that line up along `K` for this step, and do a
   block-times-block multiply-accumulate — let the matmul definition tell you which
   tile dimension pairs with `K` on each operand. Triton has a primitive that
   multiplies two 2-D blocks and accumulates into your accumulator — look for it in
   the language reference.
4. **Advance pointers** along the K dimension each iteration (mind which stride that is
   for `A` vs `B`).
5. **Masks:** mask the K tail in the loop, and the M/N edges on the final store.

## Validate & benchmark it yourself
The runner's `[PASS]` / `[PERF]` / `[REF]` lines are just the `1a` correctness-and-speed
loop. Here it is for this kernel, to run yourself in a scratch script:

```python
import torch, triton
torch.backends.cuda.matmul.allow_tf32 = False   # fair fp32 reference (torch uses TF32 by default)

torch.manual_seed(0); a = torch.randn(1024, 1024, device="cuda", dtype=torch.float32); b = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)   # as spec.py builds them

ref = a @ b                                 # reference FIRST (torch)
out = matmul(a, b)                          # your kernel
torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-2)   # a K-long inner product reorders many adds -> loose

M, K = a.shape; N = b.shape[1]
ms     = triton.testing.do_bench(lambda: matmul(a, b), warmup=25, rep=100, return_mode="median")
tflops = 2 * M * N * K / (ms * 1e-3) / 1e12    # 2·M·N·K: one multiply + one add per inner-product term
ref_ms = triton.testing.do_bench(lambda: a @ b, warmup=25, rep=100, return_mode="median")
print(f"{tflops:.1f} TFLOP/s   ({ref_ms/ms:.2f}x torch)")
```

This kernel is **compute-bound**, so compare TFLOP/s to the fp32 / tensor-core peak, not the
bandwidth roof. Two subtleties: disable TF32 so torch's reference is the *same* precision you
target (otherwise it disagrees for reasons that aren't your bug), and expect `ref_ms/ms < 1` —
beating cuBLAS is hard. Full tolerance table and traps: `7b`.

## Going for performance
- The block-matmul primitive maps to tensor cores — this is where you first feel them.
- **Precision note:** on fp32 inputs, the block-matmul primitive uses TF32
  tensor-core math by default, which rounds the mantissa. The spec's reference
  runs in true fp32 (`allow_tf32=False`), so to match it exactly, note that the
  primitive accepts an `input_precision` argument — you'll want full fp32
  (`'ieee'`). The loose tolerances here (`atol=1e-1`) also accept the faster
  default TF32, so either works — just know which one you picked.
- Tile sizes are everything: experiment with power-of-two `BLOCK_M`/`BLOCK_N` in the
  tens-to-low-hundreds and a smaller `BLOCK_K`, and watch how FLOP/s responds.
- The reuse story: each loaded A/B tile is used `BLOCK_N`/`BLOCK_M` times. Bigger tiles
  = more reuse = higher intensity, until you run out of registers/shared memory (the
  occupancy limiters from `0d`). `e10` automates this search with `@triton.autotune`.
