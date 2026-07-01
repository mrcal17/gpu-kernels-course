# e10 — Autotuned matmul (mask any shape, let the tuner pick the tile)

**Goal:** `C = A @ B` again — but now the shapes are ragged (none of `M`, `N`, `K`
is a multiple of a nice block size) and you stop hand-picking the tile. Two new
jobs on top of `e07`: be correct for **any** shape, and hand the tile choice to
`@triton.autotune`.

Unlocked by: `e07_matmul`, `2a` (autotuning).

## The spec
- Inputs: `A`, `B` float32 with deliberately awkward dimensions — `M`, `N`, **and**
  `K` are all ragged, so every tile hangs off an edge. Output: `A @ B`.
- Metric: **FLOP/s** = `2*M*N*K / time`. Same metric as `e07`.

## What to write (`kernel.py`)
- `matmul_kernel` + `matmul(a, b)`, plus an `@triton.autotune` decorator above the
  kernel with a `configs=[...]` menu **you** design and a `key=[...]`.
- Each program still computes one `BLOCK_M × BLOCK_N` tile of `C` — but now the
  block sizes come from the chosen autotune config, not constants you wrote.

## Hints — one at a time
1. This is `e07` with two new jobs: **(1)** be correct for any shape, and **(2)**
   let `@triton.autotune` pick the tile. Do the masking **first** — a fast kernel
   that is wrong on ragged shapes is a bug, not an optimization.
2. **Masking in two dimensions plus the K loop.** A program owns a band of rows and
   a band of columns that can hang off the bottom/right edge of `C`. Guard every
   load and the final store so off-edge lanes read and write nothing.
3. **The contraction dimension can be ragged too.** When `K` is not a multiple of
   your K-block, the last K-chunk is partial — mask it and load the missing lanes
   as the additive identity so they add *exactly* nothing to the dot product.
4. **The tile sizes are no longer constants you wrote** — the autotuner supplies
   them. That changes how the launch grid is computed: the grid has to ask the
   chosen config how big the tiles are. Think about where the block sizes come
   from at launch time (hint: the grid is a function of `META`).
5. **Designing the config menu.** Don't brute-force the Cartesian product. Propose
   a small, diverse set — a couple of big compute-bound tiles, a couple of small
   high-occupancy ones, with varied `num_warps` and `num_stages`. Reason about the
   shared-memory budget and occupancy (lecture `2a`) for each candidate.
6. **The `key=[...]` argument** tells the autotuner which arguments, when they
   change, should trigger a re-tune. For a matmul whose optimum depends on the
   shape, list the dimensions that define the problem size.

## Validate & benchmark it yourself
The runner's `[PASS]` / `[PERF]` / `[REF]` lines are just the `1a` correctness-and-speed
loop. Here it is for this kernel, to run yourself in a scratch script:

```python
import torch, triton
torch.backends.cuda.matmul.allow_tf32 = False   # fair fp32 reference

torch.manual_seed(0); a = torch.randn(1023, 1025, device="cuda", dtype=torch.float32); b = torch.randn(1025, 769, device="cuda", dtype=torch.float32)   # as spec.py builds them (ragged M/K/N)

ref = a @ b                                 # reference FIRST (torch)
out = matmul(a, b)                          # your kernel
torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-2)   # long inner product -> loose

M, K = a.shape; N = b.shape[1]
ms     = triton.testing.do_bench(lambda: matmul(a, b), warmup=25, rep=100, return_mode="median")
tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
ref_ms = triton.testing.do_bench(lambda: a @ b, warmup=25, rep=100, return_mode="median")
print(f"{tflops:.1f} TFLOP/s   ({ref_ms/ms:.2f}x torch)")
```

Same compute-bound matmul math as `e07`, but the shapes are deliberately **ragged**
(1023×1025×769) so your masking is exercised, and `@triton.autotune` means `do_bench` reports
your *best* config's time. Compare TFLOP/s to the tensor-core peak; TF32 off for a fair torch
bar. Full tolerance table and traps: `7b`.

## Going for performance
- **Precision note (same as `e07`):** on fp32 inputs `tl.dot` uses TF32 tensor
  cores by default, which rounds the mantissa. The loose tolerances here
  (`atol=1e-1`) accept that — but if you want to match the true-fp32 reference more
  tightly, `tl.dot` takes an `input_precision` argument (`'ieee'`). Pick one and
  know which you picked.
- Autotune caches its choice per `key`, so the first launch pays for the search and
  later launches reuse the winner. Keep the menu small and diverse — a huge menu
  just makes the first call slow without finding a better tile.
- Bigger tiles = more reuse = higher arithmetic intensity, until you run out of
  registers / shared memory and occupancy drops. That tension is exactly what the
  tuner is exploring for you.
