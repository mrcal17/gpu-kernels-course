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
1. **The tile a program owns:** 2-D program ids → a `BLOCK_M × BLOCK_N` block of `C`.
2. **Accumulator:** allocate a `(BLOCK_M, BLOCK_N)` fp32 accumulator initialized to 0.
   You add into it across the K loop, then write once at the end.
3. **The K loop:** step `k` from `0` to `K` by `BLOCK_K`. Each iteration: load an
   `(BLOCK_M, BLOCK_K)` tile of `A` and a `(BLOCK_K, BLOCK_N)` tile of `B`, and do a
   block-times-block multiply-accumulate. Triton has a primitive that multiplies two
   2-D blocks and accumulates — look for `tl.dot`.
4. **Advance pointers** by `BLOCK_K` along the K dimension each iteration (mind which
   stride that is for `A` vs `B`).
5. **Masks:** mask the K tail in the loop, and the M/N edges on the final store.

## Going for performance
- `tl.dot` maps to tensor cores — this is where you first feel them.
- **Precision note:** on fp32 inputs, `tl.dot` uses TF32 tensor-core math by
  default, which rounds the mantissa. The spec's reference runs in true fp32
  (`allow_tf32=False`), so to match it exactly use
  `tl.dot(a, b, input_precision='ieee')`. The loose tolerances here
  (`atol=1e-1`) also accept the faster default TF32, so either works — just
  know which one you picked.
- Tile sizes are everything: try `BLOCK_M=BLOCK_N=64 or 128`, `BLOCK_K=32 or 64`.
- The reuse story: each loaded A/B tile is used `BLOCK_N`/`BLOCK_M` times. Bigger tiles
  = more reuse = higher intensity, until you run out of registers/shared memory (the
  occupancy limiters from `0d`). `e10` automates this search with `@triton.autotune`.
