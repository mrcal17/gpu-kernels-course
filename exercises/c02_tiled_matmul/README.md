# c02 — Tiled matmul (shared memory by hand)

**Goal:** `C = A @ B` for square `N × N × N`, row-major, fp32. This is the
by-hand version of `e07` — Triton's `tl.dot` and automatic tiling are gone, and
you stage the tiles into `__shared__` yourself. Tiling is what raises arithmetic
intensity enough to stop being bandwidth-bound — connect it back to the roofline
in `0d`.

Maps to lecture `3b_shared_tiling`.

## The spec
- Inputs: `A (N,N)`, `B (N,N)` float32 (device pointers, deterministic). Output: `A @ B`.
- The harness checks **two** sizes: `N = 257` first (correctness only — this is the
  one that catches a missing ragged-tail guard), then `N = 512` (correctness, then
  timed). `512` is a multiple of both tile 16 and tile 32.
- Metric: **FLOP/s** = `2·N³ / time`, reported as TFLOP/s.

## What to write (`kernel.cu`)
- `matmul_tiled` — the `__global__` kernel: each block owns one output tile,
  each thread owns one output element of it.
- `solve(d_A, d_B, d_C, N)` — the host launcher (signature already in the stub).
  Configure the grid/block and launch; return `0`.

Run it (auto re-runs on save):

```
python -m harness.runner c02 --watch
```

While `solve()` returns the sentinel you'll see `[TODO] ...`. Once it launches a
real kernel you'll get `[PASS] correct` / `[FAIL] ...` and a `[PERF]` line.

## Hints — one at a time
1. **Who owns what.** Each block owns one output tile; each thread owns one
   output element of that tile. The reuse you're buying: load a `TILE × TILE`
   patch once, then use every element of it `TILE` times (lecture 3b).
2. **Stage, sync, reuse — in that order.** Cooperatively load a tile into
   `__shared__`, hit a barrier, then compute *entirely* from on-chip data. No
   thread touches global memory during the inner-product loop.
3. **Two barriers per K-step, and reason out why.** You need `__syncthreads()`
   *twice* each iteration. One stops a thread from reading a half-loaded tile;
   the other stops a fast warp from clobbering a tile a slow warp is still
   reading. Say out loud which is which before you place them.
4. **The crux is the global address this thread loads.** For tile-step `t`, which
   row/col of `A` and of `B` does this tile touch? Work it out in raw
   `row*N + k` arithmetic — that derivation is the exercise; don't reach for a
   formula.
5. **Barriers must not be divergent.** Every thread in the block has to reach
   every `__syncthreads()` — a barrier inside a branch some threads skip
   deadlocks. So put the ragged-`N` guard on the *final store*, not around the
   loads or the barriers. (This is exactly what the `N = 257` size is testing.)

## Going for performance
- **Tile size is a resource budget.** Two `TILE × TILE` float tiles live in shared
  memory per block, against the ~48 KB/block and ~100 KB/SM limits. Bigger tiles
  buy more reuse — until you run out of shared memory or occupancy collapses.
  Try the obvious power-of-two edges and watch the `[PERF]` number move.
- You won't touch cuBLAS here — the point is to *feel* where the reuse comes from
  with the staging spelled out to the metal. `e07` is the same math handed to
  Triton; compare the two once this passes.
