# e06 — Transpose

**Goal:** `out = x.T`, contiguous. Your first **2-D tiling** and the cleanest lesson in
coalescing: you cannot make both the read and the write contiguous, so one side fights
you.

Unlocked by: `1e_tiling_matmul` (the 2-D-indexing warmup before matmul).

## The spec
- Input: `x`, `(4096, 4096)` float32. Output: `x.t().contiguous()`, `(4096, 4096)`.
- Metric: **bandwidth**.

## What to write (`kernel.py`)
- `transpose_kernel` + `transpose(x)`. Use a 2-D grid; each program owns a
  `BLOCK_M × BLOCK_N` tile.

## Hints — one at a time
1. **Two program ids:** a 2-D launch grid gives each program two index coordinates, one
   per grid axis. Find the Triton call that returns the current program's index along a
   given axis, and use it twice to identify which row-block and which col-block this
   program owns.
2. **Tile offsets:** turn each block index into the range of element indices that block
   covers (offset the block's start by a `0..BLOCK` range). Then form a 2-D index grid
   from the two 1-D ranges by broadcasting one along rows and the other along columns.
3. **Two strides each:** address into `x` with x's strides, into `out` with out's
   strides — and the row/col roles swap between them. That swap *is* the transpose.
4. **Edge mask:** tiles at the matrix edge can run past the real bounds in either
   dimension. Build a 2-D boolean mask that is true only where both the row and the col
   index are still in range, and pass it to the load/store.

## Validate & benchmark it yourself
The runner's `[PASS]` / `[PERF]` / `[REF]` lines are just the `1a` correctness-and-speed
loop. Here it is for this kernel, to run yourself in a scratch script:

```python
import torch, triton

torch.manual_seed(0); x = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)   # as spec.py builds it

ref = x.t().contiguous()                    # reference FIRST (torch)
out = transpose(x)                          # your kernel
torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)   # transpose relocates exact values -> 0/0

ms     = triton.testing.do_bench(lambda: transpose(x), warmup=25, rep=100, return_mode="median")
gbps   = 2 * x.numel() * x.element_size() / (ms * 1e-3) / 1e9   # read x, write x-transposed
ref_ms = triton.testing.do_bench(lambda: x.t().contiguous(), warmup=25, rep=100, return_mode="median")
print(f"{gbps:.0f} GB/s   ({ref_ms/ms:.2f}x torch)")
```

Bandwidth-bound, but the challenge is coalescing, not arithmetic: a naive transpose makes
either the read or the write strided, so GB/s against the ~896 roof is really a
coalescing / bank-conflict grade (`3d`). Exact values -> `0/0`. Full table and traps: `7b`.

## Going for performance — the coalescing puzzle
A naive transpose makes the writes strided (each lane writes a different row of `out`),
which serializes them and tanks bandwidth. The classic fix is to stage the tile through
**shared memory** so both the global read and the global write are coalesced — that's
what you'll do by hand in CUDA exercise `c04`. In Triton, experiment with tile shapes
and see how close to your `e03` copy bandwidth you can get; transpose will trail it.
