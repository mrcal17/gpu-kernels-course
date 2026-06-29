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
1. **Two program ids:** `tl.program_id(0)` and `tl.program_id(1)` pick the tile's row
   and column blocks.
2. **Tile offsets:** build `rows = pid_m*BLOCK_M + arange(0, BLOCK_M)` and similarly for
   cols; combine into 2-D offset arrays with broadcasting (`rows[:, None]`,
   `cols[None, :]`).
3. **Two strides each:** address into `x` with x's strides, into `out` with out's
   strides — and the row/col roles swap between them. That swap *is* the transpose.
4. **Edge mask:** `(rows < M) & (cols < N)` for ragged tiles.

## Going for performance — the coalescing puzzle
A naive transpose makes the writes strided (each lane writes a different row of `out`),
which serializes them and tanks bandwidth. The classic fix is to stage the tile through
**shared memory** so both the global read and the global write are coalesced — that's
what you'll do by hand in CUDA exercise `c04`. In Triton, experiment with tile shapes
and see how close to your `e03` copy bandwidth you can get; transpose will trail it.
