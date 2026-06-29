# e01 — Vector add

**Goal:** `out = a + b` for two big 1-D float32 tensors, with a Triton kernel you
write yourself. This is the "hello world" that forces every core idea from `1a`.

Unlocked by: lecture `1a_triton_model` (and the execution model in `0b`).

## The spec
- Inputs: `a`, `b` — each `N = 2**24` float32 on CUDA.
- Output: `a + b`, elementwise. Must match torch exactly.
- Metric: **bandwidth** (memory-bound — 2 reads + 1 write). Aim for a large fraction
  of your card's ~896 GB/s.

## What to write (`kernel.py`)
1. A `@triton.jit` kernel `vector_add_kernel(...)`.
2. A launch wrapper `vector_add(a, b)` that allocates the output, computes the grid,
   and launches the kernel.

## Run it
```bash
python -m harness.runner e01 --watch
```
Save `kernel.py` and watch it check correctness + print GB/s.

## Hints — peek only when stuck (one at a time)
1. **Which block am I?** Each program instance handles one contiguous chunk. There's
   a function that gives you the current program's index along an axis.
2. **Which elements?** From your program index and the block size, build the range of
   element offsets this program owns. There's a `tl.arange`-style helper.
3. **The ragged tail.** `N` may not divide evenly by your block size, so the last
   program would read/write out of bounds. Build a boolean *mask* and pass it to both
   the loads and the store.
4. **Load, add, store.** Two masked loads, one add (in registers), one masked store.
5. **The grid.** You need enough programs to cover all `N` elements — a ceiling
   division of `N` by your block size. Look for a Triton ceil-div helper.
6. **BLOCK_SIZE.** Make it a `tl.constexpr`. A power of two (1024–4096) is conventional
   — think about why (warps, coalescing).

## Going for performance
This kernel is pure memory traffic, so the ceiling is bandwidth, not math. Once it
passes, try a few `BLOCK_SIZE` values and watch the GB/s. You should get within
shouting distance of `torch`'s own add. If you're far below ~896 GB/s, revisit
`0c`/`1b` (coalescing) — though for a simple contiguous add you'll get most of it for
free.
