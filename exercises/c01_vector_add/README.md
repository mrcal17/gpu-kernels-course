# c01 — Vector add (the nvcc hello-world)

**Goal:** `out = a + b` for two big 1-D float32 arrays, with a CUDA C++ kernel
you write yourself. This is the "hello world" that forces every core idea from
`3a_cuda_model`: the launch geometry, the global-index idiom, and the tail
guard.

Unlocked by: lecture `3a_cuda_model`.

## The spec
- Inputs: `a`, `b` — each `n = 2**24` float32, already on the device.
- Output: `a + b`, elementwise. Must match the host reference exactly
  (`atol=1e-5`; fp32 add is exact, so the slack absorbs nothing special).
- Metric: **bandwidth** (memory-bound — 2 reads + 1 write). Aim for a large
  fraction of your card's ~896 GB/s.
- `solve()` receives **device** pointers, so `[PERF]` times only the add, not
  the PCIe copies. The harness owns every `cudaMalloc`/`cudaMemcpy`.

## What to write (`kernel.cu`)
1. The `__global__` kernel `vector_add(a, b, out, n)` — the body is the code of
   **one thread**.
2. `solve(...)` — choose the launch geometry and fire the kernel. Exact
   signature (already stubbed for you):
   ```cpp
   extern "C" int solve(const float* d_a, const float* d_b, float* d_out, int n)
   ```
   Return `0` once you launch; it ships returning the sentinel `77`.

## Run it
```bash
python -m harness.runner c01 --watch
```
Save `kernel.cu` and watch it check correctness (a small ragged size first,
then the big one) + print GB/s.

## Hints — peek only when stuck (one at a time)
1. The body you write is the code of **one** thread, not a loop over `n` — the
   launch *is* the loop (lecture `3a`).
2. Turn `(block, lane)` into a flat position: combine `blockIdx.x`,
   `blockDim.x`, and `threadIdx.x` into a single global index. Re-derive the
   idiom; don't look it up.
3. The last block runs past the end when `blockDim` doesn't divide `n`. One
   plain `if` is your mask — decide what condition keeps a thread in bounds.
   (The harness checks a ragged `n=1000` first, precisely to catch a missing
   guard.)
4. Pick threads-per-block as a whole number of warps (32s); `gridDim` is a
   ceil-division of `n` by that. The harness owns all `cudaMalloc`/`cudaMemcpy`
   — you only configure and launch.
5. Remember to `return 0` from `solve()` once you actually launch; while it
   still returns the sentinel the harness will only ever print `[TODO]`.

## Going for performance
This kernel is pure memory traffic, so the ceiling is bandwidth, not math.
Once it passes, try a couple of block sizes (256, 512, 1024) and watch the
GB/s. Contiguous, coalesced accesses get you most of the roofline for free; if
you land far below ~896 GB/s, revisit coalescing in `3a`. There's no reuse to
exploit here — every byte is touched exactly once — so don't expect tiling or
shared memory to help. The interesting knob is just occupancy via block size.
