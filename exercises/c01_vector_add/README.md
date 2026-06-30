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
2. Turn `(block, lane)` into a flat position: combine your block's index, the
   number of threads per block, and your index within the block into a single
   global index. Re-derive the idiom; don't look it up.
3. The last block runs past the end when `blockDim` doesn't divide `n`. One
   plain `if` is your mask — decide what condition keeps a thread in bounds.
   (The harness checks a ragged `n=1000` first, precisely to catch a missing
   guard.)
4. Pick threads-per-block as a whole number of warps (32s); `gridDim` is a
   ceil-division of `n` by that. The harness owns all `cudaMalloc`/`cudaMemcpy`
   — you only configure and launch.
5. Remember to `return 0` from `solve()` once you actually launch; while it
   still returns the sentinel the harness will only ever print `[TODO]`.

## Validate & benchmark it yourself
The harness does correctness + timing for you (`harness.cu`), but the methodology is the real
lesson — and in CUDA you write it by hand, with **events** on the device and a host-side
comparison. This is what `harness.cu` is doing under those `[PASS]`/`[PERF]` lines, and the
shape you'd write yourself:

```cpp
// 1. CORRECTNESS — compute a host reference, copy your result back, compare with a tolerance.
//    fp32 add is exact, so atol is tight; reductions/matmul need more slack.
for (int i = 0; i < n; ++i) {
    float ref = h_a[i] + h_b[i];
    if (std::fabs(h_out[i] - ref) > 1e-5f) { /* fail */ }
}

// 2. TIMING — CUDA events time on the DEVICE. Warm up, then bracket many launches.
const int warmup = 10, iters = 50;
for (int i = 0; i < warmup; ++i) solve(d_a, d_b, d_out, n);   // burn the cold-start cost
cudaDeviceSynchronize();

cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);
cudaEventRecord(start);
for (int i = 0; i < iters; ++i) solve(d_a, d_b, d_out, n);
cudaEventRecord(stop);
cudaEventSynchronize(stop);                                   // wait for the device

float total_ms = 0.0f;
cudaEventElapsedTime(&total_ms, start, stop);
double ms = total_ms / iters;

// 3. THROUGHPUT — bytes moved ÷ time. Vector add moves 3·n·4 bytes (read a, read b, write out).
double gbps = 3.0 * n * sizeof(float) / (ms * 1e-3) / 1e9;
```

Two things the events buy you that a host timer can't: they measure **device** time (no
async-launch lie — a CPU `clock()` around the launch would time the *launch*, not the kernel),
and warmup hides the one-time cost of the first launch. Then judge `gbps` against the ~896
roof. The Triton track does all of this with one line — `triton.testing.do_bench` — and the
full method (median over samples, the L2-flush trap, choosing tolerances) is the reference
card, `7b`.

## Going for performance
This kernel is pure memory traffic, so the ceiling is bandwidth, not math.
Once it passes, try a couple of block sizes (256, 512, 1024) and watch the
GB/s. Contiguous, coalesced accesses get you most of the roofline for free; if
you land far below ~896 GB/s, revisit coalescing in `3a`. There's no reuse to
exploit here — every byte is touched exactly once — so don't expect tiling or
shared memory to help. The interesting knob is just occupancy via block size.
