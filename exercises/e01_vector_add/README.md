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
   element offsets this program owns. There's a helper that builds a contiguous range
   of indices — find it.
3. **The ragged tail.** `N` may not divide evenly by your block size, so the last
   program would read/write out of bounds. Build a boolean *mask* and pass it to both
   the loads and the store.
4. **Load, add, store.** Two masked loads, one add (in registers), one masked store.
5. **The grid.** You need enough programs to cover all `N` elements — a ceiling
   division of `N` by your block size. Look for a Triton ceil-div helper.
6. **The tile size.** Make it a compile-time constant. A power of two is conventional
   — think about why (warps, coalescing).

## Validate & benchmark it yourself
When your kernel passes, the runner prints three lines — `[PASS]`, `[PERF] … GB/s`, and
`[REF] torch … (N.NNx your time)`. Those aren't magic; they're the correctness-and-speed
loop from `1a`, and the skill worth keeping is running it yourself. Drop this into a scratch
script (real GPU, `import torch, triton`) and you've reproduced the harness:

```python
import torch, triton

torch.manual_seed(0)                        # reproducible inputs
N = 1 << 24
a = torch.randn(N, device="cuda"); b = torch.randn(N, device="cuda")

# 1. CORRECTNESS — reference FIRST, then your kernel, then compare with a tolerance.
ref = a + b                                 # the trusted answer
out = vector_add(a, b)                      # your kernel
torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)   # add is exact -> demand 0/0
print("[PASS] correct")

# 2. SPEED — warmup + many reps + median, all handled by do_bench -> milliseconds.
ms = triton.testing.do_bench(lambda: vector_add(a, b), warmup=25, rep=100, return_mode="median")

# 3. THROUGHPUT — bytes moved ÷ time. Vector add moves 3·N·4 bytes (read a, read b, write out).
gbps = 3 * a.numel() * a.element_size() / (ms * 1e-3) / 1e9
print(f"[PERF] {ms:.3f} ms   {gbps:.0f} GB/s")

# 4. THE TORCH BAR — time the reference the same way; ratio > 1 means you beat it.
ref_ms = triton.testing.do_bench(lambda: a + b, warmup=25, rep=100, return_mode="median")
print(f"[REF]  torch {ref_ms:.3f} ms  ({ref_ms/ms:.2f}x your time)")
```

**Read the number, don't just collect it.** A contiguous add is memory-bound, so judge your
GB/s against the **~896 GB/s** roof: 90%+ means you're saturating DRAM (done — there's no
math to optimize); far below at full occupancy means uncoalesced loads (`0c`/`1b`). And why
`atol=rtol=0`? fp32 add reorders nothing, so a correct kernel matches torch **bit for bit** —
a single mismatched element is a real bug, not rounding. Other ops loosen this; the full
tolerance table and the timing traps are the reference card, `7b`.

## Going for performance
This kernel is pure memory traffic, so the ceiling is bandwidth, not math. Once it
passes, try a few `BLOCK_SIZE` values and watch the GB/s. You should get within
shouting distance of `torch`'s own add. If you're far below ~896 GB/s, revisit
`0c`/`1b` (coalescing) — though for a simple contiguous add you'll get most of it for
free.
