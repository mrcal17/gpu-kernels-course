# e04 — Row reduction

**Goal:** sum each row of an `(M, N)` tensor → an `(M,)` vector. Your first
**reduction** and your first 2-D indexing.

Unlocked by: `1c_reductions`.

## The spec
- Input: `x`, shape `(4096, 4096)` float32. Output: `x.sum(dim=1)`, shape `(4096,)`.
- Metric: **bandwidth** (you read all `M*N` elements once).

## What to write (`kernel.py`)
- `row_sum_kernel` + `row_sum(x)`. One program per row.

## Hints — one at a time
1. **Decomposition:** launch one program per row, with program `i` owning row `i` —
   decide the grid's shape and size from that.
2. **Indexing a row:** you need the row stride and column stride of `x` to compute the
   address of element `(i, j)`. Pass them in — PyTorch tensors expose their
   per-dimension strides through a method; find it.
3. **If the row is longer than one tile:** you can't load the whole row at once. Loop
   over the row in tile-sized chunks, masking the final partial chunk, accumulating a
   running partial.
4. **Block → scalar:** Triton reduces a vector to a scalar with a reduction op along an
   axis — look in `tl` for the function that sums a block. Add each chunk's partial
   into your accumulator.
5. **Store** one value per program.

## Validate & benchmark it yourself
The runner's `[PASS]` / `[PERF]` / `[REF]` lines are just the `1a` correctness-and-speed
loop. Here it is for this kernel, to run yourself in a scratch script:

```python
import torch, triton

ref = x.sum(dim=1)                          # reference FIRST (torch)
out = row_sum(x)                            # your kernel
torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-3)   # a reduction reorders sums -> small slack

ms     = triton.testing.do_bench(lambda: row_sum(x), warmup=25, rep=100, return_mode="median")
gbps   = (x.numel() + x.shape[0]) * x.element_size() / (ms * 1e-3) / 1e9   # read M·N, write M
ref_ms = triton.testing.do_bench(lambda: x.sum(dim=1), warmup=25, rep=100, return_mode="median")
print(f"{gbps:.0f} GB/s   ({ref_ms/ms:.2f}x torch)")
```

Bandwidth-bound, so judge GB/s against the ~896 roof. The tolerance loosens here (vs. `0/0`
for an exact add) because summing a row reorders floating-point adds — bit-equality would be
the wrong test. Full tolerance table and the timing traps: `7b`.

## Going for performance
This is memory-bound: you touch every element once, so the ceiling is your `e03`
bandwidth. Compare against `torch`'s own `sum(dim=1)` (the runner prints it). If you're
much slower, your row accesses may not be coalesced — adjacent programs should read
adjacent columns *within a step* so the warp's lanes hit contiguous addresses.
