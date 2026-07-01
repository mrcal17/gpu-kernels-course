# e03 — Copy / bandwidth

**Goal:** `out = x`. No arithmetic — this exercise exists to measure your card's
*achievable* memory bandwidth and to make coalescing real.

Unlocked by: `0c_memory_hierarchy`, `1b_memory_coalescing`.

## The spec
- Input: `x`, `N = 2**25` float32 on CUDA. Output: a copy.
- Metric: **bandwidth** (1 read + 1 write). This is the number to remember — it's the
  ceiling every memory-bound kernel in the course is measured against.

## What to write (`kernel.py`)
- `copy_kernel` + `copy(x)`. It's `e01` minus the add.

## Run
```bash
python -m harness.runner e03 --watch
```

## Hints
1. Same structure as `e01`, just with no arithmetic between the load and the store —
   revisit how that kernel mapped programs to elements and guarded the tail.
2. There is nothing to compute. If your GB/s here is below your `e01`/`e02` numbers,
   something's off with your block size or grid.

## Validate & benchmark it yourself
The runner's `[PASS]` / `[PERF]` / `[REF]` lines are just the `1a` correctness-and-speed
loop. Here it is for this kernel, to run yourself in a scratch script:

```python
import torch, triton

torch.manual_seed(0); x = torch.randn(1 << 25, device="cuda", dtype=torch.float32)   # as spec.py builds it

ref = x.clone()                             # reference FIRST (torch)
out = copy(x)                               # your kernel
torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)   # a copy moves bytes unchanged -> exact

ms     = triton.testing.do_bench(lambda: copy(x), warmup=25, rep=100, return_mode="median")
gbps   = 2 * x.numel() * x.element_size() / (ms * 1e-3) / 1e9   # read x, write out
ref_ms = triton.testing.do_bench(lambda: x.clone(), warmup=25, rep=100, return_mode="median")
print(f"{gbps:.0f} GB/s   ({ref_ms/ms:.2f}x torch)")
```

This is the purest bandwidth test in the course: two buffers of traffic and no math, so GB/s
against the ~896 roof *is* your coalescing grade. Exact data movement means `0/0` — any
mismatch is a real indexing bug. Full table and timing traps: `7b`.

## Going for performance — the real lesson
- Sweep `BLOCK_SIZE` (e.g. 1024, 2048, 4096, 8192) and record peak GB/s.
- Whatever you measure here (likely ~80–90% of 896 GB/s) is your *practical* ceiling.
  GDDR/HBM never delivers 100% of theoretical peak.
- Why can't a contiguous copy hit 896? Because a warp's 32 lanes each load 4 bytes →
  one 128-byte transaction, perfectly coalesced — but DRAM efficiency, ECC-less burst
  overheads, and turnaround still cost you. Write down your number; you'll compare
  every later kernel to it.
