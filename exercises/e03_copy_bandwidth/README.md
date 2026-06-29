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
1. Identical structure to `e01`: program-id → offsets → mask → load → store.
2. There is nothing to compute. If your GB/s here is below your `e01`/`e02` numbers,
   something's off with your block size or grid.

## Going for performance — the real lesson
- Sweep `BLOCK_SIZE` (e.g. 1024, 2048, 4096, 8192) and record peak GB/s.
- Whatever you measure here (likely ~80–90% of 896 GB/s) is your *practical* ceiling.
  GDDR/HBM never delivers 100% of theoretical peak.
- Why can't a contiguous copy hit 896? Because a warp's 32 lanes each load 4 bytes →
  one 128-byte transaction, perfectly coalesced — but DRAM efficiency, ECC-less burst
  overheads, and turnaround still cost you. Write down your number; you'll compare
  every later kernel to it.
