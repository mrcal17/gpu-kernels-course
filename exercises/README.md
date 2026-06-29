# Exercises — the practical track

This is where you write kernels. Lectures (in `../notebooks/`) build the intuition;
here you do the work on the real GPU. **You write every kernel** — each `kernel.py` is
a stub with TODOs and layered hints, never a solution.

## How to run
```bash
# from the repo root
python -m harness.device_info          # your GPU's actual limits
python -m harness.runner e01 --watch   # write exercises/e01_*/kernel.py, save, watch it check
python -m harness.runner e07           # run once
python -m harness.runner --all         # the whole progress board
```
The runner verifies your output against a torch reference, then reports latency and
achieved **GB/s** (memory-bound kernels) or **TFLOP/s** (compute-bound), and how that
compares to torch.

## Each exercise folder
- `kernel.py` — **you edit this.** Stub + TODOs.
- `spec.py` — the harness contract (reference, inputs, metric). Don't edit.
- `README.md` — the brief + hints (peek one at a time).

## The progression (Triton — Parts 0–2)
| # | Exercise | Pattern | Metric | Lecture |
|---|---|---|---|---|
| `e01` | vector add | maps, masking | bandwidth | 1a |
| `e02` | fused elementwise (SiLU) | fusion | bandwidth | 1a |
| `e03` | copy | your bandwidth ceiling | bandwidth | 0c, 1b |
| `e04` | row reduce | reduction, 2-D indexing | bandwidth | 1c |
| `e05` | softmax | reduce + map, stability | bandwidth | 1d |
| `e06` | transpose | 2-D tiling, coalescing | bandwidth | 1e |
| `e07` | tiled matmul | tiling (the centerpiece) | TFLOP/s | 1e |
| `e08` | layernorm | fused norm | bandwidth | 1f |
| `e09` | cumsum *(advanced)* | scan | bandwidth | 1g |
| `e10` | autotuned matmul | `@triton.autotune`, ragged masking | TFLOP/s | 2a |
| `e11` | flash attention (fwd) | online softmax over KV blocks | TFLOP/s | 2b |
| `e12` | quantized matmul | int8 dequant in the inner loop | TFLOP/s | 2c |
| `e13` | fused SiLU + backward | `autograd.Function`, hand-written grad | bandwidth | 2d |

## CUDA C++ (Part 3+)
`c01`–`c06` re-derive these patterns in raw CUDA (compiled with `nvcc`). Scaffolded as
you reach Part 3.

> Stuck? The matching lecture has the concept; the README has layered hints. Resist the
> urge to look anything up until you've tried — the point is to build the muscle.
