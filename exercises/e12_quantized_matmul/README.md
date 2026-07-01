# e12 — Quantized matmul (dequantize int8 weights in the inner loop)

**Goal:** `C = A @ dequant(B_q)`. `A` is fp16. `B` arrives **pre-quantized** as int8
with one fp32 scale per output channel; you dequantize it back to floats inside the
kernel and matmul as usual. This is the weight-only-quantized GEMM pattern from
`lecture 2c`: cheap bytes in (int8 weights), full-precision sum out.

Unlocked by: `e07_matmul` / `e10_autotuned_matmul` (the tiling) + lecture 2c (quant).

## The spec
- Inputs: `A (M,K)` fp16, `B_q (K,N)` int8, `scale (1,N)` fp32. Output: `A @ (B_q*scale)`
  as fp16. The quantization is **symmetric** (zero-point 0) and **per-output-channel**
  (one scale per column of `B`).
- Metric: **FLOP/s** = `2*M*N*K / time`. The matmul flop count is unchanged by quant —
  the dequant multiply is lower-order and ignored (lecture 2c).

## What to write (`kernel.py`)
- `quant_matmul_kernel` + `quant_matmul(a, b_q, scale)`. 2-D grid; each program computes
  one `BLOCK_M × BLOCK_N` tile of `C`. The tiling, K loop, and masks are exactly your
  e07/e10 matmul — only the B load changes.

## Hints — one at a time
1. This is your tiled matmul with **one new step in the inner loop**: B comes in as int8
   and must be turned back into real numbers before it can be multiplied. The tiling, the
   K loop, and the masks are identical — only the B load changes.
2. The quant layout is **symmetric** and **per-output-channel**: there is no zero-point
   (`z = 0`), and there is exactly one fp32 scale per column of `B` (per output channel).
   Because it's symmetric, dequant is a single scale multiply — recover each weight by
   scaling the stored int8 value by its channel's scale. Derive from the spec what that
   one multiply looks like and which factor it uses, and convince yourself **which axis**
   the scale broadcasts over — look at which dimension `amax` reduced.
3. The scale vector is indexed by the **output (column)** dimension only — it is constant
   across the contraction dimension. So load it **once per output tile, outside the K
   loop**, not every K-chunk. Think about where that load belongs.
4. Dequantize the int8 tile to a compute type **inside the kernel**, in registers/SRAM,
   then feed the float tile to `tl.dot`. Do not try to dot int8 against fp16 directly.
5. **Accumulate wide.** Read low precision, but keep the running sum in an fp32
   accumulator — the whole point of dequant-then-accumulate (lecture 2c) is cheap bytes
   in, full-precision sum.
6. You are graded against the reference's **dequantized** answer (it multiplies the same
   `b_q` by the same `scale`), not against the original full-precision weights. So your
   dequant must match the reference's dequant exactly — same scale, same broadcasting —
   and then only float rounding separates you. A kernel that forgets the scale misses by
   thousands.

## Validate & benchmark it yourself
The runner's `[PASS]` / `[PERF]` / `[REF]` lines are just the `1a` correctness-and-speed
loop. Here it is for this kernel, to run yourself in a scratch script:

```python
import torch, triton

torch.manual_seed(0); a = torch.randn(512, 1024, device="cuda", dtype=torch.float16); b_f = torch.randn(1024, 768, device="cuda", dtype=torch.float32); scale = b_f.abs().amax(dim=0, keepdim=True) / 127.0; b_q = torch.clamp(torch.round(b_f / scale), -127, 127).to(torch.int8)   # as spec.py builds them

ref = (a.float() @ (b_q.float() * scale)).to(torch.float16)   # dequant then matmul, FIRST
out = quant_matmul(a, b_q, scale)                             # your kernel
torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-2)   # fp16 round-off + accumulation order

M, K = a.shape; N = b_q.shape[1]
ms     = triton.testing.do_bench(lambda: quant_matmul(a, b_q, scale), warmup=25, rep=100, return_mode="median")
tflops = 2 * M * N * K / (ms * 1e-3) / 1e12   # dequant is lower-order; the matmul dominates
ref_ms = triton.testing.do_bench(lambda: (a.float() @ (b_q.float() * scale)).to(torch.float16), warmup=25, rep=100, return_mode="median")
print(f"{tflops:.1f} TFLOP/s   ({ref_ms/ms:.2f}x torch)")
```

Compute-bound. The reference dequantizes the int8 weights exactly as your kernel should, so
what's left in the tolerance is fp16 rounding + accumulation order — hence the loose
`1e-1/1e-2`. Compare TFLOP/s to the tensor-core peak. Full tolerance table and traps: `7b`.

## Going for performance
- Everything from `e07`/`e10` applies: `tl.dot` hits tensor cores, tile sizes are
  everything, bigger tiles = more reuse until you run out of registers/SRAM. You can drop
  `@triton.autotune` on top exactly as in `e10`.
- The win that motivates quant lives in **bandwidth**, not flops: int8 weights are a
  quarter the bytes of fp32 (half of fp16). The flop count is identical — the payoff is
  fitting more weight into the same memory traffic / cache.
- **Stretch thought (not graded):** the integer-accumulate identity from lecture 2c lets
  you sum `q_a*q_b` in int32 and apply the scale once at the end. Here `A` is fp16, so the
  straight dequant-then-dot is the natural path — but it is worth seeing why pulling the
  scale out of the loop is legal in the first place.
