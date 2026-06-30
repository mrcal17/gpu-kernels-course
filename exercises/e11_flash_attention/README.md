# e11 — Flash attention forward

**Goal:** `O = softmax(Q @ K^T * scale) @ V` for `Q, K, V` each `(B, H, N, D)` fp16,
non-causal. The headline idea: compute attention **without ever materializing the
`N × N` scores matrix in HBM**. Each program streams over the keys/values and folds
them into a running result with the online softmax — connect this back to the
streaming softmax you derived in `2b`.

Unlocked by: `2b_flash_attention` (online softmax) and `1e_tiling_matmul` (tiling).

## The spec
- Inputs: `Q, K, V` `(2, 4, 512, 64)` fp16. Output: `(2, 4, 512, 64)` fp16.
- Reference: `F.scaled_dot_product_attention(q, k, v, is_causal=False)`. Its default
  scale is `1/sqrt(D)` and it accumulates the softmax in fp32.
- Metric: **FLOP/s** = `4*B*H*N*N*D / time` (the two matmuls `QK^T` and `PV`; the
  softmax is lower-order and ignored).
- `N = 512` is a clean multiple of common KV block sizes, so v1 need not mask the KV
  tail. `D = 64` fits a single head-dim block.

## What to write (`kernel.py`)
- `flash_kernel` + `flash_attention(q, k, v)`. One program per
  `(batch*head, query block)`; each program owns a block of query rows and loops
  over all the keys/values.

## Hints — one at a time
1. **The scores matrix is a ghost.** A program owns one block of query *rows* and
   streams over blocks of keys/values, folding each score tile into a running result
   and discarding it. Nothing `N × N` ever touches HBM.
2. **Three running statistics per query row**, carried across the KV loop: a running
   max, a running denominator (sum of exponentials), and a running *unnormalized*
   output accumulator. You derived the update in `2b` — the formula's variable names
   (`m, l, o, alpha`) won't match your code names; you map them. Derive the shapes
   yourself from what a "per query row" / "per head dim" quantity needs.
3. **The rescale is the crux.** Each KV block: score it against your query block, find
   the new running max, then rescale the OLD accumulated state by
   `exp(old_max - new_max)` before adding the new block's contribution. That single
   per-row rescale is what makes the streaming softmax *exact*, not approximate.
4. **Defer the division.** Don't divide by the denominator inside the loop — it isn't
   final until the last KV block. Divide the accumulator by it exactly once, after the
   loop, then store.
5. **Precision discipline.** Inputs and output are fp16, but do the exponentials, the
   running max/sum, and the `tl.dot` accumulation in **fp32**. Casting the score math
   down to fp16 mid-loop is the classic way to fail the tolerance.
6. **`tl.dot` orientation.** The score tile is queries-by-keys; the output update is
   probabilities-times-values. Think carefully about which operand needs transposing
   for each of the two dots.

## Validate & benchmark it yourself
The runner's `[PASS]` / `[PERF]` / `[REF]` lines are just the `1a` correctness-and-speed
loop. Here it is for this kernel, to run yourself in a scratch script:

```python
import torch, triton
import torch.nn.functional as F

ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)   # reference FIRST (torch)
out = flash_attention(q, k, v)                                   # your kernel
torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)   # fp16 data + fp32 softmax -> looser

B, H, N, D = q.shape
ms     = triton.testing.do_bench(lambda: flash_attention(q, k, v), warmup=25, rep=100, return_mode="median")
tflops = 4 * B * H * N * N * D / (ms * 1e-3) / 1e12   # the two matmuls (QK^T and PV) dominate; softmax is lower-order
ref_ms = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False), warmup=25, rep=100, return_mode="median")
print(f"{tflops:.1f} TFLOP/s   ({ref_ms/ms:.2f}x torch)")
```

Compute-bound, and the inputs are fp16 — so the tolerance is loose (fp16 data, fp32 softmax
accumulation). torch's SDPA is itself a fused flash kernel, so it's a stiff bar. The FLOP
count includes only the two matmuls. Full tolerance table and traps: `7b`.

## Going for performance
- `tl.dot` maps to tensor cores — both matmuls go through it.
- The win over a naive attention is **memory**: you never read or write the `N × N`
  scores. Bigger query blocks reuse each loaded K/V tile across more query rows; tile
  sizes trade reuse against register/SRAM pressure (the occupancy limiters from `0d`).
- **Stretch goals** once the non-causal version passes: (1) handle a ragged `N` that
  isn't a multiple of your KV block by masking the key tail; (2) add a causal mask so a
  query only attends to keys at or before its position (set masked scores to `-inf`
  before the `exp`). Neither is required for this reference (non-causal, `N` a clean
  multiple).
