"""e11: flash attention forward -- YOU write this file.

O = softmax(Q @ K^T * scale) @ V, with Q, K, V each (B, H, N, D) fp16. The whole
point is that the N x N scores matrix never exists in HBM: a program owns one
block of query rows and STREAMS over blocks of K/V, folding each score tile into
a running (max, denominator, output) triple via the online softmax from lecture
2b, then discarding it. Read README.md (and lecture 2b) carefully.

Run:  python -m harness.runner e11 --watch
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def flash_kernel(
    # TODO: q, k, v, o pointers; the strides needed to index each tensor (consider
    #       collapsing the batch and head axes into one program axis and passing
    #       per-(b,h) strides); N, D; scale;
    #       BLOCK_M, BLOCK_N, D: tl.constexpr (your query/KV block sizes + head dim)
):
    # TODO: map the program id(s) -> (which batch*head, which query block) this
    #       program owns. Load that Q block once into SRAM.
    # TODO: init the three running statistics per query row so the first KV block's
    #       contribution is taken as-is -- think about what starting value for the
    #       running max makes the first real max win, and what neutral values the
    #       two sums (denominator, output accumulator) need. Keep them in fp32, sized
    #       to what a "per query row" / "per head dim" quantity needs.
    # TODO: loop over the N keys in steps of BLOCK_N. For each KV block:
    #          load the K block and V block, then score your query block against the
    #          key block (remember the scale). That score step and the later
    #          output-accumulation step are each a matmul that maps to tensor cores --
    #          figure out which operand of each must be transposed so the dimensions
    #          line up (see hint 6). The score tile lives only in SRAM.
    #          Then apply the online-softmax update you derived in 2b: refresh the
    #          running max, exponentiate the scores against it, compute the rescale
    #          factor for the OLD state, and fold both the denominator and the output
    #          accumulator forward. Keep the score/probability math in fp32.
    # TODO: after the KV loop, normalize ONCE: acc = acc / denom; cast to fp16 and
    #       store the O block.
    pass


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    B, H, N, D = q.shape
    scale = 1.0 / math.sqrt(D)
    o = torch.empty_like(q)
    # TODO: pick BLOCK_M / BLOCK_N; build the grid (one program per
    #       (b*h, query block)); launch passing the strides, N, D, scale, and
    #       your block sizes.
    raise NotImplementedError("write the flash-attention kernel + launch")
    return o
