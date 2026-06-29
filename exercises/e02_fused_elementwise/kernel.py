"""e02: fused elementwise (SiLU) -- YOU write this file.

Compute  out = x * sigmoid(x)  in a single Triton kernel: one load, the whole
expression evaluated in registers, one store. No intermediate tensors, no extra
memory traffic. Read README.md for hints.

Run:  python -m harness.runner e02 --watch
"""
import torch
import triton
import triton.language as tl


@triton.jit
def silu_kernel(
    # TODO: pointers (in, out), n_elements, BLOCK_SIZE: tl.constexpr
):
    # TODO: program id -> offsets -> mask -> load x
    # TODO: compute sigmoid(x) and multiply by x, all in registers
    #       (tl has the exp / sigmoid building blocks you need)
    # TODO: masked store
    pass


def silu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = out.numel()
    # TODO: BLOCK_SIZE, grid, launch
    raise NotImplementedError("write the SiLU kernel + launch")
    return out
