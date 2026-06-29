"""e03: copy / bandwidth -- YOU write this file.

The simplest possible kernel: out = x. No math at all. Its only job is to teach you
what your card's *real* achievable bandwidth is, and to make coalescing concrete.
Get as close to ~896 GB/s as you can.

Run:  python -m harness.runner e03 --watch
"""
import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(
    # TODO: in ptr, out ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    # TODO: program id -> offsets -> mask -> load -> store
    pass


def copy(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = out.numel()
    # TODO: BLOCK_SIZE, grid, launch
    raise NotImplementedError("write the copy kernel + launch")
    return out
