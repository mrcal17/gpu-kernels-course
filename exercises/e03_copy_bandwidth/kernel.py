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
    # TODO: declare the kernel's parameters -- the data it reads/writes, how many
    #       elements there are, and the per-program tile size (think about which of
    #       these must be known at compile time).
):
    # TODO: each program handles one contiguous tile of the array. Work out which
    #       elements this program owns, guard against running past the end, then
    #       move the data from input to output.
    pass


def copy(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = out.numel()
    # TODO: BLOCK_SIZE, grid, launch
    raise NotImplementedError("write the copy kernel + launch")
    return out
