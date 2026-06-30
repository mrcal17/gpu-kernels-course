"""e01: vector add  --  YOU write this file.

Goal: compute out = a + b on the GPU with a Triton kernel you wrote yourself.
Read README.md in this folder first. It walks you through the four ideas you
need (program id, the block of indices, the mask, load/store) without handing
you the lines.

Run it:   python -m harness.runner e01 --watch
"""
import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    # TODO: declare the arguments this kernel needs.
    #   - the input/output pointers
    #   - the number of elements (so you can build a mask)
    #   - a per-program tile size (which of these must be known at compile time?)
):
    # TODO: write the kernel body. See README.md hints 1-4.
    #   1. which contiguous chunk does this program own? (its program index)
    #   2. from that and the tile size, build the vector of element offsets it owns
    #   3. the last program runs past the end -- build a boolean mask of the valid lanes
    #   4. read the two input chunks (masked), add them, write the result back (masked)
    pass


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Launch wrapper the harness calls. Allocate the output, decide the launch
    grid, and launch the kernel above."""
    out = torch.empty_like(a)
    n_elements = out.numel()

    # TODO: choose a tile size (a power of two is conventional -- why?).
    # TODO: compute the 1-D launch grid: how many program instances do you need
    #       so that every element is covered? (a ceiling division -- look for a
    #       Triton ceil-div helper.)
    # TODO: launch the kernel over that grid, passing the pointers, the element
    #       count, and the tile size.
    raise NotImplementedError("write the kernel + launch in kernel.py")

    return out
