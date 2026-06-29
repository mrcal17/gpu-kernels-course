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
    #   - a BLOCK_SIZE compile-time constant: tl.constexpr
):
    # TODO: write the kernel body. See README.md hints 1-4.
    #   1. which block am I?         -> tl.program_id
    #   2. which indices do I own?   -> that block id, BLOCK_SIZE, tl.arange
    #   3. don't run off the end     -> a boolean mask
    #   4. load, add, store          -> tl.load / tl.store (pass the mask!)
    pass


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Launch wrapper the harness calls. Allocate the output, decide the launch
    grid, and launch the kernel above."""
    out = torch.empty_like(a)
    n_elements = out.numel()

    # TODO: choose a BLOCK_SIZE (a power of two is conventional -- why?).
    # TODO: compute the 1-D launch grid: how many program instances do you need
    #       so that every element is covered? (Look for a Triton ceil-div helper.)
    # TODO: launch -> vector_add_kernel[grid](... , BLOCK_SIZE=...)
    raise NotImplementedError("write the kernel + launch in kernel.py")

    return out
