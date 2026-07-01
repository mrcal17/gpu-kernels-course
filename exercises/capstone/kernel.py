"""capstone: YOUR kernel -- you write this file (after you write spec.py).

Unlike e01-e13 there is no prescribed op here. Lecture 4b is the project brief:
pick fused attention (Option A) or quantized GEMM (Option B), define the
contract in spec.py (that IS milestone M0), then climb M1 -> M4 in this file.
Slow-but-correct first; never tune a kernel that hasn't passed.

Run:  python -m harness.runner capstone --watch
"""
import torch
import triton
import triton.language as tl


# TODO (M1): your @triton.jit kernel(s) go here. You built every piece already:
#   Option A = tiling (1e) + online softmax (1d/2b) + reduction fusion (1c)
#   Option B = tiled GEMM (1e/2a) + dequant-in-the-tile-loop (2c)


def capstone(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Launch wrapper the harness calls. Its name must match spec.ENTRYPOINT and
    its signature must accept exactly what spec.make_inputs() returns -- if you
    redefine the spec (you should), update this signature to match."""
    # TODO: allocate the output, build the grid, launch the kernel(s).
    raise NotImplementedError("capstone: write spec.py first (M0), then build M1 here")
