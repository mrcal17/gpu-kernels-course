"""e04: row reduction -- YOU write this file.

Given a 2-D tensor x of shape (M, N), produce out of shape (M,) where out[i] is the
sum of row i. One program per row is the natural decomposition. Read README.md.

Run:  python -m harness.runner e04 --watch
"""
import torch
import triton
import triton.language as tl


@triton.jit
def row_sum_kernel(
    # TODO: declare the parameters this kernel needs -- the input/output pointers,
    #       whatever stride and dimension info you must index a row with, and a
    #       per-program tile size (which of these has to be known at compile time?).
):
    # TODO: which row am I? (program id)
    # TODO: walk this row in chunks of your tile size, accumulating a partial sum
    #       (mask the tail when the row length is not a multiple of the tile)
    # TODO: reduce each loaded chunk to a scalar and add it into the accumulator
    # TODO: store the row's total
    pass


def row_sum(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty((M,), device=x.device, dtype=x.dtype)
    # TODO: choose a tile size; launch one program per row; pass whatever strides
    #       you need to index a row correctly.
    raise NotImplementedError("write the row-sum kernel + launch")
    return out
