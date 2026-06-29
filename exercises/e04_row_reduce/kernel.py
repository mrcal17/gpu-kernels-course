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
    # TODO: x ptr, out ptr, strides you need, N (cols), BLOCK_SIZE: tl.constexpr
):
    # TODO: which row am I? (program id)
    # TODO: walk this row in chunks of BLOCK_SIZE, accumulating a partial sum
    #       (mask the tail when N is not a multiple of BLOCK_SIZE)
    # TODO: reduce the loaded block to a scalar and add into the accumulator
    # TODO: store the row's total
    pass


def row_sum(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty((M,), device=x.device, dtype=x.dtype)
    # TODO: choose BLOCK_SIZE; grid = one program per row; launch.
    #       you'll need x's row/col strides to index correctly.
    raise NotImplementedError("write the row-sum kernel + launch")
    return out
