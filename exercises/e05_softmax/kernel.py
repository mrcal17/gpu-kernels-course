"""e05: softmax -- YOU write this file.

Row-wise softmax of an (M, N) tensor, numerically stable. One program per row.
This fuses a reduction (max, then sum) with a map (exp, divide). Read README.md.

Run:  python -m harness.runner e05 --watch
"""
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    # TODO: declare the kernel parameters you need -- the data pointers, whatever stride
    #       info you need to walk rows, the row length, and a compile-time tile width.
):
    # TODO: one program per row.
    # TODO: load the row (assume N <= BLOCK_SIZE for your first version; mask the tail).
    # TODO: subtract the row max BEFORE exp  (this is the whole game -- see README).
    # TODO: exp, sum, divide. Store the row.
    pass


def softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty_like(x)
    # TODO: BLOCK_SIZE must cover a row (a power of two >= N for the simple version);
    #       grid = one program per row; pass strides; launch.
    raise NotImplementedError("write the softmax kernel + launch")
    return out
