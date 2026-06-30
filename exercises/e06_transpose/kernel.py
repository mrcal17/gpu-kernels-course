"""e06: transpose -- YOU write this file.

Produce out = x.T for an (M, N) tensor, contiguous. Your first 2-D program grid and
your first real coalescing puzzle: the read and the write can't both be contiguous.
Read README.md.

Run:  python -m harness.runner e06 --watch
"""
import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(
    # TODO: declare the kernel parameters you need -- the two pointers, the matrix
    #       dims, the strides for each tensor, and the tile sizes (think about which
    #       params must be compile-time constants).
):
    # TODO: 2-D program ids -> this program owns a BLOCK_M x BLOCK_N tile of x
    # TODO: build row/col offset vectors, a 2-D mask for edge tiles
    # TODO: load the tile from x, store it transposed into out
    pass


def transpose(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty((N, M), device=x.device, dtype=x.dtype)
    # TODO: choose tile sizes and build a 2-D grid with enough programs to cover the
    #       whole matrix in both dimensions (round each dimension up by your tile size
    #       -- look for a Triton ceil-div helper). Pass strides for both x and out;
    #       launch. (e10 revisits this grid once the tile sizes come from
    #       @triton.autotune -- the grid becomes a callable that reads them from the
    #       chosen config.)
    raise NotImplementedError("write the transpose kernel + launch")
    return out
