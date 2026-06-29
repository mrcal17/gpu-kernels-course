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
    # TODO: x ptr, out ptr, M, N, strides, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # TODO: 2-D program ids -> this program owns a BLOCK_M x BLOCK_N tile of x
    # TODO: build row/col offset vectors, a 2-D mask for edge tiles
    # TODO: load the tile from x, store it transposed into out
    pass


def transpose(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty((N, M), device=x.device, dtype=x.dtype)
    # TODO: BLOCK_M, BLOCK_N; 2-D grid = (cdiv(M,BLOCK_M), cdiv(N,BLOCK_N));
    #       pass strides for both x and out; launch.
    raise NotImplementedError("write the transpose kernel + launch")
    return out
