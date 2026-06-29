"""e07: tiled matmul -- YOU write this file.

C = A @ B, with A (M, K) and B (K, N). This is THE kernel. The whole point is tiling:
each program computes a BLOCK_M x BLOCK_N tile of C by looping over K in steps of
BLOCK_K, accumulating in registers. Tiling is what turns a memory-bound dot product
into a compute-bound kernel. Read README.md (and lecture 1e) carefully.

Run:  python -m harness.runner e07 --watch
"""
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    # TODO: a, b, c pointers; M, N, K; strides for a, b, c;
    #       BLOCK_M, BLOCK_N, BLOCK_K: tl.constexpr
):
    # TODO: 2-D program ids -> which BLOCK_M x BLOCK_N tile of C this program owns
    # TODO: row/col offset vectors for this tile
    # TODO: an accumulator of shape (BLOCK_M, BLOCK_N), start at zero (fp32)
    # TODO: loop k from 0 to K in steps of BLOCK_K:
    #          load an (BLOCK_M x BLOCK_K) tile of A and a (BLOCK_K x BLOCK_N) tile of B
    #          (mask the K tail), accumulate the tile product (look for tl.dot)
    # TODO: write the accumulator to C (mask the M/N edges)
    pass


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # TODO: BLOCK_M/N/K; 2-D grid = (cdiv(M,BLOCK_M), cdiv(N,BLOCK_N));
    #       pass all strides; launch.
    raise NotImplementedError("write the tiled matmul kernel + launch")
    return c
