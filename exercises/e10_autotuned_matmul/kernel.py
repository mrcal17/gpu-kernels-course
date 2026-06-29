"""e10: autotuned matmul over ragged shapes -- YOU write this file.

C = A @ B again, but with two new jobs on top of e07:
  (1) be correct for ANY shape -- M, N, and K can all be ragged, so every load
      and the final store needs masking; and
  (2) stop hand-picking the tile -- hand the choice to @triton.autotune and let
      the tuner search a menu of configs you design.

Read README.md (and lecture 2a on occupancy / shared memory) carefully.

Run:  python -m harness.runner e10 --watch
"""
import torch
import triton
import triton.language as tl


# TODO: add an @triton.autotune decorator here.
#   configs=[ triton.Config({...block sizes...}, num_warps=..., num_stages=...),
#             ... a small, diverse menu YOU design ... ],
#   key=[...]   # which args, when they change, trigger a re-tune
@triton.jit
def matmul_kernel(
    # TODO: a, b, c pointers; M, N, K; strides for a, b, c
    #       (stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn);
    #       BLOCK_M, BLOCK_N, BLOCK_K: tl.constexpr
):
    # TODO: 2-D program ids -> which BLOCK_M x BLOCK_N tile of C this program owns
    # TODO: row/col offset vectors for this tile
    # TODO: an accumulator of shape (BLOCK_M, BLOCK_N), start at zero (fp32)
    # TODO: loop k from 0 to K in steps of BLOCK_K:
    #          load an (BLOCK_M x BLOCK_K) tile of A and a (BLOCK_K x BLOCK_N) tile of B
    #          MASK the K tail with other=0.0 so out-of-range lanes contribute nothing,
    #          accumulate the tile product (look for tl.dot)
    # TODO: write the accumulator to C, masking the M/N edges
    #       (row < M) & (col < N) so the ragged last tiles do not write out of bounds
    pass


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # TODO: the tile sizes now live in the autotune configs, so the grid lambda
    #       must read them from the META dict:
    #         grid = lambda META: (cdiv(M, META['BLOCK_M']), cdiv(N, META['BLOCK_N']))
    #       launch matmul_kernel[grid](...) passing all six strides.
    raise NotImplementedError("write the masked, autotuned matmul kernel + launch")
    return c
