"""e08: layernorm -- YOU write this file.

Row-wise LayerNorm of (M, N): normalize each row to zero mean / unit variance, then
apply the per-column affine (weight, bias). The win is FUSION -- mean, variance,
normalize and affine in one pass over the data instead of several kernels. EPS = 1e-5.
Read README.md.

Run:  python -m harness.runner e08 --watch
"""
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    # TODO: x, w, b, out pointers; strides; N; eps; BLOCK_SIZE: tl.constexpr
):
    # TODO: one program per row; load the row (mask the tail; assume N <= BLOCK_SIZE
    #       for the first version).
    # TODO: mean = sum(row)/N ; var = sum((row-mean)^2)/N
    # TODO: norm = (row - mean) * rsqrt(var + eps)
    # TODO: out = norm * w + b   (w, b indexed by column)
    pass


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty_like(x)
    eps = 1e-5
    # TODO: BLOCK_SIZE covering a row; grid = one program per row; pass strides; launch.
    raise NotImplementedError("write the layernorm kernel + launch")
    return out
