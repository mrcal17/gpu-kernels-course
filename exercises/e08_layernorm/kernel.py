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
    # TODO: reduce the row to its mean, then to its variance (deviation-squared,
    #       averaged). Decide what to divide by.
    # TODO: center the row and rescale it to unit variance (remember eps for numerical
    #       safety -- think about where it belongs).
    # TODO: apply the affine transform; figure out which axis weight/bias are indexed
    #       along and which offsets that reuses.
    pass


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty_like(x)
    eps = 1e-5
    # TODO: BLOCK_SIZE covering a row; grid = one program per row; pass strides; launch.
    raise NotImplementedError("write the layernorm kernel + launch")
    return out
