"""e09: cumulative sum / scan -- YOU write this file.  (ADVANCED)

Row-wise inclusive prefix sum of (M, N): out[i, j] = sum(x[i, 0..j]). Scan is the
trickiest of the parallel patterns -- unlike a reduction, every output depends on a
running total. For this version each row fits in one block, so you can use Triton's
in-block scan primitive. Read README.md (and lecture 1g).

Run:  python -m harness.runner e09 --watch
"""
import torch
import triton
import triton.language as tl


@triton.jit
def cumsum_kernel(
    # TODO: declare the kernel parameters. Think about what each program needs: the
    #       input/output base pointers, enough stride info to walk to its row, the row
    #       length, and a compile-time tile width.
):
    # TODO: one program per row; load the row (mask the tail to 0 so it doesn't
    #       pollute the running sum).
    # TODO: inclusive scan along the block axis (Triton provides a cumulative-sum /
    #       associative-scan primitive for in-block scans -- find it in the Triton
    #       language module).
    # TODO: store the scanned row.
    pass


def cumsum(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty_like(x)
    # TODO: BLOCK_SIZE covering a row; grid = one program per row; pass strides; launch.
    raise NotImplementedError("write the cumsum kernel + launch")
    return out
