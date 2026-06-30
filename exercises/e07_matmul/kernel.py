"""e07: tiled matmul -- YOU write this file.

C = A @ B, with A (M, K) and B (K, N). This is THE kernel. The whole point is tiling:
each program computes a tile of C by looping over K in steps, accumulating in
registers. Tiling is what turns a memory-bound dot product into a compute-bound
kernel. Read README.md (and lecture 1e) carefully.

Run:  python -m harness.runner e07 --watch
"""
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    # TODO: declare the kernel's parameters -- the input/output pointers, the three
    #       problem dimensions, the strides needed to address each 2-D operand (how
    #       many strides does a 2-D tensor need?), and the tile sizes (which must be
    #       compile-time constants).
):
    # TODO: from the 2-D program ids, work out which output tile this program owns
    # TODO: build the row/col offset vectors for that tile
    # TODO: allocate a register accumulator sized to the output tile, zero-initialized;
    #       pick a dtype that preserves accumulation accuracy
    # TODO: loop over K one tile-width at a time:
    #         load the slice of A and the slice of B that line up along K for this
    #         step (mask the K tail), then multiply-accumulate the two blocks. Triton
    #         has a primitive that does a block-by-block matmul into an accumulator --
    #         find it.
    # TODO: write the accumulator out to C, masking the M/N edges
    pass


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # TODO: choose tile sizes and build a 2-D launch grid so that, together, the
    #       programs cover every output tile (each program owns one tile -- round up
    #       so ragged edges still get a tile). Pass all strides; launch.
    #       (e10 revisits this grid once the tile sizes come from @triton.autotune.)
    raise NotImplementedError("write the tiled matmul kernel + launch")
    return c
