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


# TODO: this kernel should be autotuned. Find Triton's autotune decorator and give
#       it a menu of candidate configurations to search over. Each candidate fixes a
#       set of tile sizes plus the scheduling knobs you met in lecture 2a. You'll
#       also tell it which arguments, when they change, force a fresh search.
@triton.jit
def matmul_kernel(
    # TODO: declare the kernel parameters. You'll need the input/output pointers, the
    #       three problem dimensions, the strides needed to address each 2-D operand
    #       (think about how many strides a 2-D tensor needs), and the tile sizes --
    #       which must be compile-time constants so they can come from the autotune
    #       config.
):
    # TODO: from the 2-D program ids, work out which output tile this program owns
    # TODO: row/col offset vectors for this tile
    # TODO: create an accumulator that holds this program's output tile and
    #       initialize it to zero. Think about what shape it must be (it has to hold
    #       the tile you'll eventually store) and what precision you want to
    #       accumulate in for accuracy.
    # TODO: loop over K one tile-width at a time:
    #          load the slice of A and the slice of B that line up along K for this
    #          step, masking the K tail so out-of-range lanes contribute nothing, then
    #          multiply-accumulate the two blocks. Triton has a primitive that does a
    #          block-by-block matmul into an accumulator -- find it.
    # TODO: write the accumulator out to C, but only the lanes that actually fall
    #       inside C -- the last row-tile and last column-tile hang off the edge, so
    #       guard the store with a 2-D mask that is true only where this tile's rows
    #       AND columns are still within bounds.
    pass


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # TODO: the tile sizes now come from the chosen autotune config, not constants.
    #       That means the launch grid can't be a fixed tuple anymore -- it has to
    #       ask the config how big the tiles are. Triton lets the grid be a callable
    #       that receives the chosen meta-parameters. Build a 2-D grid that covers
    #       all row-tiles and all column-tiles (round UP so ragged edges still get a
    #       tile). Then launch the kernel, passing every pointer, the three sizes,
    #       and all six strides.
    raise NotImplementedError("write the masked, autotuned matmul kernel + launch")
    return c
