// c06 — WMMA tensor-core matmul (fp16): YOU write this file.
//
// Square C = A @ B, with A and B in fp16 (__half) and C in fp32. The whole
// point is the tensor cores: instead of one thread doing scalar FMAs, a WHOLE
// WARP cooperatively issues a 16x16x16 matrix-multiply-accumulate in a single
// hardware op (the WMMA API in <mma.h>). This is the c02 tiled-matmul shape
// with the inner product replaced by a sequence of tile-MMAs.
//
// harness.cu owns all cudaMalloc / cudaMemcpy / the fp16<->fp32 conversion /
// the fp32 reference / timing. You get device pointers that are already filled
// (d_A, d_B are __half; d_C is fp32) and you launch the matmul over them.
//
// While solve() still returns the sentinel (77), the harness prints [TODO] and
// exits without checking anything. Return 0 once you actually launch.

#include <cuda_fp16.h>
// TODO: the WMMA fragment/load/store/mma types live in a dedicated CUDA header
//       and a sub-namespace -- find which header to include alongside this one,
//       and how to bring those symbols into scope.

__global__ void wmma_matmul(const __half* A, const __half* B, float* C, int N) {
    // TODO: map each WARP (not each thread) to one 16x16 output tile of C.
    //       A block has several warps. A warp is a fixed number of lanes; from
    //       threadIdx work out which warp within the block you belong to, then
    //       derive this warp's (tileRow, tileCol) in the C grid from blockIdx and
    //       that warp index. Each of the 32 lanes runs this same code with
    //       converged control flow.

    // TODO: declare three fragments for one tile -- two input operand fragments
    //       (A and B) and one accumulator. Work out from the WMMA docs each
    //       fragment's template arguments: its operand role, the tile's three
    //       dimensions, the element precision (inputs vs accumulator), and the
    //       memory layout each operand expects. The accumulator stays fp32 for
    //       precision — that is the whole fp16-inputs / fp32-accumulate trade.

    // TODO: before the K-loop, zero the accumulator fragment (there is a WMMA
    //       helper that fills a fragment with a constant).

    // TODO: walk the K dimension one tile at a time, accumulating each tile-MMA
    //       into the accumulator (derive the step from the tile's K extent).
    //       Each iteration:
    //         - derive the base pointer of this warp's A-tile (the rows this
    //           warp's tileRow owns, at column offset k) and its B-tile (rows at
    //           offset k, the columns this warp's tileCol owns) inside the big
    //           row-major buffers, and the leading dimension to pass;
    //         - load this warp's A-tile and B-tile into their fragments (there is
    //           a load-fragment-from-memory call that takes a pointer and a
    //           leading dimension);
    //         - issue one warp-collective tile-MMA that multiplies the two
    //           operand fragments and accumulates into the accumulator. Work out
    //           the argument order, including how the running accumulator is
    //           threaded through the call.
    //       Every WMMA call ends in _sync because all 32 lanes must reach it
    //       together. A fragment is opaque — never index into it.

    // TODO: after the K-loop, write the accumulator out to this warp's tile of C
    //       (there is a store-fragment call; you supply the destination pointer,
    //       the buffer's leading dimension, and the memory layout C is stored in).
}

extern "C" int solve(const __half* d_A, const __half* d_B, float* d_C, int N) {
    // TODO: pick a launch geometry so that exactly one warp lands on each output
    //       tile and every tile is covered once. Work out how many tiles there
    //       are from N and the tile size, and remember a block's thread count
    //       must be a whole number of warps. (Here the grid is a fixed config of
    //       constants; in e10 the grid becomes a callable that reads the chosen
    //       tile sizes from the autotune config instead.)
    // TODO: launch your kernel with the grid/block you chose, forwarding the
    //       device pointers and N.
    // TODO: return 0 after launching (replace the `return 77;` sentinel).
    return 77;
}
