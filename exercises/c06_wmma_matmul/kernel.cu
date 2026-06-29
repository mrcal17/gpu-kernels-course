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
// TODO: #include <mma.h> as well, and bring the WMMA names into scope with
//       `using namespace nvcuda::wmma;` (the fragment/load/store/mma symbols
//       all live in nvcuda::wmma).

__global__ void wmma_matmul(const __half* A, const __half* B, float* C, int N) {
    // TODO: map each WARP (not each thread) to one 16x16 output tile of C.
    //       A block has several warps; derive this warp's (tileRow, tileCol)
    //       in the C grid from blockIdx and threadIdx (threadIdx.x / 32 picks
    //       the warp within the block). Each of the 32 lanes runs this same
    //       code with converged control flow.

    // TODO: declare the three fragments for a 16x16x16 tile:
    //         fragment<matrix_a, 16,16,16, half, row_major>  a_frag;
    //         fragment<matrix_b, 16,16,16, half, col_major>  b_frag;
    //         fragment<accumulator, 16,16,16, float>         acc_frag;
    //       The accumulator stays fp32 for precision — that is the whole
    //       fp16-inputs / fp32-accumulate trade.

    // TODO: fill_fragment(acc_frag, 0.0f) to zero the accumulator.

    // TODO: loop k from 0 to N in steps of 16. Each iteration:
    //         - derive the base pointer of this warp's A-tile (rows tileRow*16,
    //           cols k) and B-tile (rows k, cols tileCol*16) inside the big
    //           row-major buffers, and the leading dimension to pass;
    //         - load_matrix_sync(a_frag, A_tile_ptr, lda);
    //         - load_matrix_sync(b_frag, B_tile_ptr, ldb);
    //         - mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    //       Every WMMA call ends in _sync because all 32 lanes must reach it
    //       together. A fragment is opaque — never index into it.

    // TODO: store_matrix_sync the accumulator to C's tile (leading dim N,
    //       layout mem_row_major).
}

extern "C" int solve(const __half* d_A, const __half* d_B, float* d_C, int N) {
    // TODO: choose a launch config that gives ONE WARP per 16x16 output tile.
    //       There are (N/16) x (N/16) output tiles. A block holds several warps
    //       (blockDim.x must be a multiple of 32); size grid/block so every tile
    //       is covered exactly once.
    // TODO: launch wmma_matmul<<<grid, block>>>(d_A, d_B, d_C, N);
    // TODO: return 0 after launching (replace the `return 77;` sentinel).
    return 77;
}
