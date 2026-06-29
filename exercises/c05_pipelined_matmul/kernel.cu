// c05 — Pipelined double-buffered matmul
//
// YOU write this file: the __global__ kernel(s) plus the host launcher solve().
// harness.cu is provided and you should not edit it.
//
// Goal: C = A @ B for square N x N x N fp32 matrices. SAME contract and oracle
// as c02 (tiled matmul) -- so your correctness should already be settled. The
// ONE thing changing here is the K-loop's load/compute structure: you overlap
// the load of the NEXT K-tile with the compute on the CURRENT one, using
// asynchronous cp.async copies into a DOUBLE-buffered shared tile. This is the
// CUDA expression of Triton's num_stages (lecture 3f).
//
// Contract with the harness:
//   - d_A, d_B, d_C are DEVICE pointers (the harness owns alloc + H2D copy).
//   - A, B, C are row-major N x N. On success, write C = A @ B and return 0.
//   - While the kernel is still a stub, return the TODO sentinel 77 and do NOT
//     touch d_C -- the harness sees 77 and prints "[TODO] ..." instead of checking.

#include <cuda_runtime.h>
// TODO: #include <cuda_pipeline.h>  (NOT <cuda/pipeline> -- the __pipeline_*
//       intrinsics live in cuda_pipeline.h; <cuda/pipeline> is the separate,
//       heavier libcu++ cuda::pipeline C++ API. We want the lighter intrinsics.)

// Side length of the square tile each block walks across K. Pick a multiple of
// 32 (N=768 is a multiple of 16/32/64). TILE is yours to choose.
#ifndef TILE
#define TILE 16
#endif

// ---------------------------------------------------------------------------
// The pipelined tiled matmul kernel.
//
// TODO: declare DOUBLE-buffered shared tiles:
//          __shared__ float As[2][TILE][TILE], Bs[2][TILE][TILE];
// TODO: write a device helper that ISSUES the async copies of one A-tile and
//       one B-tile (global -> shared) for a given k-step, using
//       __pipeline_memcpy_async for each element this thread owns, then
//       __pipeline_commit() to close the batch. (Mind the ragged-N edge: copy
//       0.0f or skip out-of-range source rows/cols so the tile stays correct.)
// TODO: PROLOGUE -- before the K loop, kick off the load of tile 0 into buffer 0.
// TODO: in the loop over K-tiles, FIRST issue the async load of the NEXT tile
//       into the OTHER buffer (only if there is a next tile).
// TODO: then __pipeline_wait_prior(N) so exactly ONE copy stays in flight --
//       reason out the depth N for a double buffer (see README hint 5) --
//       __syncthreads(), and do the TILE-long multiply-accumulate on the
//       CURRENT buffer into a per-thread register accumulator.
// TODO: ping-pong the buffer index each iteration (buf ^= 1); __syncthreads()
//       before overwriting a buffer you may still be reading.
// TODO: after the loop, write the accumulator to C with the ragged-N guard.
// ---------------------------------------------------------------------------
__global__ void pipelined_matmul_kernel(const float* A, const float* B,
                                        float* C, int N) {
    // TODO: implement the double-buffered, cp.async pipelined matmul above.
    (void)A;
    (void)B;
    (void)C;
    (void)N;
}

// ---------------------------------------------------------------------------
// Host launcher. The harness calls this.
// ---------------------------------------------------------------------------
extern "C" int solve(const float* d_A, const float* d_B, float* d_C, int N) {
    // TODO: set the launch config (a 2-D block covering one TILE x TILE output
    //       tile, a 2-D grid covering the N x N output), launch the kernel,
    //       then return 0.
    (void)d_A;
    (void)d_B;
    (void)d_C;
    (void)N;

    // Stub: report "not implemented yet" to the harness without doing real work.
    return 77;
}
