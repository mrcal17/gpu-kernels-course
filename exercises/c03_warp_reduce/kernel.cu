// c03 — Warp-shuffle reduction (array sum)
//
// YOU write this file: the __global__ kernel(s) plus the host launcher solve().
// harness.cu is provided and you should not edit it.
//
// Goal: sum a large 1-D array d_in (n floats) down to a single scalar in d_out[0].
//
// Contract with the harness:
//   - d_in, d_out are DEVICE pointers (the harness owns alloc + H2D copy).
//   - On success, write the full sum into d_out[0] and return 0.
//   - While the kernel is still a stub, return the TODO sentinel 77 and do NOT
//     touch d_out -- the harness sees 77 and prints "[TODO] ..." instead of checking.
//   - IMPORTANT: the harness does NOT pre-zero d_out. If you finalize with
//     atomicAdd, you must cudaMemset(d_out, 0, sizeof(float)) inside solve()
//     yourself before launching.

#include <cuda_runtime.h>

#define FULL_MASK 0xffffffffu

// ---------------------------------------------------------------------------
// The reduction kernel(s).
//
// TODO: each thread accumulates a partial over a grid-stride range of d_in
//       (so one launch covers any n).
// TODO: reduce within the warp using __shfl_down_sync with delta = 16,8,4,2,1
//       and the FULL_MASK = 0xffffffff participation mask.
// TODO: lane 0 of each warp writes its partial to a small __shared__ array;
//       __syncthreads().
// TODO: have the first warp load those per-warp partials and warp-reduce again.
// TODO: combine block results into d_out[0] -- via atomicAdd, or by writing one
//       partial per block and reducing those in a second kernel/launch.
// ---------------------------------------------------------------------------
__global__ void warp_reduce_kernel(const float* d_in, float* d_out, int n) {
    // TODO: implement the warp-shuffle reduction described above.
    (void)d_in;
    (void)d_out;
    (void)n;
}

// ---------------------------------------------------------------------------
// Host launcher. The harness calls this.
// ---------------------------------------------------------------------------
extern "C" int solve(const float* d_in, float* d_out, int n) {
    // TODO: zero d_out (cudaMemset) if you finalize with atomicAdd,
    //       choose a launch config sized for occupancy (not one-thread-per-element),
    //       launch the kernel(s), then return 0.
    (void)d_in;
    (void)d_out;
    (void)n;

    // Stub: report "not implemented yet" to the harness without doing real work.
    return 77;
}
