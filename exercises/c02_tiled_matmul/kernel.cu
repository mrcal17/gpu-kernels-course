// c02 — Tiled matmul, shared memory by hand.
//
// YOU write this file: the __global__ kernel and the host launcher solve().
// harness.cu is provided (do not edit) — it allocates deterministic inputs,
// calls solve(), checks correctness, and times the kernel.
//
// Right now solve() returns the TODO sentinel (77) WITHOUT launching anything,
// so the harness prints "[TODO] ..." and exits. Replace the TODOs with a real
// tiled kernel and a real launch, then make solve() return 0.

// The TODO sentinel solve() returns until you implement the kernel.
// (The harness treats this exact value as "not done yet".)
#define TODO_SENTINEL 77

// TODO: pick a tile edge length and #define it (a multiple of the warp size is
//       the usual choice). Both shared tiles below are TILE x TILE.
// #define TILE ??

__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    // TODO: declare two __shared__ tiles, As[TILE][TILE] and Bs[TILE][TILE].

    // TODO: map threadIdx.x/.y + blockIdx.x/.y to this thread's output (row, col).

    // TODO: an fp32 accumulator, initialized to 0, carried across the K loop.

    // TODO: loop over K in steps of TILE. Each step:
    //   - cooperatively load ONE element of the A-tile and ONE of the B-tile
    //     (derive each from a global address: row*N + k arithmetic).
    //   - __syncthreads() AFTER the loads, BEFORE the inner-product loop.
    //   - inner loop over TILE: acc += As[ty][k] * Bs[k][tx].
    //   - __syncthreads() AFTER compute, BEFORE the next iteration overwrites
    //     the tiles.

    // TODO: write acc to C with a bounds guard for ragged N.
}

// Host launcher. d_A, d_B, d_C are DEVICE pointers (already allocated and
// filled by the harness). C is N x N row-major; A and B are N x N row-major.
extern "C" int solve(const float* d_A, const float* d_B, float* d_C, int N) {
    // TODO: launch matmul_tiled with a 2-D block dim3(TILE, TILE) and a 2-D grid
    //       that covers an N x N output (round the grid up so the ragged tail is
    //       still covered). Then return 0.
    //
    // While this stub returns the sentinel, nothing is launched and the harness
    // reports [TODO]. Delete the next line once your kernel is wired up.
    (void)d_A; (void)d_B; (void)d_C; (void)N;
    return TODO_SENTINEL;
}
