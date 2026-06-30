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

__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int N) {
    // TODO: declare two __shared__ tiles, As[TILE][TILE] and Bs[TILE][TILE].

    // TODO: map threadIdx.x/.y + blockIdx.x/.y to this thread's output (row, col).

    // TODO: an fp32 accumulator, initialized to 0, carried across the K loop.

    // TODO: loop over K one tile-width at a time. Each step you'll cooperatively
    //       stage the next A and B tiles into shared memory, place the barriers
    //       you reasoned about in the hints, and accumulate from on-chip data.
    //       Decide the order of stage/sync/compute/sync and where each barrier
    //       goes yourself.
    //   - For the inner product: walk k across the staged tile and accumulate the
    //     product of the A-tile element on this thread's tile-row and the B-tile
    //     element on this thread's tile-column. Work out which of (tile-row,
    //     tile-col, k) indexes each shared array — A and B are not indexed the
    //     same way.

    // TODO: write acc to C with a bounds guard for ragged N.
}

// Host launcher. d_A, d_B, d_C are DEVICE pointers (already allocated and
// filled by the harness). C is N x N row-major; A and B are N x N row-major.
extern "C" int solve(const float* d_A, const float* d_B, float* d_C, int N) {
    // TODO: launch matmul_tiled. Choose a 2-D block whose thread count matches
    //       one output element per thread of a single tile, and a 2-D grid sized
    //       so every output tile of the N x N result (including the ragged tail)
    //       is covered. Derive both extents yourself. Then return 0.
    //
    //       (Here the grid is a fixed tuple because the tile size is a compile-time
    //       constant. In e10 the tile sizes come from an autotune config, so the
    //       grid becomes a callable that reads the chosen config — but that's later.)
    //
    // While this stub returns the sentinel, nothing is launched and the harness
    // reports [TODO]. Delete the next line once your kernel is wired up.
    (void)d_A; (void)d_B; (void)d_C; (void)N;
    return TODO_SENTINEL;
}
