// c04 — Conflict-free transpose
//
// YOU write this file. Out-of-place transpose: d_out (cols x rows) is the
// transpose of d_in (rows x cols), both row-major.
//
// The whole point of this exercise is shared-memory STAGING + the bank-conflict
// fix. A naive transpose can coalesce the read OR the write but not both — route
// the tile through __shared__ so BOTH global accesses are coalesced, then pad the
// tile so the transposed (column) read out of shared memory doesn't collide on a
// single bank.
//
// harness.cu owns every cudaMalloc / cudaMemcpy / timing. You get device pointers
// that are already filled, and you launch the transpose over them.
//
// While solve() still returns the sentinel (77), the harness prints [TODO] and
// exits without checking anything. Return 0 once you actually launch.

__global__ void transpose_kernel(const float* in, float* out, int rows, int cols) {
    // TODO: declare a __shared__ tile sized TILE x (TILE+1) — the +1 is the
    //       bank-conflict fix (column neighbors land in consecutive banks).
    //
    // TODO: compute the global (row, col) this thread READS from `in`. Lay the
    //       block out so consecutive threadIdx.x walk consecutive `cols` — that
    //       makes the global READ coalesced.
    //
    // TODO: store the loaded value into the shared tile at [threadIdx.y][threadIdx.x],
    //       guarded by the edge check (rows/cols are NOT always multiples of TILE).
    //
    // TODO: __syncthreads() so the entire staged tile has landed before anyone
    //       reads it back transposed.
    //
    // TODO: compute the TRANSPOSED global (row, col) for the WRITE. This is a
    //       DIFFERENT index map from the read: the block's row/col roles swap so
    //       that consecutive threadIdx.x again walk consecutive output columns —
    //       keeping the global WRITE coalesced too.
    //
    // TODO: read the shared tile with the indices SWAPPED ([threadIdx.x][threadIdx.y])
    //       and write to `out`, guarded by the edge check.
}

extern "C" int solve(const float* d_in, float* d_out, int rows, int cols) {
    // TODO: choose a square block dim3(TILE, TILE) and a 2-D grid that covers the
    //       rows x cols input (ceil-div on each axis).
    // TODO: launch transpose_kernel<<<grid, block>>>(d_in, d_out, rows, cols).
    // TODO: return 0 after launching (replace the `return 77;` sentinel).
    return 77;
}
