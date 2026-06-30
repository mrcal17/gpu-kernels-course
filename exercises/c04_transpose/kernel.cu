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
    // TODO: declare a __shared__ tile big enough to stage one block's worth of
    //       data. A column read of an unpadded tile whose width is a multiple of
    //       32 collides on one bank — adjust the inner dimension so column
    //       neighbors fall in different banks.
    //
    // TODO: compute the global (row, col) this thread reads from `in`. Arrange the
    //       mapping so that, within a warp, consecutive threads read consecutive
    //       global addresses — that is what makes the read coalesced.
    //
    // TODO: store the loaded value into the staged tile, indexing it by this
    //       thread's position within the block, guarded by the edge check
    //       (rows/cols are NOT always multiples of the tile size).
    //
    // TODO: __syncthreads() so the entire staged tile has landed before anyone
    //       reads it back transposed.
    //
    // TODO: compute the TRANSPOSED global (row, col) for the WRITE. This is a
    //       DIFFERENT index map from the read: the block's row/col roles swap so
    //       that consecutive threadIdx.x again walk consecutive output columns —
    //       keeping the global WRITE coalesced too.
    //
    // TODO: read the staged tile back transposed — i.e. with the two in-block
    //       coordinates exchanged relative to how you stored it — and write to
    //       `out`, guarded by the edge check.
}

extern "C" int solve(const float* d_in, float* d_out, int rows, int cols) {
    // TODO: pick a 2-D block whose threads map onto one tile, and size a 2-D grid
    //       so the blocks cover the whole input even when a dimension isn't an
    //       exact multiple of the tile.
    // TODO: launch transpose_kernel<<<grid, block>>>(d_in, d_out, rows, cols).
    // TODO: return 0 after launching (replace the `return 77;` sentinel).
    return 77;
}
