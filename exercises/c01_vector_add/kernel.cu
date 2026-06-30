// c01 — Vector add (the nvcc hello-world)
//
// YOU write this file. Two things to fill in:
//   1) the __global__ kernel below (the body is the code of ONE thread), and
//   2) solve(), which configures the launch and fires the kernel.
//
// harness.cu owns all cudaMalloc / cudaMemcpy / timing — you only get device
// pointers that are already filled, and you launch the add over them.
//
// While solve() still returns the sentinel (77), the harness prints [TODO] and
// exits without checking anything. Return 0 once you actually launch.

__global__ void vector_add(const float* a, const float* b, float* out, int n) {
    // TODO: compute the flat global position this thread is responsible for, by
    //       combining its block index, the block size, and its index within the
    //       block. Re-derive the arithmetic; don't look it up.
    // TODO: before this thread touches memory, guard the ragged tail -- only
    //       proceed when this thread's position actually falls inside the array.
    // TODO: inside the guard, do the elementwise work: write the sum of the two
    //       inputs at this thread's position into the output at the same position.
}

extern "C" int solve(const float* d_a, const float* d_b, float* d_out, int n) {
    // TODO: choose a threads-per-block count (a whole number of warps), then size
    //       the grid so every element is covered even when the block size doesn't
    //       divide n evenly (round the block count up).
    // TODO: launch the kernel with your chosen grid/block configuration, passing
    //       the three device pointers and the length.
    // TODO: return 0 after launching (replace the `return 77;` sentinel).
    return 77;
}
