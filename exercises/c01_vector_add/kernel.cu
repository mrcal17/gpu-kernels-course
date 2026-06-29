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
    // TODO: in the __global__ kernel, compute this thread's global index
    //       i = blockIdx.x*blockDim.x + threadIdx.x
    // TODO: guard the ragged tail with if (i < n) before touching memory
    // TODO: inside the guard, out[i] = a[i] + b[i]
}

extern "C" int solve(const float* d_a, const float* d_b, float* d_out, int n) {
    // TODO: in solve(), choose blockDim (a multiple of 32) and
    //       gridDim = ceil(n / blockDim)
    // TODO: launch the kernel with <<<grid, block>>>(d_a, d_b, d_out, n)
    // TODO: return 0 after launching (replace the `return 77;` sentinel)
    return 77;
}
