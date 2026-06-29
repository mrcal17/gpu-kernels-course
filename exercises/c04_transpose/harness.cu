// TITLE: Conflict-free transpose
//
// PROVIDED HARNESS — do not edit. This file owns main(): it allocates
// deterministic inputs, calls your solve() (from kernel.cu), checks the result,
// and times the kernel. You only write kernel.cu.
//
// Contract / output format (matches the Triton runner):
//   solve() still returns the sentinel 77  -> prints "[TODO] ..."  and exits 0
//   wrong answer                            -> prints "[FAIL] ..."  and exits 1
//   correct                                 -> prints "[PASS] correct"
//                                              then "[PERF] <ms> ms   <GB/s>"
//
// The inputs use an INDEX-ENCODING fill: h_in[r*cols + c] = (float)(r*cols + c).
// That makes a transposed reference trivial to state exactly and makes any
// wrong-index bug jump out. Because the largest linear index at the timed size
// (4096*4096-1 = 2^24-1) is still exactly representable in fp32, the comparison
// is EXACT (atol = 0). Do not push rows*cols past 2^24 with this fill.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ---- self-contained error check -------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",               \
                         cudaGetErrorName(err__), __FILE__, __LINE__,          \
                         cudaGetErrorString(err__));                           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// The student implements this in kernel.cu.
extern "C" int solve(const float* d_in, float* d_out, int rows, int cols);

static const int TODO_SENTINEL = 77;

// Out-of-place transpose on the host for the reference / correctness check.
// out_ref[c*rows + r] = in[r*cols + c]
static void transpose_ref(const float* in, float* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            out[(size_t)c * rows + r] = in[(size_t)r * cols + c];
}

// Run solve() once on a given size and verify the transpose EXACTLY (atol = 0).
// Returns:  1 = pass, 0 = fail, TODO_SENTINEL = student hasn't implemented yet.
static int check_size(int rows, int cols, const char* label) {
    const size_t n = (size_t)rows * cols;
    const size_t bytes = n * sizeof(float);

    float* h_in  = (float*)std::malloc(bytes);
    float* h_out = (float*)std::malloc(bytes);
    float* h_ref = (float*)std::malloc(bytes);
    if (!h_in || !h_out || !h_ref) {
        std::fprintf(stderr, "host alloc failed\n");
        std::exit(1);
    }

    // Index-encoding fill (deterministic): exact in fp32 while n <= 2^24.
    for (size_t i = 0; i < n; ++i) h_in[i] = (float)i;
    transpose_ref(h_in, h_ref, rows, cols);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    // Poison the output so a kernel that skips elements is caught.
    CUDA_CHECK(cudaMemset(d_out, 0xFF, bytes));

    int rc = solve(d_in, d_out, rows, cols);
    if (rc == TODO_SENTINEL) {
        std::free(h_in); std::free(h_out); std::free(h_ref);
        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
        return TODO_SENTINEL;
    }

    CUDA_CHECK(cudaGetLastError());      // catch a bad launch config
    CUDA_CHECK(cudaDeviceSynchronize()); // catch an async kernel fault
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    int ok = 1;
    for (size_t i = 0; i < n && ok; ++i) {
        if (h_out[i] != h_ref[i]) {
            int r = (int)(i / rows);     // out is cols x rows
            int c = (int)(i % rows);
            std::printf("[FAIL] %s: out[%d][%d] (linear %zu) = %.1f, expected %.1f\n",
                        label, r, c, i, h_out[i], h_ref[i]);
            ok = 0;
        }
    }

    std::free(h_in); std::free(h_out); std::free(h_ref);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return ok;
}

// Time the timed (square, multiple-of-32) size with CUDA events.
static void perf(int rows, int cols) {
    const size_t n = (size_t)rows * cols;
    const size_t bytes = n * sizeof(float);

    float* h_in = (float*)std::malloc(bytes);
    if (!h_in) { std::fprintf(stderr, "host alloc failed\n"); std::exit(1); }
    for (size_t i = 0; i < n; ++i) h_in[i] = (float)i;

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Warmup.
    for (int i = 0; i < 5; ++i) (void)solve(d_in, d_out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iters = 50;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) (void)solve(d_in, d_out, rows, cols);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float ms = total_ms / iters;

    // Read every element once, write every element once.
    double moved = 2.0 * (double)rows * (double)cols * sizeof(float);
    double gbps = moved / (ms * 1e-3) / 1e9;
    std::printf("[PERF] %.3f ms   %.0f GB/s\n", ms, gbps);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::free(h_in);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

int main() {
    // 1) Non-square, non-multiple-of-TILE size FIRST: forces the edge guards so a
    //    student can't hardcode a square power-of-two and pass.
    int rc = check_size(1000, 1500, "1000x1500");
    if (rc == TODO_SENTINEL) {
        std::printf("[TODO] solve() returns the sentinel -- go write the kernel.\n");
        return 0;
    }
    if (rc == 0) return 1;

    // 2) Timed square size (multiple of 32): correctness, then bandwidth.
    const int N = 4096;
    rc = check_size(N, N, "4096x4096");
    if (rc == 0) return 1;

    std::printf("[PASS] correct\n");
    perf(N, N);
    return 0;
}
