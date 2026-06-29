// TITLE: Warp-shuffle reduction (array sum)
//
// PROVIDED test harness -- do not edit.
//
// Builds together with kernel.cu (the student's solve() is linked in via the
// extern "C" declaration below). Allocates a deterministic input, calls solve(),
// and either reports [TODO], checks correctness ([PASS]/[FAIL]), or times the
// kernel and reports bandwidth ([PERF]).
//
// Output contract (must match the Triton runner):
//   [TODO] ...                      solve() returned the sentinel; nothing checked
//   [PASS] correct                  result within tolerance
//   [FAIL] ...                      wrong answer / launch error
//   [PERF] <ms> ms   <GB/s>         timing of the bandwidth-bound size

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _err = (call);                                            \
        if (_err != cudaSuccess) {                                            \
            std::fprintf(stderr, "[FAIL] CUDA error %s at %s:%d: %s\n",       \
                         #call, __FILE__, __LINE__,                           \
                         cudaGetErrorString(_err));                          \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// The student's launcher, defined in kernel.cu.
extern "C" int solve(const float* d_in, float* d_out, int n);

// TODO sentinel: solve() returns this while the kernel is still a stub.
static const int TODO_SENTINEL = 77;

// Deterministic LCG fill in a SMALL range so the true sum stays O(n) and fp32
// round-off is bounded. Each element ~ U[-1, 1] + 0.5 (so the mean is nonzero
// and the running sum grows ~0.5*n, large enough to exercise fp32 cancellation
// while staying a fair target for a double-precision oracle).
static void fill_lcg(std::vector<float>& h, unsigned seed) {
    unsigned state = seed;
    for (size_t i = 0; i < h.size(); ++i) {
        state = 1664525u * state + 1013904223u;      // Numerical Recipes LCG
        float u = (state >> 8) * (1.0f / 16777216.0f); // 24-bit mantissa -> [0,1)
        h[i] = (2.0f * u - 1.0f) + 0.5f;               // [-1,1) shifted by +0.5
    }
}

// Host double-precision oracle: a fair, order-independent reference for a
// parallel tree (which reorders the additions).
static double host_sum(const std::vector<float>& h) {
    double acc = 0.0;
    for (float v : h) acc += static_cast<double>(v);
    return acc;
}

// Run solve() once on a given size and return the device result in *out_scalar.
// Returns the value solve() returned (so the caller can detect the sentinel).
static int run_once(const std::vector<float>& h_in, float* out_scalar) {
    const int n = static_cast<int>(h_in.size());
    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));
    // NOTE: we deliberately do NOT zero d_out -- the student must do that
    // inside solve() if they finalize with atomicAdd.

    int rc = solve(d_in, d_out, n);

    if (rc != TODO_SENTINEL) {
        // Surface any launch/runtime error the student's kernel triggered.
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out_scalar, d_out, sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return rc;
}

// Relative-tolerance compare: |a - b| <= atol + rtol*|b|.
static bool close(double got, double ref, double rtol, double atol) {
    return std::fabs(got - ref) <= atol + rtol * std::fabs(ref);
}

int main() {
    const double RTOL = 1e-3;
    const double ATOL = 1e-2;

    // --- 1. Tiny size FIRST (correctness, untimed): catches tail off-by-one.
    {
        const int n_small = 100;
        std::vector<float> h(n_small);
        fill_lcg(h, /*seed=*/12345u);

        float got = 0.0f;
        int rc = run_once(h, &got);
        if (rc == TODO_SENTINEL) {
            std::printf("[TODO] solve() returned the sentinel -- go write the "
                        "kernel in kernel.cu.\n");
            return 0;
        }
        double ref = host_sum(h);
        if (!close(static_cast<double>(got), ref, RTOL, ATOL)) {
            std::printf("[FAIL] tiny size n=%d: got %.6f, expected %.6f "
                        "(check the tail of your grid-stride loop)\n",
                        n_small, static_cast<double>(got), ref);
            return 0;
        }
    }

    // --- 2. Large size (correctness, then timed): bandwidth-bound.
    const int n = 1 << 24;  // 16,777,216 floats = 64 MB
    std::vector<float> h(n);
    fill_lcg(h, /*seed=*/987654321u);
    const double ref = host_sum(h);

    // Persistent device buffers for the timed runs.
    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Correctness on the large size.
    int rc = solve(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float got = 0.0f;
    CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    if (!close(static_cast<double>(got), ref, RTOL, ATOL)) {
        std::printf("[FAIL] n=%d: got %.6f, expected %.6f\n", n,
                    static_cast<double>(got), ref);
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        return 0;
    }
    std::printf("[PASS] correct\n");

    // --- 3. Timing with CUDA events (median of repeated launches).
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int WARMUP = 10;
    const int ITERS = 50;
    for (int i = 0; i < WARMUP; ++i) {
        solve(d_in, d_out, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times_ms(ITERS);
    for (int i = 0; i < ITERS; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        solve(d_in, d_out, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times_ms[i], start, stop));
    }

    // Median.
    for (int i = 0; i < ITERS; ++i)
        for (int j = i + 1; j < ITERS; ++j)
            if (times_ms[j] < times_ms[i]) {
                float t = times_ms[i];
                times_ms[i] = times_ms[j];
                times_ms[j] = t;
            }
    float ms = times_ms[ITERS / 2];

    // Bandwidth is dominated by the n input reads; the single output write is
    // negligible (e04 'dominated by the read' convention).
    double gbps = (static_cast<double>(n) * sizeof(float)) / (ms * 1e-3) / 1e9;
    std::printf("[PERF] %.3f ms   %.0f GB/s\n", ms, gbps);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
