// TITLE: Vector add (the nvcc build loop)
// c01 — Vector add: the test harness. DO NOT EDIT.
//
// Self-contained driver: it builds deterministic inputs, calls your solve()
// (declared extern, linked from kernel.cu), and then:
//   - if solve() returns the TODO sentinel (77), prints "[TODO] ..." and exits 0;
//   - otherwise it syncs, checks for launch errors, copies the result back,
//     compares against a host reference (atol=1e-5), and on success times the
//     kernel with cudaEvents and prints "[PERF] <ms> ms   <GB/s>".
//
// Output format matches the Triton runner: [TODO] / [PASS] correct /
// [FAIL] ... / [PERF] <ms> ms   <metric>.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

// The function YOU write in kernel.cu.
extern "C" int solve(const float* d_a, const float* d_b, float* d_out, int n);

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr, "[FATAL] %s:%d: %s\n", __FILE__, __LINE__,    \
                         cudaGetErrorString(_err));                            \
            std::exit(1);                                                      \
        }                                                                       \
    } while (0)

static const int TODO_SENTINEL = 77;

// Hardcoded HBM roofline for the %-of-peak suffix (mirrors SEGMENTATION.md /
// the runner's _CUDA_PEAK_BW_GBPS). Aim for a large fraction of this.
static const double PEAK_BW_GBPS = 896.0;

// A tiny deterministic host LCG so inputs are reproducible run-to-run and do
// not depend on any RNG library. Seeded at 0. Returns a float in roughly
// [-1, 1).
static float lcg_next(uint32_t* state) {
    // Numerical Recipes constants.
    *state = 1664525u * (*state) + 1013904223u;
    // top 24 bits -> [0,1), then shift to [-1,1).
    float u = (float)(*state >> 8) / (float)(1u << 24);
    return 2.0f * u - 1.0f;
}

static void fill_inputs(float* a, float* b, int n) {
    uint32_t sa = 0u;          // seed 0
    uint32_t sb = 0x9E3779B9u; // a different stream for b
    for (int i = 0; i < n; ++i) {
        a[i] = lcg_next(&sa);
        b[i] = lcg_next(&sb);
    }
}

// Run solve() once on size n. If it returns the sentinel, report [TODO] and
// signal the caller to stop (returns false). Otherwise verify correctness and
// return true. `time_it` controls whether we also print a [PERF] line.
static bool run_size(int n, bool time_it) {
    size_t bytes = (size_t)n * sizeof(float);

    float* h_a = (float*)std::malloc(bytes);
    float* h_b = (float*)std::malloc(bytes);
    float* h_out = (float*)std::malloc(bytes);
    if (!h_a || !h_b || !h_out) {
        std::fprintf(stderr, "[FATAL] host malloc failed for n=%d\n", n);
        std::exit(1);
    }
    fill_inputs(h_a, h_b, n);

    float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int rc = solve(d_a, d_b, d_out, n);

    if (rc == TODO_SENTINEL) {
        std::printf("[TODO] solve() returned the sentinel -- go write the "
                    "kernel and return 0.\n");
        // Clean up and tell the caller to stop here.
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
        std::free(h_a); std::free(h_b); std::free(h_out);
        return false;
    }

    // Surface async launch errors (bad config, out-of-bounds, etc.) that would
    // otherwise only show up later.
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Host reference: exact fp32 add. atol=1e-5 absorbs nothing in particular
    // (fp32 a+b is exact) but keeps the check honest.
    const float atol = 1e-5f;
    int bad = 0;
    int first_bad = -1;
    for (int i = 0; i < n; ++i) {
        float ref = h_a[i] + h_b[i];
        float diff = std::fabs(h_out[i] - ref);
        if (diff > atol) {
            if (first_bad < 0) first_bad = i;
            ++bad;
        }
    }

    if (bad != 0) {
        std::printf("[FAIL] wrong answer: %d / %d elements off (first at i=%d: "
                    "got %.7g, want %.7g)\n",
                    bad, n, first_bad, h_out[first_bad],
                    h_a[first_bad] + h_b[first_bad]);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
        std::free(h_a); std::free(h_b); std::free(h_out);
        std::exit(1);
    }

    std::printf("[PASS] correct (n=%d)\n", n);

    if (time_it) {
        // Warm up, then time the kernel alone (no PCIe copies) with cudaEvents.
        const int warmup = 10;
        const int iters = 50;
        for (int i = 0; i < warmup; ++i) {
            (void)solve(d_a, d_b, d_out, n);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i) {
            (void)solve(d_a, d_b, d_out, n);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        double ms = (double)total_ms / iters;

        // read a, read b, write out = 3 arrays.
        double bytes_moved = 3.0 * (double)n * (double)sizeof(float);
        double gbps = bytes_moved / (ms * 1e-3) / 1e9;
        double pct = 100.0 * gbps / PEAK_BW_GBPS;
        std::printf("[PERF] %.3f ms   %.0f GB/s  (%.0f%% of ~%.0f GB/s peak)\n",
                    ms, gbps, pct, PEAK_BW_GBPS);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    std::free(h_a); std::free(h_b); std::free(h_out);
    return true;
}

int main() {
    // Small ragged size FIRST: n=1000 is not a multiple of any sane blockDim,
    // so it catches a missing `if (i < n)` tail guard. Not timed.
    if (!run_size(1000, /*time_it=*/false)) {
        return 0;  // [TODO] already printed.
    }

    // The timed run: a clean power of two for tidy bandwidth numbers.
    const int N = 1 << 24;  // 16,777,216 floats = 64 MB/array, 192 MB total.
    run_size(N, /*time_it=*/true);
    return 0;
}
