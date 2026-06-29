// TITLE: Pipelined double-buffered matmul
//
// c05 — Pipelined matmul: the test harness. DO NOT EDIT.
//
// Self-contained driver: it builds deterministic inputs, calls your solve()
// (declared extern, linked from kernel.cu), and then:
//   - if solve() returns the TODO sentinel (77), prints "[TODO] ..." and exits 0;
//   - otherwise it syncs, checks for launch errors, copies the result back,
//     compares against a host triple-loop reference (atol=1e-1, rtol=1e-2), and
//     on success times the kernel with cudaEvents and prints
//     "[PERF] <ms> ms   <TFLOP/s>".
//
// Same contract and oracle as c02: square N x N x N fp32 matmul, row-major.
// The timed size is N=768 (a multiple of tiles 16/32/64; 768 = 256*3). The
// host fp32 triple-loop oracle runs ONCE, untimed -- N=768 (~0.9 GFLOP of host
// work) keeps that one-time check snappy while still giving 768/TILE K-steps,
// plenty for load-latency hiding to matter.
//
// Output format matches the Triton runner: [TODO] / [PASS] correct /
// [FAIL] ... / [PERF] <ms> ms   <metric>.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

// The function YOU write in kernel.cu.
extern "C" int solve(const float* d_A, const float* d_B, float* d_C, int N);

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

// A tiny deterministic host LCG so inputs are reproducible run-to-run and do
// not depend on any RNG library. Returns a float in roughly [-1, 1).
static float lcg_next(uint32_t* state) {
    // Numerical Recipes constants.
    *state = 1664525u * (*state) + 1013904223u;
    // top 24 bits -> [0,1), then shift to [-1,1).
    float u = (float)(*state >> 8) / (float)(1u << 24);
    return 2.0f * u - 1.0f;
}

static void fill_inputs(float* A, float* B, int N) {
    uint32_t sa = 0u;          // seed 0
    uint32_t sb = 0x9E3779B9u; // a different stream for B
    int nn = N * N;
    for (int i = 0; i < nn; ++i) A[i] = lcg_next(&sa);
    for (int i = 0; i < nn; ++i) B[i] = lcg_next(&sb);
}

// Host reference: a plain fp32 triple loop, C = A @ B (row-major). This is the
// oracle -- run once, untimed. Identical contract to c02.
static void host_matmul(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < N; ++k) {
                acc += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

// Run solve() once on an N x N x N problem. If it returns the sentinel, report
// [TODO] and signal the caller to stop (returns false). Otherwise verify
// correctness and return true. `time_it` controls whether we also print [PERF].
static bool run_size(int N, bool time_it) {
    size_t elems = (size_t)N * (size_t)N;
    size_t bytes = elems * sizeof(float);

    float* h_A = (float*)std::malloc(bytes);
    float* h_B = (float*)std::malloc(bytes);
    float* h_C = (float*)std::malloc(bytes);
    float* h_ref = (float*)std::malloc(bytes);
    if (!h_A || !h_B || !h_C || !h_ref) {
        std::fprintf(stderr, "[FATAL] host malloc failed for N=%d\n", N);
        std::exit(1);
    }
    fill_inputs(h_A, h_B, N);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    int rc = solve(d_A, d_B, d_C, N);

    if (rc == TODO_SENTINEL) {
        std::printf("[TODO] solve() returned the sentinel -- go write the "
                    "kernel and return 0.\n");
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        std::free(h_A); std::free(h_B); std::free(h_C); std::free(h_ref);
        return false;
    }

    // Surface async launch errors (bad config, out-of-bounds, etc.) that would
    // otherwise only show up later.
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Host reference (untimed) and a tolerant compare: fp32 accumulation order
    // differs between the host triple loop and a tiled GPU kernel, so we allow
    // atol=1e-1, rtol=1e-2 (matches c02).
    host_matmul(h_A, h_B, h_ref, N);

    const float atol = 1e-1f;
    const float rtol = 1e-2f;
    int bad = 0;
    int first_bad = -1;
    for (size_t i = 0; i < elems; ++i) {
        float got = h_C[i];
        float ref = h_ref[i];
        float diff = std::fabs(got - ref);
        if (diff > atol + rtol * std::fabs(ref)) {
            if (first_bad < 0) first_bad = (int)i;
            ++bad;
        }
    }

    if (bad != 0) {
        int r = first_bad / N, c = first_bad % N;
        std::printf("[FAIL] wrong answer: %d / %zu elements off (first at "
                    "(%d,%d): got %.6g, want %.6g)\n",
                    bad, elems, r, c, h_C[first_bad], h_ref[first_bad]);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        std::free(h_A); std::free(h_B); std::free(h_C); std::free(h_ref);
        std::exit(1);
    }

    std::printf("[PASS] correct (N=%d)\n", N);

    if (time_it) {
        // Warm up, then time the kernel alone (no PCIe copies) with cudaEvents.
        const int warmup = 10;
        const int iters = 50;
        for (int i = 0; i < warmup; ++i) {
            (void)solve(d_A, d_B, d_C, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i) {
            (void)solve(d_A, d_B, d_C, N);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        double ms = (double)total_ms / iters;

        // 2 * N^3 flops (one multiply + one add per inner-loop step).
        double flop = 2.0 * (double)N * (double)N * (double)N;
        double tflops = flop / (ms * 1e-3) / 1e12;
        std::printf("[PERF] %.3f ms   %.2f TFLOP/s\n", ms, tflops);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    std::free(h_A); std::free(h_B); std::free(h_C); std::free(h_ref);
    return true;
}

int main() {
    // Small ragged size FIRST: N=750 is not a multiple of any sane TILE, so it
    // catches a missing edge guard in the tile loads or the C store. Not timed.
    if (!run_size(750, /*time_it=*/false)) {
        return 0;  // [TODO] already printed.
    }

    // The timed run: N=768 = 256*3, a multiple of tiles 16/32/64. Device buffers
    // are 3 * 768 * 768 * 4 = 7 MB total.
    const int N = 768;
    run_size(N, /*time_it=*/true);
    return 0;
}
