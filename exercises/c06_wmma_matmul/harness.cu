// TITLE: WMMA tensor-core matmul (fp16)
//
// c06 — WMMA tensor-core matmul: the test harness. DO NOT EDIT.
//
// Self-contained driver. It builds deterministic fp32 inputs, converts them to
// __half for the device (keeping the fp32 originals for the reference), calls
// your solve() (declared extern, linked from kernel.cu), and then:
//   - if solve() returns the TODO sentinel (77), prints "[TODO] ..." and exits 0;
//   - otherwise it syncs, checks for launch errors, copies C back, compares
//     against a host fp32 triple-loop reference (loose tolerance — see below),
//     and on success times the kernel with cudaEvents and prints
//     "[PERF] <ms> ms   <TFLOP/s>".
//
// PRECISION NOTE (why the tolerance is loose): the device inputs are fp16, so
// each element loses ~3 mantissa bits versus the fp32 originals the reference
// uses, and that per-element error accumulates across the K dimension. The
// answer is therefore CLOSE, not exact — that is the throughput-for-precision
// trade tensor cores make, not a bug. We accept atol=1, rtol=2e-2.
//
// Output format matches the Triton runner: [TODO] / [PASS] correct /
// [FAIL] ... / [PERF] <ms> ms   <metric>.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>   // __half, __float2half — needed for the buffers AND
                         // for the extern "C" solve() declaration below.

// The function YOU write in kernel.cu. Note the __half device pointers: the
// harness owns the fp32->fp16 conversion, so solve() only ever sees fp16 A/B.
extern "C" int solve(const __half* d_A, const __half* d_B, float* d_C, int N);

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
// not depend on any RNG library. Returns a float in a small range so that the
// fp32 accumulation across K stays well-conditioned and the fp16 rounding
// error stays inside the documented tolerance.
static float lcg_next(uint32_t* state) {
    // Numerical Recipes constants.
    *state = 1664525u * (*state) + 1013904223u;
    // top 24 bits -> [0,1), then shift to a small symmetric range ~[-0.5,0.5).
    float u = (float)(*state >> 8) / (float)(1u << 24);
    return u - 0.5f;
}

static void fill_inputs(float* a, float* b, int n) {
    uint32_t sa = 0u;          // seed 0
    uint32_t sb = 0x9E3779B9u; // a different stream for b
    for (int i = 0; i < n; ++i) {
        a[i] = lcg_next(&sa);
        b[i] = lcg_next(&sb);
    }
}

int main() {
    // Timed (and only) size. N=512 is a multiple of the 16x16x16 WMMA tile
    // (512 = 32*16) AND of a 32x32 warp-tiled block, so the student can start
    // WITHOUT a K-tail special case. A ragged N (not a multiple of 16) would
    // need the student to mask the edge tiles — a later refinement, not graded
    // here.
    const int N = 512;
    const size_t elems = (size_t)N * (size_t)N;
    const size_t f32_bytes = elems * sizeof(float);
    const size_t f16_bytes = elems * sizeof(__half);

    // Host fp32 originals (kept for the reference) + host fp16 (for the device).
    float* h_A   = (float*)std::malloc(f32_bytes);
    float* h_B   = (float*)std::malloc(f32_bytes);
    float* h_C   = (float*)std::malloc(f32_bytes);   // device result copied back
    float* h_ref = (float*)std::malloc(f32_bytes);   // fp32 oracle
    __half* h_Ah = (__half*)std::malloc(f16_bytes);
    __half* h_Bh = (__half*)std::malloc(f16_bytes);
    if (!h_A || !h_B || !h_C || !h_ref || !h_Ah || !h_Bh) {
        std::fprintf(stderr, "[FATAL] host malloc failed for N=%d\n", N);
        std::exit(1);
    }

    fill_inputs(h_A, h_B, (int)elems);
    // Convert the fp32 originals to __half for the device inputs.
    for (size_t i = 0; i < elems; ++i) {
        h_Ah[i] = __float2half(h_A[i]);
        h_Bh[i] = __float2half(h_B[i]);
    }

    __half *d_A = nullptr, *d_B = nullptr;
    float  *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, f16_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, f16_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, f32_bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_Ah, f16_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_Bh, f16_bytes, cudaMemcpyHostToDevice));

    int rc = solve(d_A, d_B, d_C, N);

    if (rc == TODO_SENTINEL) {
        std::printf("[TODO] solve() returned the sentinel -- go write the "
                    "kernel and return 0.\n");
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        std::free(h_A); std::free(h_B); std::free(h_C);
        std::free(h_ref); std::free(h_Ah); std::free(h_Bh);
        return 0;
    }

    // Surface async launch errors (bad config, out-of-bounds, etc.).
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, f32_bytes, cudaMemcpyDeviceToHost));

    // Host fp32 reference: a plain triple loop over the fp32 ORIGINALS. One-time
    // and untimed; at N=512 it is ~0.13 GFLOP, so it stays snappy. Row-major.
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < N; ++k) {
                acc += h_A[(size_t)i * N + k] * h_B[(size_t)k * N + j];
            }
            h_ref[(size_t)i * N + j] = acc;
        }
    }

    // Loose tolerance BY DESIGN: fp16 inputs round the mantissa and the error
    // grows with K. atol=1, rtol=2e-2. See the precision note at the top.
    const float atol = 1.0f;
    const float rtol = 2e-2f;
    int bad = 0;
    int first_bad = -1;
    for (size_t i = 0; i < elems; ++i) {
        float got = h_C[i];
        float ref = h_ref[i];
        float tol = atol + rtol * std::fabs(ref);
        if (std::fabs(got - ref) > tol) {
            if (first_bad < 0) first_bad = (int)i;
            ++bad;
        }
    }

    if (bad != 0) {
        std::printf("[FAIL] wrong answer: %d / %zu elements off (first at "
                    "i=%d: got %.6g, want %.6g)\n",
                    bad, elems, first_bad, h_C[first_bad], h_ref[first_bad]);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        std::free(h_A); std::free(h_B); std::free(h_C);
        std::free(h_ref); std::free(h_Ah); std::free(h_Bh);
        std::exit(1);
    }

    std::printf("[PASS] correct (N=%d)\n", N);

    // --- performance: time the kernel alone (no PCIe copies) with cudaEvents.
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

    // GEMM flops = 2*N^3 (one multiply + one add per inner-product term).
    double flop = 2.0 * (double)N * (double)N * (double)N;
    double tflops = flop / (ms * 1e-3) / 1e12;
    std::printf("[PERF] %.3f ms   %.1f TFLOP/s\n", ms, tflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    std::free(h_A); std::free(h_B); std::free(h_C);
    std::free(h_ref); std::free(h_Ah); std::free(h_Bh);
    return 0;
}
