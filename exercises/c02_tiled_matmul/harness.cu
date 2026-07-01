// TITLE: Tiled matmul (shared memory)
// c02 — Tiled matmul: PROVIDED harness. DO NOT EDIT.
//
// Builds deterministic inputs, calls the student's solve() (in kernel.cu),
// checks correctness against a host triple-loop reference, then times the
// kernel with CUDA events.
//
// Output contract (matches the Triton runner):
//   [TODO] ...                 solve() returned the sentinel — not implemented
//   [PASS] correct             all sizes within tolerance
//   [FAIL] ...                 wrong answer (or a CUDA error)
//   [PERF] <ms> ms   <TFLOP/s> timing for the clean size
//
// Two sizes are exercised, ragged FIRST so a tail bug is caught before timing:
//   N = 257  correctness only, untimed  (forces the ragged-tail guard)
//   N = 512  correctness, then timed     (multiple of tiles 16/32)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _err = (call);                                         \
        if (_err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",              \
                    cudaGetErrorName(_err), __FILE__, __LINE__,            \
                    cudaGetErrorString(_err));                            \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// The student's launcher. Returns 0 when implemented, the sentinel otherwise.
extern "C" int solve(const float* d_A, const float* d_B, float* d_C, int N);

// Must match TODO_SENTINEL in kernel.cu.
static const int TODO_SENTINEL = 77;

// Deterministic LCG (Numerical Recipes constants) -> floats centered on 0,
// roughly in [-1, 1]. Keeping inputs small and centered keeps the fp32
// inner-product sums well-conditioned at N = 512.
static void fill_inputs(std::vector<float>& A, std::vector<float>& B, int N, unsigned seed) {
    unsigned state = seed;
    auto next = [&state]() -> float {
        state = 1664525u * state + 1013904223u;
        // map the 32-bit state to [-1, 1)
        return (float)((double)state / 4294967295.0) * 2.0f - 1.0f;
    };
    A.resize((size_t)N * N);
    B.resize((size_t)N * N);
    for (size_t i = 0; i < (size_t)N * N; ++i) A[i] = next();
    for (size_t i = 0; i < (size_t)N * N; ++i) B[i] = next();
}

// Host reference: plain triple loop, fp32 accumulate, row-major.
static void reference(const std::vector<float>& A, const std::vector<float>& B,
                      std::vector<float>& C, int N) {
    C.assign((size_t)N * N, 0.0f);
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float a = A[(size_t)i * N + k];
            const float* brow = &B[(size_t)k * N];
            float* crow = &C[(size_t)i * N];
            for (int j = 0; j < N; ++j) {
                crow[j] += a * brow[j];
            }
        }
    }
}

// allclose with atol=1e-1, rtol=1e-2 (mirrors e07; absorbs fp32 reordering).
static bool check_close(const std::vector<float>& got, const std::vector<float>& ref,
                        const char** why, int* bad_idx) {
    const float atol = 1e-1f, rtol = 1e-2f;
    for (size_t i = 0; i < ref.size(); ++i) {
        float g = got[i], r = ref[i];
        if (!(std::isfinite(g))) { *why = "non-finite output"; *bad_idx = (int)i; return false; }
        if (std::fabs(g - r) > atol + rtol * std::fabs(r)) {
            *why = "value mismatch"; *bad_idx = (int)i; return false;
        }
    }
    return true;
}

// Allocate device buffers, copy inputs, call solve(), copy C back.
// Returns the value solve() returned (so the caller can detect the sentinel).
static int run_size(int N, std::vector<float>& C_out) {
    std::vector<float> A, B;
    fill_inputs(A, B, N, /*seed=*/12345u + (unsigned)N);

    size_t bytes = (size_t)N * N * sizeof(float);
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));
    // Poison C with 0xFF (NaN-ish float garbage) instead of zeros so a kernel
    // that accumulates into C or relies on a pre-zeroed C fails loudly.
    CUDA_CHECK(cudaMemset(d_C, 0xFF, bytes));

    int rc = solve(d_A, d_B, d_C, N);

    if (rc != TODO_SENTINEL) {
        // A correct launch leaves no pending error; surface a bad config/run.
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        C_out.resize((size_t)N * N);
        CUDA_CHECK(cudaMemcpy(C_out.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return rc;
}

int main() {
    // --- ragged size first: N = 257 (correctness only, untimed) ---
    {
        const int N = 257;
        std::vector<float> C;
        int rc = run_size(N, C);
        if (rc == TODO_SENTINEL) {
            printf("[TODO] solve() returned the sentinel -- write the kernel in kernel.cu\n");
            return 0;
        }
        std::vector<float> A, B, ref;
        fill_inputs(A, B, N, 12345u + (unsigned)N);
        reference(A, B, ref, N);
        const char* why = ""; int bad = -1;
        if (!check_close(C, ref, &why, &bad)) {
            int r = bad / N, c = bad % N;
            printf("[FAIL] %s at N=%d (row %d, col %d): got %.5f want %.5f\n",
                   why, N, r, c, C[bad], ref[bad]);
            return 0;
        }
    }

    // --- clean size second: N = 512 (correctness, then timed) ---
    const int N = 512;
    std::vector<float> C;
    int rc = run_size(N, C);
    if (rc == TODO_SENTINEL) {
        // (Won't normally reach here -- the ragged size already short-circuited.)
        printf("[TODO] solve() returned the sentinel -- write the kernel in kernel.cu\n");
        return 0;
    }

    std::vector<float> A, B, ref;
    fill_inputs(A, B, N, 12345u + (unsigned)N);
    reference(A, B, ref, N);
    const char* why = ""; int bad = -1;
    if (!check_close(C, ref, &why, &bad)) {
        int r = bad / N, c = bad % N;
        printf("[FAIL] %s at N=%d (row %d, col %d): got %.5f want %.5f\n",
               why, N, r, c, C[bad], ref[bad]);
        return 0;
    }
    printf("[PASS] correct\n");

    // --- timing: median over several launches, kernel-only (device pointers) ---
    size_t bytes = (size_t)N * N * sizeof(float);
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warmup
    for (int i = 0; i < 5; ++i) {
        (void)solve(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const int iters = 50;
    std::vector<float> times(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        (void)solve(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times[i] = ms;
    }
    // median
    for (int i = 0; i < iters; ++i)
        for (int j = i + 1; j < iters; ++j)
            if (times[j] < times[i]) { float t = times[i]; times[i] = times[j]; times[j] = t; }
    float ms = times[iters / 2];

    // GFLOP/s = 2*N^3 / time ; one mul + one add per inner-product term. Render as TFLOP/s.
    double flop = 2.0 * (double)N * (double)N * (double)N;
    double tflops = flop / (ms * 1e-3) / 1e12;
    printf("[PERF] %.3f ms   %.1f TFLOP/s\n", ms, tflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
