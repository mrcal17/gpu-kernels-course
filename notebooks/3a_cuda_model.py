import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3A: The CUDA C++ Execution Model

    > *"Triton hid the machine to teach you the patterns. CUDA hands it back — same
    > patterns, now every index, every byte, every barrier is yours to spell out."*

    Part 1 taught you to think in blocks, warps, masks, and tiles — through Triton,
    which wrote the boilerplate for you. Part 3 re-derives every one of those patterns
    **by hand in CUDA C++**, the language the hardware actually speaks. Nothing about
    the *machine* changes: it is the same `sm_120` GPU, the same warps of 32, the same
    SMs and shared memory from Part 0. What changes is that **you** now write the index
    arithmetic Triton generated, manage the host/device memory split explicitly, and
    drive `nvcc` instead of a Python JIT.

    This first CUDA lecture maps the Triton programming model from `1a` onto its CUDA
    primitives one-to-one. By the end you'll be able to read a `__global__` kernel,
    compute a thread's global index, move data across the host/device boundary, and run
    the `nvcc` build loop on your Windows box. Then you write vector-add — again, but
    this time to the metal.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. `__global__`: a kernel is a function

    In Triton you decorated a Python function with `@triton.jit`. In CUDA you mark a
    function with the `__global__` qualifier. It declares a function that **runs on the
    device** and is **launched from the host**:

    ```cpp
    __global__ void add(const float* a, const float* b, float* out, int n) {
        // ... body runs once per thread ...
    }
    ```

    Three function-space qualifiers carve up the language:

    - `__global__` — entry point for a launch. Returns `void`, called with the
      `<<<...>>>` syntax (§4). This is your kernel.
    - `__device__` — a helper callable **only from device code** (from inside a
      `__global__` or another `__device__` function). Your inlined math helpers.
    - `__host__` — ordinary CPU code (the default; usually left implicit). You can mark
      a function `__host__ __device__` to compile it for both.

    The crucial mental shift from CPU C++: **the body of a `__global__` function is the
    code of one thread.** You do not write a loop over `n` elements. You write what a
    *single* thread does, and the launch (§4) stamps out thousands of copies. This is
    exactly the Triton model — a `@triton.jit` body was the code of one *program* — but
    in Triton one program processed a whole `BLOCK_SIZE` vector at once. In CUDA the
    default grain is finer: **one thread, one element.** (You can give each thread a
    strip of elements; that's a tuning choice, not the default.)

    > [CUDA C++ Programming Guide §7.1, "Function Execution Space Specifiers"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-execution-space-specifiers)
    > defines `__global__`/`__device__`/`__host__`. PMPP Ch. 2 introduces the kernel.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Who am I? `blockIdx`, `threadIdx`, `blockDim`, `gridDim`

    Inside a kernel, four built-in variables tell each thread where it sits in the
    grid → block → thread hierarchy from `0b`. Each is a `dim3` with `.x`, `.y`, `.z`
    fields (use `.x` for 1-D problems):

    | built-in | meaning | Triton analogue |
    |---|---|---|
    | `threadIdx.x` | this thread's index *within its block* | (the lane offset inside `tl.arange`) |
    | `blockIdx.x`  | this block's index *within the grid* | `tl.program_id(0)` |
    | `blockDim.x`  | threads per block (block size) | `BLOCK_SIZE` (a `tl.constexpr`) |
    | `gridDim.x`   | number of blocks in the grid | the launch grid size |

    The correspondence is direct. In Triton `tl.program_id(0)` told a program which
    block it was; in CUDA that's `blockIdx.x`. But CUDA exposes one more level Triton
    hid: `threadIdx.x`, the lane *within* the block. Triton folded that into the
    vectorized `tl.arange(0, BLOCK_SIZE)`; in CUDA every thread is a scalar program and
    must ask "which lane am I?" for itself.

    These are **read-only**, set by the hardware for each thread before your code runs.
    A thread cannot change its own coordinates — it can only read them and compute from
    them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. The global-index idiom

    A thread knows its block (`blockIdx.x`), its block's size (`blockDim.x`), and its
    lane (`threadIdx.x`). To turn those into a single position in the flat array, you
    combine them — the **most important line in all of CUDA**:

    $$i \;=\; \texttt{blockIdx.x}\,\times\,\texttt{blockDim.x}\;+\;\texttt{threadIdx.x}.$$

    Read it as: *skip past all the threads in the blocks before me*
    ($\texttt{blockIdx.x}\times\texttt{blockDim.x}$), *then add my offset within my own
    block* ($\texttt{threadIdx.x}$). Every thread in the launch gets a distinct $i$, and
    together they tile $0,1,2,\dots$ contiguously across the array.

    This is the scalar unrolling of the Triton offset line from `1a`:

    ```python
    # Triton (1a): one program, a whole vector of offsets at once
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ```
    ```cpp
    // CUDA: one thread, one scalar offset
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    ```

    Triton's `pid * BLOCK_SIZE` is CUDA's `blockIdx.x * blockDim.x`; Triton's
    `tl.arange(0, BLOCK_SIZE)` vector becomes CUDA's scalar `threadIdx.x` — because each
    CUDA thread *is* one element of that range.

    **The ragged-tail guard.** If `n` isn't a multiple of `blockDim.x`, the last block
    has threads whose $i \ge n$. They must not touch memory. Triton handled this with a
    `mask`; CUDA handles it with a plain `if`:

    ```cpp
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {                       // CUDA's version of Triton's `mask = offs < n`
        out[i] = a[i] + b[i];
    }
    ```

    Forgetting this `if` is the single most common CUDA bug: it reads/writes out of
    bounds in the tail block, corrupting memory or faulting.

    > [CUDA C++ Programming Guide §2.2, "Thread Hierarchy"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy)
    > and PMPP Ch. 2 derive this indexing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Picture: the global-index mapping

    The figure below lays out a tiny grid — 3 blocks of 8 threads each — and shows how
    `blockIdx.x * blockDim.x + threadIdx.x` maps every `(block, lane)` pair onto a
    distinct global index $0\dots23$. The labels under the cells are the global $i$; the
    color groups threads by their block. This is the contiguous tiling the idiom buys
    you.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        n_blocks = 3
        block_dim = 8
        colors = ["#5b8def", "#4c9f70", "#e0a458"]

        _fig, _ax = plt.subplots(figsize=(9.2, 3.0))
        _ax.set_xlim(-0.5, n_blocks * block_dim + 0.5)
        _ax.set_ylim(-1.4, 2.2)
        _ax.axis("off")
        _ax.set_title(
            "global index  i = blockIdx.x * blockDim.x + threadIdx.x   "
            "(blockDim.x = 8)", fontsize=10)

        for _b in range(n_blocks):
            for _t in range(block_dim):
                _i = _b * block_dim + _t
                _ax.add_patch(mpatches.Rectangle(
                    (_i, 0.0), 0.92, 1.0, facecolor=colors[_b],
                    edgecolor="white", linewidth=1.2))
                # threadIdx.x label (top, inside cell)
                _ax.text(_i + 0.46, 0.72, str(_t), ha="center", va="center",
                         color="white", fontsize=7)
                # global index label (below cell)
                _ax.text(_i + 0.46, -0.35, str(_i), ha="center", va="center",
                         color="#333", fontsize=8, weight="bold")
            # block bracket + label
            _cx = _b * block_dim + block_dim / 2
            _ax.text(_cx, 1.55, f"blockIdx.x = {_b}", ha="center",
                     color=colors[_b], fontsize=9, weight="bold")
            _ax.annotate("", xy=(_b * block_dim + 0.05, 1.25),
                         xytext=(_b * block_dim + block_dim - 0.1, 1.25),
                         arrowprops=dict(arrowstyle="<->", color=colors[_b]))

        _ax.text(-0.3, 0.72, "threadIdx.x:", ha="right", va="center",
                 fontsize=7, color="#666")
        _ax.text(-0.3, -0.35, "global i:", ha="right", va="center",
                 fontsize=8, color="#666")
        _ax.text(n_blocks * block_dim / 2, -1.05,
                 "every (block, lane) -> a distinct, contiguous global index",
                 ha="center", fontsize=8, color="#666", style="italic")
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactive: watch the indices renumber

    Slide `blockDim.x` (threads per block). For a fixed problem of `N = 24` elements,
    the grid uses $\lceil N/\texttt{blockDim.x}\rceil$ blocks, and each thread's global
    index is recomputed by the idiom. Notice two things: the indices stay a contiguous
    $0\dots23$ no matter how you slice the blocks, and when `blockDim.x` doesn't divide
    24 the **last block runs off the end** — those over-`N` threads (drawn hollow) are
    exactly the ones the `if (i < n)` guard must mask off.
    """)
    return


@app.cell
def _(mo):
    blockdim_slider = mo.ui.slider(start=2, stop=12, step=1, value=8,
                                   label="blockDim.x (threads / block)")
    blockdim_slider
    return (blockdim_slider,)


@app.cell
def _(blockdim_slider):
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        N = 24
        bd = int(blockdim_slider.value)
        n_blocks = -(-N // bd)          # ceil div
        total = n_blocks * bd
        palette = ["#5b8def", "#4c9f70", "#e0a458", "#d65f5f",
                   "#9b7fd4", "#54b7c4", "#c97fa5", "#7f9b54",
                   "#b06f4f", "#5f7fd6", "#4fae8a", "#d49a3a"]

        _fig, _ax = plt.subplots(figsize=(9.2, 2.6))
        _ax.set_xlim(-0.5, total + 0.5)
        _ax.set_ylim(-1.2, 1.9)
        _ax.axis("off")

        n_masked = total - N
        _ax.set_title(
            f"N=24, blockDim.x={bd}  ->  {n_blocks} blocks, "
            f"{total} threads launched, {n_masked} masked by  if (i < n)",
            fontsize=9.5)

        for _b in range(n_blocks):
            _col = palette[_b % len(palette)]
            for _t in range(bd):
                _i = _b * bd + _t
                _live = _i < N
                _ax.add_patch(mpatches.Rectangle(
                    (_i, 0.0), 0.92, 1.0,
                    facecolor=_col if _live else "white",
                    edgecolor=_col, linewidth=1.4,
                    hatch=None if _live else "////"))
                _lbl = str(_i) if _live else "x"
                _ax.text(_i + 0.46, 0.5, _lbl, ha="center", va="center",
                         color="white" if _live else _col, fontsize=7,
                         weight="bold")
            _cx = _b * bd + bd / 2
            _ax.text(_cx, 1.45, f"block {_b}", ha="center", color=_col,
                     fontsize=8, weight="bold")
        _ax.text(total / 2, -0.85,
                 "solid = does real work (i < N)    hollow = guarded off (i >= N)",
                 ha="center", fontsize=8, color="#666", style="italic")
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. The launch: `<<<grid, block>>>`

    Triton launched a kernel with `kernel[grid](args...)`, where `grid` was a tuple of
    block counts. CUDA uses the **triple-angle-bracket** execution-configuration syntax,
    placed between the function name and its arguments:

    ```cpp
    int threads = 256;                       // blockDim.x  (threads per block)
    int blocks  = (n + threads - 1) / threads;   // ceil-div: gridDim.x

    add<<<blocks, threads>>>(d_a, d_b, d_out, n);
    ```

    - The **first** `<<<...>>>` argument is the **grid** (`gridDim`) — how many blocks.
    - The **second** is the **block** (`blockDim`) — how many threads per block.
    - Either can be a `dim3` for 2-D/3-D launches: `dim3 block(16,16); kernel<<<grid,
      block>>>(...)` gives you `threadIdx.x` and `threadIdx.y` (you'll use this for
      tiled matmul in `3b`).

    The `(n + threads - 1) / threads` is integer **ceil-division** — the same
    `ceil_div(n, BLOCK_SIZE)` you wrote for the Triton grid in `1a`, spelled out. It
    rounds *up* so the last (possibly ragged) block is covered; the `if (i < n)` guard
    then disables the surplus threads.

    Two facts to carry from `0b`, now concrete in CUDA:

    1. **The launch is asynchronous.** `add<<<...>>>(...)` returns to the CPU
       *immediately* — it only *queues* the kernel on the GPU's stream. The CPU keeps
       running. To wait for completion you call `cudaDeviceSynchronize()` (or copy
       results back, which synchronizes implicitly).
    2. **Choosing `threads`** is the block-size decision from `0b`/`0d`: a multiple of
       32 (whole warps), commonly 128–256, balancing occupancy against registers and
       shared memory.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Host vs device memory, and `cudaMemcpy`

    This is the biggest thing Triton hid. In Triton you passed PyTorch tensors and the
    framework had already placed them on the GPU. In raw CUDA, **the CPU and GPU have
    separate address spaces**, and you move data across the boundary by hand:

    ```cpp
    float *h_a = (float*)malloc(n * sizeof(float));   // HOST  (CPU) memory
    // ... fill h_a on the CPU ...

    float *d_a;                                       // DEVICE (GPU) pointer
    cudaMalloc(&d_a, n * sizeof(float));              // allocate in GPU DRAM

    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);  // up
    // ... launch kernel that reads d_a, writes d_out ...
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost); // down

    cudaFree(d_a);   free(h_a);                       // release both
    ```

    The convention `h_` / `d_` (host / device) pointer prefixes is universal and worth
    adopting — **a device pointer dereferenced on the host (or vice versa) is undefined
    behavior**, usually a crash, and the prefix is your only visual guard.

    The lifecycle of nearly every CUDA program is this five-step dance:

    $$\texttt{cudaMalloc} \;\to\; \texttt{cudaMemcpy (H2D)} \;\to\; \texttt{kernel}\langle\langle\langle\rangle\rangle\rangle \;\to\; \texttt{cudaMemcpy (D2H)} \;\to\; \texttt{cudaFree}.$$

    **Why this matters for performance:** those copies cross the PCIe bus (~tens of
    GB/s), an order of magnitude slower than the GPU's ~896 GB/s DRAM. A kernel that's
    bandwidth-bound on-device can be *transfer*-bound end-to-end. The roofline lesson
    from `0d` extends: for one-shot work the H2D/D2H copies can dominate, which is why
    real pipelines keep data resident on the GPU across many kernels rather than
    bouncing it.

    > **Unified memory** (`cudaMallocManaged`) gives a *single* pointer valid on both
    > host and device; the driver migrates pages on demand, so you skip the explicit
    > `cudaMemcpy`. It's convenient — and how PyTorch-style frameworks feel — but the
    > migration still happens under the hood, so for performance-critical code you'll
    > usually manage transfers explicitly and overlap them with compute (Part 3 later).
    > We use explicit `cudaMalloc`/`cudaMemcpy` in the exercises so the cost is visible.

    > [CUDA C++ Programming Guide §3.2.2, "Device Memory"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory)
    > and §3.2.4 (Unified Memory). PMPP Ch. 2–3 walk the copy lifecycle.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6. Error checking: the macro you always paste

    CUDA API calls return a `cudaError_t` status code; **kernel launches report errors
    asynchronously** and silently unless you ask. Ignoring these is how a CUDA bug hides
    for an hour. The standard defense is a check macro wrapped around every API call:

    ```cpp
    #include <cstdio>
    #include <cstdlib>

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

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    ```

    For the **kernel launch** itself (which has no return value to wrap), check the two
    error channels right after:

    ```cpp
    add<<<blocks, threads>>>(d_a, d_b, d_out, n);
    CUDA_CHECK(cudaGetLastError());        // catches bad launch config (e.g. too many threads)
    CUDA_CHECK(cudaDeviceSynchronize());   // catches errors *during* execution (e.g. illegal address)
    ```

    The two checks catch different failures: `cudaGetLastError()` catches a malformed
    *launch* (immediately), while `cudaDeviceSynchronize()` waits for the kernel and
    surfaces faults that happened *while it ran* (the classic "illegal memory access"
    from a missing bounds guard). The `do { ... } while (0)` wrapper is the idiom that
    lets the macro act as a single statement even after an `if` without braces. **Paste
    this macro into every CUDA file you write** — the harness expects you to check your
    calls.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 7. The `nvcc` build loop (Windows, CUDA 13.1)

    Triton compiled in-process at the first call. CUDA C++ is **ahead-of-time
    compiled** by `nvcc`, NVIDIA's compiler driver. On Windows `nvcc` does *not* contain
    its own C++ host compiler — it shells out to **MSVC** (`cl.exe`, from Visual Studio
    Build Tools) for the host half of each file and compiles the device half itself. So
    the build has one prerequisite beyond the CUDA Toolkit: the MSVC toolchain must be on
    `PATH`.

    The reliable way to get that environment is the **"x64 Native Tools Command Prompt
    for VS"** (it runs `vcvars64.bat`, putting `cl.exe` on `PATH`), or call `vcvars64.bat`
    from your shell first. Then:

    ```bat
    REM 1. compile for your GPU's architecture (Blackwell = sm_120)
    nvcc -arch=sm_120 -O2 vector_add.cu -o vector_add.exe

    REM 2. run it
    vector_add.exe
    ```

    What the flags mean:

    - `-arch=sm_120` — generate code for **compute capability 12.0**, your RTX 5070 Ti's
      Blackwell architecture (`sm_120`). Get this wrong (e.g. an older `sm_86`) and the
      binary may refuse to run or miss Blackwell features. This is the CUDA equivalent of
      Triton targeting your device automatically.
    - `-O2` — optimize host code (device code is optimized by default).
    - `.cu` — the CUDA source extension; `nvcc` splits host/device and routes each half.

    A common first-run failure on Windows is **`cl.exe not found`** — that means MSVC
    isn't on `PATH`; open the x64 Native Tools prompt (or point `nvcc` at it with
    `-ccbin "<path to MSVC>"`). A quick sanity check that the toolchain is wired up:

    ```bat
    nvcc --version
    cl
    ```

    The exercise harness (`harness.runner`) wraps this build loop for you — it invokes
    `nvcc` with the right `-arch=sm_120` and host-compiler flags, compiles your `.cu`,
    runs it against the reference, and reports the metric. You write the kernel; it
    drives `nvcc`. Knowing the manual loop is still essential for when you compile a
    scratch file to debug.

    > [CUDA C++ Programming Guide §6.1, "Compilation with NVCC"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compilation-with-nvcc)
    > and the [NVCC docs](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).
    > For the Windows host-compiler requirement see the
    > [CUDA Installation Guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    Everything in Part 3 is built from the pieces in this lecture. The Triton → CUDA
    map you'll lean on constantly:

    | Triton (`1a`) | CUDA C++ (this lecture) |
    |---|---|
    | `@triton.jit` | `__global__` |
    | `tl.program_id(0)` | `blockIdx.x` |
    | `BLOCK_SIZE` (`tl.constexpr`) | `blockDim.x` |
    | grid size in `kernel[grid](...)` | `gridDim.x`, the `<<<grid, ...>>>` arg |
    | `pid*BLOCK_SIZE + tl.arange(0,B)` | `blockIdx.x*blockDim.x + threadIdx.x` |
    | `mask = offs < n` | `if (i < n)` |
    | tensors already on GPU | `cudaMalloc` + `cudaMemcpy` by hand |
    | JIT at first call | `nvcc -arch=sm_120` ahead of time |

    - **Write the body of one thread.** Resist the urge to loop over `n` — the launch is
      the loop.
    - **Always guard the tail.** `if (i < n)` is not optional; the ragged last block
      *will* run threads past the end.
    - **Respect the memory boundary.** Host pointers and device pointers are not
      interchangeable; copies cost real bandwidth.
    - **Check every CUDA call.** The `CUDA_CHECK` macro turns silent corruption into a
      file:line error you can fix.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Now go write it

    Time to write your first CUDA kernel — vector-add, the same problem as Triton `e01`,
    now spelled out to the metal. You'll write the `__global__` function, the global-index
    line, the `if (i < n)` guard, the `cudaMalloc`/`cudaMemcpy` lifecycle, and the
    `<<<blocks, threads>>>` launch:

    ```bash
    python -m harness.runner c01 --watch
    ```

    `c01` is the CUDA counterpart of `e01`: everything in this lecture, in one `.cu`
    file you compile with `nvcc -arch=sm_120`. The harness drives the build and checks
    your output against the reference. When the bandwidth number lands near the Part-0
    roofline, you've translated the whole Part-1 mental model into CUDA C++.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [2D: Autograd Integration](../2d_autograd/) &nbsp;|&nbsp; Next: [3B: Shared-Memory Tiling](../3b_shared_tiling/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
