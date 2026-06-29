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
    # 3B: Shared-Memory Tiling

    > *"Global memory is a warehouse across town; shared memory is the workbench at your
    > elbow. Tiling is the art of carrying one box over and using every part of it before
    > going back."*

    In `1e` you tiled a matmul in Triton: 2-D program ids, blocks of A and B, an
    accumulator, a loop over K. Triton's `tl.dot` and block pointers staged the tiles
    into fast on-chip memory **for** you. Here you do it by hand. The reuse arithmetic is
    identical — what changes is that **you** declare the on-chip scratchpad with
    `__shared__`, load each tile cooperatively, place a `__syncthreads()` barrier so the
    whole block agrees the tile is ready, and only then compute.

    This is the lecture where shared memory earns its keep. A naive matmul re-reads every
    element of A and B from global memory $N$ times; tiling reads each element *once* into
    shared memory and reuses it across a whole tile. That single change moves a matmul
    from bandwidth-starved to compute-bound — the roofline shift you measured in `0d`,
    now built from primitives. By the end you'll be able to read and reason about a
    hand-tiled GEMM, and then write one.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. The problem: naive matmul re-reads everything

    Compute $C = A B$ for $N\times N$ matrices. The textbook one-thread-per-output kernel
    is the scalar version of the `0b` model:

    ```cpp
    __global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < N && col < N) {
            float acc = 0.0f;
            for (int k = 0; k < N; ++k)
                acc += A[row * N + k] * B[k * N + col];   // every k: two global loads
            C[row * N + col] = acc;
        }
    }
    ```

    Count the global-memory traffic. Each output element runs the K-loop, issuing $2N$
    global loads, and there are $N^2$ outputs — so $2N^3$ loads for $2N^3$ FLOPs. The
    **operational intensity** (FLOP per byte, from `0c`) is

    $$I_{\text{naive}} \;=\; \frac{2N^3 \text{ FLOP}}{2N^3 \cdot 4 \text{ bytes}}
      \;=\; \frac{1}{4}\ \text{FLOP/byte}.$$

    That's a *terrible* intensity — pinned to the far-left, memory-bound corner of the
    `0d` roofline, no matter how big $N$ is. The math units starve while the kernel waits
    on DRAM. The waste is obvious once named: **the same row of A is re-read for every
    one of the $N$ columns it multiplies, and the same column of B for every row.** Each
    value is fetched from global memory $N$ times when once would do.

    > PMPP Ch. 5 ("Memory architecture and data locality") develops exactly this
    > naive-vs-tiled comparison; the Triton tiled-matmul tutorial is its `1e` analogue.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. `__shared__`: a per-block scratchpad

    Shared memory (from `0c`) is a small, fast, on-chip SRAM **private to a block** and
    shared by all its threads — ~100 KB per SM on your 5070 Ti, with a 48 KB-per-block
    default. Latency is a handful of cycles versus hundreds for DRAM. You declare it
    inside a kernel with the `__shared__` qualifier:

    ```cpp
    __shared__ float As[TILE][TILE];   // a TILE x TILE patch of A, on chip
    __shared__ float Bs[TILE][TILE];   // a TILE x TILE patch of B, on chip
    ```

    Two properties make it the engine of reuse:

    1. **Block-scoped & cooperative.** Every thread in the block sees the *same* `As` and
       `Bs`. So 256 threads can *together* load a 16×16 tile — one element each — and then
       every thread reads from the whole tile. A value loaded once is reused by many
       threads.
    2. **Fast.** Reading `As[i][j]` is an on-chip access, not a DRAM round-trip. Once a
       tile is staged, the inner products run at SRAM speed.

    This is the manual version of what Triton's block pointers gave you in `1e`: a named,
    reusable on-chip copy of a sub-block of the operands. In Triton it was implicit in
    `tl.dot` over loaded blocks; in CUDA you spell out the staging.

    > Shared memory is also where **bank conflicts** live (32 banks) — covered in `3d`.
    > For now treat it as fast scratch; the padding trick comes later.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. `__syncthreads()`: the block barrier

    Shared memory is cooperative, which means it needs a **handshake**. If thread A loads
    `As[2][3]` and thread B wants to read it, B must *wait until A's store is done*.
    Without enforcement the threads of a block run at their own pace (warps interleave on
    the SM, §`0b`), so B might read garbage. The barrier is `__syncthreads()`:

    > `__syncthreads()` — every thread in the block stops here until **all** threads in
    > the block have arrived. After it returns, all shared-memory writes issued before it
    > are visible to all threads.

    The tiling loop needs **two** barriers per tile, and getting them right is the crux:

    ```cpp
    for (int t = 0; t < N / TILE; ++t) {       // advance over K, one tile at a time
        As[ty][tx] = A[...];                    // (a) each thread loads one element
        Bs[ty][tx] = B[...];                    //     of the current A-tile and B-tile
        __syncthreads();                        // (1) WAIT: tile fully loaded before use

        for (int k = 0; k < TILE; ++k)          // (b) compute partial products
            acc += As[ty][k] * Bs[k][tx];       //     entirely from fast shared memory
        __syncthreads();                        // (2) WAIT: all done reading before reload
    }
    ```

    - **Barrier (1)** sits *after the loads, before the compute*: no thread may read a
      tile element until every thread has finished writing its share. Omit it and you
      read a half-loaded tile.
    - **Barrier (2)** sits *after the compute, before the next iteration's loads*: no
      thread may overwrite `As`/`Bs` for tile $t{+}1$ until every thread has finished
      consuming tile $t$. Omit it and a fast warp clobbers the tile a slow warp is still
      reading.

    **Hard rule:** `__syncthreads()` must be reached by *all* threads in the block. Put
    it inside a data-dependent `if` that only some threads take and you deadlock — the
    arrivers wait forever for the threads that branched away. This is the CUDA-level
    cousin of the divergence warning from `0b`.

    > [CUDA C++ Programming Guide §7.6, "Synchronization Functions"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Picture: loading a tile into shared memory

    Below is one iteration of the K-loop. The block (here 4×4 threads) cooperatively
    copies one `TILE×TILE` patch of A and one of B from **global memory** (left) into
    **shared memory** (right) — each thread fetches exactly one element of each. The
    `__syncthreads()` barrier between the load and the inner-product loop guarantees the
    whole patch has landed before any thread multiplies. After the barrier, every thread
    reads a full row of `As` and a full column of `Bs` — *that* shared read is where the
    one-load-many-uses reuse happens.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        T = 4
        _fig, _ax = plt.subplots(figsize=(9.4, 4.2))
        _ax.set_xlim(0, 14)
        _ax.set_ylim(0, 6.4)
        _ax.axis("off")
        _ax.set_title("one K-iteration: cooperative tile load  global -> shared",
                      fontsize=10)

        def tile(x0, y0, color, edge, label, sub):
            _ax.text(x0 + T * 0.45, y0 + T * 0.9 + 0.5, label, ha="center",
                     fontsize=9.5, weight="bold", color=edge)
            _ax.text(x0 + T * 0.45, y0 + T * 0.9 + 0.18, sub, ha="center",
                     fontsize=7.2, color="#777")
            for _r in range(T):
                for _c in range(T):
                    _ax.add_patch(mpatches.Rectangle(
                        (x0 + _c * 0.9, y0 + (T - 1 - _r) * 0.9), 0.82, 0.82,
                        facecolor=color, edgecolor=edge, linewidth=1.0))

        tile(0.4, 3.4, "#cfe0f7", "#5b8def", "A tile (global)", "slow DRAM")
        tile(0.4, 0.2, "#d6ecdc", "#4c9f70", "B tile (global)", "slow DRAM")
        tile(9.0, 3.4, "#9bc0f0", "#3060c0", "As (shared)", "fast on-chip SRAM")
        tile(9.0, 0.2, "#a7d8b6", "#2e6b48", "Bs (shared)", "fast on-chip SRAM")

        # arrows: each thread copies one element
        _ax.annotate("", xy=(8.7, 4.6), xytext=(4.4, 4.6),
                     arrowprops=dict(arrowstyle="->", color="#5b8def", lw=2))
        _ax.annotate("", xy=(8.7, 1.4), xytext=(4.4, 1.4),
                     arrowprops=dict(arrowstyle="->", color="#4c9f70", lw=2))
        _ax.text(6.5, 4.9, "each thread loads\none element", ha="center",
                 fontsize=8, color="#5b8def")
        _ax.text(6.5, 1.7, "each thread loads\none element", ha="center",
                 fontsize=8, color="#4c9f70")
        _ax.text(6.5, 2.95, "__syncthreads()", ha="center", fontsize=9,
                 color="#d65f5f", weight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fde9e9", ec="#d65f5f"))
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Why tiling raises operational intensity

    Here is the payoff, as arithmetic. With a `TILE×TILE` block, each thread still
    produces one output, but the loads are now **amortized across the tile**. Loading one
    `TILE×TILE` patch of A into shared costs $\text{TILE}^2$ global loads — and every one
    of those elements is then reused by `TILE` different threads (a whole tile-row of
    output). The reuse factor is exactly the tile dimension:

    $$\boxed{\ \text{reuse factor} \;=\; \text{TILE}\ }$$

    Concretely, the tiled kernel reads each input element from global memory only once
    per *tile-step* instead of once per *output*, cutting global traffic by a factor of
    `TILE`. So the operational intensity climbs from the naive $\tfrac14$ to

    $$I_{\text{tiled}} \;\approx\; \frac{\text{TILE}}{4}\ \text{FLOP/byte}
      \quad\text{(}\,\text{TILE}\times\text{ better than naive}\,\text{)}.$$

    A 16×16 tile turns $I=\tfrac14$ into $I\approx 4$; a 32×32 tile into $I\approx 8$.
    On the `0d` roofline that's the dot sliding right, off the bandwidth slope, toward
    the compute roof. **This is the whole reason shared memory exists:** to convert
    expensive, repeated global reads into cheap on-chip reuse. Tie it back to `0c`'s
    memory hierarchy — you are deliberately staging data one level up (DRAM → SRAM) and
    cashing in the bandwidth difference.

    The limit on `TILE` is resources, not desire. Bigger tiles reuse more but consume
    more shared memory per block, which caps how many blocks fit per SM (the
    shared-memory occupancy limiter from `0d`). The interactive below makes that tradeoff
    concrete.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactive: tile size vs shared memory vs reuse

    Slide `TILE`. The kernel needs two `TILE×TILE` float tiles per block (`As` and
    `Bs`), so shared memory per block is $2\cdot\text{TILE}^2\cdot 4$ bytes. The bars
    show that against the **48 KB/block default** and the **100 KB/SM** budget from
    `0d`; the title reports the reuse factor (= `TILE`) and how many such blocks could
    fit per SM by the shared-memory limit alone. Watch a big tile buy more reuse but
    crowd the SM — the same threads-vs-resources tug-of-war as the occupancy lecture.
    """)
    return


@app.cell
def _(mo):
    tile_slider = mo.ui.slider(start=4, stop=64, step=4, value=16,
                               label="TILE (tile dimension)")
    tile_slider
    return (tile_slider,)


@app.cell
def _(tile_slider):
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        TILE = int(tile_slider.value)
        bytes_per_block = 2 * TILE * TILE * 4          # As + Bs, float32
        kb_per_block = bytes_per_block / 1024.0

        DEFAULT_BLOCK_KB = 48.0
        SM_KB = 100.0
        threads_per_block = TILE * TILE                # one thread per output tile cell

        # how many blocks fit per SM by shared-mem rule (using opt-in ~100KB ceiling)
        blocks_by_smem = int(SM_KB // kb_per_block) if kb_per_block > 0 else 0
        reuse = TILE

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6),
                                          gridspec_kw={"width_ratios": [1, 1]})

        # left: shared mem per block vs the two ceilings
        over_default = kb_per_block > DEFAULT_BLOCK_KB
        bar_col = "#d65f5f" if over_default else "#5b8def"
        _ax1.bar(["smem / block"], [kb_per_block], color=bar_col, width=0.5)
        _ax1.axhline(DEFAULT_BLOCK_KB, color="#e0a458", linestyle="--",
                     linewidth=1.5, label="48 KB/block default")
        _ax1.axhline(SM_KB, color="#4c9f70", linestyle=":", linewidth=1.5,
                     label="100 KB/SM ceiling")
        _ax1.set_ylabel("KB")
        _ax1.set_ylim(0, max(SM_KB * 1.15, kb_per_block * 1.15))
        _ax1.set_title(f"As+Bs = {kb_per_block:.1f} KB/block")
        _ax1.legend(fontsize=7.5, loc="upper left")

        # right: reuse factor
        xs = np.arange(4, 65, 4)
        _ax2.plot(xs, xs, color="#4c9f70", linewidth=2)
        _ax2.scatter([TILE], [reuse], color="#d65f5f", s=80, zorder=5)
        _ax2.set_xlabel("TILE")
        _ax2.set_ylabel("reuse factor  (= TILE)")
        _ax2.set_title(f"reuse x{reuse}   ~I = {reuse/4:.1f} FLOP/byte")
        _ax2.grid(True, alpha=0.15)

        _fitnote = (f"{blocks_by_smem} block(s)/SM by smem"
                    if not over_default
                    else "exceeds 48KB default -> needs opt-in")
        _fig.suptitle(
            f"TILE={TILE}: {threads_per_block} threads/block, "
            f"{kb_per_block:.1f} KB smem, {_fitnote}", y=1.03, fontsize=9.5)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. The hand-tiled GEMM, assembled

    Putting §2–§4 together, the skeleton of a tiled matmul reads as follows. Study the
    shape of it — the two-barrier loop, the global-index math for *where in A/B this
    tile lives*, the accumulator carried across tiles — but **the index expressions
    inside the loads are deliberately left as `...` for you to derive in `c02`:**

    ```cpp
    #define TILE 16

    __global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
        __shared__ float As[TILE][TILE];
        __shared__ float Bs[TILE][TILE];

        int tx = threadIdx.x, ty = threadIdx.y;
        int row = blockIdx.y * TILE + ty;     // this thread's output row
        int col = blockIdx.x * TILE + tx;     // this thread's output col
        float acc = 0.0f;

        for (int t = 0; t < N / TILE; ++t) {      // march the tile across K
            As[ty][tx] = A[ /* ... row, t, tx ... */ ];   // <- you derive this
            Bs[ty][tx] = B[ /* ... t, ty, col ... */ ];   // <- and this
            __syncthreads();                              // (1) tile ready

            for (int k = 0; k < TILE; ++k)
                acc += As[ty][k] * Bs[k][tx];             // inner product, all on-chip
            __syncthreads();                              // (2) safe to reload
        }
        C[row * N + col] = acc;                           // (+ a bounds guard for ragged N)
    }
    ```

    The launch uses a 2-D block, the natural extension of the `<<<grid, block>>>` syntax
    from `3a`:

    ```cpp
    dim3 block(TILE, TILE);                              // TILE*TILE threads
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    ```

    The crux left for you: the global addresses of `As[ty][tx]` and `Bs[ty][tx]` for tile
    `t`. They follow from "which row/col of A and B does *this* tile-step touch?" — the
    same 2-D indexing logic as `1e`, expressed in raw `row*N + k` arithmetic. The ragged
    guard (non-multiple-of-`TILE` $N$) is also yours to add, as in `3a`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    Tiling is *the* technique that separates a toy kernel from a fast one, and every
    later Part-3 kernel is a variation on it.

    - **Stage, sync, reuse — in that order.** Cooperative load into `__shared__`,
      `__syncthreads()`, then compute entirely from on-chip data. The barrier discipline
      (two per tile-step) is non-negotiable.
    - **Tiling is a roofline move.** It raises operational intensity by the reuse factor
      (`TILE`), sliding a memory-bound kernel toward the compute roof from `0d`. When a
      kernel is bandwidth-starved, ask "what data am I re-reading, and can I stage it?"
    - **Tile size is a resource budget.** Bigger tiles reuse more but spend more shared
      memory, capping blocks-per-SM. Pick `TILE` against the 48 KB/block and 100 KB/SM
      limits, exactly as you balanced occupancy in `0d`.
    - **Barriers must be uniform.** Every thread in the block must hit every
      `__syncthreads()`; a barrier behind a divergent branch deadlocks. (The next
      lecture, `3c`, shows the *warp*-level shuffles that skip the barrier for the last
      reduction steps.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Now go write it

    You've seen the tiled GEMM's shape; now fill in the index arithmetic and make it run.
    Write the cooperative loads, place both barriers, accumulate across K, and guard the
    ragged tail:

    ```bash
    python -m harness.runner c02 --watch
    ```

    `c02` is the CUDA tiled matmul — the by-hand version of Triton's `e07`. The harness
    builds it with `nvcc -arch=sm_120`, checks correctness against the reference, and
    reports **FLOP/s**. Tune `TILE` and watch the number climb as your dot slides right
    on the `0d` roofline toward the compute roof. When a 16×16 or 32×32 tile beats the
    naive kernel by the reuse factor, you've built shared-memory tiling from scratch.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [3A: The CUDA C++ Execution Model](../3a_cuda_model/) &nbsp;|&nbsp; Next: [3C: Warp-Level Primitives](../3c_warp_primitives/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
