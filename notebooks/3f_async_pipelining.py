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
    # 3F: Async Copy & Pipelining

    > *"The fastest load is the one that already finished while you were busy with the
    > last one."*

    > **Advanced / off the critical path.** This builds on shared-memory tiling (`3b`)
    > and the latency-hiding model (`0b`, `0d`). It unlocks the pipelined matmul exercise
    > `c05` — the payoff for everything in Part 3.

    In `3b` you tiled a matmul: load a tile of A and B into shared memory, `__syncthreads`,
    multiply, repeat over the K dimension. That kernel is correct and fast — but it has a
    structural stall built into every iteration. The threads issue the global loads,
    then **wait at the barrier doing nothing** until the bytes arrive, then compute. Load
    and compute happen *in sequence*, so the long DRAM latency is fully exposed once per
    K-step.

    This lecture removes that stall. The idea is **software pipelining**: while the SM
    computes on tile $k$, asynchronously *prefetch* tile $k{+}1$ into a second shared-
    memory buffer. The copy and the compute overlap, and the load latency disappears
    behind useful math. This is what Triton's `num_stages` does for you automatically —
    here you'll see the machinery underneath it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. The exposed-latency problem in a tiled loop

    Strip the `3b` matmul to its K-loop skeleton. Each iteration does three phases:

    ```cuda
    for (int k = 0; k < K; k += TILE) {
        // (L) LOAD: global -> shared, every thread copies its element
        As[ty][tx] = A[row * K + (k + tx)];
        Bs[ty][tx] = B[(k + ty) * N + col];
        __syncthreads();                 // <-- wait for ALL loads to land

        // (C) COMPUTE: the inner product over this tile
        for (int t = 0; t < TILE; ++t)
            acc += As[ty][t] * Bs[t][tx];
        __syncthreads();                 // <-- wait before overwriting the tile
    }
    ```

    The timeline per iteration is **L then C, strictly serial**:

    $$t_{\text{iter}} \;=\; t_{\text{load}} + t_{\text{compute}}.$$

    The barrier after the load is the killer: a synchronous `As[...] = A[...]` actually
    routes the data **global → register → shared**, and the thread *blocks on the load's
    result* before it can write shared memory. Every thread in the block stalls on its
    DRAM load each K-step. With $K/\text{TILE}$ iterations you pay the load latency
    $K/\text{TILE}$ times, in full, with nothing hiding it but whatever other warps
    happen to be resident.

    If you could instead make $t_{\text{iter}} \approx \max(t_{\text{load}},
    t_{\text{compute}})$ — overlapping the two phases — you would hide *almost the entire
    load cost* whenever compute is the longer of the two. That is the whole game.

    > [PMPP Ch. 6](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)
    > (memory-bound tiling) and [Simon Boehm's CUDA matmul writeup](https://siboehm.com/articles/22/CUDA-MMM)
    > both motivate prefetching as the next step after basic tiling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. `cp.async`: copy global → shared without a register stop

    The first ingredient is a hardware feature (Ampere and newer, so your Blackwell
    5070 Ti has it): an **asynchronous copy** instruction, `cp.async` in PTX, that moves
    data **global → shared memory directly**, *bypassing the register file*, and — the key
    word — **without blocking the issuing thread**. The thread fires the copy and keeps
    going; the bytes arrive later, and a separate *commit/wait* mechanism tells you when.

    In CUDA C++ you reach it through the pipeline primitives in `<cuda/pipeline>` (or the
    lower-level `__pipeline_*` intrinsics):

    ```cuda
    #include <cuda/pipeline>
    #include <cooperative_groups.h>

    // fire an async copy of one element global -> shared (does NOT block):
    __pipeline_memcpy_async(&As[ty][tx], &A[row * K + (k + tx)], sizeof(float));
    __pipeline_commit();          // group the copies issued so far into a "stage"

    // ... later, when you actually need that tile ...
    __pipeline_wait_prior(0);     // block until the oldest outstanding stage lands
    __syncthreads();              // then the usual barrier before reading shared
    ```

    Two properties make this worth the extra ceremony:

    1. **No register detour.** A synchronous `As = A[...]` occupies a register for the
       in-flight value (register pressure — recall `3e`) and forces the thread to wait on
       it. `cp.async` writes shared memory directly, so the load is *in flight in the
       background* and costs no register.
    2. **Non-blocking issue + explicit wait.** You separate *starting* the copy from
       *needing* the result. That gap is exactly where you slot compute.

    > [CUDA C++ Programming Guide — Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies)
    > documents `cp.async` and the `cuda::pipeline` API.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Double-buffering: two shared-memory tiles, ping-ponging

    `cp.async` *enables* overlap; **double-buffering** *arranges* it. The trick: allocate
    **two** shared-memory tiles instead of one, and at any moment have one being *read by
    compute* while the other is being *filled by an async copy*. They swap roles each
    iteration — ping, pong.

    ```cuda
    __shared__ float As[2][TILE][TILE];   // two buffers
    __shared__ float Bs[2][TILE][TILE];

    int buf = 0;
    // ---- prologue: prefetch tile 0 into buffer 0 ----
    load_async(As[0], Bs[0], /*k=*/0);
    __pipeline_commit();

    for (int k = 0; k < K; k += TILE) {
        // start prefetching the NEXT tile into the OTHER buffer
        if (k + TILE < K) {
            load_async(As[buf ^ 1], Bs[buf ^ 1], k + TILE);
            __pipeline_commit();
        }
        // wait for THIS tile (issued last iteration) and compute on it
        __pipeline_wait_prior(/*keep 1 stage in flight=*/1);
        __syncthreads();
        compute_tile(As[buf], Bs[buf]);   // overlaps with the copy above

        buf ^= 1;                          // ping-pong
    }
    ```

    The structure that makes it work:

    - **Prologue** primes the pipe: kick off the load of tile 0 *before* the loop.
    - **Inside the loop**, the order is *issue next load → wait for current → compute*.
      Because the next load was issued *before* the wait/compute, it runs **concurrently**
      with the compute on the current buffer.
    - `__pipeline_wait_prior(1)` keeps **one** copy outstanding (the depth-2 pipeline),
      rather than draining everything.

    With one extra buffer the per-iteration cost collapses from $t_L + t_C$ toward
    $\max(t_L, t_C)$. The cost is **2× the shared memory** for the tiles — and recall
    from `3e` that shared memory is an occupancy limiter, so deeper pipelines trade
    occupancy for overlap. That tension is the tuning knob.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Multi-buffering: more stages, deeper pipeline

    Two buffers hide *one* load behind *one* compute. If a single compute step is too
    short to fully cover the load latency, you go deeper: **N buffers**, with up to
    $N{-}1$ copies in flight at once. This is a classic software pipeline of depth $N$ —
    issue load for tile $k{+}N{-}1$, compute on tile $k$. More stages = more latency
    hidden, at the cost of $N\times$ the shared memory (lower occupancy) and a longer
    prologue/epilogue to fill and drain the pipe.

    The sweet spot is the smallest $N$ that fully hides the load — past that you spend
    shared memory (occupancy) for no extra overlap. That is precisely the parameter you'll
    sweep below, and precisely what Triton exposes as `num_stages`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Visualizing the overlap: a Gantt timeline

    Below is the timeline of a tiled K-loop as Gantt bars — **load** phases on top,
    **compute** phases below, time running left to right. The single-buffered version
    serializes them ($t_L + t_C$ per step); the double-buffered version slides the loads
    left to run *under* the previous compute. Watch the total length shrink: the
    overlapped loads stop adding to the critical path (except the first, in the prologue).
    """)
    return


@app.cell
def _():
    def _run():
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        N_TILES = 5
        t_load = 3.0      # load phase length (illustrative units)
        t_comp = 2.0      # compute phase length

        _fig, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(9.2, 4.4), sharex=True)

        def gantt(ax, single):
            # returns total time; draws load (top lane) + compute (bottom lane)
            t = 0.0
            comp_start = []
            load_start = []
            if single:
                for _i in range(N_TILES):
                    load_start.append(t); t += t_load
                    comp_start.append(t); t += t_comp
                total = t
            else:
                # prologue: first load
                load_start.append(0.0)
                t = t_load           # first compute can't start until tile 0 is in
                for _i in range(N_TILES):
                    comp_start.append(t)
                    # next load runs concurrently with this compute
                    if _i + 1 < N_TILES:
                        load_start.append(t)
                    t += t_comp
                total = t
            for _i, _ls in enumerate(load_start):
                ax.add_patch(mpatches.Rectangle(
                    (_ls, 1.15), t_load, 0.7, facecolor="#5b8def",
                    edgecolor="white"))
                ax.text(_ls + t_load / 2, 1.5, f"L{_i}", ha="center",
                        va="center", color="white", fontsize=8)
            for _i, _cs in enumerate(comp_start):
                ax.add_patch(mpatches.Rectangle(
                    (_cs, 0.25), t_comp, 0.7, facecolor="#4c9f70",
                    edgecolor="white"))
                ax.text(_cs + t_comp / 2, 0.6, f"C{_i}", ha="center",
                        va="center", color="white", fontsize=8)
            ax.axvline(total, color="#d65f5f", linestyle="--", linewidth=1.3)
            return total

        tot1 = gantt(_ax1, single=True)
        tot2 = gantt(_ax2, single=False)

        for _ax in (_ax1, _ax2):
            _ax.set_ylim(0, 2.1)
            _ax.set_yticks([0.6, 1.5])
            _ax.set_yticklabels(["compute", "load"])
            _ax.set_xlim(0, tot1 + 0.5)
        _ax1.set_title(f"single-buffered: load THEN compute, total = {tot1:.0f}")
        _ax2.set_title(
            f"double-buffered: loads overlap compute, total = {tot2:.0f} "
            f"({(1 - tot2/tot1):.0%} shorter)")
        _ax2.set_xlabel("time (illustrative units)")
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. How many stages? The diminishing-returns sweep

    The interactive below sweeps the **number of pipeline stages** (buffers). It models
    one SM running $K/\text{TILE}$ K-steps, each with a fixed load latency and a fixed
    compute time, and asks: with $N$ buffers, how much of the load latency is hidden, and
    what is the effective per-step time?

    The model: with $N$ stages you can have $N{-}1$ loads in flight, hiding up to
    $(N{-}1)\times t_{\text{compute}}$ of load latency. The per-step cost is

    $$t_{\text{step}}(N) = \max\!\big(t_{\text{compute}},\; t_{\text{load}} -
      (N-1)\,t_{\text{compute}}\big),$$

    i.e. compute fully covers the load once $N$ is large enough, after which **more
    stages buy nothing** — they only cost shared memory (and thus occupancy, `3e`). Slide
    it and find the knee.
    """)
    return


@app.cell
def _(mo):
    stages_slider = mo.ui.slider(start=1, stop=6, step=1, value=2,
                                 label="pipeline stages (buffers)")
    latency_slider = mo.ui.slider(start=1.0, stop=8.0, step=0.5, value=4.0,
                                  label="load latency / compute-time ratio")
    mo.vstack([stages_slider, latency_slider])
    return latency_slider, stages_slider


@app.cell
def _(latency_slider, stages_slider):
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        K_STEPS = 12
        t_comp = 1.0
        t_load = float(latency_slider.value) * t_comp   # latency in compute-units
        chosen = int(stages_slider.value)

        def per_step(n):
            # n-1 computes overlap the load; clamp the hidden part at the load cost
            hidden = (n - 1) * t_comp
            return max(t_comp, t_load - hidden)

        stages = np.arange(1, 7)
        # total = prologue (fill) + steady state. Approx: prologue ~ t_load,
        # steady state ~ K_STEPS * per_step.
        totals = np.array([t_load + K_STEPS * per_step(int(n)) for n in stages])
        baseline = (t_load + t_comp) * K_STEPS   # single-buffered serial
        speedups = baseline / totals

        eff_latency_hidden = np.array(
            [min(t_load, (int(n) - 1) * t_comp) / t_load for n in stages])

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9.6, 3.7))

        _bars = _ax1.bar(stages, speedups, color="#cbd6ee", edgecolor="none")
        _bars[chosen - 1].set_color("#d65f5f")
        _ax1.axhline(1.0, color="#999", linestyle=":", linewidth=1)
        _ax1.set_xlabel("pipeline stages (buffers)")
        _ax1.set_ylabel("speedup vs single-buffered")
        _ax1.set_title(f"chosen N={chosen}: {speedups[chosen-1]:.2f}x")

        _ax2.plot(stages, eff_latency_hidden, color="#4c9f70", marker="o",
                  linewidth=2)
        _ax2.axvline(chosen, color="#d65f5f", linestyle="--", linewidth=1.2)
        _ax2.set_ylim(0, 1.05)
        _ax2.set_xlabel("pipeline stages (buffers)")
        _ax2.set_ylabel("fraction of load latency hidden")
        _ax2.set_title(f"latency hidden: {eff_latency_hidden[chosen-1]:.0%}")
        _ax2.grid(True, alpha=0.15)

        _fig.suptitle(
            f"load latency = {t_load:.1f} compute-units; knee where compute "
            f"fully covers load", y=1.04, fontsize=10)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the sweep

    - At **N=1** (single-buffered) there is no overlap — speedup 1.0, baseline.
    - Each extra stage hides another $t_{\text{compute}}$ of the load, so speedup climbs
      **until compute fully covers the load**. Past that knee the curve is flat: more
      buffers hide no more latency, they just consume shared memory.
    - Crank the **latency/compute ratio** up (memory-bound, slow loads) and the knee
      moves *right* — you need more stages to hide a longer load. Crank it down
      (compute-bound) and even N=2 saturates: a single compute step already covers the
      load. **This is why the optimal `num_stages` is problem-dependent**, and why an
      autotuner sweeps it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6. The Triton connection: `num_stages`

    You met the *outcome* of this lecture back in Part 2 without the plumbing. When you
    write a Triton matmul and pass `num_stages` to `@triton.autotune` (or the kernel's
    config):

    ```python
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
                          num_stages=3, num_warps=8),
            triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32},
                          num_stages=4, num_warps=4),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def matmul_kernel(...):
        ...
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs)      # Triton emits cp.async + multi-buffering
            b = tl.load(b_ptrs)      # across `num_stages` shared-mem buffers
            acc += tl.dot(a, b)      # ...automatically pipelined for you
    ```

    `num_stages` is **exactly the buffer count from §3–§5**. Triton's compiler takes your
    plain `tl.load`/`tl.dot` K-loop and rewrites it into the double/multi-buffered
    `cp.async` pipeline you just hand-built in CUDA — prologue, ping-pong, `wait_prior`
    and all. Autotuning `num_stages` *is* sweeping the §5 plot to find the knee for your
    shape. Now you know what the compiler is doing on your behalf, and why the optimum is
    shape-dependent: it is the depth that hides *your* load behind *your* compute without
    over-spending shared memory.

    > [Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
    > exposes `num_stages`; [CUTLASS pipelines](https://github.com/NVIDIA/cutlass) implement
    > the same idea by hand at production quality.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    - **Overlap is the next lever after tiling.** Once a kernel is tiled and coalesced, a
      synchronous load-then-compute loop still exposes the full load latency each K-step.
      Prefetching the next tile while computing the current one is where the next chunk of
      performance lives.
    - **`cp.async` + double-buffering is the pattern.** Two shared buffers, a prologue,
      and *issue-next → wait-current → compute* ordering inside the loop. The async copy
      avoids the register detour (`3e`) *and* lets the load run in the background.
    - **More stages until the knee, not beyond.** Depth hides latency only up to the
      point where compute covers the load; past that it just spends shared memory and
      costs occupancy. Sweep it; the optimum is shape-dependent.
    - **You now understand Triton's `num_stages`.** It is this exact pipeline,
      auto-generated. When you autotune it, you're finding the §5 knee.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Now go write it

    Time to build the pipeline by hand. Exercise `c05` takes your tiled CUDA matmul from
    `3b` and asks you to add `cp.async` double-buffering to its K-loop:

    ```bash
    python -m harness.runner c05
    ```

    You'll write the prologue prefetch, the two-buffer ping-pong, and the
    `__pipeline_commit` / `__pipeline_wait_prior` dance — then profile it against the
    single-buffered `c02` and watch the `Long Scoreboard` stalls (`3e`) shrink as the
    loads slide under the compute. The skeleton in §3 is the *shape* of the answer; the
    indexing, the masking of the ragged K-tail, and getting `wait_prior` depth right are
    yours to work out.

    *(If `c05` has no on-disk stub yet, the command is still your forward pointer — it's
    the next CUDA exercise after the conflict-free transpose `c04`.)*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [3E: Occupancy Tuning](../3e_occupancy_tuning/) &nbsp;|&nbsp; Next: [3G: Tensor Cores](../3g_tensor_cores/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
