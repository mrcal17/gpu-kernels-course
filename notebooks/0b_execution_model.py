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
    # 0B: The Execution Model

    > *"A CPU is a sports car; a GPU is a freight train. The train is slower to start,
    > but it moves a thousand tonnes at once."*

    Before a single kernel makes sense you need the mental model of **how the GPU runs
    your code**. This is the most important lecture in the course — everything later
    is a corollary. We will build it from one design decision: the GPU is built to
    **hide latency with parallelism**, not to minimize it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. Throughput, not latency

    A CPU core spends most of its transistors making *one* stream of instructions
    finish fast: big caches, branch prediction, out-of-order execution. It hates
    waiting.

    A GPU makes the opposite bet. It has thousands of simple lanes and *expects* to
    wait — a load from DRAM costs hundreds of cycles. Instead of avoiding the stall,
    it **parks the stalled work and runs other work**. With enough independent work
    resident, the memory latency is completely hidden behind useful computation.

    So the GPU's whole personality is: **oversubscribe me.** Give me far more
    parallel work than I have lanes, and I will keep the lanes busy while memory
    catches up. A kernel that does *not* give it enough parallel work leaves the
    machine idle — this is the failure mode you will learn to recognize.

    > [PMPP](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)
    > Ch. 1–4 develops this latency-hiding model in depth.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. The thread hierarchy

    Your kernel is launched as a **grid** of **thread blocks**, each block a group of
    **threads**. Threads are scheduled in hardware bundles of 32 called **warps**.

    $$\underbrace{\text{thread}}_{\text{1 lane}}
      \;\xrightarrow{\times 32}\;
      \underbrace{\text{warp}}_{\text{lockstep}}
      \;\xrightarrow{}\;
      \underbrace{\text{block}}_{\le 1024 \text{ threads}}
      \;\xrightarrow{}\;
      \underbrace{\text{grid}}_{\text{your whole launch}}$$

    The contract:

    - **Threads in a block** can cooperate: they share **shared memory** and can
      **synchronize** (`__syncthreads()` / a Triton barrier). A block runs entirely
      on **one SM**.
    - **Blocks are independent.** Different blocks may run in any order, on any SM,
      possibly not at the same time. You may *not* assume two blocks run together or
      synchronize globally within a launch.
    - **Warps** are the real unit of execution: 32 threads issuing the *same*
      instruction together (next section).

    On your card: a block is at most **1024 threads** (32 warps), and an SM holds up
    to **1536 resident threads** (48 warps) drawn from one or more blocks.
    """)
    return


@app.cell
def _():
    def _run():
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        _fig, _ax = plt.subplots(figsize=(8, 3.6))
        _ax.set_xlim(0, 10)
        _ax.set_ylim(0, 4)
        _ax.axis("off")
        _ax.set_title("Grid -> Blocks -> Warps -> Threads")

        # grid box
        _ax.add_patch(mpatches.Rectangle(
            (0.1, 0.2), 9.8, 3.6, fill=False, edgecolor="#333", linewidth=2))
        _ax.text(0.25, 3.55, "grid", color="#333", fontsize=11, weight="bold")

        # blocks
        for _b in range(3):
            _x = 0.5 + _b * 3.1
            _ax.add_patch(mpatches.Rectangle((_x, 0.6), 2.7, 2.7,
                          fill=True, facecolor="#eef3ff", edgecolor="#5b8def", linewidth=1.5))
            _ax.text(_x + 0.1, 3.05, f"block {_b}", color="#3060c0", fontsize=9, weight="bold")
            # warps inside a block
            for _w in range(2):
                _wy = 0.85 + _w * 1.05
                _ax.add_patch(mpatches.Rectangle((_x + 0.15, _wy), 2.4, 0.8,
                              fill=True, facecolor="#dff0e4", edgecolor="#4c9f70"))
                _ax.text(_x + 0.25, _wy + 0.55, f"warp = 32 threads", color="#2e6b48", fontsize=7)
                # threads as ticks
                for _t in range(16):
                    _ax.add_patch(mpatches.Rectangle((_x + 0.2 + _t * 0.145, _wy + 0.08), 0.1, 0.28,
                                  fill=True, facecolor="#4c9f70", edgecolor="none"))
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Mapping onto the SMs

    The GPU is a grid of **Streaming Multiprocessors (SMs)** — your card has **70**.
    The hardware scheduler hands blocks to SMs. An SM runs as many resident blocks as
    its limits allow, interleaving their warps cycle-by-cycle. When one warp stalls on
    a load, the SM issues from another ready warp **in the same cycle** — that is the
    latency hiding from §1, made concrete.

    This is why you launch *thousands* of blocks for a big array even though there are
    only 70 SMs: the extra blocks queue up, and the surplus of resident warps is
    exactly what keeps the lanes fed. **More independent warps resident = more latency
    hidden.** The fraction of the SM's warp capacity you actually fill is called
    **occupancy**, and we devote `0d` to it.

    > A block is sticky to one SM for its whole life (so its threads can share memory
    > and sync). Blocks themselves are fire-and-forget across the 70 SMs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. SIMT: warps execute in lockstep

    The 32 threads of a warp share **one instruction stream** — Single Instruction,
    Multiple Thread. In the same cycle all 32 lanes execute the same instruction on
    their own data. This is why warp size pervades everything.

    The catch is **branch divergence**. If a data-dependent `if` sends some lanes down
    the `then` path and others down the `else`, the warp cannot do both at once — the
    hardware executes *both* paths and masks off the inactive lanes each time:

    $$\text{warp time} \;\approx\; \text{time}(\text{then}) + \text{time}(\text{else})
      \quad\text{when the warp diverges}$$

    A branch that is *uniform across the warp* (all 32 lanes agree) is free — no
    divergence. The lesson you will apply in every kernel: **structure work so a
    warp's 32 lanes follow the same path and touch contiguous data.** The worked
    example below shows the cost.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        print("=== Divergence cost in a 32-lane warp ===")
        print("Branch taken by lanes where (lane_id % stride == 0):\n")
        for _stride in [1, 2, 4, 32]:
            _lanes = np.arange(32)
            _take = (_lanes % _stride == 0)
            _n_then = int(_take.sum())
            # If any lane takes 'then' and any takes 'else', both paths run.
            _diverges = (0 < _n_then < 32)
            _cost = "BOTH paths run (masked)" if _diverges else "uniform -> 1 path (free)"
            print(f"  stride={_stride:2d}:  {_n_then:2d}/32 lanes take 'then'  ->  {_cost}")

        print("\nTakeaway: make the whole warp agree, or keep per-lane work identical.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. The launch boundary

    Kernels run on the **device**; you configure and launch them from the **host**.
    A launch specifies the **grid** (how many blocks) and the **block** (how many
    threads each). In CUDA C++ that is `kernel<<<grid, block>>>(args)`; in Triton you
    pass a `grid` and the framework derives the block from your `BLOCK_SIZE`.

    Two facts to carry forward:

    1. **You choose the decomposition.** How you map your data onto grid/block
       coordinates *is* the kernel design. Get it wrong and you serialize or diverge.
    2. **Launches are asynchronous and not free.** The CPU queues the kernel and moves
       on; a kernel launch costs microseconds of overhead. Tiny kernels are dominated
       by launch + memory traffic, not math — which is why we *fuse* operations later.

    The canonical 1-D mapping (you will write it in exercise `e01`):

    ```python
    # one program (block) handles BLOCK_SIZE contiguous elements
    pid   = program_id(0)                 # which block am I?
    offs  = pid * BLOCK_SIZE + arange(0, BLOCK_SIZE)
    mask  = offs < n_elements             # guard the ragged last block
    grid  = ceil_div(n_elements, BLOCK_SIZE)   # this many blocks
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6. Worked example: choosing a block size

    Suppose you launch over $N = 1\,000\,000$ elements with a block of `BLOCK_SIZE`
    threads. Drag the slider. Watch three things:

    - **#blocks** = $\lceil N / \text{BLOCK\_SIZE}\rceil$ — must cover all elements.
    - **warps/block** = $\text{BLOCK\_SIZE}/32$ — should divide evenly (no wasted lanes).
    - **occupancy** (threads-only model) = resident warps / 48, capped by the SM's
      1536-thread and up-to-32-blocks-per-SM (sm_120) limits.

    Occupancy here ignores registers and shared memory — the *other* two limiters,
    covered in `0d`. Even so, notice block sizes that aren't multiples of 32 waste
    lanes, and very small blocks can't fill the SM.
    """)
    return


@app.cell
def _(mo):
    block_slider = mo.ui.slider(start=32, stop=1024, step=32, value=256,
                                label="BLOCK_SIZE (threads/block)")
    block_slider
    return (block_slider,)


@app.cell
def _(block_slider):
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        N = 1_000_000
        MAX_THREADS_SM = 1536      # your 5070 Ti
        MAX_BLOCKS_SM = 32         # typical per-SM block cap
        WARPS_CAP = MAX_THREADS_SM // 32   # 48

        def occupancy(tpb):
            # resident blocks limited by thread budget AND block-count cap
            by_threads = MAX_THREADS_SM // tpb
            resident_blocks = max(0, min(by_threads, MAX_BLOCKS_SM))
            resident_warps = resident_blocks * (tpb // 32)
            return resident_warps / WARPS_CAP

        tpb = int(block_slider.value)
        n_blocks = -(-N // tpb)            # ceil div
        warps_per_block = tpb / 32

        xs = np.arange(32, 1025, 32)
        ys = [occupancy(int(x)) for x in xs]

        _fig, _ax = plt.subplots(figsize=(7.5, 3.4))
        _ax.plot(xs, ys, color="#5b8def", linewidth=2)
        _ax.axvline(tpb, color="#d65f5f", linestyle="--")
        _ax.scatter([tpb], [occupancy(tpb)], color="#d65f5f", zorder=5)
        _ax.set_xlabel("BLOCK_SIZE (threads/block)")
        _ax.set_ylabel("occupancy (threads-only)")
        _ax.set_ylim(0, 1.05)
        _ax.set_title(
            f"BLOCK_SIZE={tpb}:  {n_blocks:,} blocks,  "
            f"{warps_per_block:.1f} warps/block,  occ={occupancy(tpb):.0%}"
        )
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    - **Launch enough work.** A grid with too few blocks starves the 70 SMs and the
      latency-hiding collapses. When in doubt, more independent blocks.
    - **Think in warps of 32.** Block sizes that are multiples of 32, contiguous data
      per warp, uniform branches. Coalescing (`1b`) is the same idea applied to memory.
    - **Avoid divergence on the hot path.** Push data-dependent branches out of the
      inner loop, or make them warp-uniform.
    - **Fuse to amortize launch + traffic.** Many tiny kernels lose to one fused one.

    Hold these four and most "why is my kernel slow?" questions answer themselves.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Now go write it

    The next lecture (`0c`) drills into the memory hierarchy, but the execution model
    is best cemented by *doing*. Once you've read `1a` (the Triton programming model),
    open the terminal and write your first kernel:

    ```bash
    python -m harness.runner e01 --watch
    ```

    `e01` is vector-add — the "hello world" that forces you to use `program_id`, build
    an index range, mask the ragged tail, and choose a launch grid. Everything in this
    lecture, in ~10 lines you write yourself.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [0A: Orientation](../0a_orientation/) &nbsp;|&nbsp; Next: [0C: Memory Hierarchy](../0c_memory_hierarchy/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
