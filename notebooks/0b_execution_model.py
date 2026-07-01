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

    This lecture is deliberately heavy on definitions. Every italicized term below is
    one you will use in every kernel you write, so we pin each one down the first time
    it appears — and collect them all in the glossary box right under this paragraph.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The words you'll need (keep this open)

    | term | one-line meaning |
    |---|---|
    | **kernel** | a function *you* write that runs on the GPU, launched as a grid of many threads |
    | **thread / lane** | the smallest unit of execution; runs your kernel on its *own* slice of data |
    | **warp** | a bundle of **32 threads** that execute the *same* instruction in lockstep — the real scheduling unit |
    | **block** | a group of threads (≤ 1024) that share fast on-chip memory and can synchronize; lives on one SM |
    | **grid** | *all* the blocks of a single launch |
    | **SM** (Streaming Multiprocessor) | a physical cluster of lanes on the chip — your card has **70**; each runs many warps at once |
    | **host / device** | host = the CPU running your Python/C++ program; device = the GPU that runs the kernel |
    | **cycle** | one tick of the GPU clock; latencies are quoted in cycles (a register read ≈ 1, a DRAM read ≈ hundreds) |
    | **latency** | how long *one* operation takes to finish |
    | **throughput** | how *many* operations finish per unit time, counting everything in flight at once |
    | **occupancy** | how full an SM's warp slots are — i.e. how much parallel work is resident to hide latency (lecture `0d`) |

    Don't memorize the table — just know it's here. Each term is reintroduced in
    context below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. Throughput, not latency

    Two words run through this entire course, so pin them down first (they're in the
    glossary too):

    - **Latency** — how long *one* operation takes to finish. A load from DRAM costs
      **hundreds of clock cycles**, where a *cycle* is a single tick of the GPU's clock.
    - **Throughput** — how *many* operations finish per unit time, counting everything
      in flight at once.

    A CPU core spends most of its transistors minimizing **latency** for *one* stream
    of instructions: big caches, branch prediction, out-of-order execution. It hates
    waiting.

    A GPU makes the opposite bet. It maximizes **throughput** and simply *accepts* that
    any single operation is slow. It has thousands of simple lanes and *expects* to
    wait. Instead of working hard to avoid a memory stall, it **parks the stalled work
    and runs other work**. With enough independent work resident, the memory latency is
    completely **hidden** behind useful computation — note the word: hidden, not
    *reduced*. The DRAM load is just as slow; the machine is simply never sitting around
    waiting for it.

    So the GPU's whole personality is: **oversubscribe me.** Give it far more parallel
    work than it has lanes, and it keeps the lanes busy while memory catches up. A
    kernel that does *not* supply enough parallel work leaves the machine idle — that is
    the failure mode you will learn to recognize, and the picture below shows exactly
    what it looks like.

    > [PMPP](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)
    > Ch. 1–4 develops this latency-hiding model in depth.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Seeing latency hiding

    Here is the single most important mechanism in the course, simulated. Each **warp**
    repeats the same rhythm: issue a little math, then **stall** waiting for a DRAM load
    to return. The simulation deliberately models a single **warp scheduler** that can
    issue from only one ready warp per cycle. (A real `sm_120` SM has 4 processing
    partitions, each with its own scheduler, so the SM as a whole issues from up to 4
    warps per cycle — the latency-hiding intuition below is per-scheduler.) Green = the
    scheduler is issuing math; red = it sat **idle** because *every* resident warp was
    stalled on memory.

    - **Left — 1 warp resident:** the lone warp computes for a moment, then stalls for
      the whole memory latency while the SM has nothing else to run. The machine is idle
      most of the time. This is the starved kernel.
    - **Right — 6 warps resident:** while warp 0 waits on its load, warps 1–5 issue
      their math. The stalls are still there — they're just *hidden* behind other warps'
      work, and the SM stays busy.

    Same slow memory, same per-warp stalls. The only thing that changed is **how much
    independent work was resident** — and that is the entire game.
    """)
    return


@app.cell
def _():
    def _run():
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        COMPUTE = 2     # cycles of math a warp issues before its next memory load
        STALL = 10      # cycles a DRAM load takes to return (the latency we hide)
        T = 36          # cycles simulated

        # Cycle-by-cycle scheduler for one SM that issues one warp per cycle.
        # Returns: per-warp list of cycles it issued math, list of idle cycles,
        # and the fraction of cycles the SM was busy.
        def simulate(n_warps):
            rem_c = [0] * n_warps      # compute cycles left in current chunk
            rem_s = [0] * n_warps      # stall cycles left on the in-flight load
            ready = [True] * n_warps   # not stalled, has work to issue
            comp = [[] for _ in range(n_warps)]
            idle = []
            busy = 0
            rr = 0                     # round-robin pointer
            for t in range(T):
                # memory is served in parallel: every stall ticks down each cycle
                for w in range(n_warps):
                    if rem_s[w] > 0:
                        rem_s[w] -= 1
                        if rem_s[w] == 0:
                            ready[w] = True
                # pick a warp to issue: continue one mid-compute, else start a ready one
                pick = -1
                for off in range(n_warps):
                    w = (rr + off) % n_warps
                    if rem_c[w] > 0:
                        pick = w
                        break
                if pick == -1:
                    for off in range(n_warps):
                        w = (rr + off) % n_warps
                        if ready[w]:
                            pick = w
                            rem_c[w] = COMPUTE
                            ready[w] = False
                            break
                if pick == -1:
                    idle.append(t)                 # nothing ready: SM stalls
                else:
                    busy += 1
                    rem_c[pick] -= 1
                    comp[pick].append(t)
                    if rem_c[pick] == 0:           # finished a chunk -> issue a load
                        rem_s[pick] = STALL
                    rr = pick + 1
            return comp, idle, busy / T

        _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.9))
        for _ax, _n in zip(_axes, [1, 6]):
            _comp, _idle, _util = simulate(_n)
            for _w in range(_n):
                for _t in _comp[_w]:
                    _ax.add_patch(mpatches.Rectangle(
                        (_t, _w + 0.1), 1, 0.8, facecolor="#4c9f70", edgecolor="none"))
            # SM-busy/idle strip beneath the warps
            _yidle = -1.2
            for _t in range(T):
                _is_busy = _t not in _idle
                _ax.add_patch(mpatches.Rectangle(
                    (_t, _yidle), 1, 0.8,
                    facecolor=("#cfe8d8" if _is_busy else "#d65f5f"),
                    edgecolor="white", linewidth=0.3))
            _ax.text(-0.5, _yidle + 0.4, "SM", fontsize=8, ha="right", va="center")
            _ax.set_xlim(-3, T)
            _ax.set_ylim(_yidle - 0.4, _n + 0.4)
            _ax.set_yticks([_w + 0.5 for _w in range(_n)])
            _ax.set_yticklabels([f"warp {_w}" for _w in range(_n)], fontsize=8)
            _ax.set_xlabel("clock cycle")
            _ax.set_title(f"{_n} warp(s) resident  ->  SM busy {_util:.0%} of the time")
        _fig.suptitle(
            "Latency hiding: green = SM issuing math,  red = SM idle (every warp stalled)",
            y=1.03)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. The thread hierarchy

    Your kernel is launched as a **grid** of **thread blocks**, each block a group of
    **threads**. A single **thread** is one **lane** of execution: it runs your kernel
    code on its *own* slice of the data, and it is the smallest unit there is. Threads
    are scheduled by the hardware in fixed bundles of 32 called **warps**.

    $$\underbrace{\text{thread}}_{\text{1 lane}}
      \;\xrightarrow{\times 32}\;
      \underbrace{\text{warp}}_{\text{lockstep}}
      \;\xrightarrow{}\;
      \underbrace{\text{block}}_{\le 1024 \text{ threads}}
      \;\xrightarrow{}\;
      \underbrace{\text{grid}}_{\text{your whole launch}}$$

    Why 32? It is the hardware's fixed bundle width — the warp is what the scheduler
    actually issues, so 32 is the granularity of *everything*: how lanes are grouped,
    how memory requests are batched, how branches diverge. You will round to multiples
    of 32 constantly.

    The contract between the levels:

    - **Threads in a block** can cooperate. They share **shared memory** — a small, fast
      on-chip scratchpad that every thread in the block can read and write (detailed in
      `0c`) — and they can **synchronize**: a *barrier* where every thread in the block
      waits until all of them have arrived (written `__syncthreads()` in CUDA, a barrier
      call in Triton). A block runs entirely on **one SM** — a *Streaming
      Multiprocessor*, the physical cluster of lanes on the chip that we unpack in §3.
    - **Blocks are independent.** Different blocks may run in any order, on any SM,
      possibly not even at the same time. You may *not* assume two blocks run together,
      and there is no barrier *across* blocks within a launch.
    - **Warps** are the real unit of execution: 32 threads issuing the *same* instruction
      together (next section).

    On your card: a block is at most **1024 threads** (32 warps), and an SM holds up to
    **1536 resident threads** (48 warps) drawn from one or more blocks.
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
                # threads as ticks (one per lane)
                for _t in range(32):
                    _ax.add_patch(mpatches.Rectangle((_x + 0.2 + _t * 0.0715, _wy + 0.08), 0.05, 0.28,
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

    The GPU is an array of **Streaming Multiprocessors (SMs)** — your card has **70**.
    An SM is the physical unit that owns a pool of lanes, a register file, and a slab of
    shared memory. The hardware scheduler hands blocks to SMs; each SM runs as many
    resident blocks as its limits allow and interleaves their warps cycle-by-cycle. When
    one warp stalls on a load, the SM issues from another ready warp **in the very next
    cycle** — that is the latency hiding from §1, now located in real silicon.

    This is why you launch *thousands* of blocks for a big array even though there are
    only 70 SMs. The blocks that don't fit run later, in **waves**: a first batch fills
    every SM to capacity, and as those blocks finish, the next batch moves in. The print
    below works out how many blocks are resident at once and how many waves a launch
    takes — and the surplus of resident warps in each wave is exactly what keeps the
    lanes fed.

    > A block is *sticky* to one SM for its whole life (so its threads can share memory
    > and synchronize). Blocks themselves are fire-and-forget across the 70 SMs — order
    > and timing between blocks are not yours to control.
    """)
    return


@app.cell
def _():
    def _run():
        # How blocks of a given size pack onto the 70 SMs, and how many "waves"
        # a launch of N blocks takes. Same limits used by the slider in section 6.
        SMS = 70
        MAX_THREADS_SM = 1536      # resident-thread budget per SM (5070 Ti)
        MAX_BLOCKS_SM = 24         # hardware cap on resident blocks per SM (CC 12.x)

        BLOCK_SIZE = 256           # threads/block = 8 warps
        by_threads = MAX_THREADS_SM // BLOCK_SIZE        # blocks an SM can hold (threads)
        resident_per_sm = min(by_threads, MAX_BLOCKS_SM)  # ... capped by the block limit
        resident_total = resident_per_sm * SMS            # blocks live across the whole GPU

        print("=== One wave on the RTX 5070 Ti ===")
        print(f"  BLOCK_SIZE = {BLOCK_SIZE} threads ({BLOCK_SIZE // 32} warps)")
        print(f"  per SM:  min(1536//{BLOCK_SIZE}, {MAX_BLOCKS_SM}) = "
              f"{resident_per_sm} resident blocks")
        print(f"  whole GPU:  {resident_per_sm} x {SMS} SMs = "
              f"{resident_total} blocks resident at once  (= one wave)\n")

        print(f"  {'N blocks launched':>18s} {'waves = ceil(N / wave)':>24s}")
        print("  " + "-" * 44)
        for _N in [70, resident_total, 10_000, 1_000_000]:
            _waves = -(-_N // resident_total)   # ceil div
            print(f"  {_N:>18,} {_waves:>24,}")

        print("\n  Takeaway: a launch is many waves of blocks streaming across 70 SMs.")
        print("  Each wave wants a surfeit of resident warps -- that surplus is the")
        print("  latency hiding from section 1, paid for in parallel work.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. SIMT: warps execute in lockstep

    The 32 threads of a warp share **one instruction stream** — Single Instruction,
    Multiple Thread. In the same cycle all 32 lanes execute the same instruction, each
    on its own data. (Contrast SIMD on a CPU, where *you* pack the lanes into one vector
    register by hand; under SIMT you write scalar per-thread code and the hardware gangs
    32 threads into a warp for you.) This is why the warp size of 32 pervades everything.

    The catch is **branch divergence**. If a data-dependent `if` sends some lanes down
    the `then` path and others down the `else`, the warp cannot run both at once — it has
    only one instruction stream. The hardware executes *both* paths in sequence and
    **masks off** the inactive lanes each time:

    $$\text{warp time} \;\approx\; \text{time}(\text{then}) + \text{time}(\text{else})
      \quad\text{when the warp diverges}$$

    "Masks off" means the inactive lanes still ride along through the path — they occupy
    the warp but their results are thrown away. They are *predicated off*, not skipped,
    which is exactly why both paths cost time. A branch that is **uniform across the
    warp** (all 32 lanes agree) takes only one path and is free — no divergence.

    The lesson you will apply in every kernel: **structure work so a warp's 32 lanes
    follow the same path and touch contiguous data.** The worked example below counts
    the cost.
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

    Kernels run on the **device** (the GPU); you configure and launch them from the
    **host** (your CPU, running the Python or C++ program). A launch specifies the
    **grid** (how many blocks) and the **block** (how many threads each). In CUDA C++
    that is `kernel<<<grid, block>>>(args)`; in Triton you pass a `grid` and the
    framework derives the block from your `BLOCK_SIZE`.

    Two facts to carry forward:

    1. **You choose the decomposition.** How you map your data onto grid/block
       coordinates *is* the kernel design. Get it wrong and you serialize the work or
       force a warp to diverge.
    2. **Launches are asynchronous and not free.** The host *queues* the kernel and
       moves on without waiting; a launch costs microseconds of overhead before any of
       your code runs. Tiny kernels are dominated by that launch overhead plus memory
       traffic, not by math — which is why we *fuse* operations together later.

    The canonical 1-D mapping (you will write it in exercise `e01`):

    ```python
    # one program (block) handles BLOCK_SIZE contiguous elements
    pid   = program_id(0)                 # which block am I?
    offs  = pid * BLOCK_SIZE + arange(0, BLOCK_SIZE)
    mask  = offs < n_elements             # guard the ragged last block
    grid  = ceil_div(n_elements, BLOCK_SIZE)   # this many blocks
    ```

    Read that as: each block claims a contiguous `BLOCK_SIZE`-wide window of the array,
    `offs` are the global indices that window covers, and `mask` switches off the lanes
    that would run past the end when `n_elements` isn't a clean multiple of `BLOCK_SIZE`.
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
    - **occupancy** = resident warps / 48. *Occupancy* is how full the SM's warp slots
      are — the §1 surplus, quantified. Here it is capped by the SM's 1536-thread budget
      and its 24-blocks-per-SM (`sm_120`) limit.

    This occupancy model ignores registers and shared memory — the *other* two limiters,
    covered in `0d`. Even so, notice that very small blocks can't put enough warps on an
    SM to hide latency: at `BLOCK_SIZE=32` the 24-block cap allows only 24 resident
    warps — 50% occupancy. (The slider moves in steps of 32 because block sizes should
    always be multiples of the warp size — a ragged block still occupies whole warp
    slots.)

    > Footnote on the 24: NVIDIA's Blackwell tuning guide erroneously says 32 blocks/SM
    > for CC 12.0 — that figure is correct only for CC 10.0. The CUDA Programming Guide
    > table and the runtime (`cudaDevAttrMaxBlocksPerMultiprocessor`) both say **24**
    > for this card.
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
        MAX_BLOCKS_SM = 24         # per-SM block cap (CC 12.x; see footnote above)
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
      latency-hiding collapses — the left panel of the §1 picture. When in doubt, more
      independent blocks.
    - **Think in warps of 32.** Block sizes that are multiples of 32, contiguous data
      per warp, uniform branches. Coalescing (`1b`) is the same idea applied to memory.
    - **Avoid divergence on the hot path.** Push data-dependent branches out of the
      inner loop, or make them warp-uniform.
    - **Fuse to amortize launch + traffic.** Many tiny kernels lose to one fused one,
      because each launch pays the host-side overhead and re-reads memory.

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
