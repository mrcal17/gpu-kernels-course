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
    # 0D: Occupancy & the Roofline

    > *"You cannot make a kernel fast. You can only stop it from being slow in the one
    > way that matters."*

    Two questions close out Part 0, and together they let you predict a kernel's
    performance before you tune it:

    1. **Occupancy** — am I keeping the SM busy? The GPU hides latency by interleaving
       many resident warps (`0b`); occupancy measures how full of warps each SM is, and
       three hardware resources cap it.
    2. **The roofline** — what is the *ceiling* I'm aiming at? Given a kernel's
       operational intensity (`0c`), the roofline model tells you whether bandwidth or
       compute limits you, and what the best achievable rate is.

    This is an **analysis** lecture: no kernel to write, but the two diagnostic tools
    you'll reach for every time a kernel underperforms.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. Occupancy: how full is the SM?

    An SM has a fixed warp capacity. On your 5070 Ti it holds **1536 resident
    threads = 48 warps**. **Occupancy** is the fraction of that capacity your kernel
    actually fills:

    $$\text{occupancy} = \frac{\text{resident warps}}{\text{max warps per SM}}
      = \frac{\text{resident warps}}{48}.$$

    Why care? Because resident warps are the raw material of latency hiding. When one
    warp stalls on a DRAM load (~500 cycles), the scheduler issues from another ready
    warp. The more warps resident, the more independent work in flight to cover those
    stalls. Low occupancy means too few warps to hide latency, so the math units sit
    idle waiting on memory.

    Occupancy is **not** the goal itself — a perfectly coalesced, well-tiled kernel at
    50% occupancy can beat a sloppy one at 100%. But low occupancy is a common *cause*
    of slowness, and it's the first thing to check. The subtlety is *what stops you
    from being fully occupied*: three resources, any of which can be the binding limit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. The three limiters

    The hardware places **as many blocks on an SM as all three budgets allow**, then
    the resident-warp count is `blocks_resident x warps_per_block`. The three budgets
    on your 5070 Ti:

    **(a) Threads / SM** — at most **1536** resident threads (48 warps). A block of
    `T` threads lets at most $\lfloor 1536 / T \rfloor$ blocks fit by this rule alone.

    **(b) Registers / SM** — **65,536** registers, partitioned across all resident
    threads. If the compiler assigns `R` registers per thread, a block of `T` threads
    needs $R\cdot T$ registers, so by this rule:
    $$\text{blocks}_{\text{reg}} = \left\lfloor \frac{65{,}536}{R \cdot T} \right\rfloor.$$
    This is why a register-hungry kernel ($R$ large) can be *limited to a handful of
    warps* even with a small block — full occupancy needs $R \lesssim 65536/1536
    \approx 42$ registers/thread.

    **(c) Shared memory / SM** — **~100 KB** usable (48 KB/block default, opt-in to
    ~99 KB). If a block declares `S` bytes of shared memory:
    $$\text{blocks}_{\text{smem}} = \left\lfloor \frac{100\,\text{KB}}{S} \right\rfloor.$$
    A block that grabs 50 KB of shared memory can have **at most one block** per SM, no
    matter how few threads or registers it uses.

    The resident block count is the **minimum** across all three (and a hardware cap of
    ~16–32 blocks/SM):

    $$\text{blocks}_{\text{resident}} = \min\!\big(
      \text{blocks}_{\text{threads}},\,
      \text{blocks}_{\text{reg}},\,
      \text{blocks}_{\text{smem}},\,
      \text{blocks}_{\text{cap}}\big).$$

    > [CUDA C++ Best Practices Guide — Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy)
    > and PMPP Ch. 4 ("Compute architecture and scheduling") derive these limits.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Worked occupancy math

    Take a block of **256 threads** (8 warps) using **32 registers/thread** and
    **16 KB shared memory**. Work each limiter, then take the min:

    - **Threads:** $\lfloor 1536/256 \rfloor = 6$ blocks.
    - **Registers:** $\lfloor 65536 / (32\cdot256) \rfloor = \lfloor 65536/8192\rfloor
      = 8$ blocks.
    - **Shared mem:** $\lfloor 100\text{KB} / 16\text{KB} \rfloor = 6$ blocks.

    Min = **6 blocks** = $6\times 8 = 48$ resident warps = **100% occupancy** — here the
    *threads* and *shared-mem* rules tie as the binding limit. Now bump registers to
    **64/thread**: registers allow only $\lfloor 65536/(64\cdot256)\rfloor = 4$ blocks
    = 32 warps = **67%**. One register decision moved you off the peak. The calculator
    below lets you feel exactly that.
    """)
    return


@app.cell
def _():
    def _run():
        MAX_THREADS = 1536
        MAX_REGS = 65536
        SMEM_KB = 100.0

        def occ(tpb, regs, smem_kb):
            by_threads = MAX_THREADS // tpb
            by_regs = MAX_REGS // (regs * tpb)
            by_smem = int(SMEM_KB // smem_kb) if smem_kb > 0 else 999
            blocks = min(by_threads, by_regs, by_smem)
            warps = blocks * (tpb // 32)
            return by_threads, by_regs, by_smem, blocks, warps, warps / 48

        print("=== Worked occupancy: block of 256 threads ===\n")
        print(f"  {'regs/thd':>8s} {'smem KB':>7s} {'b_thr':>6s} {'b_reg':>6s} "
              f"{'b_smem':>7s} {'blocks':>7s} {'warps':>6s} {'occ':>6s}")
        print("  " + "-" * 62)
        for _regs, _smem in [(32, 16), (64, 16), (32, 50), (96, 8)]:
            _bt, _br, _bs, _blk, _w, _o = occ(256, _regs, _smem)
            print(f"  {_regs:>8d} {_smem:>7.0f} {_bt:>6d} {_br:>6d} "
                  f"{_bs:>7d} {_blk:>7d} {_w:>6d} {_o:>5.0%}")
        print("\n  The binding limiter is whichever column is smallest.")
        print("  64 regs/thread halves blocks (8 -> 4); 50 KB smem caps at 6 by SRAM.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactive occupancy calculator

    Two sliders: **threads per block** and **registers per thread**. The plot computes
    each limiter on your 5070 Ti's numbers, shows which one binds, and reports resident
    warps and occupancy. (Shared memory is fixed at a modest 8 KB/block here so you can
    focus on the threads-vs-registers tug-of-war — the most common one in practice.)
    Watch the binding limiter switch as you push registers up.
    """)
    return


@app.cell
def _(mo):
    tpb_slider = mo.ui.slider(start=32, stop=1024, step=32, value=256,
                              label="threads / block")
    regs_slider = mo.ui.slider(start=16, stop=128, step=4, value=32,
                               label="registers / thread")
    mo.vstack([tpb_slider, regs_slider])
    return regs_slider, tpb_slider


@app.cell
def _(regs_slider, tpb_slider):
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        MAX_THREADS = 1536
        MAX_REGS = 65536
        SMEM_KB = 100.0
        SMEM_PER_BLOCK = 8.0   # fixed in this demo
        BLOCK_CAP = 32         # hardware-ish per-SM block cap

        tpb = int(tpb_slider.value)
        regs = int(regs_slider.value)

        by_threads = MAX_THREADS // tpb
        by_regs = MAX_REGS // (regs * tpb)
        by_smem = int(SMEM_KB // SMEM_PER_BLOCK)
        limits = {
            "threads/SM": by_threads,
            "registers/SM": by_regs,
            "smem/SM": by_smem,
            "block cap": BLOCK_CAP,
        }
        blocks = min(limits.values())
        binder = min(limits, key=limits.get)
        warps = blocks * (tpb // 32)
        occ = warps / 48.0

        names = list(limits.keys())
        vals = [limits[n] for n in names]
        colors = ["#d65f5f" if n == binder else "#cbd6ee" for n in names]

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6),
                                          gridspec_kw={"width_ratios": [1.4, 1]})

        _ax1.bar(names, vals, color=colors, edgecolor="none")
        _ax1.axhline(blocks, color="#4c9f70", linestyle="--", linewidth=1.5,
                     label=f"resident = {blocks} blocks")
        _ax1.set_ylabel("max blocks/SM by this rule")
        _ax1.set_title(f"binding limiter: {binder}")
        _ax1.legend(loc="upper right", fontsize=8)
        _ax1.tick_params(axis="x", labelrotation=15)

        _ax2.bar(["occupancy"], [occ], color="#5b8def", width=0.5)
        _ax2.axhline(1.0, color="#999", linestyle=":", linewidth=1)
        _ax2.set_ylim(0, 1.05)
        _ax2.set_title(f"{warps} warps -> {occ:.0%}")
        _ax2.set_ylabel("resident warps / 48")

        _fig.suptitle(
            f"block={tpb} thr ({tpb // 32} warps),  {regs} regs/thread", y=1.03)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Latency hiding & Little's Law

    Why does occupancy translate into speed? Because hiding a long memory latency
    requires *enough independent work in flight*. This is Little's Law from queueing
    theory, borrowed for hardware:

    $$\text{required concurrency} = \text{latency} \times \text{throughput}.$$

    To keep a unit busy that takes `L` cycles per operation and can retire one result
    per cycle, you need ~`L` independent operations outstanding at all times. For DRAM:
    if a load takes ~500 cycles and you want to sustain many bytes/cycle, you need
    *hundreds of loads in flight* — which means many resident warps each with
    independent loads. Too few warps → the pipeline drains → the SM stalls.

    A back-of-envelope: to hide a ~500-cycle latency, you want on the order of
    $500 / (\text{cycles between independent loads per warp})$ warps' worth of
    outstanding work. This is *why* the GPU wants oversubscription, and why occupancy is
    the lever: more resident warps = more memory requests in flight = more latency
    hidden. It is the §1 idea of `0b`, now quantified.

    > Vasily Volkov's "Better performance at lower occupancy" (GTC 2010) is the classic
    > demonstration that *instruction-level* parallelism per thread can substitute for
    > occupancy — concurrency is what matters, and occupancy is one way to buy it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. The roofline model

    Occupancy tells you if the SM is fed. The **roofline** tells you the *ceiling* that
    feeding is chasing. It plots, for a given kernel:

    - **x-axis:** operational intensity $I$ (FLOP/byte) from `0c` — log scale.
    - **y-axis:** attainable performance (FLOP/s) — log scale.

    Two hardware limits form the roof. A kernel of intensity $I$ can never exceed:

    $$P(I) = \min\big(\underbrace{P_{\text{peak}}}_{\text{compute roof}},\;
      \underbrace{B \cdot I}_{\text{memory roof}}\big),$$

    where $P_{\text{peak}}$ is peak FLOP/s and $B$ is peak DRAM bandwidth
    (~896 GB/s). On the log-log plot the memory term $B\cdot I$ is a **slanted line of
    slope 1** (bandwidth), and the compute term is a **flat ceiling**. They meet at the
    **ridge point**:

    $$I_{\text{ridge}} = \frac{P_{\text{peak}}}{B}.$$

    - **Left of the ridge** ($I < I_{\text{ridge}}$): you're on the slanted roof —
      **memory-bound**. Performance $= B\cdot I$; the only ways up are *raise $I$*
      (reuse data, fuse) or *raise effective $B$* (coalesce).
    - **Right of the ridge** ($I > I_{\text{ridge}}$): you're under the flat roof —
      **compute-bound**. You're limited by the math units; use faster math (tensor
      cores, lower precision) or you're already near peak.

    The roofline turns "is my kernel slow?" into "is my kernel near *its* roof?" — a
    much more answerable question.

    > [Williams, Waterman & Patterson, "Roofline: an insightful visual performance
    > model" (CACM 2009)](https://dl.acm.org/doi/10.1145/1498765.1498785) is the
    > original; the [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
    > frames every optimization as moving toward one of these two roofs.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        # 5070 Ti: bandwidth exact; FP32 peak is a representative ~44 TFLOP/s
        # (tensor-core peaks are far higher and measured later in Parts 2-3).
        B = 896e9               # bytes/s
        P_PEAK = 44e12          # FLOP/s, FP32 (illustrative)
        I_ridge = P_PEAK / B

        I = np.logspace(-2, 3, 400)          # FLOP/byte
        roof = np.minimum(P_PEAK, B * I)     # attainable FLOP/s

        # Example kernels placed at their operational intensities.
        kernels = [
            ("vector add",   1 / 12,   "#d65f5f"),
            ("matmul n=256", 256 / 6,  "#e0a458"),
            ("matmul n=4096", 4096 / 6, "#4c9f70"),
        ]

        _fig, _ax = plt.subplots(figsize=(8, 4.2))
        _ax.plot(I, roof, color="#333", linewidth=2.2, zorder=3)
        _ax.axvline(I_ridge, color="#999", linestyle=":", linewidth=1.2)
        _ax.text(I_ridge * 1.1, B * 0.02,
                 f"ridge\nI={I_ridge:.0f}", color="#666", fontsize=8)

        # Annotate the two regimes.
        _ax.text(0.05, P_PEAK * 0.5, "memory-bound\n(slope = bandwidth)",
                 fontsize=9, color="#5b8def")
        _ax.text(I_ridge * 2.5, P_PEAK * 1.25, "compute-bound (flat = peak FLOP/s)",
                 fontsize=9, color="#4c9f70")

        for _name, _Ik, _col in kernels:
            _perf = min(P_PEAK, B * _Ik)
            _ax.scatter([_Ik], [_perf], color=_col, s=70, zorder=5)
            _ax.annotate(_name, (_Ik, _perf), textcoords="offset points",
                         xytext=(6, -12), fontsize=8, color=_col)

        _ax.set_xscale("log")
        _ax.set_yscale("log")
        _ax.set_xlabel("operational intensity  I  (FLOP / byte)")
        _ax.set_ylabel("attainable performance (FLOP/s)")
        _ax.set_title("RTX 5070 Ti roofline: 896 GB/s slope, ~44 TFLOP/s ceiling")
        _ax.set_ylim(1e10, P_PEAK * 3)
        _ax.grid(True, which="both", alpha=0.15)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the picture

    The three dots tell the whole Part-1-to-2 story:

    - **Vector add** ($I\approx0.08$) sits far left, glued to the slanted memory roof.
      No amount of compute tuning helps; its ceiling is `896 GB/s × 0.08`. The win is
      coalescing and fusion — exactly `0c`'s lesson.
    - **Small matmul** (n=256, $I\approx43$) is near the ridge — borderline. Tiling to
      raise reuse pushes it right, toward compute-bound.
    - **Big matmul** (n=4096, $I\approx680$) is deep in compute-bound territory, pinned
      to the flat roof. Here the lever is *faster math* — tensor cores and lower
      precision (Parts 2–3), which lift the ceiling itself.

    Same hardware, same plot, opposite optimization strategies — chosen entirely by
    where the kernel lands relative to the ridge.
    """)
    return


@app.cell
def _(mo):
    intensity_slider = mo.ui.slider(start=-2.0, stop=3.0, step=0.1, value=0.0,
                                    label="log10(operational intensity)  [FLOP/byte]")
    intensity_slider
    return (intensity_slider,)


@app.cell
def _(intensity_slider):
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        B = 896e9
        P_PEAK = 44e12
        I_ridge = P_PEAK / B

        I = np.logspace(-2, 3, 400)
        roof = np.minimum(P_PEAK, B * I)

        my_I = 10.0 ** float(intensity_slider.value)
        my_perf = min(P_PEAK, B * my_I)
        regime = "memory-bound" if my_I < I_ridge else "compute-bound"
        frac_peak = my_perf / P_PEAK

        _fig, _ax = plt.subplots(figsize=(8, 4.0))
        _ax.plot(I, roof, color="#333", linewidth=2.2, zorder=3)
        _ax.axvline(I_ridge, color="#999", linestyle=":", linewidth=1.2)
        _ax.scatter([my_I], [my_perf], color="#d65f5f", s=90, zorder=5)
        _ax.annotate(f"I={my_I:.2f}\n{my_perf/1e12:.1f} TFLOP/s",
                     (my_I, my_perf), textcoords="offset points",
                     xytext=(8, -18), fontsize=8, color="#d65f5f")
        _ax.set_xscale("log")
        _ax.set_yscale("log")
        _ax.set_xlabel("operational intensity  I  (FLOP / byte)")
        _ax.set_ylabel("attainable performance (FLOP/s)")
        _ax.set_ylim(1e10, P_PEAK * 3)
        _ax.set_title(f"{regime}:  {frac_peak:.0%} of peak compute attainable")
        _ax.grid(True, which="both", alpha=0.15)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    - **Diagnose before you optimize.** Compute the kernel's operational intensity,
      place it on the roofline. Left of the ridge → chase bandwidth; right → chase
      compute. The plot tells you which knobs can possibly help.
    - **Occupancy is a means, not an end.** Check it when a kernel underperforms — low
      occupancy starves latency hiding — but don't chase 100% blindly. Enough warps to
      hide latency is the real target; registers and shared memory are what you trade
      against it.
    - **Know your three limiters.** When occupancy is low, identify *which* of
      threads/registers/shared-memory binds, and spend the right resource. A
      register-heavy kernel and a shared-memory-heavy kernel need opposite fixes.
    - **Aim at *the* roof, not *a* number.** "My matmul hits 80% of the compute roof"
      and "my copy hits 90% of the bandwidth roof" are both wins. The roofline defines
      what winning even means for each kernel.

    With Part 0 complete you have the full mental model: execution (`0b`), memory
    (`0c`), and now the two diagnostics that connect them to performance. Everything in
    Parts 1–3 is an instance of moving a dot up toward its roof.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Where this is used

    There's no kernel to write for this lecture — it's the lens you'll look through for
    every kernel that follows. You'll apply it immediately:

    - In `1b` and exercise `e03`, you'll measure achieved GB/s and check it against the
      *memory* roof (the bandwidth slope).
    - In `1e`/`2a` and exercises `e07`/`e10`, you'll measure TFLOP/s and check it
      against the *compute* roof, watching tiling push a matmul rightward past the
      ridge.

    Keep this notebook open as a reference — the roofline plot is the scoreboard for
    the rest of the course.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [0C: Memory Hierarchy](../0c_memory_hierarchy/) &nbsp;|&nbsp; Next: [1A: The Triton Programming Model](../1a_triton_model/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
