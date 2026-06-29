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
    # 3E: Occupancy Tuning

    > *"Registers are the fastest memory on the chip, and the scarcest. Every one you
    > spend on a thread is one fewer thread the SM can keep resident."*

    > **Advanced / off the critical path.** This is the practical follow-up to `0d`:
    > there we *defined* occupancy and its three limiters; here we drive the one knob
    > that bites hardest in real CUDA C++ kernels — **register pressure** — and learn to
    > read the verdict in Nsight Compute. No new kernel to write; the exercise is
    > *profile an existing kernel and explain its occupancy*.

    In `0d` you computed occupancy from threads, registers, and shared memory as if the
    register count were a given. It is not — it is an *output of the compiler*, and one
    you can negotiate with. This lecture is about that negotiation: how the compiler
    decides how many registers each thread gets, how that decision caps the warps an SM
    can hold, the two levers (`__launch_bounds__`, `-maxrregcount`) for forcing its
    hand, and the surprising fact that **lower occupancy is sometimes faster**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. Registers are the binding limiter in real kernels

    Recall the three occupancy budgets on your 5070 Ti (`0d`): **1536** resident
    threads/SM, **65,536** registers/SM, **~100 KB** shared memory/SM. Of these, the
    register budget is the one you trip over without meaning to — because *you never
    asked for registers*. The compiler allocates them silently for every live local
    variable, loop-carried value, and address it needs to keep in flight.

    The arithmetic that matters: for the SM to hold all 48 warps (full occupancy), the
    registers-per-thread budget is

    $$R_{\max} = \frac{65{,}536}{1536} \approx 42.6 \;\Rightarrow\; R \le 40
      \text{ registers/thread for 100% occupancy.}$$

    That 42.6 is the *continuous* bound; warp-granularity rounding (256 regs/warp,
    below) tightens the real ceiling to **40 regs/thread** — 41–42 already round up to
    1536 regs/warp = 42 warps = 88%. Cross 40 and you fall off the peak in **discrete
    steps**, because the hardware
    allocates registers in *granularity-rounded blocks* and packs whole warps. A kernel
    the compiler decides needs 64 registers/thread can keep at most

    $$\left\lfloor \frac{65{,}536}{64 \cdot 32} \right\rfloor = 32 \text{ warps}
      = \frac{32}{48} \approx 67\% \text{ occupancy,}$$

    counting in warps of 32 threads. The kernel did not get slower instruction-by-
    instruction — it just got *lonelier*, with fewer sibling warps to hide its memory
    latency behind.

    > [CUDA C++ Best Practices Guide — Register Pressure](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#register-pressure)
    > and PMPP Ch. 5 cover how register allocation feeds back into occupancy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How to *see* the register count

    The compiler will tell you exactly how many registers it spent if you ask. Two ways,
    both at build time — no profiler needed yet:

    ```bash
    # per-kernel resource usage: registers, shared mem, spills, constant mem
    nvcc -arch=sm_120 --ptxas-options=-v -c my_kernel.cu
    # ptxas info : Used 64 registers, 8192 bytes smem, 0 bytes spill stores ...
    ```

    The line you watch is **`Used N registers`** and, crucially, **`spill stores /
    spill loads`**. Spills mean the kernel needed *more* live values than registers
    available, so the compiler parked some in **local memory** — which lives in DRAM.
    A spilling kernel pays a double tax: register pressure *and* extra global traffic on
    the hot path. The first occupancy fix is often "stop spilling," not "raise
    occupancy."
    """)
    return


@app.cell
def _():
    def _run():
        MAX_THREADS = 1536
        MAX_REGS = 65536
        WARP = 32
        WARPS_CAP = MAX_THREADS // WARP  # 48

        # Register allocation granularity on recent NVIDIA archs: 256 regs/warp.
        # (regs/thread are rounded up so warp totals land on this granularity.)
        GRAN = 256

        def warps_resident(regs_per_thread):
            regs_per_warp = ((regs_per_thread * WARP + GRAN - 1) // GRAN) * GRAN
            by_regs = MAX_REGS // regs_per_warp
            return min(by_regs, WARPS_CAP)

        print("=== Register pressure -> resident warps (5070 Ti, 65536 regs/SM) ===\n")
        print(f"  {'regs/thd':>8s} {'regs/warp':>10s} {'warps':>6s} {'occ':>6s}  note")
        print("  " + "-" * 56)
        for _r in [24, 32, 40, 42, 48, 56, 64, 80, 96, 128, 168, 255]:
            _w = warps_resident(_r)
            _note = ""
            if _r <= 40:
                _note = "full occupancy"
            elif _r >= 168:
                _note = "very low — needs strong ILP to win"
            _print = f"  {_r:>8d} {((_r*WARP+GRAN-1)//GRAN)*GRAN:>10d} " \
                     f"{_w:>6d} {_w/WARPS_CAP:>5.0%}  {_note}"
            print(_print)

        print("\n  Occupancy drops in STEPS, not smoothly — granularity rounding +")
        print("  whole-warp packing. 40 regs/thd is the last rung at 100%.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. The two levers: `__launch_bounds__` and `-maxrregcount`

    By default `ptxas` minimizes register count only loosely — it will happily spend 64+
    registers if that lets it keep more values live and avoid recomputation, *not
    knowing or caring* what that does to your occupancy. You have two ways to put a
    ceiling on it.

    **(a) `__launch_bounds__` — per-kernel, the preferred tool.** You annotate the
    kernel with the *maximum block size* you will launch it with, and (optionally) the
    *minimum number of blocks per SM* you want resident. The compiler then has enough
    information to cap registers so that target is achievable:

    ```cuda
    //                          maxThreadsPerBlock, minBlocksPerSM
    __global__ void __launch_bounds__(256, 6) my_kernel(const float* a, float* b) {
        // ptxas now budgets registers so that 6 blocks of 256 threads
        // (= 48 warps = full occupancy) can be co-resident on one SM.
        ...
    }
    ```

    Asking for 6 blocks * 256 threads = 1536 threads = 100% occupancy forces the
    compiler to fit each thread in `<= 40` registers — the continuous `65536 / 1536 ~=
    42.6` is an upper bound that granularity rounding tightens to 40 — spilling the rest
    if it must. This is *local* (only this kernel) and *intentful* (you state the goal,
    the compiler solves for registers).

    **(b) `-maxrregcount=N` — the blunt, whole-translation-unit hammer.**

    ```bash
    nvcc -arch=sm_120 -maxrregcount=40 my_kernel.cu
    ```

    Caps *every* kernel in the file at N registers. Coarser than `__launch_bounds__`
    (no per-kernel control, no block-count target) and easy to misuse — set it too low
    and you force spills everywhere. Reach for `__launch_bounds__` first; keep
    `-maxrregcount` for quick experiments and legacy code.

    > [CUDA C++ Programming Guide — Launch Bounds](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launch-bounds).
    > Both levers trade *register comfort per thread* for *more resident threads*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The catch: capping registers can cause spills

    Forcing the register count down does not delete the kernel's need for live values —
    it relocates the overflow to **local memory** (DRAM-backed). So both levers have a
    sweet spot, not a "lower is better" curve:

    - **Too high** a register count → few resident warps → poor latency hiding.
    - **Too low** a register count → spills to local memory → extra DRAM traffic on the
      hot path, often *worse* than the occupancy you bought.

    The right move is to find the register count where occupancy is high enough to hide
    latency *and* the kernel is not spilling — usually by reading `--ptxas-options=-v`
    while nudging `__launch_bounds__`. The next section shows why "high enough" is not
    the same as "100%."
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. The occupancy / ILP tradeoff (the Volkov result)

    Here is the counter-intuitive fact that this whole lecture builds to. **Occupancy is
    not the thing you actually need — *concurrency* is.** Occupancy (more resident warps)
    is just *one* way to buy concurrency. The other is **instruction-level parallelism
    (ILP)**: independent instructions *within a single thread* that the SM can keep in
    flight at once.

    Recall Little's Law from `0d`: to hide a latency $L$ at throughput $T$ you need

    $$\text{concurrency} = L \times T \text{ independent operations in flight.}$$

    That concurrency can come from **many warps each with one op outstanding** (high
    occupancy, low ILP) **or** from **few warps each with many independent ops
    outstanding** (low occupancy, high ILP). The product is what matters. So a kernel
    that does, say, 4 independent FMAs per thread before any dependency can saturate the
    math units at *one quarter* the occupancy of a kernel that does one.

    This is **Vasily Volkov's** result — *"Better Performance at Lower Occupancy"*
    (GTC 2010): hand-tuned GEMM and FFT kernels that hit peak at **well under 50%
    occupancy** by giving each thread more independent work (register blocking: each
    thread computes a small tile of outputs, reusing operands held in registers). Those
    kernels *want* the extra registers — capping them to chase occupancy would *slow them
    down*.

    The practical upshot: **do not blindly chase 100% occupancy.** When a kernel is
    register-blocked for ILP, the registers are doing useful work. Measure; the roofline
    and the profiler decide, not the occupancy percentage.

    > Volkov, *"Better Performance at Lower Occupancy"* (GTC 2010) — the classic
    > demonstration. Echoed in the [Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy):
    > "higher occupancy does not always equate to higher performance."
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Picturing it: occupancy vs registers, and where ILP rescues you

    The plot below is the central figure of this lecture. The **stepped curve** is
    achievable occupancy as a function of registers/thread on your 5070 Ti — the
    discrete drops are the granularity rounding from §1. Overlaid are two regimes:

    - **The naive view** (occupancy *is* performance): performance tracks the steps
      down — every extra register hurts.
    - **The Volkov view** (concurrency = occupancy x ILP): with enough ILP per thread,
      performance stays near peak *even as occupancy falls*, because each thread keeps
      more independent work in flight.

    The numbers are illustrative — the *shape* is the lesson: the gap between the two
    curves is the room ILP buys you.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        MAX_REGS = 65536
        WARP = 32
        WARPS_CAP = 48
        GRAN = 256  # regs/warp allocation granularity

        regs = np.arange(20, 256)

        def warps_for(r):
            rpw = int(np.ceil(r * WARP / GRAN)) * GRAN
            return min(MAX_REGS // rpw, WARPS_CAP)

        occ = np.array([warps_for(int(r)) / WARPS_CAP for r in regs])

        # Naive model: performance == occupancy (normalized).
        perf_naive = occ.copy()

        # ILP model: each extra register buys a little independent work per thread,
        # so achieved concurrency = occupancy * ilp_factor(regs), clipped at 1.
        # (illustrative: ILP rises with register blocking, saturating the units)
        ilp = 1.0 + 0.9 * (regs - 20) / (255 - 20)   # 1.0x -> ~1.9x ILP
        perf_ilp = np.clip(occ * ilp, 0, 1.0)

        _fig, _ax = plt.subplots(figsize=(8.4, 4.2))
        _ax.plot(regs, occ, color="#5b8def", linewidth=2.2,
                 label="occupancy (hardware limit)")
        _ax.plot(regs, perf_naive, color="#d65f5f", linewidth=1.6,
                 linestyle="--", label="perf if occupancy were everything")
        _ax.plot(regs, perf_ilp, color="#4c9f70", linewidth=2.2,
                 label="perf with ILP (concurrency = occ x ILP)")

        _ax.axvline(40, color="#999", linestyle=":", linewidth=1.2)
        _ax.text(41, 0.05, "40 regs/thd\n(last 100% rung)",
                 color="#666", fontsize=8)
        _ax.fill_between(regs, perf_naive, perf_ilp, where=(perf_ilp > perf_naive),
                         color="#4c9f70", alpha=0.12)
        _ax.text(150, 0.78, "room ILP buys", color="#2e6b48", fontsize=9)

        _ax.set_xlabel("registers / thread")
        _ax.set_ylabel("fraction of peak (occupancy / perf)")
        _ax.set_ylim(0, 1.05)
        _ax.set_title("Occupancy falls in steps; ILP can hold performance up")
        _ax.legend(loc="upper right", fontsize=8)
        _ax.grid(True, alpha=0.15)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Interactive occupancy calculator (your 5070 Ti)

    Three sliders — **registers/thread**, **shared memory/block**, **block size** — feed
    the same min-of-three-limiters model from `0d`, but now with the register
    *granularity rounding* that makes the steps real. The plot reports resident warps,
    occupancy, and — the part to watch — **which limiter binds**. Push registers up and
    watch the binder flip to "registers"; grab a big shared-memory tile and watch it flip
    to "shared mem." This is the calculator you reach for before touching
    `__launch_bounds__`.
    """)
    return


@app.cell
def _(mo):
    regs_slider = mo.ui.slider(start=16, stop=168, step=4, value=32,
                               label="registers / thread")
    smem_slider = mo.ui.slider(start=0, stop=100, step=4, value=8,
                               label="shared memory / block (KB)")
    tpb_slider = mo.ui.slider(start=32, stop=1024, step=32, value=256,
                              label="threads / block")
    mo.vstack([regs_slider, smem_slider, tpb_slider])
    return regs_slider, smem_slider, tpb_slider


@app.cell
def _(regs_slider, smem_slider, tpb_slider):
    def _run():
        import matplotlib.pyplot as plt

        MAX_THREADS = 1536
        MAX_REGS = 65536
        SMEM_KB = 100.0
        WARP = 32
        WARPS_CAP = 48
        GRAN = 256       # regs/warp granularity
        BLOCK_CAP = 32   # hardware-ish per-SM block cap

        regs = int(regs_slider.value)
        smem = float(smem_slider.value)
        tpb = int(tpb_slider.value)
        warps_per_block = tpb // WARP

        # register limiter, with granularity rounding per warp
        regs_per_warp = ((regs * WARP + GRAN - 1) // GRAN) * GRAN
        warps_by_regs = MAX_REGS // regs_per_warp
        blocks_by_regs = warps_by_regs // warps_per_block if warps_per_block else 0

        blocks_by_threads = MAX_THREADS // tpb
        blocks_by_smem = int(SMEM_KB // smem) if smem > 0 else BLOCK_CAP

        limits = {
            "threads": blocks_by_threads,
            "registers": blocks_by_regs,
            "shared mem": blocks_by_smem,
            "block cap": BLOCK_CAP,
        }
        blocks = max(0, min(limits.values()))
        binder = min(limits, key=limits.get)
        warps = blocks * warps_per_block
        occ = warps / WARPS_CAP

        names = list(limits.keys())
        vals = [limits[n] for n in names]
        colors = ["#d65f5f" if n == binder else "#cbd6ee" for n in names]

        _fig, (_ax1, _ax2) = plt.subplots(
            1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})

        _ax1.bar(names, vals, color=colors, edgecolor="none")
        _ax1.axhline(blocks, color="#4c9f70", linestyle="--", linewidth=1.5,
                     label=f"resident = {blocks} blocks")
        _ax1.set_ylabel("max blocks/SM by this rule")
        _ax1.set_title(f"binding limiter: {binder}")
        _ax1.legend(loc="upper right", fontsize=8)
        _ax1.tick_params(axis="x", labelrotation=12)

        _ax2.bar(["occupancy"], [occ], color="#5b8def", width=0.5)
        _ax2.axhline(1.0, color="#999", linestyle=":", linewidth=1)
        _ax2.set_ylim(0, 1.05)
        _ax2.set_title(f"{warps} warps -> {occ:.0%}")
        _ax2.set_ylabel("resident warps / 48")

        _fig.suptitle(
            f"block={tpb} thr ({warps_per_block} warps), {regs} regs/thd, "
            f"{smem:.0f} KB smem", y=1.03)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Reading the verdict in Nsight Compute (`ncu`)

    Build-time `--ptxas-options=-v` tells you the register count; **Nsight Compute**
    (`ncu`) tells you what that count *did* to a real run, and — more importantly — *why
    the warps stalled*. Profile a single kernel invocation:

    ```bash
    # full set, then inspect interactively in the ncu UI:
    ncu -o report ./my_program
    # or a focused, headless dump of the two sections that matter here:
    ncu --set basic \
        --section Occupancy \
        --section WarpStateStats \
        ./my_program
    ```

    Two things to read, in order:

    **(1) Occupancy section.** It reports **Achieved Occupancy** vs **Theoretical
    Occupancy**. *Theoretical* is the §4 calculator's answer — the limit your registers/
    smem/block-size impose. *Achieved* is what actually happened. A big gap (achieved
    much below theoretical) means tail effects or load imbalance — not enough blocks, or
    uneven block runtimes — *not* a register problem. The "Occupancy Limiter" subline
    names the binding resource, exactly the `binder` from the calculator above.

    **(2) Warp State Statistics — the stall reasons.** This is the real diagnosis. Each
    cycle a warp is either *issuing* or *stalled*, and `ncu` attributes the stalls to
    named reasons. The ones you will see:

    | Stall reason | Means | Typical fix |
    |---|---|---|
    | `Long Scoreboard` | waiting on a **global/local memory** load | more warps (occupancy), coalesce, prefetch, `cp.async` (`3f`) |
    | `Short Scoreboard` | waiting on **shared memory** / fixed-latency | reduce bank conflicts (`3d`), more ILP |
    | `MIO Throttle` / `LG Throttle` | memory **pipe saturated** | already bandwidth-bound — fewer/larger requests |
    | `Stall Wait` | waiting on a **fixed-latency math** dependency | more ILP / independent work |
    | `Barrier` | waiting at `__syncthreads()` | reduce sync, balance work across threads |

    The logic: if `Long Scoreboard` dominates **and** occupancy is low, raising
    occupancy will likely help (more warps to hide the load). If `Long Scoreboard`
    dominates **but** occupancy is already high, you are memory-bound — go to the
    roofline, not the register knob. If `Stall Wait` dominates at low occupancy, you
    need *ILP*, and *that* register-heavy kernel might be the Volkov case where you leave
    occupancy alone. The stall histogram, not the occupancy number, tells you which.

    > [Nsight Compute — Kernel Profiling Guide / Warp Scheduler States](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#statistical-sampler).
    > Read stalls *first*, occupancy *second* — the percentage alone never tells you the fix.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Worked reading

    Two kernels, same hardware, opposite prescriptions:

    - **Kernel A:** theoretical occupancy 100%, achieved 94%, dominant stall
      `Long Scoreboard` (70% of stalled cycles), achieving 60% of the bandwidth roof.
      → Already well-occupied and memory-bound. Occupancy is *not* the problem; chase
      **bandwidth** — coalescing, fewer bytes, or overlap the loads (`3f`). The register
      knob would do nothing.
    - **Kernel B:** theoretical occupancy 44% (limiter: **registers**, 96/thread),
      achieved 42%, dominant stall `Stall Wait` (math dependencies), but already at 85%
      of the *compute* roof. → This is the Volkov case. The 96 registers are buying ILP
      that is paying off. Forcing `__launch_bounds__` to raise occupancy would cause
      spills and *lose* performance. Leave it.

    Same "occupancy is low!" headline, but only one is a problem — and it is the one the
    **stall reasons** flagged, not the percentage.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    - **Watch the register line.** `--ptxas-options=-v` on every build. `Used N
      registers` and `spill stores` are the two numbers that predict occupancy and the
      hidden DRAM tax. A spilling kernel is leaking bandwidth on the hot path.
    - **State intent with `__launch_bounds__`.** When you *do* want occupancy, ask for
      it by block-size and blocks-per-SM and let the compiler solve for registers —
      don't hand-tune `-maxrregcount` blindly.
    - **Concurrency, not occupancy, is the target.** A register-blocked, low-occupancy
      kernel with strong ILP can sit at the compute roof. Don't "fix" it. The min-of-
      three calculator tells you the ceiling; the roofline and stall histogram tell you
      whether you're near it.
    - **Diagnose with stalls first.** In `ncu`, read Warp State Statistics before the
      occupancy number. The dominant stall reason chooses the fix — more warps, more
      ILP, fewer bytes, or fewer conflicts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Now go write it

    This lecture's exercise is **analysis, not a new kernel**: take a kernel you have
    already written (the tiled matmul `c02` is the ideal subject — it has real register
    and shared-memory pressure) and *profile and explain its occupancy*.

    ```bash
    # 1. build with resource reporting and read the register/spill line
    nvcc -arch=sm_120 --ptxas-options=-v -O3 exercises/c02_tiled_matmul/kernel.cu

    # 2. profile occupancy + stall reasons on a real launch
    ncu --set basic --section Occupancy --section WarpStateStats \
        python -m harness.runner c02

    # 3. write up: what is the binding limiter? the dominant stall? is raising
    #    occupancy the right fix, or is this an ILP / bandwidth case? predict the
    #    effect of __launch_bounds__(BLOCK, 2) and then test it.
    ```

    The deliverable is the *explanation* — binder, dominant stall, and the
    occupancy-vs-ILP-vs-bandwidth verdict — backed by the numbers. That reasoning is the
    skill; the register knob is just the lever you reach for once you've made the call.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [3D: Coalescing & Bank Conflicts](../3d_memory_banks/) &nbsp;|&nbsp; Next: [3F: Async Copy & Pipelining](../3f_async_pipelining/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
