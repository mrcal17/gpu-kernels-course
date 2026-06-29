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
    # 0C: Memory Hierarchy

    > *"Flops are free; bytes are expensive."* — the modern hardware aphorism

    The execution model (`0b`) told you the GPU hides latency by oversubscribing the
    SMs with parallel work. This lecture tells you *what* it is hiding latency **to** —
    the staircase of memories between a thread's registers and the 16 GB of DRAM on the
    far side of the bus. Every memory you touch costs a different number of cycles and
    delivers a different number of bytes per second, and almost every fast kernel is a
    story about keeping data in the cheap, fast levels for as long as possible.

    The single fact that organizes the whole course: **arithmetic got cheap, moving
    data did not.** A modern GPU can do *far* more math per second than it can read
    operands from DRAM. So the first question you ask of any kernel is not "how many
    FLOPs?" but "how many **bytes** does this move, and from how far away?"
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. The staircase: capacity vs. speed

    Memory on a GPU is a hierarchy of trade-offs. Each step down is **bigger but
    slower and shared by more threads**. From the closest to the farthest:

    - **Registers** — private to one thread, ~1 cycle, the fastest storage there is.
      Your 5070 Ti has **65,536 registers per SM** (32-bit each). They are a *budget*:
      every live variable in your kernel costs registers, and using too many per
      thread caps how many warps can be resident (the register limiter in `0d`).
    - **Shared memory / L1** — on-chip SRAM, per **block**, tens of cycles. The 5070 Ti
      has **100 KB per SM** (48 KB per block by default, opt-in up to ~99 KB), carved
      from the same physical SRAM as L1. This is the *programmable* cache — you stage
      data here by hand to get reuse. Tiling (`1e`, `3b`) lives entirely here.
    - **L2 cache** — one **48 MB** cache shared by all 70 SMs, a couple hundred cycles.
      Automatic; you don't manage it, but data that fits in 48 MB and is reused is
      effectively free on the second touch.
    - **Global memory / DRAM (GDDR7)** — the 16 GB the whole device sees, hundreds of
      cycles away, fed through a **256-bit bus at ~896 GB/s**. This is the floor of the
      staircase and the bottleneck of most kernels.

    The pattern: as capacity grows from kilobytes to gigabytes, latency grows from
    ~1 cycle to ~500, and the storage goes from private to globally shared.

    > [PMPP](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)
    > Ch. 5 ("Memory architecture and data locality") is the canonical treatment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The numbers, on one card

    These are your 5070 Ti's actual levels (queried via `python -m harness.device_info`,
    latencies are order-of-magnitude typical values). Notice capacity and speed pull in
    opposite directions — that tension *is* the hierarchy.
    """)
    return


@app.cell
def _():
    def _run():
        # RTX 5070 Ti (sm_120). Capacities are exact; latencies are
        # order-of-magnitude typical GPU values (cycles).
        rows = [
            ("Registers",        "256 KB/SM (65,536 x 32-bit)", "~1",   "per thread"),
            ("Shared mem / L1",  "100 KB/SM (48 KB/block def)",  "~30",  "per block"),
            ("L2 cache",         "48 MB (whole device)",         "~200", "all 70 SMs"),
            ("Global / DRAM",    "16 GB GDDR7, 256-bit",         "~500", "all threads"),
        ]
        print("=== RTX 5070 Ti memory staircase ===")
        print(f"  {'level':18s} {'capacity':30s} {'cycles':>7s}  scope")
        print("  " + "-" * 70)
        for _lvl, _cap, _lat, _scope in rows:
            print(f"  {_lvl:18s} {_cap:30s} {_lat:>7s}  {_scope}")
        print("\n  Bandwidth: DRAM ~896 GB/s; L2 several TB/s; shared/regs ~TB/s/SM.")
        print("  Down the staircase: bigger, slower, shared by more threads.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Capacity vs. bandwidth, to scale

    It is hard to feel "kilobytes vs. gigabytes" and "~1 cycle vs. ~500 cycles" until
    you see them on a log axis. The chart below puts the four levels on the same
    picture: **capacity** (how much fits) against **relative bandwidth** (how fast it
    streams). The on-chip levels are tiny but enormously fast; DRAM is vast but slow.

    The design lesson is right there in the gap: if you can shrink your working set so
    it lives in shared memory or L2 and reuse it many times, you trade a slow level for
    a fast one. That trade — **reuse on-chip instead of re-reading DRAM** — is the
    engine behind tiling, fusion, and flash-attention.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        levels = ["Registers", "Shared/L1", "L2", "DRAM"]
        # Capacity in KB (whole-device for L2/DRAM, per-SM for regs/shared).
        capacity_kb = np.array([256, 100, 48 * 1024, 16 * 1024 * 1024], dtype=float)
        # Relative bandwidth (DRAM = 1x); on-chip levels are vastly higher.
        rel_bw = np.array([8000.0, 4000.0, 20.0, 1.0])
        colors = ["#4c9f70", "#5b8def", "#e0a458", "#d65f5f"]

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))

        _ax1.barh(levels, capacity_kb, color=colors)
        _ax1.set_xscale("log")
        _ax1.set_xlabel("capacity (KB, log scale)")
        _ax1.set_title("Capacity: tiny on-chip, vast off-chip")
        _ax1.invert_yaxis()

        _ax2.barh(levels, rel_bw, color=colors)
        _ax2.set_xscale("log")
        _ax2.set_xlabel("relative bandwidth (DRAM = 1x, log scale)")
        _ax2.set_title("Bandwidth: on-chip is orders faster")
        _ax2.invert_yaxis()

        _fig.suptitle("The same four levels, two opposing axes", y=1.02)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. The memory wall

    Why obsess over bytes? Because compute and bandwidth have grown apart for decades.
    A GPU's peak math rate vastly exceeds the rate at which DRAM can supply operands,
    so most simple kernels finish their math long before the data arrives and spend
    their life **waiting on memory**. This is the *memory wall*.

    Make it concrete. Consider a vector add, $c_i = a_i + b_i$, over $N$ float32
    elements. Per element it does **1 FLOP** but moves **12 bytes** (read $a$, read $b$,
    write $c$ — 4 bytes each). The lower bound on its runtime is set by bandwidth:

    $$t_{\min} = \frac{\text{bytes moved}}{\text{bandwidth}}
      = \frac{12N}{896\,\text{GB/s}}.$$

    No matter how clever the arithmetic, you cannot beat the time it takes to stream
    $12N$ bytes through the bus. The achievable ceiling is therefore an **effective
    bandwidth** number, not a FLOP number — which is exactly why exercises like
    `e01`/`e03` are scored in GB/s against the 896 GB/s line, not in FLOP/s.

    > [CUDA C++ Best Practices Guide — Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
    > makes bandwidth the first-class optimization target for exactly this reason.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        BW = 896e9  # bytes/s, 5070 Ti peak DRAM bandwidth

        print("=== Vector add: bandwidth sets the floor ===")
        print("  c[i] = a[i] + b[i]   ->   1 FLOP, 12 bytes per element\n")
        print(f"  {'N elements':>14s} {'bytes moved':>14s} {'t_min (us)':>12s} {'GFLOP/s':>10s}")
        print("  " + "-" * 54)
        for _N in [1_000, 1_000_000, 100_000_000]:
            _bytes = 12 * _N
            _t = _bytes / BW          # seconds, bandwidth-bound floor
            _gflops = (_N / _t) / 1e9  # FLOP/s achieved AT that floor
            print(f"  {_N:>14,} {_bytes:>14,} {_t * 1e6:>12.2f} {_gflops:>10.1f}")

        print("\n  Even at the bandwidth floor, the FLOP rate is laughably low:")
        print("  the chip can do TFLOP/s, but only ~75 GFLOP/s of *useful* add work")
        print("  fits behind 896 GB/s. The arithmetic is free; the bytes are the wall.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Operational intensity — the one ratio to internalize

    The quantity that decides whether a kernel hits the memory wall is its
    **operational** (or **arithmetic**) **intensity**: useful work per byte moved from
    DRAM.

    $$I = \frac{\text{FLOPs}}{\text{bytes moved from DRAM}}
      \qquad \left[\frac{\text{FLOP}}{\text{byte}}\right]$$

    Low intensity means you do little math per byte, so DRAM bandwidth caps you —
    **memory-bound**. High intensity means you reuse each loaded byte for many FLOPs,
    so the math units cap you — **compute-bound**. Two anchors:

    - **Vector add:** $I = \dfrac{1\ \text{FLOP}}{12\ \text{bytes}} \approx 0.08$ —
      deeply memory-bound. Bandwidth is destiny.
    - **Square matmul** $C = AB$ for $n \times n$: it does $2n^3$ FLOPs but, *if you
      tile so each input is read from DRAM only once*, moves only $\sim 3 n^2 \cdot 4$
      bytes. So $I \sim \dfrac{2n^3}{12 n^2} = \dfrac{n}{6}$ — it **grows with $n$**.
      Big matmuls become compute-bound, which is the whole reason tiling pays off.

    Intensity is the $x$-axis of the **roofline model**, which `0d` builds in full. For
    now just hold the reflex: *compute the FLOP-per-byte ratio first.* It predicts the
    ceiling before you write a line of kernel code.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        print("=== Operational intensity I = FLOPs / bytes-from-DRAM ===\n")
        # Vector add: 1 FLOP per 12 bytes.
        _I_vadd = 1.0 / 12.0
        print(f"  vector add        I = 1 / 12         = {_I_vadd:6.3f} FLOP/byte  (memory-bound)")

        # Tiled square matmul: 2 n^3 FLOPs, ~3 n^2 * 4 bytes (each input read once).
        print(f"  {'tiled matmul n':>16s} {'FLOPs':>14s} {'bytes':>14s} {'I':>10s}")
        for _n in [256, 1024, 4096]:
            _flops = 2 * _n ** 3
            _bytes = 3 * _n ** 2 * 4
            _I = _flops / _bytes
            print(f"  {_n:>16d} {_flops:>14.2e} {_bytes:>14.2e} {_I:>10.1f}")
        print("\n  I for matmul ~ n/6: it climbs with size, crossing from")
        print("  memory-bound (small n) to compute-bound (large n).")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Coalescing: bandwidth is only *peak* if you ask nicely

    The 896 GB/s figure assumes the warp asks for memory the way the hardware likes.
    DRAM is not byte-addressable in practice — the memory system serves data in
    aligned **transactions** (think 32- or 128-byte segments). When the 32 threads of a
    warp read **32 consecutive, aligned** addresses, the hardware fuses them into one
    (or a few) transactions: this is **coalescing**, and it delivers near-peak
    bandwidth.

    When the warp's threads read **strided** or **scattered** addresses, each thread
    pulls down a separate segment, most of which is thrown away. With stride $s$ you
    fetch up to $s\times$ the bytes you use, so the *effective* bandwidth drops by
    roughly $1/s$:

    $$\text{effective BW} \approx \frac{\text{peak BW}}{s}
      \quad (\text{ideal coalesced } s = 1).$$

    The numpy simulation below doesn't measure a GPU — it *counts the 128-byte segments
    a warp would touch* for a given stride, and reports the fraction of fetched bytes
    you actually use. That efficiency is the coalescing penalty, made visible.

    > [CUDA C++ Best Practices Guide — Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory).
    > You will *measure* this on the real card in `1b` and exercise `e03`.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        WARP = 32          # threads per warp
        DTYPE = 4          # float32 bytes
        SEG = 128          # bytes per memory transaction (segment)

        def used_and_fetched(stride):
            # Addresses (in elements) the 32 lanes of a warp touch for this stride.
            lanes = np.arange(WARP) * stride
            byte_addrs = lanes * DTYPE
            used_bytes = WARP * DTYPE                       # what the warp actually wants
            segments = np.unique(byte_addrs // SEG)         # distinct 128B segments hit
            fetched_bytes = segments.size * SEG             # what DRAM must deliver
            return used_bytes, fetched_bytes

        print("=== Coalescing: bytes used vs. bytes fetched (one warp) ===")
        print(f"  warp={WARP} lanes, float32, {SEG}-byte segments\n")
        print(f"  {'stride':>7s} {'segments':>9s} {'used B':>8s} {'fetched B':>10s} {'efficiency':>11s}")
        print("  " + "-" * 50)
        for _s in [1, 2, 4, 8, 16, 32]:
            _used, _fetched = used_and_fetched(_s)
            _eff = _used / _fetched
            _tag = "  <- coalesced" if _s == 1 else ""
            print(f"  {_s:>7d} {_fetched // SEG:>9d} {_used:>8d} {_fetched:>10d} {_eff:>10.0%}{_tag}")
        print("\n  stride 1: one tidy burst, ~100% of fetched bytes used.")
        print("  big stride: many segments, most bytes wasted -> bandwidth collapses.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### See the effective bandwidth fall off

    Drag the stride. The bar shows the effective DRAM bandwidth a strided access
    pattern leaves on the table, starting from the 896 GB/s peak. Coalesced ($s=1$)
    sits at the ceiling; every step in stride scatters the warp across more segments
    and the usable bandwidth craters.
    """)
    return


@app.cell
def _(mo):
    stride_slider = mo.ui.slider(start=1, stop=32, step=1, value=1,
                                 label="access stride (elements between consecutive lanes)")
    stride_slider
    return (stride_slider,)


@app.cell
def _(stride_slider):
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        WARP = 32
        DTYPE = 4
        SEG = 128
        PEAK_BW = 896.0  # GB/s

        def efficiency(stride):
            byte_addrs = (np.arange(WARP) * stride) * DTYPE
            used = WARP * DTYPE
            fetched = np.unique(byte_addrs // SEG).size * SEG
            return used / fetched

        strides = np.arange(1, 33)
        effs = np.array([efficiency(int(s)) for s in strides])
        bws = effs * PEAK_BW

        s = int(stride_slider.value)
        s_bw = efficiency(s) * PEAK_BW

        _fig, _ax = plt.subplots(figsize=(7.8, 3.6))
        _ax.bar(strides, bws, color="#cbd6ee", edgecolor="none", width=0.85)
        _ax.bar([s], [s_bw], color="#d65f5f", edgecolor="none", width=0.85)
        _ax.axhline(PEAK_BW, color="#4c9f70", linestyle="--", linewidth=1.5,
                    label=f"peak {PEAK_BW:.0f} GB/s")
        _ax.set_xlabel("stride (elements)")
        _ax.set_ylabel("effective DRAM bandwidth (GB/s)")
        _ax.set_ylim(0, PEAK_BW * 1.08)
        _ax.set_title(f"stride={s}: ~{s_bw:.0f} GB/s  ({efficiency(s):.0%} of peak)")
        _ax.legend(loc="upper right")
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    - **Count bytes before FLOPs.** Most kernels you write in Parts 1–2 are
      memory-bound; their score is GB/s against the 896 ceiling. Knowing the byte
      traffic tells you the best you can do *before* you tune anything.
    - **Compute operational intensity.** FLOP-per-byte predicts which ceiling you'll
      hit. Low $I$ → fight for bandwidth (coalesce, fuse). High $I$ → fight for the
      math units (tiling, tensor cores).
    - **Coalesce every global access.** Lay out data so a warp's 32 lanes read 32
      contiguous, aligned elements. This is the memory-side twin of the "think in
      warps of 32" rule from `0b`, and it is the difference between 896 GB/s and a
      fraction of it.
    - **Reuse on-chip.** When a byte is expensive to fetch, use it many times before
      letting it fall out of registers / shared memory. That instinct becomes tiling
      (`1e`, `3b`) and fusion.

    The memory hierarchy is the budget; the next lecture (`0d`) turns it into a
    *ceiling* you can plot and aim at — the roofline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Now go write it

    The payoff for this lecture is *measuring* bandwidth on your own card. After you've
    worked through the Triton model (`1a`–`1b`), open the harness and write the
    copy/bandwidth kernel:

    ```bash
    python -m harness.runner e03 --watch
    ```

    `e03` does the simplest possible thing — copy an array — so that the *only* thing
    being measured is how close your memory access pattern gets to 896 GB/s. Coalesce
    it and you'll see the ceiling; stride it and you'll watch this lecture's slider
    happen for real.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [0B: The Execution Model](../0b_execution_model/) &nbsp;|&nbsp; Next: [0D: Occupancy &amp; the Roofline](../0d_occupancy_and_roofline/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
