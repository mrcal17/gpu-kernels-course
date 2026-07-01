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
    # 1B: Memory & Coalescing

    > *"The bus is 256 bits wide and runs at a fixed clock. Everything else is just how
    > much of each transaction you waste."*

    `0c` named the bottleneck — DRAM bandwidth, ~896 GB/s on your 5070 Ti — and `1a` gave
    you the tool to move data (`tl.load`/`tl.store` over a vector of addresses). This
    lecture connects them: **the access pattern your `offs` describe decides how much of
    that 896 GB/s you actually get.** For a memory-bound kernel (most of Part 1),
    bandwidth *is* the budget, and coalescing is how you spend it well.

    We'll do three things: see how a warp's 32 addresses fuse into hardware transactions
    — counted in the **32-byte sectors** that DRAM actually moves; quantify what
    contiguous vs. strided vs. misaligned access costs in sectors and therefore in
    effective bandwidth; and learn to compute **achieved
    GB/s** — bytes moved over time — so you can score your own kernel against the ceiling.
    That number is the metric for `e03` and most of the exercises that follow.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. Bandwidth is the budget

    For a memory-bound kernel, runtime is set almost entirely by how many bytes cross the
    DRAM bus and how fast the bus runs. The peak is fixed by physics:

    $$B_{\text{peak}} = \text{bus width} \times \text{memory clock}
      \;\approx\; 896\ \text{GB/s} \quad (\text{256-bit bus, GDDR7}).$$

    You cannot raise $B_{\text{peak}}$ — it's silicon. What you *can* control is the
    **fraction of each transaction you actually use**. The bus always moves data in
    fixed-size chunks; if your kernel uses every byte of every chunk, you ride the
    ceiling, and if it uses a quarter of each chunk, you get a quarter of the bandwidth.
    So the design target for a memory-bound kernel is simply:

    $$\boxed{\;\text{achieved BW} = B_{\text{peak}} \times (\text{bytes used} / \text{bytes fetched})\;}$$

    Everything in this lecture is about that efficiency factor. The lower bound on runtime
    for moving $M$ useful bytes is $M / B_{\text{peak}}$; coalescing is how you get close
    to it instead of paying $2\times$, $8\times$, or $32\times$ that.

    > [CUDA C++ Best Practices Guide — Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
    > treats effective bandwidth as the primary optimization target for exactly this
    > reason.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. A warp's 32 addresses → sectors

    Global memory is not byte-addressable in hardware. The memory system services a warp's
    loads in fixed-size aligned chunks, and two sizes matter:

    - The **L1 cache line** is **128 bytes** — exactly one warp's worth of float32
      ($32 \times 4$ B). It's a handy unit for *thinking about* a warp's footprint, and
      it's the "transaction" size older CUDA texts quote.
    - The chunk that actually crosses **L2/DRAM** — the one that costs you bandwidth — is
      the **32-byte sector**. A 128-byte line is 4 sectors, and the hardware only moves
      the sectors a warp actually touches.

    The key fact, stated in the unit that bills you:

    > The hardware looks at the 32 addresses a warp's lanes request, finds the set of
    > aligned **32-byte sectors** they fall in, and moves **one sector per distinct
    > sector touched.**

    So the cost of a warp's load is *the number of distinct 32-byte sectors its 32
    addresses span* — nothing to do with the order of the lanes, only which sectors get
    touched. For one warp of float32 (32 lanes wanting $32\times4 = 128$ useful bytes
    = exactly 4 sectors' worth):

    - **Coalesced** (32 contiguous, aligned floats): the 128 bytes fill exactly **4
      sectors**, every fetched byte used → **100%** efficiency. This is the ceiling.
    - **Strided by $s$**: consecutive lanes are $s$ elements apart, so each sector
      delivers 32 bytes of which the warp uses fewer and fewer — until at **stride 8**
      (32 bytes between lanes) every lane sits in its **own sector**: 32 sectors for 128
      useful bytes → $4/32 = 1/8$ efficiency. Larger strides can't do worse — a lane
      can't waste more than its whole sector — so the floor is **12.5%, hit already at
      stride 8**.
    - **Misaligned** (contiguous but the base address isn't sector-aligned): the warp's
      128 bytes straddle one extra boundary → **5 sectors instead of 4** → 80% for that
      warp in isolation (and §3 shows why a streaming read makes it milder still).

    The numpy cell counts exactly this — distinct sectors per warp — for several
    patterns. It is a simulation of the *addressing rule*, not a GPU measurement, but the
    sector counts are what the hardware would move. For comparison it also counts
    **128-byte lines** (the classic transaction model): note how the line model
    *over-bills* large strides (predicting 3% where the real floor is 12.5%) and
    misalignment (predicting 50% where the truth is 80%).
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        WARP = 32          # lanes per warp
        DTYPE = 4          # float32 bytes
        SECTOR = 32        # bytes per L2/DRAM sector -- the unit that costs bandwidth
        LINE = 128         # bytes per L1 cache line  -- the classic "transaction" unit

        def chunks(addrs_bytes, size):
            # distinct aligned `size`-byte chunks the warp's addresses fall in
            return np.unique(np.asarray(addrs_bytes) // size).size

        _lanes = np.arange(WARP)
        _patterns = [
            ("coalesced (stride 1, aligned)", _lanes * DTYPE),
            ("misaligned by 1 element",       _lanes * DTYPE + DTYPE),
            ("stride 2",                       _lanes * 2 * DTYPE),
            ("stride 8",                       _lanes * 8 * DTYPE),
            ("stride 32",                      _lanes * 32 * DTYPE),
        ]

        _used = WARP * DTYPE   # bytes the warp actually wants (128)
        print("=== One warp of float32: 32B sectors (ground truth) vs 128B lines ===\n")
        print(f"  {'pattern':30s} {'sectors':>7s} {'fetched B':>10s} {'eff':>5s}"
              f"  |  {'128B lines':>10s} {'line eff':>9s}")
        print("  " + "-" * 82)
        for _name, _addrs in _patterns:
            _ns = chunks(_addrs, SECTOR)
            _nl = chunks(_addrs, LINE)
            print(f"  {_name:30s} {_ns:>7d} {_ns * SECTOR:>10d} "
                  f"{_used / (_ns * SECTOR):>5.0%}  |  {_nl:>10d} "
                  f"{_used / (_nl * LINE):>8.0%}")
        print(f"\n  Useful bytes per warp = {_used} = 4 full sectors; 4 sectors/warp is")
        print("  the ceiling. Sectors are what DRAM bills: the 128B-line model over-")
        print("  charges big strides (3% vs the real 1/8 floor) and misalignment")
        print("  (50% vs the real 80%).")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Contiguous vs. strided vs. misaligned

    The three patterns above are the three things that happen to real kernels. Tie each to
    how you'd produce it in a Triton kernel (`1a`):

    - **Contiguous / coalesced.** `offs = pid*BLOCK + tl.arange(0, BLOCK)` — consecutive
      lanes, consecutive elements. Each warp's 128 bytes fill 4 fully-used sectors. This
      is the *default* you get from the standard 1-D pattern, and why that pattern is the
      standard. Efficiency ≈ 100%.
    - **Strided.** `offs = base + tl.arange(0, BLOCK) * s`, or implicitly when you walk a
      tensor along its non-contiguous axis (reading **columns** of a row-major matrix has
      stride = number of columns). Efficiency falls as $1/s$ — but only until **stride 8**
      (fp32), where each lane already occupies its own 32-byte sector and the damage
      saturates. This is still the classic transpose / column-reduction trap.
    - **Misaligned.** Contiguous data, but the starting address isn't sector-aligned
      (e.g. you sliced an array at an odd offset). One warp *in isolation* touches 5
      sectors instead of 4 → 80%. But in a **streaming** read the straddled sector isn't
      wasted — the *next* warp uses the rest of it. $N$ consecutive warps touch $4N + 1$
      sectors for $4N$ sectors' worth of useful data, so aggregate efficiency is
      $\approx N/(N+1)$: a **percent-level penalty**, independent of `BLOCK_SIZE`.

    The effective-bandwidth model from §1 turns each into a number. With element stride
    $s$ (fp32), a warp touches $\min(4s,\, 32)$ sectors for its 4 sectors' worth of
    useful bytes, so

    $$\text{achieved BW} \approx \frac{B_{\text{peak}}}{\min(s,\ 8)}
      \qquad\Rightarrow\qquad
      \text{floor} = \frac{896}{8} \approx 112\ \text{GB/s at stride} \ge 8.$$

    Misalignment adds one sector per warp — amortized to ~nothing by streaming. The
    takeaway is blunt, with honest numbers: **stride is the expensive mistake (up to 8×
    on fp32); misalignment is the cheap one (~1%)** — free to avoid by aligning
    allocations, and mostly harmless when you can't.

    > [CUDA C++ Best Practices — Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
    > works through aligned, misaligned, and strided cases with the same accounting
    > (quoted there in 128-byte-line units; on modern GPUs the L2 sector — what profiler
    > metrics like `sectors/request` count — is the unit that matches measurement).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sectors per warp as stride grows

    The plot sweeps stride and shows two curves: the **32-byte sectors a warp must
    fetch** (left, lower is better — 4 is ideal for fp32) and the resulting **effective
    bandwidth** (right, starting from the 896 GB/s ceiling). Watch how a handful of
    stride steps already halve, then quarter the usable bandwidth — and how the damage
    *saturates* at stride 8, where every lane owns a whole sector and the floor of
    $1/8$ of peak (~112 GB/s) is reached.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        WARP = 32
        DTYPE = 4
        SECTOR = 32
        PEAK = 896.0   # GB/s

        def sectors(stride):
            addrs = (np.arange(WARP) * stride) * DTYPE
            return np.unique(addrs // SECTOR).size

        strides = np.arange(1, 33)
        ns = np.array([sectors(int(s)) for s in strides])
        # efficiency = used bytes / fetched bytes = (32*4) / (sectors*32) = 4/sectors
        eff = (WARP * DTYPE) / (ns * SECTOR)
        bw = eff * PEAK

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4))

        _ax1.plot(strides, ns, color="#5b8def", linewidth=2, marker="o", markersize=3)
        _ax1.axhline(4, color="#4c9f70", linestyle="--", linewidth=1.3,
                     label="ideal = 4 sectors/warp")
        _ax1.axhline(32, color="#999", linestyle=":", linewidth=1.2,
                     label="max = 32 (1 lane per sector)")
        _ax1.set_xlabel("stride (elements between lanes)")
        _ax1.set_ylabel("32B sectors per warp")
        _ax1.set_title("Strided access fragments the warp")
        _ax1.legend(loc="lower right", fontsize=8)

        _ax2.plot(strides, bw, color="#d65f5f", linewidth=2, marker="o", markersize=3)
        _ax2.axhline(PEAK, color="#4c9f70", linestyle="--", linewidth=1.3,
                     label=f"peak {PEAK:.0f} GB/s")
        _ax2.axhline(PEAK / 8, color="#999", linestyle=":", linewidth=1.2,
                     label=f"floor = peak/8 ~ {PEAK / 8:.0f} GB/s")
        _ax2.set_xlabel("stride (elements between lanes)")
        _ax2.set_ylabel("effective bandwidth (GB/s)")
        _ax2.set_ylim(0, PEAK * 1.08)
        _ax2.set_title("...and craters bandwidth (until stride 8)")
        _ax2.legend(loc="upper right", fontsize=8)

        _fig.suptitle("One warp, float32: 32B sectors and bandwidth vs. stride", y=1.03)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Slide the stride, feel the ceiling drop

    Pick a stride and read off how far below 896 GB/s it leaves you. Stride 1 is glued to
    the ceiling; by stride 8 you're at an eighth of peak — and that's the **floor**: at
    stride 8 every lane already lands in its own 32-byte sector, the warp is fetching the
    maximum 32 sectors, and larger strides can't make it worse (~112 GB/s on this card,
    not the ~28 GB/s the 128-byte-line model would predict). This is the same simulation
    you'll see happen *for real* in `e03` when you stride a copy kernel on the GPU.
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
        SECTOR = 32
        PEAK = 896.0

        def stats(stride):
            addrs = (np.arange(WARP) * stride) * DTYPE
            ns = np.unique(addrs // SECTOR).size
            eff = (WARP * DTYPE) / (ns * SECTOR)
            return ns, eff

        strides = np.arange(1, 33)
        bws = np.array([stats(int(s))[1] for s in strides]) * PEAK

        s = int(stride_slider.value)
        s_ns, s_eff = stats(s)
        s_bw = s_eff * PEAK

        _fig, _ax = plt.subplots(figsize=(8.0, 3.6))
        _ax.bar(strides, bws, color="#cbd6ee", edgecolor="none", width=0.85)
        _ax.bar([s], [s_bw], color="#d65f5f", edgecolor="none", width=0.85)
        _ax.axhline(PEAK, color="#4c9f70", linestyle="--", linewidth=1.5,
                    label=f"peak {PEAK:.0f} GB/s")
        _ax.axhline(PEAK / 8, color="#999", linestyle=":", linewidth=1.3,
                    label=f"floor = peak/8 ~ {PEAK / 8:.0f} GB/s (stride >= 8)")
        _ax.set_xlabel("stride (elements)")
        _ax.set_ylabel("effective DRAM bandwidth (GB/s)")
        _ax.set_ylim(0, PEAK * 1.08)
        _ax.set_title(
            f"stride={s}:  {s_ns} sectors/warp  ->  ~{s_bw:.0f} GB/s  "
            f"({s_eff:.0%} of peak)"
        )
        _ax.legend(loc="upper right")
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Computing achieved GB/s

    To score a kernel you need a number you can compare to 896. It is just **bytes moved
    over wall-clock time**:

    $$\text{achieved GB/s} = \frac{\text{bytes read} + \text{bytes written}}{t}
      \;\Big/\; 10^9.$$

    The *bytes moved* is a property of the operation, not the implementation — count the
    minimum DRAM traffic the algorithm requires. For a few Part-1 kernels over $N$
    float32 elements:

    - **copy** ($c=a$): read $4N$ + write $4N$ = $8N$ bytes.
    - **vector add** ($c=a+b$): read $4N$ + read $4N$ + write $4N$ = $12N$ bytes.
    - **row reduce** (sum each row of an $R\times C$ matrix): read $4RC$ + write $4R$
      $\approx 4RC$ bytes (the output is negligible).

    The *time* $t$ comes from a benchmark. On the real GPU you'll use
    `triton.testing.do_bench`, which runs the kernel many times, discards warmup, and
    returns a robust median — never time a single launch (launch overhead and the clock's
    resolution dominate). Then:

    $$\text{efficiency} = \frac{\text{achieved GB/s}}{896}
      \quad(\text{how close to the memory roof you got}).$$

    A coalesced copy should land in the high 80–90%+ of peak; if yours reads 20%, the
    access pattern (stride or misalignment) is the first suspect. The cell below does the
    arithmetic for a copy kernel at a few hypothetical timings so the formula is concrete.

    > [`triton.testing.do_bench`](https://triton-lang.org/main/python-api/triton.testing.html)
    > is the standard timer; the [vector-add tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
    > reports GB/s exactly this way.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        PEAK = 896.0   # GB/s
        N = 64_000_000        # elements
        DTYPE = 4             # float32
        bytes_moved = 2 * DTYPE * N   # copy: read N + write N

        print("=== Achieved GB/s for a copy of 64M float32 (512 MB moved) ===")
        print(f"  bytes moved = 2 * 4 * N = {bytes_moved / 1e6:.0f} MB\n")
        print(f"  {'time (ms)':>10s} {'GB/s':>9s} {'% of peak':>10s}")
        print("  " + "-" * 32)
        for _t_ms in [0.60, 0.80, 1.20, 2.40, 4.80]:
            _t = _t_ms * 1e-3
            _gbps = (bytes_moved / _t) / 1e9
            print(f"  {_t_ms:>10.2f} {_gbps:>9.0f} {_gbps / PEAK:>9.0%}")
        print("\n  Same bytes; only the time changes. The fastest row (~0.6 ms) rides")
        print("  the coalesced ceiling; the ~4.8 ms row (~1/8 of peak) is the stride-8")
        print("  sector floor -- the worst a strided fp32 pattern can do.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    - **Bandwidth is the score.** For every memory-bound exercise (`e01`–`e05`, `e08`,
      `e09`, `e13`) the metric is GB/s vs. 896. Before optimizing, compute the bytes the
      operation *must* move — that fixes your ceiling and tells you what "done" looks like.
    - **Coalesce by construction.** Use the contiguous `pid*BLOCK + arange` indexing from
      `1a` so each warp's 128 bytes fill 4 fully-used 32-byte sectors. The moment your
      `offs` become strided (reading a column, an un-transposed axis), expect the
      bandwidth to fall like the slider shows — as $1/s$, down to the $1/8$-of-peak
      floor at stride 8.
    - **Mind alignment — but don't fear it.** A misaligned base costs one extra sector
      per warp, and streaming warps share the straddled sector, so the aggregate tax is
      ~1%. Keep hot-path allocations aligned because it's free to do, not because
      misalignment is catastrophic — stride is the mistake that costs 8×.
    - **Always `do_bench`.** Achieved GB/s is meaningless from a single timed launch.
      Median-of-many, warmup discarded — then divide bytes by time. That's the one number
      that tells you whether you're near the roof from `0d`.

    Coalescing is the bridge from the abstract roofline to a concrete kernel: it's *how*
    you move a memory-bound dot up to its bandwidth ceiling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Now go write it

    Time to measure bandwidth on your own card. Open the harness and write the
    copy/bandwidth kernel:

    ```bash
    python -m harness.runner e03 --watch
    ```

    `e03` copies an array — the simplest possible kernel, so the *only* thing being scored
    is how close your access pattern gets to 896 GB/s. Write the coalesced version and
    watch the GB/s climb toward the ceiling; then deliberately stride the indices and
    watch this lecture's slider play out on real silicon. That contrast — coalesced vs.
    strided, in numbers from your own GPU — is the whole point of the exercise.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [1A: The Triton Programming Model](../1a_triton_model/) &nbsp;|&nbsp; Next: [1C: Reductions](../1c_reductions/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
