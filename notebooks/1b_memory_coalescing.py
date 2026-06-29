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

    We'll do three things: see how a warp's 32 addresses fuse into hardware
    **transactions**; quantify what contiguous vs. strided vs. misaligned access costs in
    transactions and therefore in effective bandwidth; and learn to compute **achieved
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

    ## 2. A warp's 32 addresses → transactions

    Global memory is not byte-addressable in hardware. The memory system services a warp's
    loads in **aligned transactions** — natural sizes are **32, 64, and 128 bytes**, and a
    128-byte transaction is the common unit (it matches the L1/global cache line and a full
    warp's worth of float32; L2/DRAM is accessed in finer 32-byte sectors). The key fact:

    > The hardware looks at the 32 addresses a warp's lanes request, finds the set of
    > aligned 128-byte segments they fall in, and issues **one transaction per distinct
    > segment touched.**

    So the cost of a warp's load, in transactions, is *the number of distinct 128-byte
    segments its 32 addresses span* — nothing to do with the order of the lanes, only
    which segments get touched. The best case and worst case for one warp of float32
    (4 bytes each, so 32 lanes want exactly $32\times4 = 128$ bytes = one segment's worth):

    - **Coalesced** (32 contiguous, aligned floats): all 32 addresses land in **one**
      128-byte segment → **1 transaction**, 100% used. This is the ceiling.
    - **Strided by $s$**: consecutive lanes are $s$ elements apart, scattering them across
      **up to 32 different segments** → up to **32 transactions** for the same 128 useful
      bytes → as little as $1/32 \approx 3\%$ efficiency.
    - **Misaligned** (contiguous but the base address isn't a multiple of 128): the 32
      floats straddle a segment boundary → **2 transactions** instead of 1 → ~50% on that
      warp.

    The numpy cell counts exactly this — distinct segments per warp — for several
    patterns. It is a simulation of the *addressing rule*, not a GPU measurement, but the
    transaction counts are what the hardware would issue.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        WARP = 32          # lanes per warp
        DTYPE = 4          # float32 bytes
        SEG = 128          # bytes per aligned transaction (segment)

        def transactions(addrs_bytes):
            # distinct aligned 128B segments the warp's addresses fall in
            segs = np.unique(np.asarray(addrs_bytes) // SEG)
            used = WARP * DTYPE                 # bytes the warp actually wants
            fetched = segs.size * SEG           # bytes DRAM must deliver
            return segs.size, used, fetched

        _lanes = np.arange(WARP)
        _patterns = [
            ("coalesced (stride 1, aligned)", _lanes * DTYPE),
            ("misaligned by 8 bytes",         _lanes * DTYPE + 8),
            ("stride 2",                       _lanes * 2 * DTYPE),
            ("stride 8",                       _lanes * 8 * DTYPE),
            ("stride 32",                      _lanes * 32 * DTYPE),
        ]

        print("=== Transactions for one warp (float32, 128B segments) ===\n")
        print(f"  {'pattern':32s} {'txns':>5s} {'used B':>7s} {'fetched B':>10s} {'eff':>6s}")
        print("  " + "-" * 64)
        for _name, _addrs in _patterns:
            _t, _u, _f = transactions(_addrs)
            print(f"  {_name:32s} {_t:>5d} {_u:>7d} {_f:>10d} {_u / _f:>5.0%}")
        print("\n  1 transaction = the whole warp in one burst (the ceiling).")
        print("  Each extra segment is bandwidth fetched and thrown away.")

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
      lanes, consecutive elements. Each warp reads one aligned run. This is the *default*
      you get from the standard 1-D pattern, and why that pattern is the standard.
      Efficiency ≈ 100%.
    - **Strided.** `offs = base + tl.arange(0, BLOCK) * s`, or implicitly when you walk a
      tensor along its non-contiguous axis (reading **columns** of a row-major matrix has
      stride = number of columns). Efficiency falls roughly as $1/s$ until the warp is
      spread over 32 separate segments. This is the classic transpose / column-reduction
      trap.
    - **Misaligned.** Contiguous data, but the starting address isn't a multiple of 128
      bytes (e.g. you sliced an array at an odd offset). Each warp straddles one extra
      segment boundary → roughly a constant $\sim$2× transaction count, so ~50–90%
      depending on `BLOCK_SIZE`. Less catastrophic than striding, but free to avoid by
      aligning allocations.

    The effective-bandwidth model from §1 turns each into a number. With stride $s$, in
    the worst case a warp fetches $s\times$ the bytes it uses (capped at 32 segments), so

    $$\text{achieved BW} \approx \frac{B_{\text{peak}}}{\min(s,\ \text{segments per warp})}.$$

    Misalignment adds roughly one segment per warp regardless of stride. The takeaway is
    blunt: **stride is the expensive mistake; alignment is the cheap one.**

    > [CUDA C++ Best Practices — Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
    > works through aligned, misaligned, and strided cases with the same transaction
    > accounting.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Transactions per warp as stride grows

    The plot sweeps stride and shows two curves: the **transactions a warp must issue**
    (left, lower is better — 1 is ideal) and the resulting **effective bandwidth** (right,
    starting from the 896 GB/s ceiling). Watch how a handful of stride steps already
    halve, quarter, then decimate the usable bandwidth — long before stride reaches the
    32-segment floor.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        WARP = 32
        DTYPE = 4
        SEG = 128
        PEAK = 896.0   # GB/s

        def txns(stride):
            addrs = (np.arange(WARP) * stride) * DTYPE
            return np.unique(addrs // SEG).size

        strides = np.arange(1, 33)
        nt = np.array([txns(int(s)) for s in strides])
        # efficiency = used bytes / fetched bytes = (32*4) / (txns*128) = 1/txns
        eff = (WARP * DTYPE) / (nt * SEG)
        bw = eff * PEAK

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4))

        _ax1.plot(strides, nt, color="#5b8def", linewidth=2, marker="o", markersize=3)
        _ax1.axhline(1, color="#4c9f70", linestyle="--", linewidth=1.3,
                     label="ideal = 1 txn/warp")
        _ax1.set_xlabel("stride (elements between lanes)")
        _ax1.set_ylabel("transactions per warp")
        _ax1.set_title("Strided access fragments the warp")
        _ax1.legend(loc="lower right", fontsize=8)

        _ax2.plot(strides, bw, color="#d65f5f", linewidth=2, marker="o", markersize=3)
        _ax2.axhline(PEAK, color="#4c9f70", linestyle="--", linewidth=1.3,
                     label=f"peak {PEAK:.0f} GB/s")
        _ax2.set_xlabel("stride (elements between lanes)")
        _ax2.set_ylabel("effective bandwidth (GB/s)")
        _ax2.set_ylim(0, PEAK * 1.08)
        _ax2.set_title("...and so craters the bandwidth")
        _ax2.legend(loc="upper right", fontsize=8)

        _fig.suptitle("One warp, float32: transactions and bandwidth vs. stride", y=1.03)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Slide the stride, feel the ceiling drop

    Pick a stride and read off how far below 896 GB/s it leaves you. Stride 1 is glued to
    the ceiling; by stride 8 you're at an eighth of peak; past the point where every lane
    lands in its own 128-byte segment, the warp is issuing the maximum 32 transactions and
    bandwidth bottoms out. This is the same simulation you'll see happen *for real* in
    `e03` when you stride a copy kernel on the GPU.
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
        PEAK = 896.0

        def stats(stride):
            addrs = (np.arange(WARP) * stride) * DTYPE
            nt = np.unique(addrs // SEG).size
            eff = (WARP * DTYPE) / (nt * SEG)
            return nt, eff

        strides = np.arange(1, 33)
        bws = np.array([stats(int(s))[1] for s in strides]) * PEAK

        s = int(stride_slider.value)
        s_nt, s_eff = stats(s)
        s_bw = s_eff * PEAK

        _fig, _ax = plt.subplots(figsize=(8.0, 3.6))
        _ax.bar(strides, bws, color="#cbd6ee", edgecolor="none", width=0.85)
        _ax.bar([s], [s_bw], color="#d65f5f", edgecolor="none", width=0.85)
        _ax.axhline(PEAK, color="#4c9f70", linestyle="--", linewidth=1.5,
                    label=f"peak {PEAK:.0f} GB/s")
        _ax.set_xlabel("stride (elements)")
        _ax.set_ylabel("effective DRAM bandwidth (GB/s)")
        _ax.set_ylim(0, PEAK * 1.08)
        _ax.set_title(
            f"stride={s}:  {s_nt} txns/warp  ->  ~{s_bw:.0f} GB/s  "
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
        for _t_ms in [0.60, 0.80, 1.20, 2.40, 8.00]:
            _t = _t_ms * 1e-3
            _gbps = (bytes_moved / _t) / 1e9
            print(f"  {_t_ms:>10.2f} {_gbps:>9.0f} {_gbps / PEAK:>9.0%}")
        print("\n  Same bytes; only the time changes. The fastest row (~0.6 ms) rides")
        print("  the coalesced ceiling; the 8 ms row is what a badly strided")
        print("  (worse than stride-8) pattern costs.")

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
      `1a` so each warp reads one aligned segment. The moment your `offs` become strided
      (reading a column, an un-transposed axis), expect the bandwidth to fall like the
      slider shows.
    - **Mind alignment.** Don't slice tensors at odd offsets on the hot path; a misaligned
      base costs a near-constant extra transaction per warp. It's the cheapest penalty to
      avoid.
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
