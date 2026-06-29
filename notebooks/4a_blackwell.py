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
    # 4A: Blackwell (sm_120)

    > *"Every generation moves the roof. Your job is to know which roof just moved,
    > and by how much — before you trust a benchmark."*

    You've spent three parts learning the patterns. They all rest on a model of the
    hardware: an SM with warps, shared memory, tensor cores, and a memory hierarchy
    you feed by hand. That model is *generation-relative*. The card under this course
    is an **RTX 5070 Ti** — **Blackwell**, compute capability **`sm_120`** — and a few
    of its primitives are genuinely new versus the Ada (`sm_89`) and Hopper (`sm_90`)
    generations you'll read about in tutorials and papers.

    This lecture is a **map, not a kernel**. We'll mark what actually changed —
    5th-generation tensor cores, narrow floating-point (FP8 and FP4 with *microscaling*),
    the **Tensor Memory Accelerator (TMA)** for bulk async copies, and thread-block
    **clusters** with distributed shared memory — and, just as important, we'll be honest
    about which of those a *consumer* `sm_120` part actually exposes versus which live on
    the data-center `sm_90`/`sm_100` line. When a capability is version- or SKU-dependent,
    the right move is not to guess: **check the arch docs for your toolkit**. We'll flag
    every such spot.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. Where sm_120 sits

    NVIDIA's recent consumer/data-center lineage, by compute capability:

    | Gen | Arch | Example CC | Example part |
    |---|---|---|---|
    | Ampere | GA10x | `sm_86` | RTX 3090 |
    | Ada Lovelace | AD10x | `sm_89` | RTX 4090 |
    | Hopper | GH100 | `sm_90` | H100 (data-center) |
    | Blackwell (DC) | GB100/GB200 | `sm_100` | B100/B200 (data-center) |
    | **Blackwell (consumer)** | **GB20x** | **`sm_120`** | **RTX 50-series (your 5070 Ti)** |

    Two cautions that this whole lecture turns on:

    1. **"Blackwell" is two families.** The data-center Blackwell (`sm_100`, B200) and
       the consumer Blackwell (`sm_120`, GeForce 50-series) share branding and a lot of
       DNA, but **not every feature is on both**. Papers and CUTLASS examples that say
       "Blackwell" often mean `sm_100`. Don't assume a `sm_100` feature is on your
       `sm_120` card without checking.
    2. **Compute capability gates features at compile time.** `nvcc -arch=sm_120` (or
       Triton's target for your device) decides which PTX instructions are even legal.
       The CUDA 13 toolchain is what actually knows your card's menu — query it, don't
       recall it.

    > [CUDA C++ Programming Guide — Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
    > is the authoritative per-CC feature table. The
    > [NVIDIA Blackwell architecture whitepaper](https://resources.nvidia.com/en-us-blackwell-architecture)
    > is the architectural overview. Read both against *your* toolkit version (CUDA 13.1
    > here) — feature availability is version-dependent.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. 5th-generation tensor cores

    Tensor cores are fixed-function matrix-multiply-accumulate units (you met them in
    `3g`): each issues a small `D = A·B + C` on tiles, far denser than the FP32 FMA
    pipes. Blackwell carries the **5th-generation** tensor core, and the headline change
    is **how narrow the operand types can go** while staying *useful* for ML.

    The lineage of what tensor cores natively accelerate:

    - **Ampere (3rd gen):** FP16 / BF16 / TF32, plus INT8/INT4 for inference.
    - **Ada / Hopper (4th gen):** add **FP8** (E4M3 / E5M2); Hopper additionally adds the
      *warpgroup* `wgmma` instruction — a much larger MMA driven by a whole warpgroup.
    - **Blackwell (5th gen):** adds **FP4 / FP6** narrow floats and **microscaling
      (MX) formats** (next section), roughly *doubling* low-precision throughput per SM
      versus the prior gen on the data-center part.

    The pattern across generations is consistent: **the cheapest way to more FLOP/s is
    fewer bits per operand.** Halving the operand width roughly doubles the matrix
    throughput, because you push twice as many elements through the same datapath each
    cycle — at the cost of dynamic range and precision, which the scaling tricks below
    exist to claw back.

    A caution specific to your card: the *exact* peak tensor throughput and *which*
    narrow formats are exposed at full rate on **consumer `sm_120`** differ from the
    data-center `sm_100` headline numbers. Treat the bars below as **illustrative of the
    trend**, not as your card's datasheet. Measure with `triton.testing.do_bench` on the
    real device (Part 2) before quoting a number.

    > [NVIDIA Blackwell architecture whitepaper](https://resources.nvidia.com/en-us-blackwell-architecture)
    > §"Fifth-Generation Tensor Cores" covers the MMA generations and the narrow-float
    > additions.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        # ILLUSTRATIVE relative tensor-core throughput by generation, normalized so
        # Ampere-FP16 = 1. These encode the *trend* (narrower type ~ more throughput,
        # newer gen ~ more per SM), NOT a datasheet for any specific SKU. Measure your
        # own card with do_bench before quoting absolute TFLOP/s.
        gens = ["Ampere\n(sm_86)", "Ada\n(sm_89)", "Hopper\n(sm_90)", "Blackwell\n(sm_120)"]

        # rows: dtype tiers; columns: generations. NaN = not natively accelerated.
        fp16 = [1.0, 1.3, 2.0, 2.6]
        fp8 = [np.nan, 2.6, 4.0, 5.2]      # FP8 lands at Ada/Hopper
        fp4 = [np.nan, np.nan, np.nan, 10.4]  # FP4 is the Blackwell addition

        x = np.arange(len(gens))
        w = 0.26

        _fig, _ax = plt.subplots(figsize=(8.4, 4.0))
        _ax.bar(x - w, fp16, w, label="FP16/BF16", color="#5b8def")
        _ax.bar(x, fp8, w, label="FP8 (E4M3/E5M2)", color="#e0a458")
        _ax.bar(x + w, fp4, w, label="FP4 (MX)", color="#4c9f70")

        # mark "not supported" gaps
        for _i, _v in enumerate(fp8):
            if _v != _v:  # NaN
                _ax.text(_i, 0.15, "n/a", ha="center", fontsize=7, color="#999")
        for _i, _v in enumerate(fp4):
            if _v != _v:
                _ax.text(_i + w, 0.15, "n/a", ha="center", fontsize=7, color="#999")

        _ax.set_xticks(x)
        _ax.set_xticklabels(gens)
        _ax.set_ylabel("relative tensor throughput\n(Ampere FP16 = 1, illustrative)")
        _ax.set_title("Tensor-core throughput & supported dtypes by generation (illustrative)")
        _ax.legend(loc="upper left", fontsize=8)
        _ax.grid(True, axis="y", alpha=0.15)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Narrow floats: FP8, FP4, and microscaling (MX)

    To use 8 or 4 bits per element you have to spend those bits wisely. A floating-point
    number splits its bits into **sign + exponent + mantissa**; the exponent buys
    *dynamic range*, the mantissa buys *precision*. The narrow ML formats:

    - **FP8 — E4M3:** 1 sign, 4 exponent, 3 mantissa. More precision, less range — used
      for **weights/activations** in the forward pass.
    - **FP8 — E5M2:** 1 sign, 5 exponent, 2 mantissa. More range, less precision — used
      for **gradients**, which span a wider scale.
    - **FP4 — E2M1:** 1 sign, 2 exponent, 1 mantissa. Only **16 representable values**.
      Unusable as a raw tensor type — which is exactly why microscaling exists.

    With so few bits, the format alone can't track a tensor whose values span several
    orders of magnitude. **Microscaling (MX)** fixes this by attaching a **shared scale
    to a small block** of elements (e.g. 32 values share one 8-bit power-of-two scale).
    Each element stores only its narrow mantissa/exponent; the block's scale restores the
    magnitude:

    $$x_i \;\approx\; s_{\text{block}} \cdot \hat{x}_i,
      \qquad \hat{x}_i \in \text{FP4/FP6/FP8},
      \quad s_{\text{block}} \text{ shared by 32 elements.}$$

    This is **per-block** scaling — finer than one scale for the whole tensor (too coarse,
    loses small values) but far cheaper than per-element FP16 (defeats the point). It's the
    same idea as the group-wise quantization you built in `2c`, standardized into a
    hardware format the 5th-gen tensor core reads directly. The supported MX variants
    (MXFP8, MXFP6, MXFP4) and their exact block sizes are version- and CC-dependent
    (verify per §6).

    > [Open Compute Project — OCP Microscaling (MX) Formats spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
    > defines E4M3/E5M2/E2M1 and the shared-scale block layout; the
    > [Blackwell whitepaper](https://resources.nvidia.com/en-us-blackwell-architecture)
    > describes the hardware support.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        # Show how many distinct magnitudes each format can represent, and the
        # range/precision tradeoff, using the IEEE-style (sign, exp, mantissa) split.
        # We enumerate the *positive* normal magnitudes only, for intuition.
        def fmt_stats(name, exp_bits, man_bits, bias):
            # number of positive values (normals + subnormals), rough count
            n_mantissa = 2 ** man_bits
            n_exp = 2 ** exp_bits
            total_pos = n_exp * n_mantissa  # ballpark (ignores special codes)
            # dynamic range from smallest normal to largest finite, in decades
            max_exp = (n_exp - 1) - bias - 1   # leave one code for inf/nan-ish
            min_exp = 1 - bias
            decades = (max_exp - min_exp) * np.log10(2)
            # relative precision near 1.0 ~ 2^-man_bits
            rel_prec = 2.0 ** (-man_bits)
            print(f"  {name:10s} | {1:>1d}+{exp_bits}+{man_bits} bits | "
                  f"~{total_pos:>5d} codes | ~{decades:5.1f} decades range | "
                  f"step near 1.0 = {rel_prec:6.3f}")

        print("=== Narrow-float range vs precision (illustrative) ===\n")
        print("  format     | bits      | codes      | dynamic range | mantissa step")
        print("  " + "-" * 74)
        fmt_stats("FP16",      5, 10, 15)
        fmt_stats("FP8 E4M3",  4,  3,  7)
        fmt_stats("FP8 E5M2",  5,  2, 15)
        fmt_stats("FP4 E2M1",  2,  1,  1)
        print("\n  Fewer mantissa bits -> coarser steps (lower precision).")
        print("  Fewer exponent bits -> narrower dynamic range.")
        print("  FP4's ~16 codes are why a per-block MX scale is mandatory.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactive: the precision / throughput tradeoff

    Pick a compute format. The left panel shows what a fixed-magnitude signal looks like
    after rounding to that format's grid (precision); the right panel shows the
    *illustrative* relative matrix throughput — note this panel is normalized
    *within Blackwell* (FP16 = 1), unlike the cross-generation chart in §2 (Ampere FP16 = 1).
    The whole game of narrow-float kernels is
    reading this picture: **how far right can I push throughput before the left panel's
    error breaks my model?** That answer is empirical — you measure accuracy on the real
    workload — which is why the capstone (`4b`) makes you benchmark *both* axes.
    """)
    return


@app.cell
def _(mo):
    dtype_dropdown = mo.ui.dropdown(
        options=["FP16 (baseline)", "FP8 E4M3", "FP8 E5M2", "FP4 E2M1 (MX block)"],
        value="FP8 E4M3",
        label="compute format",
    )
    dtype_dropdown
    return (dtype_dropdown,)


@app.cell
def _(dtype_dropdown):
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        # (mantissa_bits, exp_bits, relative throughput, uses MX block scale)
        table = {
            "FP16 (baseline)":      (10, 5, 1.0, False),
            "FP8 E4M3":             (3, 4, 2.0, False),
            "FP8 E5M2":             (2, 5, 2.0, False),
            "FP4 E2M1 (MX block)":  (1, 2, 4.0, True),
        }
        man_bits, _exp_bits, thru, uses_mx = table[dtype_dropdown.value]

        # A smooth signal we will quantize to the chosen mantissa grid.
        _x = np.linspace(0, 4 * np.pi, 400)
        _sig = 0.6 * np.sin(_x) + 0.3 * np.sin(2.7 * _x)

        # Simulate per-block MX scaling: split into blocks of 32, scale each block to
        # ~[-1, 1] by its max, quantize the mantissa, then rescale. With one global
        # scale (no MX) small blocks lose their detail.
        def quantize(sig, mbits, mx):
            step = 2.0 ** (-mbits)
            out = np.empty_like(sig)
            if mx:
                bs = 32
                for _s in range(0, len(sig), bs):
                    _blk = sig[_s:_s + bs]
                    _scale = max(np.abs(_blk).max(), 1e-9)
                    _q = np.round((_blk / _scale) / step) * step
                    out[_s:_s + bs] = _q * _scale
            else:
                _scale = max(np.abs(sig).max(), 1e-9)
                _q = np.round((sig / _scale) / step) * step
                out[:] = _q * _scale
            return out

        _q = quantize(_sig, man_bits, uses_mx)
        _err = np.sqrt(np.mean((_q - _sig) ** 2))

        _fig, (_axL, _axR) = plt.subplots(
            1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.7, 1]})

        _axL.plot(_x, _sig, color="#999", linewidth=1.2, label="true signal")
        _axL.plot(_x, _q, color="#d65f5f", linewidth=1.4, label="after rounding")
        _axL.set_title(f"{dtype_dropdown.value}: {man_bits} mantissa bits, RMSE={_err:.3f}")
        _axL.set_xlabel("element index (arb.)")
        _axL.legend(loc="upper right", fontsize=8)
        _axL.set_ylim(-1.1, 1.1)

        _names = ["FP16", "FP8", "FP8", "FP4"]
        _thru = [1.0, 2.0, 2.0, 4.0]
        _cur_idx = list(table.keys()).index(dtype_dropdown.value)
        _cols = ["#cbd6ee"] * 4
        _cols[_cur_idx] = "#4c9f70"
        _axR.bar(range(4), _thru, color=_cols)
        _axR.set_xticks(range(4))
        _axR.set_xticklabels(["FP16", "E4M3", "E5M2", "FP4"], fontsize=8)
        _axR.set_ylabel("rel. throughput (illustrative)")
        _axR.set_title(f"{thru:.0f}x")
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. The Tensor Memory Accelerator (TMA)

    Through Part 3 you moved data into shared memory **by hand**: every thread issued
    its own `cp.async` load, you computed addresses, masked the edges, and double-buffered
    the pipeline yourself (`3f`). That works, but it burns registers on address math and
    couples your copy code to your tile shape.

    The **Tensor Memory Accelerator (TMA)**, introduced on Hopper (`sm_90`) and carried
    forward on Blackwell, makes a **bulk async tensor copy** a *single* instruction. You
    describe the tensor once in a **tensor map** (a descriptor: base pointer, global
    shape, strides, the tile/box shape you want), then one thread issues:

    ```
    // schematic — descriptor built on host, copy issued on device
    cp.async.bulk.tensor   [smem_tile], [tensor_map, {tile_coords}], [mbarrier];
    // ... do other work ...
    mbarrier.wait          // copy completes asynchronously, signals the barrier
    ```

    What the hardware copy engine buys you versus the by-hand loop:

    - **Addresses are computed in hardware** from the descriptor — no per-thread index
      arithmetic, fewer registers, more occupancy headroom.
    - **One thread launches the whole tile copy**; it lands asynchronously and signals an
      **mbarrier** when done, so the rest of the warp keeps computing (true overlap, the
      `3f` software-pipeline idea but engine-driven).
    - **Edge masking is handled** by the descriptor's bounds — the ragged last tile
      "just works" instead of needing your guard logic.

    In Triton you rarely write this by hand: the compiler **emits TMA copies for you**
    when it recognizes the access pattern (block pointers / `tl.make_block_ptr` style
    loads on a supported target). The point of knowing TMA exists is to *recognize when
    you're leaving it on the table* — if your kernel is doing visible per-thread address
    math to stage a tile, a TMA path may be available.

    A caution: whether the Triton build for your CUDA 13.1 / `sm_120` setup emits TMA for
    a given kernel, and which TMA variants are legal on consumer `sm_120` versus
    data-center parts, is toolchain-version-dependent (verify per §6).

    > [CUDA C++ Programming Guide — Tensor Memory Accelerator](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-memory-accelerator)
    > and the [PTX ISA — `cp.async.bulk.tensor`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
    > are the references; CUTLASS/CuTe wrap TMA as the `TiledCopy` abstraction.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        # Illustrative timeline: staging a tile by-hand (per-thread cp.async + sync)
        # vs one TMA bulk copy that overlaps with compute. Bars are schematic units.
        _fig, _ax = plt.subplots(figsize=(8.6, 3.4))

        # By-hand: address math, then a wave of cp.async, then a barrier, then compute.
        _ax.broken_barh([(0, 1.2)], (6, 0.8), facecolors="#d65f5f")   # addr math
        _ax.broken_barh([(1.2, 2.6)], (6, 0.8), facecolors="#e0a458")  # cp.async wave
        _ax.broken_barh([(3.8, 0.6)], (6, 0.8), facecolors="#999")     # syncthreads
        _ax.broken_barh([(4.4, 3.0)], (6, 0.8), facecolors="#5b8def")  # compute
        _ax.text(0, 7.1, "by hand (3f-style):", fontsize=9, weight="bold")

        # TMA: one issue, copy runs in background, compute overlaps after a short setup.
        _ax.broken_barh([(0, 0.5)], (3, 0.8), facecolors="#4c9f70")    # issue TMA
        _ax.broken_barh([(0.5, 3.0)], (3.95, 0.35), facecolors="#cbd6ee")  # async copy (bg)
        _ax.broken_barh([(0.7, 3.0)], (3, 0.8), facecolors="#5b8def")  # compute overlaps
        _ax.text(0, 4.1, "with TMA:", fontsize=9, weight="bold")

        _ax.set_xlim(0, 8)
        _ax.set_ylim(2.5, 7.6)
        _ax.set_yticks([])
        _ax.set_xlabel("time (schematic)")
        _ax.set_title("Staging a tile: per-thread loads + barrier vs one async TMA copy")

        # legend
        import matplotlib.patches as mpatches
        _handles = [
            mpatches.Patch(color="#d65f5f", label="per-thread address math"),
            mpatches.Patch(color="#e0a458", label="cp.async wave"),
            mpatches.Patch(color="#999", label="__syncthreads"),
            mpatches.Patch(color="#4c9f70", label="TMA issue (1 thread)"),
            mpatches.Patch(color="#cbd6ee", label="async copy (background)"),
            mpatches.Patch(color="#5b8def", label="compute"),
        ]
        _ax.legend(handles=_handles, loc="lower right", fontsize=7, ncol=2)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Thread-block clusters & distributed shared memory

    A second Hopper-era addition, also present in the Blackwell programming model, sits
    *between* the block and the grid in the hierarchy you learned in `0b`:

    $$\text{thread} \to \text{warp} \to \text{block}
      \to \underbrace{\textbf{cluster}}_{\text{new tier}} \to \text{grid}.$$

    A **thread-block cluster** is a small group of blocks **guaranteed to be co-resident
    on neighboring SMs** (within a GPC), which unlocks two things plain blocks can't do:

    - **Distributed shared memory (DSMEM):** a block can directly read/write *another
      block's* shared memory in the same cluster, over a fast on-chip network — turning a
      cluster's shared memory into one larger pooled tile without a round trip to global.
    - **Cluster-wide barriers:** the blocks in a cluster can synchronize with each other,
      a synchronization scope between "within a block" (`__syncthreads`) and "not at all"
      (across the grid).

    This matters for very large tiles (big GEMMs, attention) that want more cooperating
    shared memory than one SM holds. But — the important honesty for *this* course —
    **clusters and DSMEM are primarily a data-center (`sm_90` / `sm_100`) story.** Whether,
    and to what extent, **consumer `sm_120`** exposes cluster launch and distributed shared
    memory is **not something to assume**: it is gated by compute capability and CUDA
    version, and the consumer parts have historically been more limited here than the
    H100/B200 line — so confirm against the arch docs (per §6) before you design a kernel
    around them.

    > [CUDA C++ Programming Guide — Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters)
    > and [Distributed Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#distributed-shared-memory)
    > define the model and, crucially, the **compute-capability requirements** — read
    > those requirement lines against your card.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6. How to actually find out what your card does

    This lecture deliberately refuses to hand you a feature checklist for `sm_120` —
    because the honest answer is **version-dependent**, and a wrong checklist is worse
    than none. The reliable workflow:

    1. **Query the device.** `python -m harness.device_info` reports your compute
       capability, SM count, shared-memory size, and limits straight from the driver —
       ground truth for *this* machine.
    2. **Read the per-CC table** in the CUDA C++ Programming Guide's "Compute
       Capabilities" appendix, for **CUDA 13.1** specifically. Feature rows are gated by
       CC *and* toolkit version.
    3. **Confirm the instruction is legal** for your arch: compile a tiny probe with
       `nvcc -arch=sm_120`, or check the PTX ISA doc for the instruction's supported
       targets. If it assembles for `sm_120`, it's real on your card.
    4. **Measure, don't trust the headline.** Even when a format is *supported*, its peak
       rate on consumer silicon may differ from the data-center number in the whitepaper.
       `triton.testing.do_bench` on the real workload (Part 2) is the only number you
       should quote.

    The bars and timelines in this lecture are **illustrative of trends**, by design. The
    numbers that go in your capstone report come from step 4, on your hardware.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    - **Narrower is the throughput lever now.** When a kernel is compute-bound (right of
      the roofline ridge), the biggest single win on Blackwell is dropping operand
      precision — FP8, then FP4-with-MX — and the cost is *accuracy*, which you must
      measure, not assume. This is the entire premise of the quantized-GEMM path in `4b`.
    - **Let the engine move your tiles.** If a kernel is spending registers and
      instructions on per-thread address math to stage tiles, a TMA path (often emitted
      by Triton for block-pointer loads) may exist — recognizing that is how you reclaim
      occupancy.
    - **Don't port data-center assumptions blindly.** Clusters, DSMEM, and some MX
      variants are `sm_90`/`sm_100` features that may be limited or absent on consumer
      `sm_120`. The discipline is: query, read the CC table for CUDA 13.1, confirm, then
      design.
    - **Generations move the roof; benchmarks anchor it.** Every "Blackwell does X
      TFLOP/s" claim is generation- and SKU-relative. Your roofline scoreboard from `0d`
      is only meaningful with *your card's measured* ceilings — which the capstone makes
      you produce.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Where this is used

    There's no exercise for this lecture — it's the architectural context for the final
    project. You'll put it to work immediately in the capstone (`4b`):

    - If you take the **quantized-GEMM** path, the §2–3 throughput levers above are the
      menu you'll trade against accuracy.
    - Whichever path you take, section 6's "query → read CC table → confirm → measure"
      loop is exactly how you'll establish the roofline target you benchmark against.

    Keep `python -m harness.device_info` handy — it's the one source of truth for what
    *your* `sm_120` actually offers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [3G: Tensor Cores](../3g_tensor_cores/) &nbsp;|&nbsp; Next: [4B: Capstone](../4b_capstone/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
