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
    # 2A: Autotuning

    > *"The best block size is a property of the hardware, the shape, and the
    > dtype — not a number you can guess. So stop guessing: measure."*

    Part 1 ended with a tiled matmul whose performance hinged on a handful of
    constants — `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps`, `num_stages`. We
    picked them by hand and by feel. That works for one shape on one card. It
    falls apart the moment the matrix shape changes, the dtype changes, or you move
    to a different GPU: the *same* kernel source can swing 2–3× in throughput
    depending on those constants.

    This lecture is about making that choice **empirical and automatic**. Triton's
    `@triton.autotune` compiles a *menu* of configurations, benchmarks them on the
    first call for a given shape, and caches the winner. Your job shrinks to two
    things you can reason about: (1) write the kernel so it is **correct for any
    shape** (masking), and (2) propose a **sensible menu** of configs the roofline
    says are worth trying. We'll build the intuition for both, then visualize the
    tuning surface you're searching over.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. Why a single block size is never right

    A tiled matmul (recall `1e`) computes a `BLOCK_M x BLOCK_N` output tile per
    program by streaming over the $K$ dimension in chunks of `BLOCK_K`. The tile
    shape sets three things at once, and they pull in opposite directions:

    - **Arithmetic intensity.** A bigger tile reuses each loaded element across more
      multiply-adds, pushing the kernel rightward on the roofline (`0d`) toward
      compute-bound. Tiny tiles stay memory-bound.
    - **Shared-memory / register pressure.** A `BLOCK_M x BLOCK_K` plus
      `BLOCK_K x BLOCK_N` staging area must fit in the SM's ~100 KB of shared
      memory, and the accumulator lives in registers. Bigger tiles → fewer resident
      blocks → lower occupancy (`0d` again).
    - **Wave quantization.** With 70 SMs, a grid of, say, 73 tiles runs as two
      "waves": 70 tiles, then 3 tiles while 67 SMs idle. The tile shape sets the
      tile *count*, so it silently decides how badly the tail wastes the machine.

    The optimum is the tile that **balances reuse against occupancy for this exact
    shape**. Move from a $4096^3$ GEMM to a skinny $4096\times4096\times64$ one and
    the balance shifts entirely. There is no constant that wins everywhere — which
    is precisely why we search.

    > [Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
    > builds the autotuned GEMM this lecture mirrors; PMPP Ch. 5 derives the
    > tiling/reuse tradeoff.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Masking: correct for *arbitrary* shapes

    Autotuning is only useful if the kernel is correct for whatever shape you throw
    at it — including shapes that aren't multiples of the block. The mechanism is
    the same boundary mask you met in `1a`, now in two dimensions plus the K-loop.

    A program owns rows $[m_0, m_0 + \text{BLOCK\_M})$ and columns
    $[n_0, n_0 + \text{BLOCK\_N})$. When $M$ or $N$ isn't a multiple of the block,
    the last tile hangs off the edge. We guard **every** load and the final store:

    $$\text{mask}_{ij} = (i < M)\;\wedge\;(j < N), \qquad
      \text{mask}^{K}_{k} = (k < K).$$

    Loads use `other=0.0` for the out-of-range lanes so the masked-off products
    contribute exactly zero to the accumulator — the dot product is unchanged. The
    K-loop needs its own mask because the *contraction* dimension can also be ragged
    (think $K = 65$ with `BLOCK_K = 32`: two full chunks plus a 1-wide tail).

    The padding-to-zero trick is what lets a single config like `BLOCK_M=128` run on
    an $M=130$ matrix without a separate "remainder" kernel. **Mask first, tune
    second** — a fast kernel that's wrong on ragged shapes is worthless.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        # A tiny, exact simulation of the boundary mask in a tiled matmul.
        # We "compute" one output tile that hangs off the right/bottom edge and
        # show that masked loads (other=0) reproduce the true result.
        _rng = np.random.default_rng(0)
        M, N, K = 6, 5, 7              # deliberately not multiples of the block
        BLOCK_M, BLOCK_N, BLOCK_K = 4, 4, 4

        A = _rng.integers(-3, 4, size=(M, K)).astype(np.float32)
        B = _rng.integers(-3, 4, size=(K, N)).astype(np.float32)
        C_true = A @ B

        # The tile at (m0, n0) = (4, 4): rows 4..7, cols 4..7 -> partly off-edge.
        m0, n0 = 4, 4
        rows = m0 + np.arange(BLOCK_M)         # [4,5,6,7] -- 6,7 are OOB (M=6)
        cols = n0 + np.arange(BLOCK_N)         # [4,5,6,7] -- 5,6,7 are OOB (N=5)
        acc = np.zeros((BLOCK_M, BLOCK_N), dtype=np.float32)

        for k0 in range(0, K, BLOCK_K):
            ks = k0 + np.arange(BLOCK_K)       # may run past K=7
            # Masked load of A[rows, ks] with other=0.0
            a_mask = (rows[:, None] < M) & (ks[None, :] < K)
            a = np.where(a_mask, A[np.clip(rows, 0, M - 1)][:, np.clip(ks, 0, K - 1)], 0.0)
            # Masked load of B[ks, cols] with other=0.0
            b_mask = (ks[:, None] < K) & (cols[None, :] < N)
            b = np.where(b_mask, B[np.clip(ks, 0, K - 1)][:, np.clip(cols, 0, N - 1)], 0.0)
            acc += a @ b

        # Compare the in-range part of our tile to the truth.
        valid_r = rows < M
        valid_c = cols < N
        got = acc[np.ix_(valid_r, valid_c)]
        want = C_true[np.ix_(rows[valid_r], cols[valid_c])]

        print("=== Masked tile vs ground truth (ragged 6x5x7 matmul) ===")
        print(f"  tile origin (m0,n0) = ({m0},{n0}), block = "
              f"{BLOCK_M}x{BLOCK_N}x{BLOCK_K}")
        print(f"  rows {rows.tolist()}  ->  in-range {rows[valid_r].tolist()}")
        print(f"  cols {cols.tolist()}  ->  in-range {cols[valid_c].tolist()}")
        print(f"  reconstructed in-range tile:\n{got}")
        print(f"  ground-truth in-range tile:\n{want}")
        print(f"  max abs error = {np.abs(got - want).max():.1f}  "
              f"(0 == masking is exact)")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. The `@triton.autotune` decorator

    With masking in place, you hand Triton a **list of `Config`s** and the
    **argument names that change the problem size** (`key`). On the first call for a
    new key, Triton benchmarks every config, caches the winner against that key, and
    reuses it forever after. Subsequent calls with the same key pay nothing.

    ```python
    import triton
    import triton.language as tl

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
                          num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32},
                          num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32},
                          num_warps=2, num_stages=4),
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32},
                          num_warps=8, num_stages=3),
        ],
        key=['M', 'N', 'K'],          # re-tune when any of these changes
    )
    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr):
        ...
    ```

    Four knobs live in a `Config`:

    - **`BLOCK_M / BLOCK_N / BLOCK_K`** — the tile shape from §1. These are
      `tl.constexpr` *meta-parameters*: each distinct value compiles a separate
      kernel, so the compiler can unroll and allocate registers/shared memory for
      exactly that size.
    - **`num_warps`** — warps per program (block). More warps = more lanes to
      cover a big tile and more parallelism to hide latency, but each warp competes
      for registers. Powers of two, typically 2–8.
    - **`num_stages`** — depth of the **software pipeline** over the K-loop. Triton
      prefetches the next K-chunk while computing on the current one (`cp.async`
      under the hood — you'll do this by hand in `3f`). More stages hide more memory
      latency but cost more shared memory for the in-flight buffers.

    > [`triton.autotune` reference](https://triton-lang.org/main/python-api/generated/triton.autotune.html).
    > Note: because tuning runs on the *first* call per key, it shows up as a
    > one-time latency spike, and `key` should list everything that changes the
    > optimal config.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. A roofline-guided menu

    Don't throw the whole Cartesian product at the autotuner — a 200-config search
    is slow to compile and mostly wasted. The roofline (`0d`) tells you *which*
    configs are even plausible, so you can prune to a dozen good candidates.

    Reason about each candidate before adding it:

    1. **Reuse / intensity.** Bigger `BLOCK_M x BLOCK_N` raises arithmetic intensity
       (each loaded element feeds more FMAs), moving a memory-bound shape rightward.
       For large square GEMMs you *want* the big tiles. For skinny / memory-bound
       shapes the small tiles win — don't bother compiling the giant ones.
    2. **Occupancy budget.** Estimate shared memory per config:
       $S \approx \text{num\_stages}\cdot(\text{BLOCK\_M}+\text{BLOCK\_N})
       \cdot\text{BLOCK\_K}\cdot\text{bytes}$. If $S$ blows past ~100 KB it won't
       even launch; if it's large you'll get one block per SM. Keep a spread of
       sizes so *some* config keeps occupancy up.
    3. **Wave quantization.** Prefer tiles whose count divides cleanly into multiples
       of 70 SMs for the shapes you care about, so the last wave isn't mostly idle.
    4. **Warps vs tile.** Match `num_warps` to tile size — a `256 x 64` tile needs
       more warps (8) than a `64 x 64` tile (2) to keep every lane fed.

    The art is proposing a *small, diverse* menu: a couple of big compute-bound
    tiles, a couple of small high-occupancy ones, varied `num_warps`/`num_stages`.
    Let the autotuner pick; you just make sure the winner is *in the room*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualizing the tuning surface

    Here is the surface the autotuner is searching: synthetic (illustrative, not
    measured) TFLOP/s for a large square GEMM as a function of `BLOCK_M` and
    `BLOCK_N`. The model captures the two competing effects from §1 — **reuse rises**
    with tile area, then **occupancy collapses** once the tile is too big to keep
    enough blocks resident — plus a penalty for tiles whose lane count doesn't divide
    32 cleanly. The peak is an interior ridge: not the smallest tile, not the biggest.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        # Synthetic model of the tuning surface (illustrative, NOT measured).
        # perf ~ reuse(tile) * occupancy(tile) * lane_efficiency(tile)
        blocks = np.array([32, 64, 96, 128, 160, 192, 224, 256])
        P_PEAK = 44.0  # TFLOP/s ceiling for this illustration

        def reuse(bm, bn):
            # arithmetic intensity rises with tile area, saturating
            area = bm * bn
            return area / (area + 4096.0)

        def occupancy(bm, bn):
            # shared mem ~ (bm+bn)*BK*stages*bytes; bigger tile -> fewer blocks
            smem = (bm + bn) * 32 * 3 * 2 / 1024.0   # KB, BK=32, stages=3, fp16
            blocks_resident = max(1, int(100.0 // max(smem, 1e-6)))
            return min(1.0, blocks_resident / 6.0)

        def lane_eff(bm, bn):
            # penalize tiles not divisible by 32 in either dim
            pen_m = 1.0 if bm % 32 == 0 else 0.7
            pen_n = 1.0 if bn % 32 == 0 else 0.7
            return pen_m * pen_n

        Z = np.zeros((len(blocks), len(blocks)))
        for i, bm in enumerate(blocks):
            for j, bn in enumerate(blocks):
                Z[i, j] = P_PEAK * reuse(bm, bn) * occupancy(bm, bn) * lane_eff(bm, bn)

        best = np.unravel_index(np.argmax(Z), Z.shape)

        _fig, _ax = plt.subplots(figsize=(6.8, 5.4))
        _im = _ax.imshow(Z, origin="lower", cmap="viridis", aspect="auto")
        _ax.set_xticks(range(len(blocks)))
        _ax.set_yticks(range(len(blocks)))
        _ax.set_xticklabels(blocks)
        _ax.set_yticklabels(blocks)
        _ax.set_xlabel("BLOCK_N")
        _ax.set_ylabel("BLOCK_M")
        _ax.set_title("Tuning surface: synthetic TFLOP/s over (BLOCK_M, BLOCK_N)")
        # mark the winner
        _ax.scatter([best[1]], [best[0]], marker="*", s=320,
                    edgecolor="white", facecolor="#ff4d4d", zorder=5)
        _ax.annotate(f"autotune winner\nBLOCK_M={blocks[best[0]]}, "
                     f"BLOCK_N={blocks[best[1]]}",
                     (best[1], best[0]), textcoords="offset points",
                     xytext=(8, 8), color="white", fontsize=8, weight="bold")
        _cb = _fig.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
        _cb.set_label("TFLOP/s (illustrative)")
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Measuring with `triton.testing.do_bench`

    The autotuner uses a benchmarking primitive you should also reach for by hand
    when you want to *understand* a config rather than just pick one. `do_bench`
    times a callable on the GPU, correctly:

    ```python
    import triton

    ms = triton.testing.do_bench(lambda: matmul(a, b))      # median ms
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12             # GEMM = 2*M*N*K FLOP
    gbytes = (a.numel() + b.numel() + M * N) * a.element_size()
    gbps   = gbytes / (ms * 1e-3) / 1e9
    ```

    What it does that a naive `time.time()` cannot:

    - **Warms up** first (the first launch pays JIT + autotune costs you don't want
      in the measurement).
    - **Synchronizes** the device — kernel launches are async (`0b`), so wall-clock
      around the launch alone measures nothing.
    - **Flushes the L2 cache** between runs (the 48 MB L2 would otherwise serve a
      small matrix from cache and report a fantasy bandwidth).
    - Reports the **median** over many reps to reject scheduler noise; you can also
      ask for quantiles.

    Once you have milliseconds, convert to the roofline's currency: TFLOP/s for
    compute-bound kernels (matmul = $2MNK$ FLOP), GB/s for memory-bound ones, and
    compare against the 5070 Ti's ~44 TFLOP/s and ~896 GB/s ceilings.

    > [`do_bench` reference](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html).
    > For sweeps and plots, `triton.testing.perf_report` + `Benchmark` wrap it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactive: walk one config dimension

    Fix the tile area and sweep a single knob to watch the autotuner's decision form.
    Pick which dimension to vary; the plot draws the synthetic perf curve along that
    axis and marks the config the autotuner would keep (the argmax). This is the
    1-D slice through the §4 surface that the search actually compares — the winner
    is just the top of this curve.
    """)
    return


@app.cell
def _(mo):
    sweep_dim = mo.ui.dropdown(
        options=["BLOCK_M", "BLOCK_N", "num_warps", "num_stages"],
        value="BLOCK_M",
        label="config dimension to sweep",
    )
    sweep_dim
    return (sweep_dim,)


@app.cell
def _(sweep_dim):
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        P_PEAK = 44.0
        dim = sweep_dim.value

        if dim in ("BLOCK_M", "BLOCK_N"):
            xs = np.array([32, 64, 96, 128, 160, 192, 224, 256])
            other = 128  # the fixed dimension

            def perf(v):
                bm, bn = (v, other) if dim == "BLOCK_M" else (other, v)
                area = bm * bn
                reuse = area / (area + 4096.0)
                smem = (bm + bn) * 32 * 3 * 2 / 1024.0
                occ = min(1.0, max(1, int(100.0 // max(smem, 1e-6))) / 6.0)
                lane = (1.0 if bm % 32 == 0 else 0.7) * (1.0 if bn % 32 == 0 else 0.7)
                return P_PEAK * reuse * occ * lane

            xlabel = f"{dim}  (other tile dim fixed at {other})"
        elif dim == "num_warps":
            xs = np.array([1, 2, 4, 8, 16])

            def perf(w):
                # too few warps -> can't fill a 128x128 tile; too many -> reg spill
                fill = min(1.0, w / 4.0)          # need ~4 warps to cover the tile
                spill = 1.0 if w <= 8 else 0.55   # >8 warps starts spilling regs
                return P_PEAK * 0.85 * fill * spill

            xlabel = "num_warps  (tile fixed at 128x128)"
        else:  # num_stages
            xs = np.array([1, 2, 3, 4, 5, 6])

            def perf(s):
                # more stages hide latency, then smem for buffers runs out
                hide = 1.0 - 0.6 / s              # diminishing latency hiding
                smem_ok = 1.0 if s <= 4 else (0.7 if s == 5 else 0.45)
                return P_PEAK * 0.92 * hide * smem_ok

            xlabel = "num_stages  (tile fixed at 128x128)"

        ys = np.array([perf(int(x)) for x in xs])
        win = int(np.argmax(ys))

        _fig, _ax = plt.subplots(figsize=(7.6, 3.8))
        _ax.plot(xs, ys, "-o", color="#5b8def", linewidth=2, zorder=3)
        _ax.scatter([xs[win]], [ys[win]], marker="*", s=300,
                    color="#d65f5f", zorder=5)
        _ax.annotate(f"winner: {dim}={xs[win]}\n{ys[win]:.1f} TFLOP/s",
                     (xs[win], ys[win]), textcoords="offset points",
                     xytext=(8, -28), color="#d65f5f", fontsize=9, weight="bold")
        _ax.axhline(P_PEAK, color="#999", linestyle=":", linewidth=1)
        _ax.text(xs[0], P_PEAK * 0.97, "compute roof (~44 TFLOP/s)",
                 fontsize=7, color="#777", va="top")
        _ax.set_xlabel(xlabel)
        _ax.set_ylabel("TFLOP/s (illustrative)")
        _ax.set_ylim(0, P_PEAK * 1.1)
        _ax.set_title(f"do_bench would crown the argmax along {dim}")
        _ax.grid(True, alpha=0.15)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    - **Make it correct for any shape, then tune.** Two-dimensional masks plus a
      K-loop mask let one config handle ragged $M$, $N$, *and* $K$. A fast kernel
      that's wrong on non-multiple-of-block shapes is a bug, not an optimization.
    - **Propose a small, roofline-justified menu.** A handful of configs spanning
      big-compute-bound to small-high-occupancy, with varied `num_warps`/
      `num_stages`. Don't brute-force the Cartesian product — prune with the roofline.
    - **`key` everything that changes the optimum.** List `M, N, K` (and dtype, if it
      varies) so the autotuner re-tunes when the problem actually changes — and
      accept the one-time tuning cost on the first call per key.
    - **Measure honestly with `do_bench`.** Warmup, sync, L2 flush, median. Convert to
      TFLOP/s or GB/s and read it against the ceiling — that number, not vibes, tells
      you whether a config is good.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Now go write it

    Time to let the autotuner do the searching for you. Open the harness and write
    the autotuned matmul:

    ```bash
    python -m harness.runner e10 --watch
    ```

    `e10` takes your tiled GEMM from `1e` and wraps it in `@triton.autotune` over a
    config menu you design. You'll add the two-dimensional + K masks so it's correct
    for arbitrary shapes, propose configs the roofline justifies, and let `do_bench`
    crown the winner. The metric is FLOP/s — watch your matmul climb toward the
    compute roof as the autotuner finds the tile this shape actually wants.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [1G: Scan / Prefix-Sum](../1g_scan/) &nbsp;|&nbsp; Next: [2B: Flash Attention](../2b_flash_attention/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
