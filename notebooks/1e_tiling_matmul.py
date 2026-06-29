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
    # 1E: Tiling & Matmul

    > *"A matmul moves the same number every time. Tiling is the art of making it
    > pay you back before you put it down."*

    This is the heart of Part 1. Everything you've built so far — `program_id`,
    index ranges, masking the ragged tail (`0b`, `1a`), coalesced loads (`1b`),
    reductions (`1c`) — converges here, in **two dimensions**, on the single most
    important kernel in all of deep learning: matrix multiply.

    The plot from `0d` is the whole story in miniature. A naive matmul sits glued to
    the slanted **memory roof**: it does $O(N^3)$ FLOPs but also moves $O(N^3)$ bytes,
    so its arithmetic intensity is $\approx O(1)$ and it never lifts off. **Tiling**
    is the lever that drags that dot rightward, off the memory roof and up toward the
    compute roof. By the end of this lecture you'll know exactly *why* — and you'll
    have the structure to write it yourself.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. Warm-up: transpose, your first 2-D grid

    Before matmul, get comfortable indexing a 2-D problem. **Transpose** is the
    gentlest 2-D kernel there is: read element $(i, j)$ of the input, write it to
    $(j, i)$ of the output. No math — pure memory movement. It is **bandwidth-bound**
    by construction, so it lives on the memory roof and *stays* there. The interesting
    part isn't the FLOPs (there are none); it's the **index arithmetic** of a 2-D grid.

    Launch a **2-D grid** of programs, one per output tile. Each program asks the
    hardware for two coordinates:

    $$\texttt{pid\_m} = \text{program\_id}(0), \qquad
      \texttt{pid\_n} = \text{program\_id}(1).$$

    From those you build a **row range** and a **column range**, then combine them into
    a full 2-D block of offsets with broadcasting — `[:, None]` makes a column vector,
    `[None, :]` makes a row vector, and addition broadcasts them into a tile:

    $$\texttt{offs} = \underbrace{\texttt{offs\_m[:, None]}}_{\text{rows}} \cdot
      \texttt{stride\_row} \;+\; \underbrace{\texttt{offs\_n[None, :]}}_{\text{cols}}
      \cdot \texttt{stride\_col}.$$

    That single expression is the 2-D indexing pattern you will use in **every** matrix
    kernel for the rest of the course. Mask **both** dimensions (rows *and* columns can
    overrun a ragged edge), read the input tile, and write it back with the row/column
    strides **swapped** — that swap *is* the transpose.

    ```python
    # 2-D indexing skeleton (illustrative — NOT the exercise solution)
    pid_m = tl.program_id(0)                       # which tile-row
    pid_n = tl.program_id(1)                       # which tile-col
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # rows this tile owns
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # cols this tile owns

    # build a BLOCK_M x BLOCK_N grid of input addresses by broadcasting
    in_ptrs  = in_ptr  + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    mask     = (offs_m[:, None] < M) & (offs_n[None, :] < N)   # guard BOTH dims
    tile     = tl.load(in_ptrs, mask=mask)

    # write transposed: swap which range drives which stride of the output
    # out_ptrs = out_ptr + offs_n[..] * stride_o? + offs_m[..] * stride_o?   <-- your crux
    ```

    The skeleton stops where the learning starts. Working out which range drives which
    output stride — and getting the output mask right — is the crux left to you in `e06`.
    (In Part 3's CUDA version, the transpose gets *interesting* for a different reason:
    naive transpose hits **shared-memory bank conflicts**, and the fix is a padded tile.
    Here in Triton you just get the indexing right.)

    > [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
    > use exactly this `offs[:, None] + offs[None, :]` broadcast idiom throughout.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        # Visualize the 2-D broadcast that builds a tile's offsets.
        _BM, _BN = 4, 6
        _offs_m = np.arange(_BM)            # row range  -> [:, None]
        _offs_n = np.arange(_BN)            # col range  -> [None, :]
        _stride_row = _BN                   # row-major input, stride = #cols
        _offs = _offs_m[:, None] * _stride_row + _offs_n[None, :]

        _fig, _axes = plt.subplots(1, 3, figsize=(10, 3.2))

        # column vector offs_m[:, None]
        _axes[0].imshow(_offs_m[:, None], cmap="Blues", aspect="auto")
        _axes[0].set_title("offs_m[:, None]\n(rows, shape 4x1)", fontsize=9)
        for _i in range(_BM):
            _axes[0].text(0, _i, str(_offs_m[_i]), ha="center", va="center")

        # row vector offs_n[None, :]
        _axes[1].imshow(_offs_n[None, :], cmap="Greens", aspect="auto")
        _axes[1].set_title("offs_n[None, :]\n(cols, shape 1x6)", fontsize=9)
        for _j in range(_BN):
            _axes[1].text(_j, 0, str(_offs_n[_j]), ha="center", va="center")

        # broadcast sum -> 2-D tile of flat addresses
        _axes[2].imshow(_offs, cmap="Purples", aspect="auto")
        _axes[2].set_title("offs_m*stride + offs_n\n(tile, shape 4x6)", fontsize=9)
        for _i in range(_BM):
            for _j in range(_BN):
                _axes[2].text(_j, _i, str(_offs[_i, _j]), ha="center",
                              va="center", fontsize=8)

        for _ax in _axes:
            _ax.set_xticks([])
            _ax.set_yticks([])
        _fig.suptitle("Broadcasting a row range and a column range into a 2-D tile",
                      y=1.02, fontsize=11)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Matmul, the naive view — and why it's stuck on the memory roof

    The definition. For $A \in \mathbb{R}^{M\times K}$ and $B \in \mathbb{R}^{K\times N}$,
    the product $C = AB \in \mathbb{R}^{M\times N}$ is

    $$C_{ij} = \sum_{k=0}^{K-1} A_{ik}\,B_{kj}.$$

    Each output element is a **dot product** of one row of $A$ and one column of $B$.
    The obvious kernel gives one program (or thread) each $C_{ij}$: it streams a full
    row of $A$ ($K$ elements) and a full column of $B$ ($K$ elements) straight from
    DRAM, multiplies and accumulates, writes one number, and **throws all $2K$ loaded
    values away**.

    Count the traffic for square $N\times N$ matrices (fp32, 4 bytes each). There are
    $N^2$ output elements; each reads $2N$ values from DRAM:

    $$\text{bytes} \;=\; N^2 \cdot 2N \cdot 4 \;=\; 8N^3 \text{ bytes},
      \qquad \text{FLOPs} \;=\; N^2 \cdot 2N \;=\; 2N^3.$$

    Now the arithmetic intensity — FLOP per byte, the $x$-axis of the `0d` roofline:

    $$I_{\text{naive}} \;=\; \frac{2N^3}{8N^3} \;=\; \frac{1}{4}\ \text{FLOP/byte}.$$

    **It doesn't depend on $N$.** Make the matrix a thousand times bigger and the
    intensity is *still* $\tfrac14$. That is the signature of an $O(1)$-intensity
    kernel: it is pinned to the slanted **memory roof** no matter what. On your 5070 Ti
    (896 GB/s) the ceiling is $896\,\text{GB/s} \times 0.25 \approx 224$ GFLOP/s — about
    **0.5% of the ~44 TFLOP/s compute roof**. The math units sit idle while the kernel
    drowns in redundant DRAM traffic. The same row of $A$ gets reloaded once for every
    column of $B$; the same column of $B$ once for every row of $A$. *That* redundancy is
    what tiling kills.

    > [PMPP](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)
    > Ch. 5 ("Memory architecture and data locality") opens with exactly this naive
    > matmul and its DRAM-traffic problem before introducing tiling.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        # DRAM traffic: naive (no reuse) vs tiled (reuse factor ~ T) for square NxN.
        _N = np.array([256, 512, 1024, 2048, 4096], dtype=float)
        _flops = 2 * _N**3
        _bytes_naive = 8 * _N**3                 # each output reads 2N values, 4B
        _T = 64.0                                # illustrative square tile side
        # tiled: each element of A,B loaded ~ N/T times instead of N times
        _bytes_tiled = 8 * _N**3 / _T

        _I_naive = _flops / _bytes_naive         # constant 0.25
        _I_tiled = _flops / _bytes_tiled         # ~ T/4, grows with reuse

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))

        _ax1.plot(_N, _bytes_naive / 1e9, "o-", color="#d65f5f",
                  label="naive (no reuse)")
        _ax1.plot(_N, _bytes_tiled / 1e9, "s-", color="#4c9f70",
                  label=f"tiled (T={int(_T)})")
        _ax1.plot(_N, _flops / 1e9, "^--", color="#999", label="FLOPs (work done)")
        _ax1.set_xlabel("matrix side N")
        _ax1.set_ylabel("billions (bytes or FLOPs)")
        _ax1.set_title("DRAM traffic vs work")
        _ax1.set_yscale("log")
        _ax1.legend(fontsize=8)
        _ax1.grid(True, alpha=0.15)

        _ax2.plot(_N, _I_naive, "o-", color="#d65f5f", label="naive  I = 1/4")
        _ax2.plot(_N, _I_tiled, "s-", color="#4c9f70",
                  label=f"tiled  I ~ T/4 = {int(_T)//4}")
        _ax2.set_xlabel("matrix side N")
        _ax2.set_ylabel("arithmetic intensity (FLOP/byte)")
        _ax2.set_title("Intensity: flat (naive) vs lifted (tiled)")
        _ax2.set_yscale("log")
        _ax2.legend(fontsize=8)
        _ax2.grid(True, alpha=0.15)

        _fig.suptitle("Tiling cuts DRAM traffic by ~T and raises intensity by ~T",
                      y=1.03, fontsize=11)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Blocking $C$ into tiles + the inner K-loop (the core idea)

    Here is the move that defines high-performance matmul. Stop computing one output
    element at a time. Instead, **block the output $C$ into tiles** of size
    $\text{BLOCK\_M} \times \text{BLOCK\_N}$, and assign **one program per output tile**.
    The grid is now 2-D over tiles:

    $$\text{grid} = \Big(\big\lceil M / \text{BLOCK\_M}\big\rceil,\;
      \big\lceil N / \text{BLOCK\_N}\big\rceil\Big).$$

    A single program owns one $\text{BLOCK\_M}\times\text{BLOCK\_N}$ patch of $C$. To
    fill it, it needs the matching rows of $A$ and columns of $B$ — but those are huge
    ($\text{BLOCK\_M}\times K$ and $K\times\text{BLOCK\_N}$). So it **walks the K
    dimension in chunks** of $\text{BLOCK\_K}$, the inner **K-loop**:

    $$C_{\text{tile}} \;=\; \sum_{k_0 = 0,\,\text{BLOCK\_K},\,2\,\text{BLOCK\_K},\dots}^{K}
      A[\,\text{tile rows},\, k_0{:}\,k_0{+}\text{BLOCK\_K}\,]\;
      B[\,k_0{:}\,k_0{+}\text{BLOCK\_K},\, \text{tile cols}\,].$$

    At each step: load a small $\text{BLOCK\_M}\times\text{BLOCK\_K}$ tile of $A$ and a
    small $\text{BLOCK\_K}\times\text{BLOCK\_N}$ tile of $B$ into fast on-chip memory
    (SRAM / registers), do a **tiny tile-matmul** into an accumulator that lives in
    registers, advance the K pointers, and repeat. The accumulator never touches DRAM
    until the loop ends — then you write the whole $C$ tile **once**.

    ```python
    # Tiled matmul SKELETON (illustrative — NOT the e07 solution; body is yours)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # this tile's rows of C
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # this tile's cols of C
    offs_k = tl.arange(0, BLOCK_K)                      # the K window, slides each step

    # 1. set up an fp32 accumulator that stays in registers across the whole K-loop
    #    -> tl.zeros((BLOCK_M, BLOCK_N), ...)
    ...
    # 2. build the initial A and B pointer tiles by broadcasting offs_m/offs_k and
    #    offs_k/offs_n against the right strides
    ...
    # 3. the K-loop: for k0 in range(0, K, BLOCK_K):
    #      - load a BLOCK_M x BLOCK_K tile of A and a BLOCK_K x BLOCK_N tile of B
    #        (mask the ragged last K-chunk)
    #      - accumulate the tile-matmul into acc via tl.dot(...)
    #      - advance the A and B pointers along K
    ...
    # 4. build c_ptrs from offs_m/offs_n, mask BOTH the M and N edges, and store acc ONCE
    ...
    ```

    The skeleton hands you only the *frame* — 2-D pids and the broadcast row/col offsets.
    It deliberately leaves the **accumulator**, the **pointer arithmetic**, the entire
    **K-loop body** (the masked loads, `tl.dot`, and pointer advances), and the **final
    masked store** for you. Those are the crux of `e07`.

    > [Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
    > builds precisely this kernel (and adds the L2-cache-friendly "group-major"
    > program ordering on top).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Why tiling raises arithmetic intensity (the payoff)

    Now the punchline — and the single most important quantitative idea in this module.
    Tie it straight back to the `0d` roofline.

    The win is **reuse**. When you load one $\text{BLOCK\_M}\times\text{BLOCK\_K}$ tile
    of $A$ into SRAM, you do **not** use it for a single output element — you use it for
    **all $\text{BLOCK\_N}$ columns** of the output tile. Likewise each loaded tile of
    $B$ feeds **all $\text{BLOCK\_M}$ rows**. Every byte you pull from DRAM now does
    *many* FLOPs before it's discarded, instead of one. That is the entire game.

    Count it for square tiles of side $T$ (so $\text{BLOCK\_M}=\text{BLOCK\_N}
    =\text{BLOCK\_K}=T$), square $N\times N$ matmul. The number of output tiles is
    $(N/T)^2$, and each runs a K-loop of $N/T$ steps. Each step loads two $T\times T$
    tiles = $2T^2$ elements $= 8T^2$ bytes (fp32). Total DRAM traffic:

    $$\text{bytes} \;\approx\; \underbrace{(N/T)^2}_{\text{output tiles}} \cdot
      \underbrace{(N/T)}_{\text{K-steps}} \cdot \underbrace{8T^2}_{\text{2 tiles/step}}
      \;=\; \frac{8N^3}{T}.$$

    The FLOPs are unchanged at $2N^3$. So the arithmetic intensity becomes

    $$\boxed{\,I_{\text{tiled}} \;=\; \frac{2N^3}{8N^3 / T} \;=\; \frac{T}{4}
      \;=\; O(T)\,}$$

    Compare with $I_{\text{naive}} = \tfrac14 = O(1)$. **Tiling multiplies intensity by
    the tile side $T$.** A naive matmul sits at $I=0.25$, welded to the memory roof. A
    $T=64$ tile lifts it to $I=16$; a $T=128$ tile to $I=32$ — and on the `0d` roofline
    that dot **slides right**, off the slanted bandwidth roof, climbing toward the flat
    compute ceiling. Cross the ridge point ($I_{\text{ridge}} = P_{\text{peak}}/B
    \approx 44\text{e}12 / 896\text{e}9 \approx 49$ FLOP/byte) and you've gone from
    memory-bound to **compute-bound** — the regime where tensor cores and lower
    precision (Parts 2–3) become the levers that matter.

    Bigger tiles → more reuse → higher intensity → faster. The only thing stopping you
    from $T = \infty$ is that the A tile, B tile, and accumulator must *fit* in SRAM and
    registers — which is exactly the **occupancy** budget from `0d`. Tiling is the trade:
    spend on-chip memory to buy arithmetic intensity. The interactive below makes this
    concrete.

    > Simon Boehm's
    > ["How to optimize a CUDA matmul kernel for cuBLAS-like performance"](https://siboehm.com/articles/22/CUDA-MMM)
    > walks the same intensity argument step by step from a naive kernel up to a tiled
    > one, with measured numbers; the
    > [Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
    > is the framework-level twin.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        # "Animated-style" sweep: the C tile stays FIXED while an A tile sweeps
        # left->right over K and a B tile sweeps top->bottom over K. Each panel is
        # one K-step; the accumulator fills in.
        _M, _K, _N = 6, 12, 6      # logical element grids (small for clarity)
        _T = 2                     # BLOCK size in this picture
        _n_ksteps = _K // _T       # number of K-loop iterations = panels (6)

        _fig, _axes = plt.subplots(2, 3, figsize=(11, 6.4))
        _axes = _axes.ravel()

        # fixed output tile = rows 2..4, cols 2..4 of C
        _ci0, _cj0 = 2, 2

        _hot_a = "#e0a458"     # current A tile
        _hot_b = "#5b8def"     # current B tile
        _hot_c = "#d65f5f"     # the fixed destination C tile
        _done = "#4c9f70"      # accumulator coverage so far

        for _step in range(_n_ksteps):
            _ax = _axes[_step]
            _ax.set_xlim(-_K - 1, _N + 1)
            _ax.set_ylim(-_M - 1, _K + 1)
            _ax.set_aspect("equal")
            _ax.axis("off")

            # --- draw A to the LEFT of C (A is M x K), spanning x in [-K, 0)
            for _i in range(_M):
                for _j in range(_K):
                    _x = -_K + _j
                    _y = -_i
                    _face = "#f3f4f6"
                    # current A tile: rows of C-tile, K-window of this step
                    if (_ci0 <= _i < _ci0 + _T) and (_step * _T <= _j < _step * _T + _T):
                        _face = _hot_a
                    _ax.add_patch(mpatches.Rectangle((_x, _y - 1), 1, 1,
                                  facecolor=_face, edgecolor="#cccccc", linewidth=0.4))

            # --- draw B ABOVE C (B is K x N), spanning y in (0, K]
            for _i in range(_K):
                for _j in range(_N):
                    _x = _j
                    _y = _K - _i
                    _face = "#f3f4f6"
                    # current B tile: K-window of this step, cols of C-tile
                    if (_step * _T <= _i < _step * _T + _T) and (_cj0 <= _j < _cj0 + _T):
                        _face = _hot_b
                    _ax.add_patch(mpatches.Rectangle((_x, _y - 1), 1, 1,
                                  facecolor=_face, edgecolor="#cccccc", linewidth=0.4))

            # --- draw C (M x N) at origin, x in [0, N), y in [-M, 0)
            for _i in range(_M):
                for _j in range(_N):
                    _x = _j
                    _y = -_i
                    _face = "#f3f4f6"
                    if (_ci0 <= _i < _ci0 + _T) and (_cj0 <= _j < _cj0 + _T):
                        # accumulator: shade by how much of K has been summed so far
                        _frac = (_step + 1) / _n_ksteps
                        _face = _done if _step < _n_ksteps - 1 else _hot_c
                    _ax.add_patch(mpatches.Rectangle((_x, _y - 1), 1, 1,
                                  facecolor=_face, edgecolor="#cccccc", linewidth=0.4))

            _ax.text(-_K, 1.2, "A", fontsize=11, weight="bold", color="#b07a30")
            _ax.text(0, _K + 0.4, "B", fontsize=11, weight="bold", color="#3a63b8")
            _ax.text(_N - 0.4, -_M - 0.4, "C", fontsize=11, weight="bold",
                     color="#b03a3a")
            _ax.set_title(f"K-step {_step + 1}/{_n_ksteps}  "
                          f"(k = {_step * _T}:{_step * _T + _T})", fontsize=9)

        _legend = [
            mpatches.Patch(color=_hot_a, label="A tile (sweeps right over K)"),
            mpatches.Patch(color=_hot_b, label="B tile (sweeps down over K)"),
            mpatches.Patch(color=_done, label="C accumulator (filling)"),
            mpatches.Patch(color=_hot_c, label="C tile (final, written once)"),
        ]
        _fig.legend(handles=_legend, loc="lower center", ncol=4, fontsize=8,
                    bbox_to_anchor=(0.5, -0.02))
        _fig.suptitle(
            "One C tile stays FIXED while A and B tiles stream over the K dimension",
            y=1.0, fontsize=12)
        _fig.tight_layout(rect=(0, 0.03, 1, 0.98))
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The climax: slide the tile along the roofline

    Now make the payoff interactive. Pick a **tile side $T$** and watch two things move
    together: the arithmetic intensity it buys, and where the resulting dot lands on the
    *same* `0d` roofline (896 GB/s memory slope, ~44 TFLOP/s compute ceiling).

    The intensity uses the exact square-tile result derived above, $I = T/4$. (The fully
    general rectangular form, when you allow $\text{BLOCK\_M}$ and $\text{BLOCK\_N}$ to
    differ, is the harmonic-mean expression $I \approx \tfrac{2}{1/T_m +
    1/T_n}$ FLOP/element — for square tiles $T_m = T_n = T$ that is $T$ FLOP/element, and
    dividing by 4 bytes (fp32) gives $T/4$ FLOP/byte.) Drag $T$ from tiny to large and watch the dot climb the memory
    roof toward the ridge. At the slider's max ($T=128 \Rightarrow I=32$) you're a $128\times$
    above naive but still short of the ridge ($I\approx49$) — tiles bigger than SRAM can
    hold would carry you across into the compute-bound regime. The whole optimization story
    is in that one sliding dot.
    """)
    return


@app.cell
def _(mo):
    tile_slider = mo.ui.slider(start=8, stop=128, step=8, value=32,
                               label="tile side T  (BLOCK_M = BLOCK_N = BLOCK_K)")
    tile_slider
    return (tile_slider,)


@app.cell
def _(tile_slider):
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        # SAME hardware numbers as 0d's roofline, on purpose.
        _B = 896e9              # bytes/s  (5070 Ti DRAM bandwidth)
        _P_PEAK = 44e12         # FLOP/s   (illustrative FP32 peak, matches 0d)
        _I_ridge = _P_PEAK / _B

        _T = int(tile_slider.value)
        # square-tile intensity derived in section 4:  I = T / 4  (fp32)
        _I_tiled = _T / 4.0
        _I_naive = 0.25
        _perf = min(_P_PEAK, _B * _I_tiled)
        _regime = "memory-bound" if _I_tiled < _I_ridge else "compute-bound"
        _frac_peak = _perf / _P_PEAK

        _I = np.logspace(-2, 3, 400)
        _roof = np.minimum(_P_PEAK, _B * _I)

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                          gridspec_kw={"width_ratios": [1, 1.25]})

        # --- left: intensity vs tile side, current T marked
        _Ts = np.arange(8, 129, 8)
        _ax1.plot(_Ts, _Ts / 4.0, "-", color="#4c9f70", linewidth=2,
                  label="tiled  I = T/4")
        _ax1.axhline(_I_naive, color="#d65f5f", linestyle="--", linewidth=1.5,
                     label="naive  I = 1/4")
        _ax1.axhline(_I_ridge, color="#999", linestyle=":", linewidth=1.2,
                     label=f"ridge  I = {_I_ridge:.0f}")
        _ax1.axvline(_T, color="#333", linestyle="-", linewidth=0.8, alpha=0.5)
        _ax1.scatter([_T], [_I_tiled], color="#4c9f70", s=80, zorder=5)
        _ax1.set_xlabel("tile side T")
        _ax1.set_ylabel("arithmetic intensity I (FLOP/byte)")
        _ax1.set_title(f"T={_T}  ->  I = {_I_tiled:.1f}  "
                       f"({_I_tiled / _I_naive:.0f}x naive)")
        _ax1.legend(fontsize=8, loc="upper left")
        _ax1.grid(True, alpha=0.15)

        # --- right: the 0d roofline with the current tile's dot on it
        _ax2.plot(_I, _roof, color="#333", linewidth=2.2, zorder=3)
        _ax2.axvline(_I_ridge, color="#999", linestyle=":", linewidth=1.2)
        _ax2.text(_I_ridge * 1.1, _B * 0.02, f"ridge\nI={_I_ridge:.0f}",
                  color="#666", fontsize=8)

        # naive dot (stuck) for reference
        _ax2.scatter([_I_naive], [min(_P_PEAK, _B * _I_naive)], color="#d65f5f",
                     s=55, zorder=5)
        _ax2.annotate("naive\n(stuck)", (_I_naive, _B * _I_naive),
                      textcoords="offset points", xytext=(4, 8), fontsize=7,
                      color="#d65f5f")
        # current tiled dot
        _ax2.scatter([_I_tiled], [_perf], color="#4c9f70", s=110, zorder=6)
        _ax2.annotate(f"T={_T}\nI={_I_tiled:.0f}\n{_perf / 1e12:.1f} TFLOP/s",
                      (_I_tiled, _perf), textcoords="offset points",
                      xytext=(8, -28), fontsize=8, color="#2e6b48")

        _ax2.set_xscale("log")
        _ax2.set_yscale("log")
        _ax2.set_xlabel("operational intensity  I  (FLOP / byte)")
        _ax2.set_ylabel("attainable performance (FLOP/s)")
        _ax2.set_ylim(1e10, _P_PEAK * 3)
        _ax2.set_title(f"{_regime}:  {_frac_peak:.0%} of the ~44 TFLOP/s roof")
        _ax2.grid(True, which="both", alpha=0.15)

        _fig.suptitle("Turn the tile-size knob: the dot slides off the memory roof "
                      "toward compute", y=1.02, fontsize=11)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    - **2-D decomposition is *the* pattern for matrix kernels.** The 2-D broadcast +
      both-dims mask from section 1 recurs verbatim in matmul and every matrix kernel after.
    - **Tiling is the universal lever to raise intensity.** Any time a kernel reloads
      the same data from DRAM, blocking it into reused tiles raises $I$ and walks the dot
      off the memory roof. Matmul is the cleanest example, not the only one.
    - **Tile size trades occupancy for reuse.** Bigger tiles buy more intensity ($I = T/4$)
      but consume more SRAM and registers — straight back to the three limiters of `0d`.
      The "best" tile is the largest that still keeps enough warps resident.
    - **This exact structure recurs.** The fixed-output-tile + streaming-K-loop +
      register accumulator shape is the skeleton of **flash attention** (`2b` — where the
      "K-loop" streams keys/values and the accumulator is the running softmax) and of the
      **CUDA tiled matmul** (`3b` — same idea, with shared memory and `__syncthreads()`
      made explicit). Learn it once here.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Now go write it

    This is a big lecture, so it unlocks **two** exercises — the 2-D warm-up and then
    the main event:

    ```bash
    python -m harness.runner e06 --watch   # transpose (the 2-D warmup), metric bandwidth
    python -m harness.runner e07 --watch   # tiled matmul, metric flops
    ```

    Write `e06` first — get the 2-D indexing and the both-dims mask under your fingers
    on a kernel with no math to distract you. Then `e07`: the grid over output tiles, the
    sliding K-loop, the register accumulator, and the single masked store at the end.
    Watch the `--watch` metric for `e07` (TFLOP/s) climb toward the compute roof as your
    tile sizes grow — the roofline from `0d`, live.

    (If a stub isn't on disk yet, treat the command as a forward pointer — the kernel
    you'll write is exactly the structure in §3.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [1D: Softmax](../1d_softmax/) &nbsp;|&nbsp; Next: [1F: Fused Norms](../1f_fused_norms/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
