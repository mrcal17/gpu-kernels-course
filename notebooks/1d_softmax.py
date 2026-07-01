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
    # 1D: Softmax

    > *"Softmax is a reduction wearing a map's coat. Find the right constant to subtract,
    > and an operation that overflows becomes one that can't."*

    Softmax turns a row of raw scores into a probability distribution — and it is
    everywhere a network has to *choose*: the attention weights over keys, the class
    probabilities over logits. Mathematically it is trivial. Numerically it is a trap:
    written the obvious way, a single large score sends $e^x$ past the float32 ceiling
    and poisons the whole row with `inf`/`nan`.

    This lecture builds the **numerically-stable** softmax and shows why it is *exact*,
    not approximate. Then we frame it the way a kernel writer must: a **reduce** (the
    denominator) feeding a **map** (the divide), three logical passes over a row fused
    into one DRAM load. It is the bridge from the reductions of `1c` to the streaming
    online-softmax that powers Flash Attention in `2b`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. What softmax is, and where it lives

    Given a row of real numbers $x = (x_1, \dots, x_n)$, softmax maps it to a
    probability distribution:

    $$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j} e^{x_j}}.$$

    Every output is in $(0, 1)$ and the row sums to exactly 1. Exponentiation makes the
    values positive and monotonically amplifies gaps; the normalization makes them a
    distribution. You will meet it in two places constantly:

    - **Attention scores.** After $QK^\top/\sqrt{d}$ you softmax each row to get the
      weights that mix the values. This is the inner loop of every transformer — and the
      reason `2b` exists.
    - **Classifier logits.** The final layer emits one score per class; softmax turns
      them into class probabilities for the loss and the prediction.

    Look at the shape of the computation. The denominator $\sum_j e^{x_j}$ is a
    **reduction** — exactly the pattern from `1c`, an associative sum collapsing a row to
    one scalar. The output is then a per-element **map**: each $e^{x_i}$ divided by that
    one scalar. So softmax is a **reduce feeding a map**:

    $$\underbrace{Z = \sum_j e^{x_j}}_{\text{reduce (1C)}}
      \;\longrightarrow\;
      \underbrace{y_i = e^{x_i} / Z}_{\text{map}}.$$

    Hold that decomposition. Everything in this lecture is about computing $Z$ correctly
    and computing it *once*.

    > [PMPP](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)
    > Ch. 10 develops reductions; the
    > [Triton fused-softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
    > is the kernel this lecture unlocks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Why the naive version overflows

    Write the definition literally and you compute $e^{x_i}$ for every element *before*
    you normalize. That intermediate is the problem. In float32 the largest finite value
    is about $3.4\times10^{38}$, and

    $$e^{x} > 3.4\times10^{38} \quad\text{once}\quad x \gtrsim 88,$$

    because $e^{88} \approx 1.6\times10^{38}$ and $e^{89}$ already overflows. So a single
    logit above ~88 — utterly ordinary after an un-scaled matmul or a confident
    classifier — makes $e^{x_i}$ return `inf`.

    And `inf` is contagious. The denominator becomes `inf`; every output becomes
    `inf / inf = nan`. **One large element poisons the entire row.** The math says the
    answer is a clean distribution near a one-hot; the naive float32 evaluation says
    `nan`. The demo cell below shows it happening on a concrete row.

    > This is not a corner case. Attention logits and classifier logits routinely exceed
    > 88 in normal training, which is why *every* production softmax uses the trick in
    > §3 — it is not an optimization, it is correctness.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        print("=== Naive softmax overflows in float32 ===\n")
        print(f"float32 max  ~ {np.finfo(np.float32).max:.3e}")
        print(f"e^88         ~ {np.exp(np.float32(88)):.3e}  (still finite)")
        print(f"e^89         ~ {np.exp(np.float32(89)):.3e}  (overflowed)\n")

        # A perfectly ordinary row: a few modest scores and one confident logit.
        _x = np.array([1.0, 2.0, 3.0, 95.0], dtype=np.float32)
        print(f"row x = {_x}\n")

        def _naive_softmax(x):
            _e = np.exp(x)               # <-- the overflow happens here
            return _e / _e.sum()

        _e = np.exp(_x)
        print(f"naive  exp(x)      = {_e}")
        print(f"naive  sum(exp(x)) = {_e.sum()}")
        print(f"naive  softmax(x)  = {_naive_softmax(_x)}")
        print("\n-> one logit at 95 sent exp() to inf, and inf/inf = nan.")
        print("   The whole row is destroyed, not just the big element.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. The max-subtraction trick — and why it's *exact*

    The fix is one line: subtract the row max before exponentiating. Pick **any**
    constant $m$ and multiply numerator and denominator by $e^{-m}$:

    $$\text{softmax}(x)_i
      = \frac{e^{x_i}}{\sum_j e^{x_j}}
      = \frac{e^{x_i}\,e^{-m}}{\sum_j e^{x_j}\,e^{-m}}
      = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}.$$

    The $e^{-m}$ is a common factor top and bottom — it **cancels exactly**. This is an
    algebraic identity, true for every real $m$. It is **not** an approximation: the
    output is bit-for-bit the same distribution the math defines (up to the usual
    floating-point rounding, which the trick *reduces*, never introduces).

    Now choose the constant well. Take

    $$m = \max_j x_j.$$

    Then every shifted value $x_i - m \le 0$, so every exponent is non-positive and

    $$e^{x_i - m} \in (0, 1].$$

    No term can exceed 1, so the sum cannot overflow. And the largest element hits
    $x_i - m = 0$, giving $e^0 = 1$ exactly — so the denominator is **at least 1** and
    can never underflow to zero. Overflow is impossible from above; division-by-zero is
    impossible from below. The stable formula:

    $$\boxed{\;\text{softmax}(x)_i = \dfrac{e^{\,x_i - m}}{\displaystyle\sum_j e^{\,x_j - m}},
      \qquad m = \max_j x_j\;}$$

    The worked demo below runs both versions on the same row: naive returns `nan`, stable
    returns a clean distribution summing to 1, and on the small elements where naive
    *doesn't* overflow, the two agree to floating-point precision — confirming the
    identity.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        print("=== Max-subtraction trick: exact, and stable ===\n")

        def _naive_softmax(x):
            _e = np.exp(x)
            return _e / _e.sum()

        def _stable_softmax(x):
            _m = x.max()                 # the only change: subtract the row max
            _e = np.exp(x - _m)
            return _e / _e.sum()

        # --- Case A: a row that overflows the naive path -------------------
        _x = np.array([1.0, 2.0, 3.0, 95.0], dtype=np.float32)
        print(f"row x          = {_x}")
        print(f"  naive  softmax = {_naive_softmax(_x)}   <- destroyed")
        _s = _stable_softmax(_x)
        print(f"  stable softmax = {_s}")
        print(f"  stable sum     = {_s.sum():.6f}   <- valid distribution\n")

        # --- Case B: a tame row where naive does NOT overflow --------------
        # Here both must agree -> proves the trick is exact, not approximate.
        _y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        _a = _naive_softmax(_y)
        _b = _stable_softmax(_y)
        print(f"row y          = {_y}")
        print(f"  naive  softmax = {_a}")
        print(f"  stable softmax = {_b}")
        print(f"  max abs diff   = {np.abs(_a - _b).max():.3e}   "
              f"<- identical to FP precision")
        print("\n-> stable fixes the overflow AND matches naive where naive is valid.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. The fused kernel: three passes, one DRAM load

    Read the stable formula as an algorithm over a single row. It needs three logical
    passes:

    1. **max** — scan the row, find $m = \max_j x_j$. (a reduction)
    2. **exp + sum** — compute $e^{x_i - m}$ for each element and accumulate
       $Z = \sum_i e^{x_i - m}$. (a map + a reduction)
    3. **divide** — write $y_i = e^{x_i - m} / Z$. (a map)

    Done naively as three separate kernels, that is **three reads of the row from DRAM**
    (plus the writes). But softmax does almost no arithmetic per byte — one `exp`, a
    compare, an add, a divide. Its operational intensity is tiny, so by the roofline of
    `0d` it sits hard against the **memory roof**: softmax is *memory-bound*. The cost is
    DRAM traffic, and the win is **fewer passes over DRAM**.

    So we fuse. Following the
    [Triton fused-softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html),
    **each program handles one row.** It does **one** `tl.load` of the whole row into
    SRAM (registers / shared memory), runs all three passes on the resident copy, and
    does **one** `tl.store` of the result. One load, one store — versus the unfused
    version the row **reads drop 3×** (three passes → one), and since the write happens
    either way, **total DRAM traffic falls ~2×** (3 reads + 1 write → 1 read + 1 write).
    On a memory-bound op that traffic cut *is* the speedup.

    The skeleton — structure only; you fill in the details in the exercise:

    ```python
    # ONE program per row. row_len fits in one BLOCK_SIZE.
    @triton.jit
    def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride,
                       n_cols, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)                       # which row am I?

        # 1. ONE load: build col range, mask the ragged tail, and bring the whole
        #    row into SRAM. Pad masked lanes with the reduction identity for max
        #    (other=-inf) so they can't win the max or add to the sum.
        ...
        # 2. pass 1 -> row max via tl.max(..., axis=0)        (a reduction)
        ...
        # 3. pass 2 -> shifted exps tl.exp(x - m) and their sum tl.sum(..., axis=0)
        #    (a map + a reduction; every exponent <= 0  =>  e in (0, 1])
        ...
        # 4. pass 3 -> normalize: y = e / z                    (a map)
        ...
        # 5. ONE store of y back to out_ptr (masked)
        ...
    ```

    The body is left to you — that is the exercise. Everything above the numbered
    comments (one program per row, the `@triton.jit` signature) is the frame; filling in
    the load, the two reduction passes, the normalize, and the masked store is the crux.
    Also not spelled out: the launch grid (one program per row), how `BLOCK_SIZE` is
    chosen relative to `n_cols`, `num_warps`, and what happens when a row is wider than
    one block. The exercise harness measures your achieved bandwidth, so the structure
    above is the map, not the territory.

    > The `other=-inf` padding matters: masked-off lanes must not win the `tl.max`, and
    > $e^{-\infty} = 0$ contributes nothing to the sum. Pad the *reduction identity*, as
    > in `1c`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Teaser: single-pass online softmax

    The fused kernel above still makes **two reduction passes** over the resident row —
    one for the max, one for the sum. Milakov & Gimelshein showed you can do both in a
    **single streaming pass**, never holding the whole row, by maintaining a running max
    $m$ and a running sum $\ell$ together.

    The trick is **rescaling on a new max.** Suppose your running sum $\ell$ was
    accumulated relative to the old max $m_\text{old}$. When a new element pushes the max
    up to $m' > m_\text{old}$, every term already in $\ell$ was scaled by the wrong
    constant — but you can correct them all at once with a single multiply:

    $$\ell' = \ell \cdot e^{\,m_\text{old} - m'} + e^{\,x_\text{new} - m'}.$$

    The factor $e^{m_\text{old} - m'} \le 1$ re-references the old partial sum to the new
    max, then you add the new term. One pass, exact result. This streaming rescale is the
    heart of **Flash Attention** (`2b`): it lets attention compute softmax over key
    blocks it never fully materializes, tiling the row through SRAM. Keep the formula in
    your pocket — you will rederive it there.

    > [Milakov & Gimelshein, "Online normalizer calculation for softmax" (2018)](https://arxiv.org/abs/1805.02867)
    > is the one-pass algorithm; [Flash Attention (Dao et al. 2022)](https://arxiv.org/abs/2205.14135)
    > builds tiled attention on top of it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Visualizing the overflow

    Take a row whose largest logit is big. Plot, on a **log-y** axis, the naive
    $e^{x_i}$ against the stable $e^{x_i - m}$. The naive curve climbs through the float32
    ceiling (~$3.4\times10^{38}$, the dashed line) and the big element's bar shoots off
    the top into `inf`; the stable curve stays bounded in $(0, 1]$ with its largest term
    pinned exactly at 1. Same distribution either way — only one of them is computable.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        # A row with modest scores and one confident logit at 95.
        _x = np.array([1.0, 12.0, 30.0, 60.0, 95.0], dtype=np.float64)
        _labels = [f"x={v:.0f}" for v in _x]
        _idx = np.arange(len(_x))

        _f32_max = float(np.finfo(np.float32).max)   # ~3.4e38 (as float64: no overflow below)

        _naive = np.exp(_x)                          # huge; last entry overflows f32
        _stable = np.exp(_x - _x.max())             # in (0, 1], last entry = 1

        # Mark which naive terms exceed the float32 ceiling (would be inf in f32).
        _overflow = _naive > _f32_max
        # For plotting, cap the bar height at the ceiling so the spike is visible.
        _naive_plot = np.minimum(_naive, _f32_max)

        _fig, _ax = plt.subplots(figsize=(8.5, 4.4))
        _w = 0.38
        _ax.bar(_idx - _w / 2, _naive_plot, width=_w, color="#d65f5f",
                label=r"naive  $e^{x_i}$")
        _ax.bar(_idx + _w / 2, _stable, width=_w, color="#4c9f70",
                label=r"stable  $e^{x_i - m}$")

        _ax.axhline(_f32_max, color="#333", linestyle="--", linewidth=1.3)
        _ax.text(len(_x) - 1.0, _f32_max * 2.5,
                 "float32 max ~ 3.4e38", color="#333", fontsize=8, ha="right")

        # Annotate the element that overflows.
        for _i in _idx[_overflow]:
            _ax.annotate("INF", (_i - _w / 2, _f32_max),
                         textcoords="offset points", xytext=(0, 6),
                         ha="center", color="#d65f5f", fontsize=9, weight="bold")

        _ax.axhline(1.0, color="#4c9f70", linestyle=":", linewidth=1.0)
        _ax.text(0.0, 1.5, "stable terms in (0, 1]", color="#2e6b48", fontsize=8)

        _ax.set_yscale("log")
        _ax.set_xticks(_idx)
        _ax.set_xticklabels(_labels)
        _ax.set_xlabel("element of the row")
        _ax.set_ylabel("exponential value (log scale)")
        _ax.set_title("Naive exp overflows float32; max-subtraction stays bounded")
        _ax.legend(loc="center left", fontsize=9)
        _ax.set_ylim(1e-1, _f32_max * 30)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Make the threshold tangible

    The slider sets the magnitude of the **largest logit** in a row; the rest of the row
    is small fixed scores. The plot tracks the **max naive `exp` value** against the
    float32 ceiling — once the big logit crosses ~88 the naive term is `inf` (annotated)
    — while the stable path's largest term sits at exactly 1 for *any* magnitude. The
    panel beside it confirms the **stable softmax distribution is unchanged** as you drag:
    the math never moved, only the naive arithmetic broke.
    """)
    return


@app.cell
def _(mo):
    max_logit = mo.ui.slider(start=0, stop=200, step=2, value=40,
                             label="magnitude of the largest logit in the row")
    max_logit
    return (max_logit,)


@app.cell
def _(max_logit):
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        _f32_max = float(np.finfo(np.float32).max)   # ~3.4e38
        _OVERFLOW_X = 88.7                           # where e^x crosses f32 max

        # Row: three small fixed scores plus one big logit set by the slider.
        _big = float(max_logit.value)
        _x = np.array([1.0, 2.0, 3.0, _big], dtype=np.float64)

        # Naive: largest exp term (in float64 so we can see how far past f32 it is).
        _naive_max = float(np.exp(_x.max()))
        _naive_overflows = _naive_max > _f32_max

        # Stable softmax: always valid, largest exp term is exactly 1.
        _s = np.exp(_x - _x.max())
        _stable_softmax = _s / _s.sum()

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10.0, 3.9),
                                          gridspec_kw={"width_ratios": [1.25, 1]})

        # --- Left: naive max exp vs the ceiling, over the full slider range ---
        _grid = np.linspace(0, 200, 400)
        _naive_curve = np.exp(_grid)
        _ax1.plot(_grid, _naive_curve, color="#d65f5f", linewidth=2,
                  label=r"naive max $e^{x}$")
        _ax1.axhline(_f32_max, color="#333", linestyle="--", linewidth=1.3,
                     label="float32 max")
        _ax1.axhline(1.0, color="#4c9f70", linestyle=":", linewidth=1.3,
                     label=r"stable max $e^{x-m}=1$")
        _ax1.axvline(_OVERFLOW_X, color="#999", linestyle=":", linewidth=1.0)
        _ax1.scatter([_big], [min(_naive_max, _f32_max * 30)],
                     color="#d65f5f", s=70, zorder=5)
        if _naive_overflows:
            _ax1.annotate("INF", (_big, _f32_max),
                          textcoords="offset points", xytext=(0, 10),
                          ha="center", color="#d65f5f", fontsize=11, weight="bold")
        _ax1.set_yscale("log")
        _ax1.set_xlabel("largest logit  x")
        _ax1.set_ylabel("max exp value (log scale)")
        _ax1.set_ylim(1e-1, _f32_max * 1e3)
        _state = "OVERFLOWS (inf)" if _naive_overflows else "finite"
        _ax1.set_title(f"largest logit = {_big:.0f}  ->  naive {_state}")
        _ax1.legend(loc="lower right", fontsize=7.5)

        # --- Right: stable softmax distribution (unchanged as slider moves) ---
        _idx = np.arange(len(_x))
        _ax2.bar(_idx, _stable_softmax, color="#4c9f70")
        _ax2.set_ylim(0, 1.05)
        _ax2.set_xticks(_idx)
        _ax2.set_xticklabels([f"{v:.0f}" for v in _x])
        _ax2.set_xlabel("logit value")
        _ax2.set_ylabel("stable softmax prob")
        _ax2.set_title(f"stable dist (sums to {_stable_softmax.sum():.3f})")

        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    - **Always subtract the max — it's free and exact.** Subtracting $m=\max_j x_j$ is an
      algebraic identity, not an approximation. It costs one extra reduction you were
      already paying for (you scan the row anyway) and it is the difference between a
      valid distribution and `nan`.
    - **Fuse the three passes to cut DRAM traffic.** Softmax is memory-bound (`0d`): one
      `exp` per byte, no reuse. Loading the row once into SRAM and doing max, exp+sum,
      and divide on the resident copy cuts DRAM passes from three to one — and on a
      memory-bound op, fewer passes *is* the speedup.
    - **The online/streaming rescale generalizes.** The running-max correction factor
      $e^{m_\text{old}-m'}$ collapses max and sum into a single pass and is the exact
      mechanism Flash Attention (`2b`) uses to softmax over blocks it never fully
      materializes.
    - **Reductions are the substrate.** Both the max and the sum are the associative
      reductions from `1c`. Get those right — including padding masked lanes with the
      reduction identity ($-\infty$ for max, $0$ for sum) — and softmax falls out.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Now go write it

    You have the structure, the stability proof, and the reduction primitives from `1c`.
    Time to fuse them into one kernel. Open the terminal and write the stable, fused
    softmax — one program per row, one load, one store — and watch the harness report
    your achieved memory bandwidth:

    ```bash
    python -m harness.runner e05 --watch
    ```

    `e05` is softmax, scored on **bandwidth** (it's memory-bound, so GB/s vs the memory
    roof from `0d` is the scoreboard). If the stub isn't on disk yet, treat this as a
    forward pointer — the command is the one you'll run once it lands.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [1C: Reductions](../1c_reductions/) &nbsp;|&nbsp; Next: [1E: Tiling & Matmul](../1e_tiling_matmul/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
