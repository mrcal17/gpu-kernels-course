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
    # 1F: Fused Norms

    > *"A norm does almost no math. It just moves your activations through DRAM —
    > so the only thing worth optimizing is how many times it touches them."*

    Every transformer layer normalizes its activations, often twice. LayerNorm and
    RMSNorm are tiny operations — a mean, a variance, a divide, a scale — but they run
    on the same enormous tensors as everything else, and they run *constantly*. Done
    naively, a norm reloads its input from DRAM several times. Done right, it touches
    each row **once**.

    This is the lecture where **fusion stops being a slogan and becomes arithmetic.**
    The shape is the reduce-then-map you already met in `1c` (reductions) and `1d`
    (softmax): collapse a row to a couple of scalars, then sweep back over the row
    applying them. Because norms are firmly **memory-bound** (`0d`'s left-of-the-ridge
    regime), the entire performance story is **DRAM passes**, and fusing collapses
    several passes into one.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. LayerNorm: reduce, then map

    Take one row $x$ of length $N$ — the feature/hidden dimension of a single token.
    LayerNorm standardizes that row to zero mean and unit variance, then applies a
    learnable per-feature affine. Four steps:

    $$\mu = \frac{1}{N}\sum_{i} x_i
      \qquad\qquad
      \sigma^2 = \frac{1}{N}\sum_{i} (x_i - \mu)^2$$

    $$\hat x_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
      \qquad\qquad
      y_i = \gamma_i\,\hat x_i + \beta_i$$

    Read each piece for what it costs the hardware:

    - **Two reductions.** $\mu$ is a sum over the row; $\sigma^2$ is a second sum (of
      squared deviations) that *needs $\mu$ first*. Both collapse $N$ values to one
      scalar — the exact reduction pattern from `1c`, run per row.
    - **Epsilon.** The $\epsilon$ inside the square root (typically $10^{-5}$) keeps the
      denominator away from zero. When a row is nearly constant, $\sigma^2 \to 0$ and
      $1/\sqrt{\sigma^2}$ would explode or divide by zero; $\epsilon$ bounds it. It is
      pure numerical hygiene, not statistics.
    - **The affine $\gamma, \beta$.** Two learned vectors of length $N$, shared across
      all rows. $\gamma$ rescales and $\beta$ shifts each *feature* — they let the
      network undo the normalization if it wants to. Tiny, broadcast loads: $N$ values
      each, reused by every row in the batch.

    Structurally this is **REDUCE → MAP**. The reduce phase computes $(\mu, \sigma^2)$
    over the row; the map phase visits every element once to compute $\hat x_i$ and then
    $y_i$. Same skeleton as softmax (`1d`): softmax reduces a max and a sum, then maps
    an exponential-and-divide; LayerNorm reduces a mean and a variance, then maps a
    standardize-and-affine. If you can write one, you can write the other.

    > [PMPP](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)
    > Ch. 10 covers the reduction primitive these norms are built on.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        _rng = np.random.default_rng(0)
        _x = _rng.normal(loc=3.0, scale=2.0, size=8).astype(np.float64)
        _gamma = _rng.normal(size=8)
        _beta = _rng.normal(size=8)
        _eps = 1e-5

        # Hand-rolled LayerNorm: reduce (mean, var) -> map (normalize, affine).
        _mu = _x.mean()
        _var = ((_x - _mu) ** 2).mean()
        _xhat = (_x - _mu) / np.sqrt(_var + _eps)
        _y = _gamma * _xhat + _beta

        print("=== LayerNorm on one row (N=8) ===")
        print(f"  x      = {np.array2string(_x, precision=3)}")
        print(f"  mu     = {_mu:.4f}   (reduction 1: mean)")
        print(f"  var    = {_var:.4f}   (reduction 2: variance, needs mu)")
        print(f"  xhat   = {np.array2string(_xhat, precision=3)}")
        print(f"           mean(xhat)={_xhat.mean():+.2e}  std(xhat)={_xhat.std():.4f}")
        print(f"  y      = {np.array2string(_y, precision=3)}   (map: gamma*xhat + beta)")
        print("\n  xhat has ~0 mean and ~unit std by construction; gamma/beta re-color it.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. RMSNorm: the cheaper cousin

    RMSNorm drops the parts of LayerNorm that cost the most and keep the least. No mean
    subtraction, no $\beta$ shift — just divide each element by the **root-mean-square**
    of the row and scale:

    $$y_i = \gamma_i \cdot \frac{x_i}{\sqrt{\dfrac{1}{N}\sum_{j} x_j^2 + \epsilon}}$$

    The denominator is one reduction — the sum of *squares* — instead of LayerNorm's
    two. Compare the work per row:

    | | reductions | per-element FLOPs (map) | learned params |
    |---|---|---|---|
    | **LayerNorm** | 2 (mean, then variance) | subtract $\mu$, multiply by rstd, $\times\gamma$, $+\beta$ | $\gamma, \beta$ (both length $N$) |
    | **RMSNorm**   | 1 (sum of squares)      | multiply by rrms, $\times\gamma$ | $\gamma$ (length $N$) |

    One reduction instead of two, one affine multiply instead of a multiply-add, and
    half the learned parameters. The arithmetic-intensity story barely changes — both
    are still a handful of FLOPs per element, both still memory-bound — but RMSNorm has
    *less serial dependency* in the reduce phase (you don't have to finish $\mu$ before
    you can start the second reduction), which makes it a touch friendlier to fuse and
    to keep in registers.

    RMSNorm is the default normalizer in most modern LLMs — LLaMA, Mistral, Qwen,
    Gemma all use it — precisely because it is "almost LayerNorm" at lower cost and no
    measurable quality loss.

    > Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019),
    > [arxiv.org/abs/1910.07467](https://arxiv.org/abs/1910.07467) — the paper that
    > showed the mean-centering term contributes little.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        _rng = np.random.default_rng(1)
        _x = _rng.normal(loc=0.5, scale=1.5, size=8).astype(np.float64)
        _gamma = _rng.normal(size=8)
        _eps = 1e-5

        # RMSNorm: one reduction (mean of squares) -> map (scale by 1/rms, then gamma).
        _ms = (_x ** 2).mean()              # the single reduction
        _rrms = 1.0 / np.sqrt(_ms + _eps)   # reciprocal RMS
        _y = _gamma * (_x * _rrms)

        print("=== RMSNorm on one row (N=8) ===")
        print(f"  x      = {np.array2string(_x, precision=3)}")
        print(f"  mean(x^2) = {_ms:.4f}   (the ONE reduction)")
        print(f"  rms       = {np.sqrt(_ms):.4f}   ->  1/rms = {_rrms:.4f}")
        print(f"  y      = {np.array2string(_y, precision=3)}   (map: gamma * x / rms)")
        print("\n  No mu, no beta: x is scaled toward unit RMS, then re-scaled by gamma.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Why fuse: it's all about DRAM passes

    Here is the whole point of this lecture. A norm does almost no arithmetic — a few
    FLOPs per element — so its operational intensity is tiny and it lives far left on
    the roofline (`0d`): **memory-bound**. When a kernel is memory-bound, its runtime is
    *literally* set by how many bytes it moves through DRAM. So the optimization is not
    "do less math" — it's **"touch the tensor fewer times."**

    Picture an activation tensor $X$ of shape $(M, N)$ where $M = B\cdot S$ rows (batch
    × sequence) and $N$ features, in fp32 (4 bytes/element). Now compare two ways to
    compute LayerNorm.

    **Unfused — a chain of separate kernels.** A naive PyTorch-style decomposition
    might launch:

    1. a kernel to compute per-row $\mu$ (reads $X$),
    2. a kernel to compute per-row $\sigma^2$ (reads $X$ again),
    3. a kernel to normalize $\hat x = (x-\mu)/\sqrt{\sigma^2+\epsilon}$ (reads $X$,
       writes $\hat X$),
    4. a kernel for the affine $y = \gamma\hat x + \beta$ (reads $\hat X$, writes $Y$).

    Every one of those kernels streams the *entire* tensor from DRAM and (for 3–4)
    streams it back. That is roughly **3–4 reads + 2 writes** of an $M\times N$ tensor —
    five-plus full passes — plus a kernel launch each (the per-launch overhead from
    `0b`). The intermediate $\hat X$ is born in DRAM and immediately re-read, pure waste.

    **Fused — one kernel.** Assign one program (block) to one row. It does:

    - **one `tl.load`** of the row into SRAM/registers,
    - both reductions, the rsqrt, the normalize, and the $\gamma/\beta$ affine *in
      SRAM*,
    - **one `tl.store`** of the finished row.

    That is exactly **1 read + 1 write** of $X$, plus a one-time trickle for $\gamma$
    and $\beta$ (each only $N$ values, cached and reused by every row). The
    intermediates never leave the chip. One launch, not four.

    Count the traffic. Unfused moves on the order of $5\times$ the tensor; fused moves
    $2\times$. Since runtime $\propto$ DRAM bytes for a memory-bound op, fusing here is a
    **~2.5× speedup for free** — no better algorithm, just fewer trips.

    $$\boxed{\;\text{speedup} \;\approx\; \frac{\text{unfused passes}}{\text{fused passes}}
      \;=\; \frac{\text{(reads}+\text{writes) unfused}}{2}\;}$$

    Tie it back: this is `0d`'s memory roof (you can't beat $B \cdot I$, so shrink the
    bytes) and `0b`'s rule (*fuse to amortize launch + traffic*), made quantitative for
    the one operation you'll meet in every layer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The fused kernel, in skeleton

    The structure of the fused LayerNorm — one program per row, load once, reduce in
    SRAM, store once. This is **illustrative only**: it shows the `@triton.jit` signature
    and where the load/store boundaries fall, but leaves the reduce → normalize → store
    body for you to write in the exercise (`e08` is forward-only — it matches
    `F.layer_norm`, no backward pass).

    ```python
    # ILLUSTRATIVE SKELETON — not the solution; the body is yours to fill in.
    @triton.jit
    def layernorm_fwd(X, Y, GAMMA, BETA, stride_row, N, eps, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)                 # one program == one row
        X += row * stride_row
        Y += row * stride_row

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        # 1. ONE read: load the row into SRAM (masked, other=0.0)
        ...
        # 2. reduction 1 -> mean = tl.sum(x) / N
        ...
        # 3. center the row (mask the tail back to 0), then
        #    reduction 2 -> var = tl.sum(xc*xc) / N, and rstd = 1/sqrt(var + eps)
        ...
        # 4. tiny broadcast loads of GAMMA and BETA, then the affine normalize in SRAM:
        #    y = xc * rstd * g + b
        ...
        # 5. ONE write: store y back to Y (masked)
        ...
    ```

    Everything between the load and the store happens on-chip. The numbered steps are the
    body you fill in; the harness scores the result on achieved bandwidth.

    > [Triton LayerNorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
    > builds exactly this kernel (plus the fused backward pass); the
    > [fused-softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
    > is the same one-row-per-program reduce→map pattern.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Welford's algorithm: a stable one-pass variance

    The two reductions in §1 are a **two-pass** variance: pass 1 finds $\mu$, then pass
    2 sums $(x_i-\mu)^2$. Correct, but it needs $\mu$ in hand before the second pass can
    start. The obvious one-pass shortcut uses

    $$\sigma^2 = \overline{x^2} - \bar x^2,$$

    accumulating $\sum x_i$ and $\sum x_i^2$ together in a single sweep. Fast — but
    numerically fragile: when the mean is large relative to the spread, $\overline{x^2}$
    and $\bar x^2$ are two big nearly-equal numbers, and subtracting them suffers
    **catastrophic cancellation** in fp32 (you lose the small true variance in the
    rounding error of the large operands). **Welford's online algorithm** gets a single,
    numerically-stable pass by updating a running mean and a running sum of squared
    deviations $M_2$ as each new $x$ arrives:

    $$\delta = x - \mu_{k-1},\quad
      \mu_k = \mu_{k-1} + \frac{\delta}{k},\quad
      M_{2,k} = M_{2,k-1} + \delta\,(x - \mu_k),\quad
      \sigma^2 = \frac{M_2}{N}.$$

    Each update folds one element into stable running statistics — no large sums to
    cancel — so it holds fp32 accuracy and never needs the whole row in memory at once
    (handy when a row is tiled across several `BLOCK_SIZE` chunks). It is the streaming
    analogue of the **online-softmax** rescaling trick from `1d`: there you carried a
    running max and a running denominator and corrected them as new blocks arrived; here
    you carry a running mean and $M_2$ and correct them the same way. Same idea —
    *combine partial statistics on the fly* — which is exactly what Flash Attention does
    for the softmax in `2b`.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        # A row with a LARGE mean and tiny spread: the cancellation trap.
        _rng = np.random.default_rng(7)
        _x32 = (1e4 + _rng.normal(scale=1e-2, size=4096)).astype(np.float32)
        _x64 = _x32.astype(np.float64)

        _truth = _x64.var()  # fp64 reference

        # Naive one-pass in fp32: E[x^2] - E[x]^2  (catastrophic cancellation).
        _mean32 = _x32.mean(dtype=np.float32)
        _msq32 = (_x32 * _x32).mean(dtype=np.float32)
        _naive = np.float32(_msq32 - _mean32 * _mean32)

        # Welford one-pass in fp32: running mean + M2.
        _mu = np.float32(0.0)
        _m2 = np.float32(0.0)
        for _k, _xi in enumerate(_x32, start=1):
            _delta = np.float32(_xi - _mu)
            _mu = np.float32(_mu + _delta / np.float32(_k))
            _m2 = np.float32(_m2 + _delta * np.float32(_xi - _mu))
        _welford = np.float32(_m2 / np.float32(len(_x32)))

        print("=== Variance of x ~ 1e4 + tiny noise (fp32), N=4096 ===")
        print(f"  fp64 reference        : {_truth:.6e}")
        print(f"  naive  E[x^2]-E[x]^2  : {_naive:.6e}   <- cancellation wrecks it")
        print(f"  Welford running M2    : {_welford:.6e}   <- holds fp32 accuracy")
        print(f"\n  naive   relative error: {abs(_naive   - _truth)/_truth:8.2%}")
        print(f"  Welford relative error: {abs(_welford - _truth)/_truth:8.2%}")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Visualizing the traffic saved

    The bar chart below counts **DRAM passes** for a realistic activation tensor and
    converts them to bytes and an estimated time at your card's **896 GB/s**. The
    unfused LayerNorm pays for ~5 passes (multiple reads of $X$ plus an intermediate
    written and re-read); the fused kernel pays for 2 (one read, one write). RMSNorm is
    shown for contrast — one fewer reduction-read in the unfused chain.

    Watch how short the fused bars are. That gap is the speedup, and you get it without
    touching a single FLOP of the math.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        # Realistic activation tensor: (B*S, N), fp32.
        _B, _S, _N = 8, 2048, 4096
        _M = _B * _S
        _bytes_per_elem = 4
        _tensor_bytes = _M * _N * _bytes_per_elem        # one full pass of X
        _BW = 896e9                                       # bytes/s, 5070 Ti DRAM

        # Pass counts (reads + writes of the M x N tensor).
        # LayerNorm unfused: read mu, read var, read+write normalize, read+write affine
        #                    ~= 3 reads + 2 writes = 5 passes.
        # RMSNorm  unfused:  read rms, read+write normalize, read+write affine
        #                    ~= 2 reads + 2 writes = 4 passes.
        # Fused (either):    1 read + 1 write       = 2 passes.
        _cfg = [
            ("LayerNorm\nunfused", 5, "#d65f5f"),
            ("LayerNorm\nfused",   2, "#4c9f70"),
            ("RMSNorm\nunfused",   4, "#e0a458"),
            ("RMSNorm\nfused",     2, "#4c9f70"),
        ]
        _names = [c[0] for c in _cfg]
        _passes = np.array([c[1] for c in _cfg])
        _colors = [c[2] for c in _cfg]

        _gb = _passes * _tensor_bytes / 1e9               # GB moved
        _ms = (_passes * _tensor_bytes / _BW) * 1e3       # estimated ms at peak BW

        _fig, _ax = plt.subplots(figsize=(8, 4.2))
        _bars = _ax.bar(_names, _gb, color=_colors, edgecolor="none")
        for _bar, _p, _t in zip(_bars, _passes, _ms):
            _ax.text(_bar.get_x() + _bar.get_width() / 2,
                     _bar.get_height() + _gb.max() * 0.015,
                     f"{_p} passes\n{_bar.get_height():.1f} GB\n~{_t:.2f} ms",
                     ha="center", va="bottom", fontsize=8)
        _ax.set_ylabel("DRAM traffic (GB)")
        _ax.set_ylim(0, _gb.max() * 1.28)
        _ax.set_title(
            f"LayerNorm / RMSNorm DRAM traffic  —  X = ({_M:,} x {_N}) fp32 "
            f"= {_tensor_bytes/1e9:.1f} GB/pass")
        _ax.text(0.5, 0.94,
                 "fused = 1 read + 1 write; runtime is memory-bound, so bytes ARE time",
                 transform=_ax.transAxes, ha="center", fontsize=8, color="#666")
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactive: fusing N kernels into 1

    Slide the number of separate kernel passes an **unfused** norm would take. The fused
    kernel is always **2 passes** (one read, one write). Because the op is memory-bound,
    runtime tracks passes directly — so collapsing $n$ passes into $2$ is an
    $n/2\times$ traffic cut and (to first order) an $n/2\times$ speedup.
    """)
    return


@app.cell
def _(mo):
    n_unfused_passes = mo.ui.slider(start=2, stop=6, step=1, value=5,
                                    label="unfused DRAM passes (reads + writes)")
    n_unfused_passes
    return (n_unfused_passes,)


@app.cell
def _(n_unfused_passes):
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        # Same realistic tensor as the bar chart.
        _M, _N = 8 * 2048, 4096
        _tensor_bytes = _M * _N * 4
        _BW = 896e9

        _n = int(n_unfused_passes.value)
        _fused = 2

        _unfused_gb = _n * _tensor_bytes / 1e9
        _fused_gb = _fused * _tensor_bytes / 1e9
        _saved_gb = _unfused_gb - _fused_gb
        _ratio = _n / _fused

        _unfused_ms = (_n * _tensor_bytes / _BW) * 1e3
        _fused_ms = (_fused * _tensor_bytes / _BW) * 1e3

        _fig, _ax = plt.subplots(figsize=(7.8, 3.6))
        _bars = _ax.barh(["unfused", "fused"], [_unfused_gb, _fused_gb],
                         color=["#d65f5f", "#4c9f70"], edgecolor="none")
        for _bar, _ms in zip(_bars, [_unfused_ms, _fused_ms]):
            _ax.text(_bar.get_width() + _unfused_gb * 0.01,
                     _bar.get_y() + _bar.get_height() / 2,
                     f"{_bar.get_width():.1f} GB  (~{_ms:.2f} ms)",
                     va="center", fontsize=9)
        _ax.set_xlim(0, _unfused_gb * 1.25)
        _ax.set_xlabel("DRAM traffic (GB)")
        _ax.set_title(
            f"{_n} passes -> 2:  saves {_saved_gb:.1f} GB,  "
            f"{_ratio:.1f}x less traffic  =>  ~{_ratio:.1f}x faster")
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell
def _(mo, n_unfused_passes):
    def _run():
        _n = int(n_unfused_passes.value)
        return mo.md(
            f"**Fused is always 2 passes.** Unfused at **{_n}** passes moves "
            f"**{_n / 2:.1f}x** the DRAM traffic of the fused kernel — and since the "
            f"op is memory-bound, that ratio is your speedup."
        )

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    - **Norms are memory-bound, so fusion is the whole game.** A handful of FLOPs per
      element means runtime $=$ DRAM bytes; the only lever is fewer passes. This is
      `0d`'s memory roof in its purest form.
    - **One load, one store, per row.** The fused kernel reads each row into SRAM once,
      does *everything* on-chip, and writes once. Intermediates never hit DRAM.
    - **Reduce → map is the same shape as softmax (`1c`/`1d`).** Collapse the row to a
      couple of scalars, then sweep back applying them. Write one, you've written them
      all.
    - **Welford when precision or streaming matters.** A stable single pass for variance
      when fp32 cancellation bites or the row won't fit — the same online-combine trick
      you'll reuse in Flash Attention (`2b`).
    - **$\gamma$ and $\beta$ are tiny broadcast loads.** Length-$N$ vectors reused by
      every row; cache them once, don't let them dominate your traffic accounting.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Now go write it

    You've now got every piece: the two reductions, the epsilon, the affine, and the
    reason a single fused kernel crushes the multi-kernel chain. Open the terminal and
    write the fused LayerNorm / RMSNorm forward pass — load the row once, reduce in
    SRAM, normalize, apply $\gamma/\beta$, store once. The harness scores you on
    **achieved bandwidth** (GB/s), so you're racing the memory roof directly.

    ```bash
    python -m harness.runner e08 --watch
    ```

    `e08` is layernorm/rmsnorm. Get the fused kernel to within striking distance of
    896 GB/s and you've turned the theory in this lecture into a number on the
    scoreboard. (If the stub isn't on disk yet, this is your forward pointer — it lands
    with this module.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [1E: Tiling & Matmul](../1e_tiling_matmul/) &nbsp;|&nbsp; Next: [1G: Scan / Prefix-Sum](../1g_scan/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
