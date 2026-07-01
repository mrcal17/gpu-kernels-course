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
    # 2C: Quantization

    > *"A weight doesn't need 32 bits of opinion. Often 8 — or 4 — carry the signal,
    > and the other bits were just bandwidth you were paying for."*

    Every time you load a 7-billion-parameter model in 4-bit to fine-tune it with
    QLoRA, you are running quantized kernels. The model's weights are stored in a
    low-precision integer format, and a kernel **dequantizes them on the fly** as it
    multiplies. This lecture is about that machinery: how a real number is packed
    into an `int8` or `fp8`, the scales and zero-points that make it reversible, and
    the **dequantize-then-accumulate** pattern at the heart of a quantized matmul.

    The motivation is bandwidth, not flops. A matmul that reads `int8` weights moves
    a quarter of the bytes of an `fp32` one. Since matmul at inference is often
    memory-bound on the weights (`0d`), **moving fewer bytes is the win** — you trade
    a little accuracy for a lot of bandwidth. We'll quantify both sides of that trade,
    then point you at the kernel.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. Affine quantization: scale and zero-point

    Quantization maps a range of real numbers onto a small set of integers. The
    standard **affine** scheme picks a scale $s$ (the size of one integer step) and a
    zero-point $z$ (the integer that represents real $0$), then:

    $$q = \mathrm{clamp}\!\Big(\mathrm{round}\big(\tfrac{x}{s}\big) + z,\;
        q_{\min},\, q_{\max}\Big),\qquad
      \hat x = s\,(q - z).$$

    The first equation **quantizes** (real → int), the second **dequantizes**
    (int → approximate real). For signed `int8`, $q\in[-128, 127]$; for `uint8`,
    $[0, 255]$. To cover an observed range $[x_{\min}, x_{\max}]$ you choose

    $$s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}},\qquad
      z = q_{\min} - \mathrm{round}\!\Big(\tfrac{x_{\min}}{s}\Big).$$

    A **symmetric** variant fixes $z = 0$ (so $\hat x = s\,q$), using a range
    $[-a, a]$ with $a = \max|x|$. Symmetric is cheaper in the kernel — no zero-point
    subtraction in the inner loop — and is the common choice for weights, which are
    roughly zero-centered. Activations, often one-sided (think post-ReLU), benefit
    from the asymmetric zero-point.

    The error you incur is **rounding to the nearest integer step**: each value is off
    by at most $s/2$. So the whole game is making $s$ small — i.e., keeping the range
    you have to cover tight.

    > [PyTorch quantization docs](https://pytorch.org/docs/stable/quantization.html)
    > and Gholami et al., ["A Survey of Quantization Methods for Efficient Neural
    > Network Inference" (2021)](https://arxiv.org/abs/2103.13630) cover the schemes
    > here in depth.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        # Quantize -> dequantize a vector and inspect the round-trip error.
        _rng = np.random.default_rng(1)
        x = _rng.normal(size=8).astype(np.float32) * 2.0

        qmin, qmax = -128, 127           # signed int8
        xmin, xmax = x.min(), x.max()
        s = (xmax - xmin) / (qmax - qmin)             # asymmetric scale
        z = qmin - round(xmin / s)                    # zero-point
        q = np.clip(np.round(x / s) + z, qmin, qmax).astype(np.int32)
        x_hat = s * (q - z)                           # dequantized

        print("=== Affine int8 round-trip (asymmetric) ===")
        print(f"  scale s = {s:.5f}   zero-point z = {z}")
        print(f"  {'x':>9s} {'q (int8)':>9s} {'x_hat':>9s} {'|err|':>9s}")
        print("  " + "-" * 40)
        for _xi, _qi, _xh in zip(x, q, x_hat):
            print(f"  {_xi:>9.4f} {_qi:>9d} {_xh:>9.4f} {abs(_xi - _xh):>9.4f}")
        print(f"\n  max |error| = {np.abs(x - x_hat).max():.4f}   "
              f"(bounded by s/2 = {s / 2:.4f})")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Per-tensor vs per-channel

    One scale for an entire weight matrix (**per-tensor**) is the cheapest to store
    and apply, but it's hostage to outliers: a single large-magnitude column forces a
    big $s$, and every other value then rounds coarsely. The fix is to use **more
    scales** — one per row or column.

    **Per-channel** (a.k.a. per-axis) quantization gives each output channel
    (typically each column of the weight) its own $s$ and $z$:

    $$\hat W_{:,j} = s_j\,(Q_{:,j} - z_j).$$

    Now a noisy channel gets a big scale without coarsening its neighbors. The cost is
    a vector of scales instead of a scalar — negligible storage — and a per-channel
    multiply in the dequant. **Group-wise** quantization (used by QLoRA's NF4) takes
    this further: a separate scale per *block* of, say, 64 weights, catching local
    outliers within a channel. More scales → less error, slightly more overhead.

    The rule of thumb: **weights tolerate aggressive quantization with per-channel /
    group-wise scales; activations are harder** (dynamic range, outliers) and often
    stay in higher precision or use per-token scales. This asymmetry is why "weight-
    only" quantization (W8A16, W4A16) is the popular sweet spot for LLM inference and
    fine-tuning — exactly what your QLoRA runs do.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        # One outlier column wrecks per-tensor; per-channel absorbs it.
        _rng = np.random.default_rng(3)
        W = _rng.normal(size=(64, 8)).astype(np.float32)
        W[:, 3] *= 12.0                  # one wild channel

        qmin, qmax = -127, 127           # symmetric int8

        # --- per-tensor: a single scale for everything ---
        s_pt = np.abs(W).max() / qmax
        q_pt = np.clip(np.round(W / s_pt), qmin, qmax)
        W_pt = s_pt * q_pt

        # --- per-channel: one scale per column ---
        s_pc = np.abs(W).max(axis=0, keepdims=True) / qmax   # (1, 8)
        q_pc = np.clip(np.round(W / s_pc), qmin, qmax)
        W_pc = s_pc * q_pc

        def rel_err(A, Ahat):
            return np.linalg.norm(A - Ahat) / np.linalg.norm(A)

        print("=== Per-tensor vs per-channel int8 (one outlier channel) ===\n")
        print(f"  per-tensor  relative error: {rel_err(W, W_pt):.4f}")
        print(f"  per-channel relative error: {rel_err(W, W_pc):.4f}")
        print("\n  Per-column error on the QUIET channels:")
        for _j in [0, 1, 2]:
            _e_pt = rel_err(W[:, _j], W_pt[:, _j])
            _e_pc = rel_err(W[:, _j], W_pc[:, _j])
            print(f"    col {_j}:  per-tensor {_e_pt:.4f}   per-channel {_e_pc:.4f}")
        print("\n  The outlier in col 3 inflates the single per-tensor scale,")
        print("  coarsening every other column. Per-channel isolates the damage.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. int8 vs fp8: two ways to spend 8 bits

    `int8` spaces its 256 levels **uniformly** — every step is the same size $s$.
    That's ideal when values are spread evenly across the range, but wasteful for the
    bell-shaped, heavy-tailed distributions of neural-net weights, where most values
    cluster near zero and a few stragglers live far out.

    `fp8` spends its 8 bits as a tiny **floating-point** number — a sign, a few
    exponent bits, a few mantissa bits. The two Blackwell-supported formats are
    **E4M3** (4 exponent, 3 mantissa: more precision, smaller range) and **E5M2**
    (5 exponent, 2 mantissa: more range, less precision). Because the exponent makes
    the steps **logarithmically spaced**, fp8 has fine resolution near zero and
    coarse resolution far out — a much better match for weight distributions. The
    price is hardware complexity and a fixed precision/range split you pick per
    format.

    Rule of thumb: `int8` for ranges that are genuinely uniform or where simple
    integer hardware is the constraint; `fp8` (E4M3 for forward weights/acts, E5M2
    where gradients need range) for the peaky distributions inside transformers. Your
    5070 Ti's 5th-gen tensor cores accelerate both (and FP4 — Part 4); `fp8` is
    increasingly the default for low-precision LLM compute.

    > NVIDIA's [FP8 Formats for Deep Learning (Micikevicius et al., 2022)](https://arxiv.org/abs/2209.05433)
    > defines E4M3/E5M2 and the rationale for log-spaced levels.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        # Compare the representable LEVELS of int8 vs a toy fp8 (E4M3-ish),
        # both normalized to [-1, 1], over a Gaussian weight distribution.
        _rng = np.random.default_rng(5)
        w = _rng.normal(size=200000).astype(np.float64)
        w = w / np.abs(w).max()          # normalize to [-1, 1]

        # int8 symmetric: 255 uniform levels in [-1, 1]
        int8_levels = np.linspace(-1, 1, 255)

        # toy fp8 E4M3-ish: log-spaced magnitudes + sign + zero
        mant = np.array([1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875])
        exps = np.arange(-6, 1)          # small exponent range for the toy
        mags = np.unique(np.concatenate(
            [mant * (2.0 ** e) for e in exps]))
        mags = mags[mags <= 1.0]
        fp8_levels = np.concatenate([-mags[::-1], [0.0], mags])

        def quant_to(levels, vals):
            idx = np.abs(vals[:, None] - levels[None, :]).argmin(axis=1)
            return levels[idx]

        # quantization error vs the value's magnitude
        x = np.linspace(-1, 1, 400)
        err_int8 = np.abs(x - quant_to(int8_levels, x))
        err_fp8 = np.abs(x - quant_to(fp8_levels, x))

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

        # (a) where the levels sit
        _ax1.vlines(int8_levels[::8], 0, 1, color="#5b8def", alpha=0.5,
                    label="int8 (uniform)")
        _ax1.vlines(fp8_levels, 1.2, 2.2, color="#d65f5f", alpha=0.7,
                    label="fp8 (log-spaced)")
        _ax1.hist(w, bins=80, density=True, bottom=2.5, color="#cccccc",
                  alpha=0.7)
        _ax1.text(-0.98, 3.3, "weight distribution", fontsize=8, color="#777")
        _ax1.set_yticks([])
        _ax1.set_xlabel("normalized value")
        _ax1.set_title("Where the 8-bit levels land")
        _ax1.legend(loc="upper right", fontsize=7)

        # (b) error vs magnitude
        _ax2.plot(x, err_int8, color="#5b8def", label="int8")
        _ax2.plot(x, err_fp8, color="#d65f5f", label="fp8")
        _ax2.set_xlabel("value")
        _ax2.set_ylabel("|quantization error|")
        _ax2.set_title("fp8 is finer near 0, coarser in the tails")
        _ax2.legend(fontsize=8)
        _ax2.grid(True, alpha=0.15)

        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. The dequantize-then-accumulate matmul

    A quantized matmul does **not** keep everything in low precision through the
    whole computation — that would lose too much in the accumulation. The pattern is:

    1. **Read** the quantized operands (`int8`/`fp8`) from HBM — this is the small,
       bandwidth-cheap part, and the whole point.
    2. **Dequantize** a tile back to a higher-precision compute type
       ($\hat x = s(q - z)$) *inside* the kernel, in SRAM/registers.
    3. **Accumulate** the dot product in a wide accumulator — `fp32` (or `int32` for
       integer matmul), so the running sum doesn't lose bits.
    4. **Re-quantize** the output only if the next layer wants low precision.

    The accumulator width is the safeguard: the *inputs* are cheap to move, but the
    *sum of many products* needs headroom. For pure-integer matmul there's an even
    slicker variant — accumulate $q_a\cdot q_b$ in `int32` and apply the scales
    **once at the end**, since

    $$\sum_k \hat a_k \hat b_k
      = s_a s_b \sum_k (q^a_k - z_a)(q^b_k - z_b),$$

    pulling $s_a s_b$ outside the sum (the cross-terms in $z$ are correction terms
    computed once per row/column). This keeps the entire inner loop in fast integer
    arithmetic and dequantizes a single scalar at the end.

    Why it's a *bandwidth* win: in the roofline (`0d`), reading `int8` instead of
    `fp32` cuts the byte count 4×, raising operational intensity and pulling a
    memory-bound matmul up its slanted roof. The flops are similar (you still do
    $2MNK$), but you feed them with a quarter of the traffic. That is exactly why
    weight-quantized inference is fast.

    > [Triton fp8 / low-precision matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
    > shows the dequant-in-kernel structure; the integer-accumulate identity is the
    > basis of `gemmlowp`-style int8 GEMM.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        # Verify the integer-accumulate identity for a symmetric int8 matmul:
        # dequantizing once at the end == dequantizing every operand first.
        _rng = np.random.default_rng(2)
        M, N, K = 3, 2, 64
        A = _rng.normal(size=(M, K)).astype(np.float32)
        B = _rng.normal(size=(K, N)).astype(np.float32)

        qmax = 127
        sa = np.abs(A).max() / qmax       # symmetric per-tensor scales (z = 0)
        sb = np.abs(B).max() / qmax
        Aq = np.clip(np.round(A / sa), -qmax, qmax).astype(np.int32)
        Bq = np.clip(np.round(B / sb), -qmax, qmax).astype(np.int32)

        # path 1: dequantize operands, then float matmul
        C_dequant_first = (sa * Aq.astype(np.float32)) @ (sb * Bq.astype(np.float32))

        # path 2: int32 accumulate, scale ONCE at the end
        C_int_accum = (sa * sb) * (Aq @ Bq).astype(np.float32)

        C_true = A @ B

        print("=== int8 matmul: dequant-first vs int-accumulate-then-scale ===")
        print(f"  fp32 reference  C[0]: "
              f"{np.array2string(C_true[0], precision=4)}")
        print(f"  dequant-first   C[0]: "
              f"{np.array2string(C_dequant_first[0], precision=4)}")
        print(f"  int-accumulate  C[0]: "
              f"{np.array2string(C_int_accum[0], precision=4)}")
        print(f"\n  the two int8 paths agree to "
              f"{np.abs(C_dequant_first - C_int_accum).max():.2e}")
        print(f"  both differ from fp32 by ~{np.abs(C_true - C_int_accum).max():.4f} "
              f"(quantization error, not a bug)")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. The perf / accuracy trade

    Fewer bits means fewer bytes to move (faster) but coarser steps (less accurate).
    The interactive sweeps **bit-width** for a per-channel symmetric quantizer and
    reports both sides at once: the relative reconstruction error of a weight matrix,
    and the bandwidth multiplier vs `fp32`. Watch the error fall off a cliff somewhere
    around 4–8 bits — the region where modern LLM quantization lives — while the
    bandwidth saving keeps climbing. This curve *is* the engineering decision.
    """)
    return


@app.cell
def _(mo):
    bits_slider = mo.ui.slider(start=2, stop=16, step=1, value=8,
                               label="bit-width (quantized weights)")
    bits_slider
    return (bits_slider,)


@app.cell
def _(bits_slider):
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        _rng = np.random.default_rng(11)
        # A weight matrix with a heavy tail (a few outliers per channel).
        W = _rng.standard_t(df=3, size=(256, 64)).astype(np.float32)

        def per_channel_err(bits):
            qmax = 2 ** (bits - 1) - 1
            s = np.abs(W).max(axis=0, keepdims=True) / qmax
            q = np.clip(np.round(W / s), -qmax, qmax)
            W_hat = s * q
            return np.linalg.norm(W - W_hat) / np.linalg.norm(W)

        all_bits = np.arange(2, 17)
        errs = np.array([per_channel_err(int(b)) for b in all_bits])
        bw = 32.0 / all_bits             # bandwidth multiplier vs fp32

        b = int(bits_slider.value)
        my_err = per_channel_err(b)
        my_bw = 32.0 / b

        _fig, _ax1 = plt.subplots(figsize=(8, 4.0))
        _ax1.plot(all_bits, errs, "-o", color="#d65f5f", linewidth=2,
                  label="reconstruction error")
        _ax1.axvline(b, color="#999", linestyle="--", linewidth=1)
        _ax1.scatter([b], [my_err], color="#d65f5f", s=90, zorder=5)
        _ax1.set_yscale("log")
        _ax1.set_xlabel("bit-width")
        _ax1.set_ylabel("relative reconstruction error (log)", color="#d65f5f")
        _ax1.tick_params(axis="y", labelcolor="#d65f5f")
        _ax1.grid(True, which="both", alpha=0.15)

        _ax2 = _ax1.twinx()
        _ax2.plot(all_bits, bw, "-s", color="#4c9f70", linewidth=2,
                  label="bandwidth saving")
        _ax2.scatter([b], [my_bw], color="#4c9f70", s=90, zorder=5)
        _ax2.set_ylabel("bandwidth vs fp32 (x fewer bytes)", color="#4c9f70")
        _ax2.tick_params(axis="y", labelcolor="#4c9f70")

        _ax1.set_title(
            f"{b}-bit:  error {my_err:.3%},  moves {my_bw:.1f}x fewer bytes than fp32")
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    - **Quantized matmul is a bandwidth play.** The flops barely change; you move a
      quarter (int8/fp8 — both are 8 bits) or an eighth (int4) of the bytes. On the roofline that lifts a
      memory-bound GEMM up its slanted roof — which is why your QLoRA model both fits
      *and* runs faster.
    - **Dequantize in the kernel, accumulate wide.** Read low precision, expand a tile
      to the compute type in SRAM, sum in an `fp32`/`int32` accumulator. The
      integer-accumulate identity lets you pull the scale out of the inner loop
      entirely — the slickest version.
    - **Match the granularity to the data.** Per-tensor is cheapest but outlier-
      fragile; per-channel / group-wise scales isolate noisy channels for a tiny cost.
      This is exactly the knob NF4 / QLoRA turns.
    - **Pick the format for the distribution.** Uniform `int8` for even ranges,
      log-spaced `fp8` for the peaky weight distributions inside transformers — and
      remember your 5070 Ti's tensor cores accelerate both.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Now go write it

    Put the dequant-then-accumulate pattern into a real kernel. Open the harness:

    ```bash
    python -m harness.runner e12 --watch
    ```

    `e12` is a quantized matmul: you'll read `int8` operands with their scales,
    dequantize a tile inside the kernel, accumulate in a wide `fp32` accumulator, and
    produce the output. It builds directly on your tiled GEMM from `1e`/`2a` — the
    tiling and masking are the same; what's new is the dequant in the inner loop and
    choosing per-tensor vs per-channel scales. The metric is FLOP/s, but keep one eye
    on the bytes moved — that's where quantization actually pays.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [2B: Flash Attention](../2b_flash_attention/) &nbsp;|&nbsp; Next: [2D: Autograd Integration](../2d_autograd/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
