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
    # 4B: Capstone

    > *"A kernel you can't benchmark against a strong baseline is a kernel you don't
    > understand yet."*

    Everything in this course has been building one skill: take a tensor operation,
    reason about where it lands on the roofline, and write a kernel that gets close to
    *its* roof. The capstone is where you do the whole loop end-to-end, unassisted —
    **design, implement, verify, benchmark, and integrate** a real kernel into a real
    model.

    This notebook is a **project brief**, not a solution. It gives you the choice of
    target, the milestones, the correctness and performance acceptance criteria, a
    roofline target to aim at, and a benchmarking checklist. It does **not** give you a
    worked kernel — that's the point. The crux is yours.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. Choose your target

    Pick **one**. Both exercise the full pipeline; pick the one whose payoff you care
    about more.

    ### Option A — Fused attention (FlashAttention-style forward)

    Implement a fused scaled-dot-product-attention **forward** kernel: given $Q, K, V$
    of shape $(B, H, N, d)$, compute

    $$\text{out} = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + \text{mask}\right) V$$

    **without ever materializing the $N\times N$ score matrix** in global memory. You
    built the pieces: tiling (`1e`), online softmax (`1d`/`2b`), reduction fusion (`1c`).
    The capstone is assembling them into one kernel that streams over KV blocks, keeps a
    running max/sum, and accumulates the output tile. **Baseline:**
    `torch.scaled_dot_product_attention`.

    ### Option B — Quantized GEMM

    Implement a **quantized matmul**: store $A$ and/or $B$ in a narrow type (int8, FP8, or
    — if you're ambitious and your `sm_120` supports it, per `4a` — FP4 with MX block
    scales), dequantize on the fly inside the tile loop, and accumulate in a wide type.
    You built the pieces: tiled GEMM (`1e`/`2a`), autotuning (`2a`), quantization scales &
    zero-points (`2c`). The capstone is making it both **fast** (narrow operands → more
    throughput, per `4a`) and **accurate enough** (the scaling has to hold up).
    **Baseline:** a `torch` matmul (full-precision reference for accuracy; a tuned
    `torch` int8/FP8 path, if available, for the speed bar).

    > Option A is memory-traffic-driven (the win is *not* writing the score matrix);
    > Option B is throughput-driven (the win is narrow operands). Both must beat — or
    > credibly approach — their torch baseline to count.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Milestones

    Don't write the fast kernel first. Climb in stages — each milestone is independently
    verifiable, so a regression tells you exactly which step broke.

    1. **M0 — Reference & harness.** Write the torch reference and a correctness check
       (`torch.testing.assert_close` against the baseline) *before* any kernel. Decide your shapes,
       dtypes, and tolerance. You can't tell if a kernel is right without this.
    2. **M1 — Naive correct kernel.** A simple, *correct*, un-tuned Triton (or CUDA)
       kernel that passes the M0 check. Slow is fine. Correct is not optional.
    3. **M2 — Tile & fuse.** Apply the structural win: tiling for reuse (both options),
       the online-softmax streaming that avoids the score matrix (A), or the
       dequant-in-loop that avoids a separate dequant pass (B). Re-verify against M0.
    4. **M3 — Tune.** `@triton.autotune` over block sizes / num_warps / num_stages (or
       hand-tune the CUDA launch). Measure with `triton.testing.do_bench`. Plot achieved
       performance against the roofline target from §4.
    5. **M4 — Integrate.** Drop the kernel into a real model and confirm end-to-end
       correctness and a measurable speedup: wrap it in a `torch.autograd.Function` if it
       needs a backward (`2d`), swap it for the torch op in a small transformer block (A)
       or a quantized linear layer (B), and run a forward (and, for A, ideally a training
       step) to confirm parity.

    Each milestone re-runs the M0 correctness check. **Never let speed work precede a
    passing correctness check** — that's how you ship a fast wrong answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Acceptance criteria

    The capstone is "done" when **all** of these hold. Treat them as a contract.

    **Correctness**

    - Matches the full-precision torch reference within tolerance across **every** tested
      shape (not just one lucky size). Suggested: `rtol=1e-2, atol=1e-2` for FP16/BF16
      attention; for quantized GEMM, a quantization-aware bound (e.g. relative error
      consistent with the operand format — FP8 is *not* FP16, so set the tolerance from
      the format, and justify it).
    - Handles **ragged shapes** (non-multiple-of-block $N$, $d$, $M$, $K$) via masking —
      the edge tile is where kernels secretly break.
    - For Option A: numerically stable softmax (the running-max trick) — verify on inputs
      with large logits, not just small random ones.

    **Performance**

    - Beats — or comes within a stated, defended fraction of — the torch baseline on
      `do_bench` median latency, at the **large** shape (small shapes are launch-bound;
      the real test is where the math dominates).
    - Reaches a defensible fraction of *its* roofline roof (see §4): name your kernel's
      bound (memory or compute), compute its roof, and report achieved-vs-roof. "80% of
      the compute roof" or "90% of the bandwidth roof" is the kind of claim you want.

    **Integration**

    - Runs inside a real model forward (and backward, if applicable) with output parity
      vs the unmodified model, and a measured end-to-end speedup (or a clear explanation
      if the op isn't the bottleneck).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. The roofline target

    Before you tune, decide **what roof you're chasing** — that's the number M3 reports
    against. Use the `0d` model on your 5070 Ti's measured ceilings.

    - **Option A (fused attention)** is the FlashAttention insight made concrete: the
      naive path is **memory-bound** because it writes and re-reads the $N\times N$ score
      matrix to global memory. Fusing keeps the scores in SRAM, which *raises operational
      intensity* and slides the kernel **rightward** on the roofline toward the compute
      roof. Your target: land near the **compute roof** at large $N$, having escaped the
      bandwidth slope.
    - **Option B (quantized GEMM)** is compute-bound at large shapes (it's a matmul). The
      narrow operands from `4a` **raise the compute roof itself** (more FLOP/s at lower
      precision), so the target is a fraction of the *narrow-precision* peak — higher than
      the FP32/FP16 ceiling you'd aim at unquantized.

    The plot below shows both journeys: a starting (naive) dot and the target dot each
    kernel should move toward. Treat the exact coordinates as illustrative — your real
    target uses *measured* roofs from `device_info` + a microbenchmark.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        # Illustrative roofline. Real targets use MEASURED roofs on your card.
        B = 896e9            # bytes/s, measured DRAM bandwidth
        P_FP16 = 120e12      # FP16/tensor compute roof (illustrative)
        P_NARROW = 240e12    # narrow-precision (FP8) roof (illustrative, ~2x)
        I_ridge_fp16 = P_FP16 / B
        I_ridge_narrow = P_NARROW / B

        I = np.logspace(-1, 4, 500)
        roof_fp16 = np.minimum(P_FP16, B * I)
        roof_narrow = np.minimum(P_NARROW, B * I)

        _fig, _ax = plt.subplots(figsize=(8.6, 4.4))
        _ax.plot(I, roof_fp16, color="#333", linewidth=2.0, zorder=3,
                 label="FP16 roof")
        _ax.plot(I, roof_narrow, color="#4c9f70", linewidth=1.8, linestyle="--",
                 zorder=3, label="narrow (FP8) roof")

        # Option A: attention moves rightward (naive memory-bound -> fused near compute).
        _A_naive_I, _A_naive_P = 2.0, min(P_FP16, B * 2.0)
        _A_tgt_I = I_ridge_fp16 * 3.0
        _A_tgt_P = min(P_FP16, B * _A_tgt_I) * 0.85
        _ax.scatter([_A_naive_I], [_A_naive_P], color="#d65f5f", s=70, zorder=5)
        _ax.scatter([_A_tgt_I], [_A_tgt_P], color="#d65f5f", s=110, marker="*", zorder=6)
        _ax.annotate("", xy=(_A_tgt_I, _A_tgt_P), xytext=(_A_naive_I, _A_naive_P),
                     arrowprops=dict(arrowstyle="->", color="#d65f5f", lw=1.4))
        _ax.text(_A_naive_I, _A_naive_P * 1.5, "A: naive\n(mem-bound)",
                 fontsize=8, color="#d65f5f")
        _ax.text(_A_tgt_I * 0.5, _A_tgt_P * 1.35, "A: fused target\n(near compute roof)",
                 fontsize=8, color="#d65f5f")

        # Option B: quantized GEMM lifts to the higher narrow roof.
        _B_fp16_I = I_ridge_fp16 * 8.0
        _B_fp16_P = P_FP16 * 0.8
        _B_tgt_I = I_ridge_narrow * 8.0
        _B_tgt_P = P_NARROW * 0.8
        _ax.scatter([_B_fp16_I], [_B_fp16_P], color="#5b8def", s=70, zorder=5)
        _ax.scatter([_B_tgt_I], [_B_tgt_P], color="#5b8def", s=110, marker="*", zorder=6)
        _ax.annotate("", xy=(_B_tgt_I, _B_tgt_P), xytext=(_B_fp16_I, _B_fp16_P),
                     arrowprops=dict(arrowstyle="->", color="#5b8def", lw=1.4))
        _ax.text(_B_fp16_I * 1.1, _B_fp16_P * 0.55, "B: FP16 GEMM",
                 fontsize=8, color="#5b8def")
        _ax.text(_B_tgt_I * 0.3, _B_tgt_P * 1.25, "B: quantized target\n(higher roof)",
                 fontsize=8, color="#5b8def")

        _ax.set_xscale("log")
        _ax.set_yscale("log")
        _ax.set_xlabel("operational intensity  I  (FLOP / byte)")
        _ax.set_ylabel("attainable performance (FLOP/s)")
        _ax.set_title("Capstone roofline targets (illustrative — use measured roofs)")
        _ax.set_ylim(1e11, P_NARROW * 3)
        _ax.grid(True, which="both", alpha=0.15)
        _ax.legend(loc="lower right", fontsize=8)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the two arrows. **Option A's red arrow goes right** — fusion raises operational
    intensity (scores never hit DRAM), sliding off the bandwidth slope toward the compute
    roof. **Option B's blue arrow goes up** — narrow operands lift the *roof itself*, so
    the target sits on a higher ceiling than the FP16 GEMM could reach. Different physics,
    different arrow direction; the roofline names which one you're playing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. The acceptance checklist (interactive)

    Pick your path. The checklist below renders the milestone gates for that option,
    rolled up from §§2–3. Print it, and don't call the capstone done until every box is
    honestly checked.
    """)
    return


@app.cell
def _(mo):
    path_dropdown = mo.ui.dropdown(
        options=["Option A — Fused attention", "Option B — Quantized GEMM"],
        value="Option A — Fused attention",
        label="capstone path",
    )
    path_dropdown
    return (path_dropdown,)


@app.cell
def _(mo, path_dropdown):
    def _build():
        _is_a = path_dropdown.value.startswith("Option A")
        _baseline = ("`torch.scaled_dot_product_attention`" if _is_a
                     else "a `torch` matmul (FP reference + tuned int8/FP8 if available)")
        _structural = ("online-softmax streaming over KV blocks — the $N\\times N$ scores "
                       "never touch global memory"
                       if _is_a else
                       "dequantize-in-the-tile-loop — no separate dequant pass to DRAM")
        _accuracy = ("`rtol=1e-2, atol=1e-2` vs the FP reference, on large-logit inputs too"
                     if _is_a else
                     "a quantization-aware tolerance derived from the operand format "
                     "(FP8/int8/FP4), and justified in the report")
        _roof = ("escape the bandwidth slope; land near the **compute roof** at large $N$"
                 if _is_a else
                 "a defensible fraction of the **narrow-precision** compute roof at large shapes")
        _integrate = ("swap into a small transformer block's attention; forward parity + "
                       "a measured speedup (backward via `torch.autograd.Function` if trained)"
                       if _is_a else
                       "swap into a quantized linear layer; forward parity vs the FP model + "
                       "a measured speedup")

        return mo.md(rf"""
    ### Acceptance checklist — {path_dropdown.value}

    **M0 — reference & harness**
    - [ ] torch reference written; baseline = {_baseline}
    - [ ] correctness check via `torch.testing.assert_close` defined *before* any kernel
    - [ ] shapes, dtypes, and tolerance fixed up front

    **M1 — naive correct kernel**
    - [ ] simple kernel passes the M0 check (slow is fine)
    - [ ] ragged / non-multiple-of-block shapes masked correctly

    **M2 — tile & fuse**
    - [ ] structural win applied: {_structural}
    - [ ] re-verified against M0 (correctness survived the rewrite)

    **M3 — tune & roofline**
    - [ ] autotuned (block sizes / `num_warps` / `num_stages`) or hand-tuned launch
    - [ ] `triton.testing.do_bench` median latency across **varied** shapes
    - [ ] roofline target: {_roof}
    - [ ] reported achieved-vs-roof and achieved-vs-baseline at the large shape

    **M4 — integrate**
    - [ ] {_integrate}
    - [ ] end-to-end output parity vs the unmodified model

    **Done = every box above is honestly checked.**
    """)

    _build()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6. Benchmarking checklist

    Performance claims are only as good as the measurement. The discipline:

    - **Use `triton.testing.do_bench`,** not a hand-rolled `time.time()` loop. It warms
      up, runs many iterations, and aggregates them — pass `return_mode="median"`
      explicitly for a robust **median** (the default aggregation is the *mean*; use
      `quantiles` if you want p20/p80). A single timed call measures launch overhead and
      clock noise, not the kernel.
    - **Warm up first.** The first launch pays JIT/autotune/cache costs. `do_bench`
      handles this, but if you roll your own, discard the first iterations.
    - **Vary the shapes.** Bench small *and* large. Small shapes are launch-bound and will
      lie about your kernel's quality; large shapes are where the math dominates and the
      roofline applies. Report a curve, not a point.
    - **Compare to the right baseline,** measured the *same way*: run torch through
      `do_bench` too. "Faster than torch" means faster on identical inputs, same dtype,
      same device, same measurement harness.
    - **Convert to the roofline unit.** Turn latency into GB/s (memory-bound kernels) or
      TFLOP/s (compute-bound) using your `bytes_moved`/`flops` counts, and place the dot on
      the `0d` roofline. Latency alone doesn't tell you if you're near the roof.
    - **Profile when stuck.** If you're far below roof and don't know why, Nsight Compute
      (`ncu`) gives you achieved occupancy, memory throughput, and stall reasons. The `7a`
      reference lists the key metrics.
    - **Report honestly.** State the shape, dtype, baseline, measurement (median over N
      iters), achieved rate, and the roof. A number without those is not a result.

    > [`triton.testing.do_bench` docs](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)
    > and the [Nsight Compute docs](https://docs.nvidia.com/nsight-compute/) are your
    > measurement references; the `0d` roofline is the scoreboard.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters

    This is the job. A research or production kernel engineer does exactly this loop:
    establish a correct reference, write the kernel, fuse the structural inefficiency,
    tune against a measured roof, beat a strong baseline, and land it in a model without
    breaking numerics. Every individual skill — tiling, online softmax, quantization,
    autotuning, autograd integration, profiling — you've practiced in isolation. The
    capstone is the integration test for *you*: can you carry an idea from math to a
    model-ready, benchmarked kernel on your own?

    The two non-negotiables, restated because they're where capstones go wrong:
    **correctness gates speed** (never tune a wrong kernel), and **a claim needs a
    measurement** (do_bench against a real baseline, placed on the roofline). Hold those
    and the rest is the craft you already have.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Now go write it

    The capstone has its own harness target. From the repo root:

    ```bash
    python -m harness.runner capstone --watch
    ```

    Unlike `e01`–`e13`, the capstone's `spec.py` is a **template you fill in** — writing
    the contract (reference, shapes, tolerance, FLOP/byte counts) *is* milestone M0. It
    ships with a small runnable placeholder so the runner works out of the box; replace
    it with your target's contract, then write `kernel.py` against it. The kernel, the
    fusion, the tuning, and the model integration are entirely yours. Pick a path from
    §1, climb the milestones in §2, and don't check a box in §5 you can't defend.

    When it passes, you've written a real GPU kernel, benchmarked it like an engineer,
    and shipped it into a model. That's the whole course, in one project.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [4A: Blackwell (sm_120)](../4a_blackwell/) &nbsp;|&nbsp; Next: [7A: Study Guide & Reference](../7a_study_guide/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
