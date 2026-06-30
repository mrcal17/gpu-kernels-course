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
    # 7B: Validating & Benchmarking — Reference Card

    > *A kernel isn't done when it runs. It's done when you've **proven** it's correct
    > and **measured** how fast — against a reference you trust, and against the roof.*

    The validate &#8594; benchmark workflow is **taught in [1A](../1a_triton_model/)**, at
    your very first kernel, and you practice it in **every exercise** — each README has a
    *"Validate & benchmark it yourself"* section. **This page is the lookup version:** the
    thing you keep open while you work. The tolerance table, the `do_bench` parameters, the
    throughput formulas, the reusable harness, the `spec.py` contract, and the traps that
    make a benchmark quietly lie. Nothing here that 1A didn't start — this is where it's all
    collected, in one place, for reference.

    Two questions, every time:

    1. **Is it correct?** — within tolerance of a trusted reference, on more than one shape.
    2. **Is it fast?** — as a *rate*, compared to the hardware roof **and** to torch.

    We'll answer each with the exact tools the harness uses: `torch.testing.assert_close`
    and `triton.testing.do_bench`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. Correctness: `assert_close`, never `==`

    Your first instinct is `(out == ref).all()`. **It will fail on a correct kernel.**
    Floating-point addition is not associative: $(a + b) + c \neq a + (b + c)$ in the
    last bits. Your kernel sums a reduction in a *different order* than torch does — so
    tiny disagreements in the low bits are not bugs, they're **expected**. Bit-equality
    is the wrong test for anything that accumulates.

    The right test is *"close enough,"* and PyTorch spells it
    `torch.testing.assert_close`:

    ```python
    import torch

    # ref computed FIRST (see the trap below), out is your kernel's result
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    ```

    Under the hood it checks, elementwise:

    $$ |\,\text{out} - \text{ref}\,| \;\le\; \texttt{atol} \;+\; \texttt{rtol}\cdot|\,\text{ref}\,| $$

    Two knobs, two jobs:

    - **`atol`** (absolute) is the **floor** — it dominates where `ref` is near zero
      (the `rtol` term vanishes there, so without `atol` *any* error is "infinitely
      relatively large").
    - **`rtol`** (relative) **scales with magnitude** — a $10^{-2}$ rtol allows bigger
      absolute slack on a value of 1000 than on a value of 1.

    Use `assert_close`, **not** `torch.allclose`. `allclose` returns a bare `True/False`;
    `assert_close` *raises with a diagnostic* — max absolute error, max relative error,
    how many elements mismatched, and the worst offender's index. When it fails you want
    the autopsy, not a `False`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Choosing the tolerance — the rule

    The tolerance is not arbitrary; it tracks **how much floating-point reordering the
    op allows** and **what precision it runs in**. The longer the reduction and the
    lower the precision, the looser it must be. This is exactly how the course's
    `spec.py` files set `TOL`:

    | Op | `atol` | `rtol` | Why |
    |---|---|---|---|
    | vector add, copy, transpose | `0` | `0` | no accumulation — **exact**, demand bit-equality |
    | fused elementwise | `1e-3` | `1e-3` | a couple of ops, mild rounding |
    | row-reduce, softmax | `1e-2` | `1e-3` | summing $N$ terms — order diverges from torch |
    | matmul (K = 1024) | `1e-1` | `1e-2` | a 1024-long inner product per output — lots of reordering |
    | fp16 / quantized matmul | `1e-1` | `1e-2` | fp16 round-off **on top of** accumulation order |

    The pattern: **exact ops get `0/0`** (a single mismatched bit *is* a bug), and
    **every reduction loosens with its length and drops in precision.** If you find
    yourself widening tolerance past these to get a "pass," stop — that's usually a real
    bug hiding behind a generous threshold, not benign rounding.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        # Illustrate the assert_close band: pass iff |out-ref| <= atol + rtol*|ref|
        _atol, _rtol = 1e-2, 1e-2

        # (ref, your_out) pairs probing different magnitude regimes
        _cases = [
            ("large value, tiny abs error", 1000.0, 1000.05),
            ("large value, big abs error", 1000.0, 1003.0),
            ("near zero, tiny abs error", 0.0, 0.005),
            ("near zero, 'small' abs error", 0.0, 0.05),
            ("unit value, borderline", 1.0, 1.018),
        ]

        print(f"=== assert_close band:  |out-ref| <= atol + rtol*|ref|   (atol={_atol}, rtol={_rtol}) ===\n")
        print(f"  {'case':<32}{'ref':>10}{'out':>10}{'|err|':>10}{'budget':>10}  verdict")
        for _name, _ref, _out in _cases:
            _err = abs(_out - _ref)
            _budget = _atol + _rtol * abs(_ref)
            _ok = "PASS" if _err <= _budget else "FAIL"
            print(f"  {_name:<32}{_ref:>10.3f}{_out:>10.3f}{_err:>10.4f}{_budget:>10.4f}  [{_ok}]")

        print("\n  Note row 3 vs 4: near zero, ONLY atol protects you -- rtol*|ref| is ~0 there.")
        print("  That's why exact-ish ops still need a nonzero atol; rtol alone can't guard zero.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Two correctness traps the harness avoids (and you must too)

    **Trap 1 — compute the reference BEFORE you run your kernel.**
    If your kernel has an aliasing or in-place bug and writes over its own inputs,
    computing the golden answer *afterward* lets the buggy kernel **corrupt the very
    thing you check against** — and you "pass." Order matters:

    ```python
    ref = reference(*inputs)   # golden answer, captured while inputs are pristine
    out = my_kernel(*inputs)   # now run yours; if it stomps inputs, ref is safe
    torch.testing.assert_close(out, ref, **tol)
    ```

    **Trap 2 — make the comparison fair (same algorithm, same precision).**
    On Ampere and later, `torch.matmul` uses **TF32** by default — a *reduced-precision*
    tensor-core path. If your kernel does true fp32, torch's "reference" is computing a
    **different, lower-precision answer**, and you'll see disagreement that is not your
    bug. Pin the reference to the precision you're actually targeting:

    ```python
    torch.backends.cuda.matmul.allow_tf32 = False   # force strict fp32 in the reference
    torch.backends.cudnn.allow_tf32 = False
    ```

    **And test more than one shape.** A clean power-of-two ($2^{20}$) can completely hide
    a missing tail guard (`mask = offs < n`) because nothing ever lands in the ragged
    region. So validate on a **ragged** size too — `1000`, `1023`, `1025` — which forces
    the masked path. Seed your inputs (`torch.manual_seed(0)`) so a failure is
    **reproducible** instead of a different random matrix every run.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Benchmarking: measuring time without lying to yourself

    Three traps turn a naive timer into a fiction generator. Each has a fix.

    **Trap A — the launch is asynchronous.** A CUDA kernel launch *returns to the CPU
    immediately*; the GPU runs later. Wrap `time.perf_counter()` around a kernel call and
    you've timed the **launch**, not the **work** — the CPU sprinted ahead while the GPU
    hadn't started. You must force the CPU to wait for the device:
    `torch.cuda.synchronize()` (or use CUDA **events**, which timestamp *on the device*).

    **Trap B — cold start.** The first call pays one-time costs that have nothing to do
    with steady-state speed: JIT / autotune compilation, cuBLAS handle creation, lazy
    allocations, and GPU clocks still ramping from idle. **Warm up** — run it a few times
    and throw those away — before you measure.

    **Trap C — noise.** GPU timings jitter: clock boosting, the OS scheduler, other
    processes. One sample is meaningless. Take **many** and report the **median**, not the
    mean — a single outlier (a context switch, a thermal blip) drags the mean to the right
    but barely moves the median. Robust statistic, every time.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        # Synthetic 50-iteration timing run: tight cluster + a couple of outliers
        # (a scheduler hiccup, a thermal blip) -- exactly what GPU timing looks like.
        np.random.seed(0)
        _base = 0.234  # ms, like a 2^24 fp32 vector-add
        _samples = _base + np.abs(np.random.normal(0, 0.004, size=50))
        _samples[7] = 0.61   # context switch
        _samples[33] = 0.52  # another blip

        _mean = _samples.mean()
        _median = np.median(_samples)

        _fig, _ax = plt.subplots(figsize=(9.5, 3.4))
        _ax.scatter(range(len(_samples)), _samples, s=22, color="#4c78a8", zorder=3,
                    label="per-iter time")
        _ax.axhline(_mean, color="#e45756", lw=2, ls="--",
                    label=f"mean = {_mean:.3f} ms  (dragged by outliers)")
        _ax.axhline(_median, color="#54a24b", lw=2,
                    label=f"median = {_median:.3f} ms  (robust)")
        _ax.set_xlabel("iteration")
        _ax.set_ylabel("time (ms)")
        _ax.set_title("Why median, not mean: two outliers in 50 samples")
        _ax.legend(loc="upper right", fontsize=8.5)
        _ax.set_ylim(0.20, 0.66)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The right way — two options

    **Option 1: CUDA events by hand.** This is what the harness's fallback does, and what
    you'd write yourself. Events record timestamps *on the device*, so they measure GPU
    time directly — no host overhead, no async lie. The shape: warm up, then bracket each
    timed call in an event pair, synchronize, take the median.

    ```python
    import statistics, torch

    def bench_ms(fn, warmup=10, iters=50):
        for _ in range(warmup):          # Trap B: burn the cold-start cost
            fn()
        torch.cuda.synchronize()         # Trap A: drain the queue before timing

        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()     # wait for THIS iter to finish
            times.append(start.elapsed_time(end))   # device-measured ms
        return statistics.median(times)  # Trap C: robust statistic
    ```

    **Option 2: `triton.testing.do_bench` — the one-liner that does all of it.** Warmup,
    sync, many reps, median — handled. This is the harness's primary path:

    ```python
    from triton.testing import do_bench

    ms = do_bench(fn, warmup=25, rep=100, return_mode="median")   # median ms
    ```

    Reach for `do_bench` by default; it also handles the L2 trap below, which the
    hand-rolled loop above does **not**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The subtle one — Trap D: the L2 cache lies about bandwidth

    Time `fn` back-to-back on the **same buffers** and here's what happens: the first call
    pulls the data from DRAM, but it leaves it sitting in **L2 (48 MB on your 5070 Ti)**.
    Every call after that reads from L2, *not* DRAM — so you measure **cache bandwidth**
    (multiples of DRAM) and **over-report** your GB/s. Your kernel looks like it beat the
    memory roof. It didn't; you benchmarked the cache.

    `triton.testing.do_bench` guards against this: it **zeros an L2-sized scratch buffer
    between reps** to evict your data, forcing each timed call to go back to DRAM. If you
    hand-roll a timer for a *bandwidth* kernel, you must do the same (allocate ~64 MB and
    `.zero_()` it each iteration) — or vary the buffers — or your bandwidth number is a
    fantasy. (For a *compute*-bound kernel whose working set blows past L2 anyway, this
    matters far less.) When in doubt: use `do_bench`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Throughput: turn milliseconds into a rate, then judge it

    A raw time can't tell you if you're *good*: 1 ms is superb for a big matmul and awful
    for a small copy. You have to **normalize the work to a rate** and compare that rate to
    a ceiling. This is the **roofline** from `0D` coming back to collect.

    **Count the work the algorithm *must* do — not what a sloppy kernel actually moved.**
    Two currencies:

    **Bytes** (memory-bound ops) — every byte that crosses DRAM, reads **plus** writes:

    | op | bytes moved | reasoning |
    |---|---|---|
    | vector add  `c = a + b` | $3 N \cdot 4$ | read $a$, read $b$, write $c$ (fp32) |
    | copy | $2 N \cdot 4$ | read, write |
    | row-reduce  $M{\times}N \to M$ | $(MN + M)\cdot 4$ | read the whole matrix, write $M$ scalars |
    | softmax (row) | $2 MN \cdot 4$ | read the matrix, write the matrix |

    **FLOPs** (compute-bound ops) — count the math:

    | op | FLOPs | reasoning |
    |---|---|---|
    | matmul  $M{\times}K \cdot K{\times}N$ | $2 M N K$ | one multiply **and** one add per inner-product term |
    | attention | $\approx 4\,B H N^2 D$ | the two matmuls dominate; softmax is lower-order |

    Then the arithmetic is just *work ÷ time*:

    $$ \text{GB/s} = \frac{\text{bytes}}{t_{\text{ms}}\cdot 10^{-3}} \cdot 10^{-9}
       \qquad
       \text{TFLOP/s} = \frac{\text{FLOPs}}{t_{\text{ms}}\cdot 10^{-3}} \cdot 10^{-12} $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Judge the rate against **two** baselines

    **Baseline 1 — the hardware roof.** Memory-bound → compare GB/s to **~896 GB/s** (the
    5070 Ti's DRAM ceiling). Compute-bound → compare TFLOP/s to the fp32 / tensor-core
    peak. This tells you *how much is left on the table*. And it diagnoses: **96% of the
    bandwidth roof** means you're saturating DRAM — you're **done**, it's memory-bound and
    there's nothing left to win. **30% of the roof at 100% occupancy** is the textbook
    fingerprint of an **uncoalesced** kernel (straight out of `0D`): occupancy got you *to*
    a roof, but a low one.

    **Baseline 2 — torch.** The practical bar. Time the torch op the *same way* and take
    the ratio:

    ```python
    ref_ms = do_bench(lambda: reference(*inputs), warmup=25, rep=100)
    print(f"torch: {ref_ms:.3f} ms  ({ref_ms / your_ms:.2f}x your kernel)")
    ```

    A ratio **> 1** means you beat torch's kernel; **< 1** means cuBLAS/cuDNN — years of
    tuning — is still ahead, which for matmul is the normal, humbling result. **Both numbers
    matter and they're different questions:** you can sit at 60% of the roofline *and* 0.8×
    torch at the same time — which simply says torch is closer to the roof than you are, and
    points you at *its* tricks (vectorized loads, better tiling) rather than at the
    hardware. The harness prints exactly this line: `[REF] torch does it in ... (N.NNx your time)`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Feel the bytes ↔ time ↔ rate triangle

    A $2^{24}$-element fp32 vector add moves a **fixed** $3 N \cdot 4 \approx 0.2$ GB. The
    only variable is your kernel's time. Drag it and watch where you land on the roof — the
    default `0.234 ms` is the real number a coalesced vector-add hits on the 5070 Ti
    (≈ 860 GB/s, ≈ 96% of peak):
    """)
    return


@app.cell
def _(mo):
    time_ms = mo.ui.slider(start=0.20, stop=2.00, step=0.01, value=0.234,
                           label="your measured time (ms) for a 2^24 fp32 vector add")
    time_ms
    return (time_ms,)


@app.cell
def _(time_ms):
    def _run():
        import matplotlib.pyplot as plt

        _N = 1 << 24
        _bytes = 3 * _N * 4              # read a, read b, write c (fp32)
        _peak = 896.0                    # GB/s, 5070 Ti DRAM roof
        _ms = float(time_ms.value)
        _gbps = _bytes / (_ms * 1e-3) / 1e9
        _pct = 100 * _gbps / _peak

        if _pct >= 85:
            _verdict = "saturating DRAM -- memory-bound and basically done."
            _color = "#54a24b"
        elif _pct >= 40:
            _verdict = "partial -- imperfect coalescing or low occupancy; room to grow."
            _color = "#f58518"
        else:
            _verdict = "uncoalesced fingerprint (the 0D case): high occupancy, low bandwidth."
            _color = "#e45756"

        _fig, _ax = plt.subplots(figsize=(9.5, 2.6))
        _ax.barh([0], [_gbps], color=_color, height=0.5, zorder=3)
        _ax.axvline(_peak, color="#333", lw=2, ls="--")
        _ax.text(_peak, 0.45, "  896 GB/s roof", va="bottom", ha="left", fontsize=9)
        _ax.set_xlim(0, 1000)
        _ax.set_ylim(-0.5, 0.7)
        _ax.set_yticks([])
        _ax.set_xlabel("achieved bandwidth (GB/s)")
        _ax.set_title(f"{_bytes/1e9:.2f} GB  /  {_ms:.3f} ms  =  {_gbps:.0f} GB/s   ({_pct:.0f}% of peak)")
        _ax.text(15, 0, f"  {_verdict}", va="center", ha="left", fontsize=9,
                 color="white", fontweight="bold")
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Put it in one reusable harness

    Sections 1–3 collapse into a single function you can paste into any script and point at
    any kernel. This **is** `harness/runner.py`, distilled to its spine — reference first,
    `assert_close`, `do_bench` on both, throughput, ref-ratio:

    ```python
    import torch
    from triton.testing import do_bench

    def validate_and_bench(my_fn, ref_fn, inputs, *, work, metric="bandwidth",
                           tol=dict(atol=1e-2, rtol=1e-2)):
        # 1. CORRECTNESS -- reference captured BEFORE the kernel can touch inputs
        ref = ref_fn(*inputs)
        out = my_fn(*inputs)
        torch.testing.assert_close(out, ref, **tol)   # raises with a diagnostic on fail
        print("[PASS] correct")

        # 2. SPEED -- median ms, warmup + L2 flush handled by do_bench
        ms = do_bench(lambda: my_fn(*inputs), warmup=25, rep=100, return_mode="median")

        # 3. THROUGHPUT -- work is bytes moved OR flops, your choice of currency
        sec = ms * 1e-3
        if metric == "bandwidth":
            print(f"[PERF] {ms:.3f} ms   {work/sec/1e9:.0f} GB/s")
        else:
            print(f"[PERF] {ms:.3f} ms   {work/sec/1e12:.1f} TFLOP/s")

        # 4. THE TORCH BAR
        ref_ms = do_bench(lambda: ref_fn(*inputs), warmup=25, rep=100, return_mode="median")
        print(f"[REF]  torch {ref_ms:.3f} ms  ({ref_ms/ms:.2f}x your time)")
        return ms

    # --- use it ---
    N = 1 << 24
    torch.manual_seed(0)                                  # reproducible inputs
    a = torch.randn(N, device="cuda"); b = torch.randn(N, device="cuda")
    validate_and_bench(my_vector_add, lambda a, b: a + b, (a, b),
                       work=3 * N * 4, metric="bandwidth",
                       tol=dict(atol=0.0, rtol=0.0))       # add is exact -> demand 0/0
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The `spec.py` contract — the same thing, formalized

    Now the harness is no longer magic. Every `exercises/*/spec.py` encodes precisely the
    arguments `validate_and_bench` needs — so reading one tells you exactly what the runner
    will check and how it'll score you:

    ```python
    import torch

    TITLE      = "Vector add (your first Triton kernel)"
    ENTRYPOINT = "vector_add"          # the function the runner imports from kernel.py
    METRIC     = "bandwidth"           # "bandwidth" | "flops" | "none"
    TOL        = {"atol": 0.0, "rtol": 0.0}   # omit -> defaults to 1e-2 / 1e-2
    N = 1 << 24

    def make_inputs():
        torch.manual_seed(0)           # seeded -> reproducible across runs & machines
        a = torch.randn(N, device="cuda", dtype=torch.float32)
        b = torch.randn(N, device="cuda", dtype=torch.float32)
        return a, b

    def reference(a, b):               # the golden answer, in plain torch
        return a + b

    def bytes_moved(a, b):             # the "work" -> GB/s   (matmul specs define flops() -> 2*M*N*K)
        return 3 * a.numel() * a.element_size()
    ```

    Field-by-field that's: `TOL` → your tolerance dict; `reference` → `ref_fn`;
    `make_inputs` → `inputs`; `METRIC` + `bytes_moved`/`flops` → `work` and which rate to
    print. The runner reports `[TODO]` until your `kernel.py` stops raising
    `NotImplementedError` (the Triton sentinel; CUDA's is `return 77`) — then it runs this
    exact pipeline. **You can now read any spec and predict your score before you run it.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. When the rate says "slow" but not "why" — profile

    GB/s-vs-roofline tells you **whether** you're leaving performance on the table. It does
    **not** tell you **why**. For that you stop computing and start profiling:

    - **`ncu` (Nsight Compute)** — per-kernel: achieved occupancy, memory-throughput
      breakdown, warp-stall reasons, bank conflicts. The tool that turns "30% of roof" into
      "your loads are uncoalesced and you're stalling on memory." Introduced in `3E`.
    - **`nsys` (Nsight Systems)** — the timeline: launch overhead, gaps between kernels,
      H2D/D2H copies, whether the GPU is even busy. The tool for "my program is slow but the
      kernel is fast" (the answer is usually CPU-side or transfer overhead).

    Profiling is its own discipline — this is the signpost. The roofline numbers you just
    learned to compute are what you bring *into* the profiler so you know what you're hunting.
    See **`3E`** (occupancy tuning, where `ncu` lands) and the profiling section of **`7A`**.

    > **The loop:** write → **`assert_close`** (correct?) → **`do_bench` + roofline**
    > (fast vs the roof? vs torch?) → **`ncu`** (why not faster?) → fix → repeat. A kernel
    > isn't done until that loop has nothing left to say.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [7A: Study Guide & Reference](../7a_study_guide/) &nbsp;|&nbsp; *Reference — dip in anytime*
    """)
    return


if __name__ == "__main__":
    app.run()
