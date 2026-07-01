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
    # 3C: Warp-Level Primitives

    > *"A warp is not 32 threads that happen to march together — it is one machine with
    > 32 registers it can shuffle among its own lanes, no memory required."*

    In `1c` you wrote a tree reduction in Triton and `tl.sum` collapsed a block to one
    value. The classic CUDA way to reduce within a block uses **shared memory** and
    `__syncthreads()` (the `3b` machinery). But the last few steps of every reduction
    happen *inside a single warp* — and there the 32 lanes already run in lockstep
    (SIMT, from `0b`), so they can exchange registers **directly**, with no shared memory
    and no barrier. That is warp-synchronous programming, and the tools are the **warp
    shuffle** intrinsics.

    This lecture is about doing the last log-steps of a reduction at register speed.
    You'll meet `__shfl_down_sync` and `__shfl_xor_sync` for moving values between lanes,
    `__ballot_sync`/`__activemask` for asking which lanes are active, and the reason warp
    shuffles beat shared memory for the tail of a reduction. Then you write a warp reduce.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. The warp as a synchronous unit

    Recall from `0b`: a **warp** is 32 threads that issue the *same* instruction every
    cycle (SIMT). Because they advance in lockstep, after any instruction all 32 lanes
    are at the same program point — they are *implicitly synchronized* within the warp.
    That is the key that unlocks everything here: if lane 0 wants a value that lane 4
    just computed, no barrier is needed; they are already in step.

    **Lane ID.** A thread's position within its warp (0–31) is its *lane*. You can derive
    it from `threadIdx.x % 32`, or read the special register:

    ```cpp
    int lane = threadIdx.x & 31;        // lane id 0..31 within the warp
    int warp = threadIdx.x >> 5;        // which warp within the block
    ```

    **The `_sync` and the mask.** Modern CUDA shuffle intrinsics all end in `_sync` and
    take a 32-bit **mask** as the first argument — a bitset of *which lanes participate*.
    For a full warp that's `0xffffffff` (all 32 bits set). The mask exists because, after
    a divergent branch, only some lanes are active; the intrinsic must know which lanes
    are present so every participant agrees on the exchange. **All lanes named in the mask
    must execute the intrinsic**, or the result is undefined — the warp-level echo of the
    "every thread hits the barrier" rule from `3b`.

    ```cpp
    const unsigned FULL = 0xffffffff;   // all 32 lanes participate
    ```

    > [CUDA C++ Programming Guide §7.22, "Warp Shuffle Functions"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)
    > defines the `_sync` family and the participation mask.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. `__shfl_down_sync`: read a neighbor's register

    A warp shuffle lets a lane **read a register value from another lane** of the same
    warp directly — register-to-register, bypassing shared memory entirely. The workhorse
    for reductions is `__shfl_down_sync`:

    ```cpp
    float __shfl_down_sync(unsigned mask, float val, int delta);
    ```

    > Lane $i$ receives the `val` held by lane $i + \texttt{delta}$. Lanes near the top
    > (where $i+\texttt{delta} \ge 32$) keep their own `val` unchanged.

    This is exactly the data movement a tree reduction needs. To sum 32 lanes, you run a
    short sequence of folding steps, each one adding in the value from a lane `delta`
    above — the crux of `c03` is choosing those `delta`s:

    ```cpp
    // sum across a full warp with a tree of __shfl_down_sync folds
    for (int delta = /* ...? */; delta > 0; /* ...? */) {
        // each step folds the top half of the surviving lanes onto the
        // bottom half, one  val += __shfl_down_sync(FULL, val, delta)  at
        // a time. Work out the delta sequence — and which lane ends up
        // holding the total. That derivation is c03's job.
        ...
    }
    ```

    Trace it on the diagram below: each step folds the surviving upper lanes onto the
    lower ones, shrinking the live span until a single lane holds everything.
    **Five steps** ($\log_2 32$) collapse 32 values to one — the same tree depth as
    `1c`'s reduction, but with *no* shared-memory traffic and *no* `__syncthreads()`.
    Each step is one instruction per lane.

    This is the CUDA-level realization of the reduction pattern from `1c`: same
    $\log_2 N$ tree, executed at the warp level in registers instead of through shared
    memory.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Picture: the butterfly reduction

    The diagram traces a warp shuffle-down reduction (drawn for 8 lanes so it's legible;
    a real warp is 32 lanes / 5 steps). Each row is one step with halving `delta`; an
    arrow from lane $i+\texttt{delta}$ into lane $i$ means "lane $i$ adds the neighbor
    `delta` above." After $\log_2 N$ steps the full sum lands in lane 0. No shared memory,
    no barrier — just register exchanges between lockstep lanes.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        n = 8                      # illustrative; a real warp is 32
        steps = int(np.log2(n))    # 3 here; 5 for a warp
        _fig, _ax = plt.subplots(figsize=(9.2, 4.4))
        _ax.set_xlim(-1.2, n)
        _ax.set_ylim(-0.8, steps + 1.2)
        _ax.axis("off")
        _ax.set_title("warp shuffle-down reduction (8 lanes shown; a warp is 32 / 5 steps)",
                      fontsize=10)

        recv_color = "#4c9f70"

        # row y for step s: top = before, descending
        def yrow(s):
            return steps - s

        # draw lane nodes at every level (steps+1 levels)
        for level in range(steps + 1):
            for lane in range(n):
                _ax.scatter([lane], [yrow(level)], s=240,
                            color="#eef3ff", edgecolor="#9bb7e8", zorder=2)
                _ax.text(lane, yrow(level), str(lane), ha="center", va="center",
                         fontsize=7, color="#3060c0", zorder=3)
            _ax.text(-1.0, yrow(level),
                     "input" if level == 0 else f"after delta={1 << (steps - level)}",
                     ha="left", va="center", fontsize=7.5, color="#666")

        # draw fold arrows between levels: each step folds lane (i+delta) into lane i
        for s in range(steps):
            delta = 1 << (steps - 1 - s)
            y0 = yrow(s)
            y1 = yrow(s + 1)
            for lane in range(n):
                src = lane + delta
                # canonical down-fold: receivers are the lower lane of each pair
                if src < n and (lane % (delta * 2) < delta):
                    _ax.annotate("", xy=(lane, y1 + 0.18),
                                 xytext=(src, y0 - 0.18),
                                 arrowprops=dict(arrowstyle="->",
                                                 color=recv_color, lw=1.6,
                                                 alpha=0.85))
        _ax.scatter([0], [yrow(steps)], s=320, color="#fde9e9",
                    edgecolor="#d65f5f", zorder=4)
        _ax.text(0, yrow(steps), "0", ha="center", va="center",
                 fontsize=8, color="#d65f5f", weight="bold", zorder=5)
        _ax.text(1.2, yrow(steps), "<- full sum in lane 0",
                 ha="left", va="center", fontsize=8.5, color="#d65f5f")
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. `__shfl_xor_sync`: the butterfly that broadcasts

    `__shfl_down_sync` leaves the answer in **one** lane (lane 0). Sometimes you want
    *every* lane to end up with the reduced value — an **all-reduce** within the warp
    (e.g. so every lane can normalize by the warp's sum, as in a softmax). For that, use
    the XOR / butterfly shuffle:

    ```cpp
    float __shfl_xor_sync(unsigned mask, float val, int laneMask);
    ```

    > Lane $i$ exchanges with lane $i \oplus \texttt{laneMask}$ (bitwise XOR). The pattern
    > is symmetric — both partners swap — so it's a *butterfly* network.

    Run the same halving tree with XOR partners and **every** lane accumulates the warp
    total:

    ```cpp
    // all-reduce sum across a warp: every lane ends with the total
    for (/* the same halving tree as the down-shuffle version */) {
        // val += __shfl_xor_sync(FULL, val, m);  — same laneMask sequence
        // you derive for c03; only the partner pattern changes
        ...
    }
    // now ALL 32 lanes hold the sum
    ```

    The difference from `__shfl_down_sync` is *who ends up with the answer*:
    `down` funnels everything into lane 0 (good when one lane writes the result); `xor`
    butterfly-broadcasts so all lanes agree (good when every lane needs the result). Same
    five steps, same $\log_2 32$ depth — pick by what happens next. The XOR butterfly is
    also the natural pattern for warp-level `max`/`min` (swap `+=` for `fmaxf`).

    > The butterfly/XOR exchange is the classic Batcher network; `__shfl_xor_sync` maps it
    > to one warp instruction per step.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. `__ballot_sync` & `__activemask`: which lanes are live?

    Reductions move *data*; sometimes you need to reduce *predicates* — "how many lanes
    satisfy a condition?", "which lanes are active right now?". Two intrinsics answer
    that at warp speed.

    **`__ballot_sync(mask, predicate)`** evaluates the boolean `predicate` on every
    participating lane and returns a 32-bit integer whose bit $i$ is set iff lane $i$'s
    predicate was true. It's a one-instruction vote across the warp:

    ```cpp
    unsigned votes = __ballot_sync(FULL, x > 0.0f);   // bit i set iff lane i has x>0
    int count = __popc(votes);                        // how many lanes voted yes
    ```

    `__popc` (population count) then turns the bitmask into a tally — a warp-wide
    count in two instructions, no shared memory. This is the engine behind **stream
    compaction** (each lane finds its output slot by counting set bits below it) and
    histogram-style kernels.

    **`__activemask()`** returns the set of lanes currently active at this exact point.
    It is tempting to use it as the participation mask after a divergent branch —
    **don't**. NVIDIA documents that as an anti-pattern: `__activemask()` merely
    *reports* which lanes happen to be executing right now, with **no guarantee that
    those lanes are converged** — the scheduler is free to have lanes on the same path
    arrive at the intrinsic separately, in which case two "participants" can compute
    *different* masks and the exchange is broken. Treat it as a **diagnostic** (log or
    assert who's present), never as the fix.

    The correct tool is the one you just met: **derive the membership mask from the
    branch condition itself** with `__ballot_sync`, taken *before* the divergent region
    while the full warp is still converged, and pass *that* mask to the shuffle:

    ```cpp
    unsigned mask = __ballot_sync(0xffffffff, cond);  // who will take the branch —
                                                      // computed while all 32 agree
    if (cond) {
        // every lane named in `mask` is here, and all of them execute this:
        val += __shfl_down_sync(mask, val, delta);
    }
    ```

    The caution from §1 applies doubly: an intrinsic's mask must name *exactly* the lanes
    that will execute it. Using a stale `FULL` mask after a divergent branch — when some
    lanes have peeled off — is undefined behavior; so is guessing the mask with
    `__activemask()`. Compute the mask from the condition, and the mask and the control
    flow cannot disagree.

    > [CUDA C++ Programming Guide §7.21, "Warp Vote Functions"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-vote-functions)
    > covers `__ballot_sync`/`__all_sync`/`__any_sync`; `__activemask()` is in §7.24.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Why shuffles beat shared memory for the tail

    A full block reduction (say 256 threads → 1 value) is usually done in **two phases**,
    and warp shuffles win the second one:

    1. **Across warps — shared memory.** Each of the 8 warps reduces its 32 lanes (one
       shuffle reduction), then writes its partial to a tiny `__shared__` array of 8
       slots, a `__syncthreads()`, and one warp finishes the 8 partials.
    2. **Within a warp — shuffles.** The per-warp reduction and the final fold-up are
       done entirely with `__shfl_down_sync`, never touching shared memory.

    Why prefer shuffles for the within-warp part? Three concrete wins over the
    shared-memory tree of `3b`:

    - **No `__syncthreads()`.** Lanes in a warp are already in lockstep, so no barrier is
      needed between steps. The shared-memory tree pays a block-wide barrier *per step*;
      the warp shuffle pays none.
    - **No shared-memory traffic or banks.** Values move register-to-register, so you
      neither spend shared-memory capacity (an occupancy resource, `0d`) nor risk bank
      conflicts (`3d`). The last log-steps of a shared-memory reduction are also where
      bank conflicts bite hardest — shuffles sidestep that entirely.
    - **Fewer instructions.** One `__shfl_down_sync` replaces a shared store + barrier +
      shared load per step.

    The mental rule: **reduce across warps through shared memory, reduce within a warp
    with shuffles.** The crossover is the warp boundary — above 32 you need shared memory
    to cross warps; at or below 32 the warp is its own synchronous machine. This is the
    same tree from `1c`, now split at the hardware's natural seam.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactive: shuffle steps vs warp size

    A shuffle reduction takes $\lceil\log_2 W\rceil$ steps for a warp of $W$ lanes
    (real hardware fixes $W=32$, so 5 steps — but seeing the log scaling is the point).
    Slide the warp size; the plot shows the halving `delta` sequence and the step count,
    next to the linear cost a naive lane-by-lane reduction would pay. The gap between the
    flat-ish $\log_2 W$ curve and the linear one is why the tree — and the shuffle that
    implements it — matters.
    """)
    return


@app.cell
def _(mo):
    warp_slider = mo.ui.slider(start=2, stop=64, step=2, value=32,
                               label="warp size W (real hardware = 32)")
    warp_slider
    return (warp_slider,)


@app.cell
def _(warp_slider):
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        W = int(warp_slider.value)
        log_steps = int(np.ceil(np.log2(W)))
        deltas = [1 << s for s in range(log_steps - 1, -1, -1)]

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6),
                                          gridspec_kw={"width_ratios": [1.1, 1]})

        # left: the delta sequence as descending bars
        idx = np.arange(len(deltas))
        _ax1.bar(idx, deltas, color="#5b8def", edgecolor="white")
        _ax1.set_xticks(idx)
        _ax1.set_xticklabels([f"s{ s+1 }" for s in idx], fontsize=8)
        _ax1.set_ylabel("shuffle delta (lanes)")
        _ax1.set_title(f"W={W}: {log_steps} steps, delta = " +
                       ", ".join(str(d) for d in deltas), fontsize=8.5)
        for _i, _d in zip(idx, deltas):
            _ax1.text(_i, _d, str(_d), ha="center", va="bottom", fontsize=7.5,
                      color="#3060c0")

        # right: log2 tree steps vs linear, over a range of W
        ws = np.arange(2, 65, 2)
        tree = np.ceil(np.log2(ws))
        linear = ws - 1
        _ax2.plot(ws, linear, color="#d65f5f", linewidth=2, label="naive: W-1 steps")
        _ax2.plot(ws, tree, color="#4c9f70", linewidth=2, label="shuffle tree: log2(W)")
        _ax2.scatter([W], [log_steps], color="#4c9f70", s=70, zorder=5)
        _ax2.axvline(32, color="#999", linestyle=":", linewidth=1)
        _ax2.text(32.5, linear.max() * 0.85, "W=32\n(a warp)", fontsize=7.5,
                  color="#666")
        _ax2.set_xlabel("warp size W")
        _ax2.set_ylabel("reduction steps")
        _ax2.set_title("tree depth vs naive")
        _ax2.legend(fontsize=7.5, loc="upper left")
        _ax2.grid(True, alpha=0.15)

        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    Warp-level primitives are how fast reductions, softmaxes, and scans are actually
    written on real hardware.

    - **Own the warp.** Within 32 lanes you have an implicitly-synchronized machine —
      exchange registers with `__shfl_*_sync`, no barrier, no shared memory.
    - **Pick `down` vs `xor` by who needs the answer.** `__shfl_down_sync` funnels to
      lane 0 (one writer); `__shfl_xor_sync` butterfly-broadcasts to all lanes
      (everyone normalizes). Same $\log_2 32$ cost.
    - **Split reductions at the warp boundary.** Across warps: shared memory +
      `__syncthreads()` (`3b`). Within a warp: shuffles. The tail of every reduction
      belongs to the shuffles — fewer instructions, no banks, no barrier.
    - **Mind the mask.** Every lane in the participation mask must execute the intrinsic;
      around divergence, derive the mask from the branch condition with `__ballot_sync`
      (`__activemask()` is a diagnostic, not a participation mask). It's the warp-level
      version of the all-threads-hit-the-barrier rule.
    - **Votes are reductions too.** `__ballot_sync` + `__popc` count predicates
      across a warp in two instructions — the basis of compaction and histograms.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Now go write it

    Time to write a reduction the way fast kernels do it — warp shuffles for the tail.
    You'll implement the $\log_2 32$ `__shfl_down_sync` fold for the within-warp reduce,
    then combine warps through a small shared-memory array:

    ```bash
    python -m harness.runner c03 --watch
    ```

    `c03` is the CUDA warp reduce — the by-hand, to-the-metal version of the reduction you
    wrote in Triton (`e04`). The harness builds it with `nvcc -arch=sm_120` and checks
    your block sum against the reference. Get the mask right, get the halving `delta`
    sequence right, and watch a barrier-free warp reduction land — then bolt on the
    shared-memory cross-warp step for a full block reduce.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [3B: Shared-Memory Tiling](../3b_shared_tiling/) &nbsp;|&nbsp; Next: [3D: Coalescing & Bank Conflicts](../3d_memory_banks/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
