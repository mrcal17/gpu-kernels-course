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
    # 1G: Scan / Prefix-Sum

    > *"A reduction throws away every step on the way to the answer. A scan keeps
    > all of them — and that is exactly what makes it hard."*

    **This module is ADVANCED / OPTIONAL — it is off the critical path.** Scan is the
    first pattern in the course with a *genuine* data dependency: output $i$ depends on
    the running result of everything before it, so you cannot just shard the array and
    work independently. It is the canonical "hard parallel pattern," and the techniques
    here — work-vs-depth trade-offs, up-sweep/down-sweep trees, block-local scan with a
    cross-block carry — generalize far beyond cumsum (linear attention and state-space
    models lean on them). If you're racing to Flash Attention, skip ahead; come back when
    you want to understand the trick that makes recurrences parallel.

    We build it from the reduction you already wrote in `1c`, then break the sequential
    dependency two different ways.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. What scan is

    A **scan** (a.k.a. prefix sum) takes an array and produces, at each position, the
    running result of an operator applied to everything up to that point. For the
    **inclusive** prefix sum, position $i$ includes itself:

    $$y_i = \sum_{j \le i} x_j.$$

    For the **exclusive** prefix sum, position $i$ excludes itself, so $y_0$ is the
    operator's **identity** (0 for sum):

    $$y_i = \sum_{j < i} x_j, \qquad y_0 = 0.$$

    The two are one shift apart. Exclusive is inclusive shifted right by one (drop the
    last, prepend the identity); equivalently:

    $$\boxed{\;\text{inclusive}_i = \text{exclusive}_i + x_i\;}$$

    **Tiny worked example.** Take $x = [3, 1, 4, 1, 5]$ and add by hand left to right:

    | index $i$ | 0 | 1 | 2 | 3 | 4 |
    |---|---|---|---|---|---|
    | $x_i$ | 3 | 1 | 4 | 1 | 5 |
    | inclusive $y_i$ | 3 | 4 | 8 | 9 | 14 |
    | exclusive $y_i$ | 0 | 3 | 4 | 8 | 9 |

    Nothing here is special to addition. Scan works for **any associative operator** —
    max, min, product, bitwise-or, even matrix multiply. That generality is why the
    pattern is called an **associative scan**; **cumsum is just the sum instance**.
    Associativity is the whole game: it's what lets you re-bracket the additions into a
    tree instead of a chain, which is how the parallel versions below break the
    left-to-right dependency.

    > [PMPP](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)
    > Ch. 11 ("Prefix sum") develops inclusive/exclusive scan and both parallel
    > algorithms below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Why scan is HARDER than reduction

    In `1c` a **reduction** collapsed $N$ values into **one**. That is forgiving: every
    intermediate partial sum can be thrown away the instant it's folded in, and a
    balanced binary tree of additions gives you $O(\log N)$ depth with $O(N)$ work
    almost for free — pair up neighbours, then pair up the pairs, and so on.

    A scan refuses to forget. It must produce **$N$ outputs**, and output $i$ depends on
    a *different-length prefix* — the running result of $x_0 \dots x_i$. That is a real
    **sequential data dependency**: naively, $y_i = y_{i-1} + x_i$ chains every output to
    the one before it. The obvious serial loop is cheap in total operations but maximally
    serial:

    $$\text{serial scan:}\quad \text{work} = O(N), \quad \text{depth} = O(N).$$

    $O(N)$ **depth** is the problem. Depth is the length of the longest dependency chain
    — the number of sequential steps that *cannot* overlap no matter how many lanes you
    have. From `0b`: the GPU wants a flood of independent parallel work to hide latency.
    A chain of $N$ dependent adds gives it exactly one thing to do at a time and starves
    every other lane. The reduction's tree had depth $\log N$; the naive scan's chain has
    depth $N$.

    So the challenge — and what makes scan the harder sibling of reduction — is producing
    **all $N$ outputs** with **low depth** ($O(\log N)$, so the GPU stays fed) **and** low
    work (close to the serial $O(N)$, so you don't burn bandwidth and energy doing
    redundant adds). The next two sections are the two classic answers, and they sit at
    opposite corners of that work-vs-depth trade.

    > This work/depth tension is the *parallel* lens on an algorithm (the PRAM model):
    > **work** = total operations, **depth** = critical-path length. Brent's theorem ties
    > the two to achievable runtime on $p$ processors. Keep both in your head — it's a
    > recurring GPU design axis, not a scan-only curiosity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Hillis-Steele (the "naive" / step-efficient scan)

    The first parallel scan is the easy one to write and the easy one to reason about.
    It computes the **inclusive** scan in place by doubling strides. At step
    $d = 0, 1, 2, \dots$ every element $i$ with $i \ge 2^d$ does:

    $$x_i \mathrel{+}= x_{i - 2^d}.$$

    All lanes do this *simultaneously* (read the old values, then write) — that's the
    parallelism. After $\lceil \log_2 N \rceil$ steps, every $x_i$ holds the sum of all
    elements up to and including $i$: the inclusive scan.

    **Walk it on $x = [3, 1, 7, 0, 4, 1, 6, 3]$** ($N = 8$, so 3 steps):

    - **start:** `[3, 1, 7, 0, 4, 1, 6, 3]`
    - **$d=0$** (stride 1, add left neighbour): `[3, 4, 8, 7, 4, 5, 7, 9]`
    - **$d=1$** (stride 2): `[3, 4, 11, 11, 12, 12, 11, 14]`
    - **$d=2$** (stride 4): `[3, 4, 11, 11, 15, 16, 22, 25]`  ← inclusive scan

    Each step doubles how far each position has "reached," so $\log_2 N$ steps reach the
    whole prefix. The diagram below makes the doubling stride visual.

    **Analysis.** The depth is wonderful, the work is not:

    $$\text{Hillis-Steele:}\quad \text{depth} = O(\log N), \quad
      \text{work} = O(N \log N).$$

    It does an add for (almost) every element at (almost) every one of the $\log N$
    steps — *more* total adds than the serial $O(N)$. It is **step-efficient** (minimal
    depth) but **work-inefficient**. For small $N$ that's fine and often fastest; for
    large $N$ the extra factor of $\log N$ in work — and the bandwidth to move all those
    values — starts to bite. It is the natural one to write in shared memory: one buffer,
    a doubling-stride loop, a barrier between steps.

    ```python
    # Hillis-Steele inclusive scan, illustrative (one block, shared array `s`)
    d = 1
    while d < N:
        # read-then-write: snapshot, then add the element `d` to the left
        prev = s.copy()              # in a kernel: double-buffer or sync between r/w
        for i in range(d, N):        # all i run in parallel on the GPU
            s[i] = prev[i] + prev[i - d]
        barrier()                    # __syncthreads(): finish the step before the next
        d *= 2
    ```

    > [GPU Gems 3, Ch. 39 "Parallel Prefix Sum (Scan) with CUDA"](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
    > presents this as the first ("naive") scan and is the reference for the
    > shared-memory version.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        _x0 = np.array([3, 1, 7, 0, 4, 1, 6, 3], dtype=int)
        _n = _x0.size
        _n_steps = int(np.log2(_n))   # 3 steps for N=8

        # Simulate Hillis-Steele, capturing the array after each step.
        _rows = [("start", _x0.copy(), -1)]
        _cur = _x0.copy()
        for _d_idx in range(_n_steps):
            _stride = 1 << _d_idx
            _prev = _cur.copy()
            for _i in range(_stride, _n):
                _cur[_i] = _prev[_i] + _prev[_i - _stride]
            _rows.append((f"d={_d_idx}  (stride {_stride})", _cur.copy(), _stride))

        _n_rows = len(_rows)
        _fig, _ax = plt.subplots(figsize=(8.5, 4.6))
        _ax.set_xlim(-0.6, _n - 0.4)
        _ax.set_ylim(-0.6, _n_rows - 0.4)
        _ax.axis("off")
        _ax.set_title("Hillis-Steele inclusive scan: doubling-stride steps "
                      "(stride $2^d$)")

        _box_w, _box_h = 0.86, 0.74
        for _r, (_label, _vals, _stride) in enumerate(_rows):
            _y = _n_rows - 1 - _r           # top row first
            _is_final = (_r == _n_rows - 1)
            _ax.text(-0.55, _y, _label, ha="right", va="center",
                     fontsize=8.5, color="#333")
            for _i in range(_n):
                _face = "#dff0e4" if _is_final else "#eef3ff"
                _edge = "#4c9f70" if _is_final else "#5b8def"
                _ax.add_patch(plt.Rectangle(
                    (_i - _box_w / 2, _y - _box_h / 2), _box_w, _box_h,
                    facecolor=_face, edgecolor=_edge, linewidth=1.4))
                _ax.text(_i, _y, str(int(_vals[_i])), ha="center", va="center",
                         fontsize=10, weight="bold", color="#22324a")

            # Arrows: element i adds element i-stride from the row ABOVE.
            if _stride > 0:
                _y_src = _y + 1
                for _i in range(_stride, _n):
                    _ax.annotate(
                        "", xy=(_i - 0.18, _y + _box_h / 2 + 0.02),
                        xytext=(_i - _stride + 0.18, _y_src - _box_h / 2 - 0.02),
                        arrowprops=dict(arrowstyle="->", color="#d65f5f",
                                        lw=1.1, alpha=0.85))

        _ax.text(_n - 0.4, -0.5, "red arrow: $x_i \\mathrel{+}= x_{i-2^d}$",
                 ha="right", va="center", fontsize=8, color="#d65f5f")
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Blelloch (work-efficient, two-phase)

    Hillis-Steele wastes work. **Blelloch's scan** recovers the optimal $O(N)$ work by
    running the computation as a balanced tree *twice* — once up, once down — and it
    naturally produces the **exclusive** scan. Two phases over the same in-place array:

    **Up-sweep (reduce).** This is exactly the `1c` reduction tree, but you *keep* the
    partial sums in place instead of discarding them. With doubling stride
    $d = 1, 2, 4, \dots$, each right-hand node of a pair absorbs its left sibling:

    $$x[i + 2d - 1] \mathrel{+}= x[i + d - 1].$$

    After the up-sweep the last element holds the total sum, and every internal node
    holds the sum of its subtree. Cost: $O(N)$ work, $O(\log N)$ depth — same as a
    reduction.

    **Down-sweep.** Now push the partials back down to turn subtree-sums into prefixes:

    1. **Set the root (last element) to the identity** ($0$ for sum). It seeds the
       exclusive scan.
    2. Walk the strides back down. At each node, the rule is **swap-and-add**: the node
       passes its current value **left** to its left child, and the **sum** (old left
       child + node's value) goes to the **right** child:

       ```text
       temp        = x[i + d - 1]        # left child's value
       x[i + d - 1] = x[i + 2d - 1]      # left child receives the node's value
       x[i + 2d - 1] = x[i + 2d - 1] + temp   # right child gets node + old left
       ```

    When the down-sweep finishes, every position holds the sum of everything strictly to
    its left — the **exclusive** scan. (Want inclusive? Add $x_i$ back, per §1, or shift.)
    Cost again: $O(N)$ work, $O(\log N)$ depth.

    **Totals and the trade.**

    $$\text{Blelloch:}\quad \text{work} = O(N)\ (\approx 2N), \quad
      \text{depth} = O(\log N)\ (\approx 2\log N).$$

    Versus Hillis-Steele:

    - **Blelloch is work-efficient** — $\approx 2N$ adds total, matching the serial
      work, so it moves far less data. That's the win for **large $N$**, and for
      bandwidth-/energy-bound settings (the common GPU case).
    - **The cost is depth and complexity** — two passes means $\approx 2\times$ the
      sync barriers and the fiddly swap-and-add indexing. **Hillis-Steele** is simpler
      and lower-latency, so it often wins for **small $N$** (a single warp or small
      block) where the extra work is cheap and the second pass isn't worth it.

    Same problem, two answers at opposite corners of the work/depth plane — exactly the
    trade §2 set up. The interactive below lets you see the crossover.

    ```python
    # Blelloch exclusive scan, illustrative structure (one block, array `s`, size N)
    # --- up-sweep (reduce): build subtree sums in place ---
    d = 1
    while d < N:
        for i in range(0, N, 2 * d):     # parallel over the stride
            s[i + 2 * d - 1] += s[i + d - 1]
        barrier()
        d *= 2

    s[N - 1] = 0                          # seed the exclusive scan with the identity

    # --- down-sweep: push partials down, swap-and-add ---
    d = N // 2
    while d >= 1:
        for i in range(0, N, 2 * d):     # parallel over the stride
            t              = s[i + d - 1]
            s[i + d - 1]   = s[i + 2 * d - 1]      # left child <- node
            s[i + 2 * d - 1] += t                  # right child <- node + old left
        barrier()
        d //= 2
    # s now holds the EXCLUSIVE scan
    ```

    > [Blelloch, "Prefix Sums and Their Applications" (1990)](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)
    > is the original work-efficient algorithm;
    > [GPU Gems 3, Ch. 39](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
    > gives the up-sweep/down-sweep CUDA version (and the bank-conflict fixes you'll
    > meet in `1b`'s spirit).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Work vs depth, made tangible

    Slide $\log_2 N$ to set the array size $N = 2^{\log_2 N}$. The left panel plots
    **total work** (operations) and the right plots **depth** (critical-path steps), each
    over a sweep of $N$, with your current $N$ marked. Three algorithms:

    - **serial** — $\text{work} = N$ (optimal), $\text{depth} = N$ (terrible).
    - **Hillis-Steele** — $\text{work} = N\log_2 N$ (worse), $\text{depth} = \log_2 N$
      (best).
    - **Blelloch** — $\text{work} \approx 2N$ (optimal order), $\text{depth} \approx
      2\log_2 N$ (still log, ~2× HS).

    Each algorithm spends one axis to save the other — which one you'd rather spend is
    the whole choice.
    """)
    return


@app.cell
def _(mo):
    log2n_slider = mo.ui.slider(start=2, stop=20, step=1, value=10,
                                label="log2(N)  ->  array size N = 2**log2N")
    log2n_slider
    return (log2n_slider,)


@app.cell
def _(log2n_slider):
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        _log2n = int(log2n_slider.value)
        _N = 2 ** _log2n

        # Sweep of sizes for the curves.
        _exps = np.arange(2, 21)
        _Ns = 2.0 ** _exps
        _logs = _exps.astype(float)            # log2(N) for each size

        # Work and depth models.
        _work = {
            "serial": _Ns,
            "Hillis-Steele": _Ns * _logs,
            "Blelloch": 2.0 * _Ns,
        }
        _depth = {
            "serial": _Ns,
            "Hillis-Steele": _logs,
            "Blelloch": 2.0 * _logs,
        }
        _colors = {
            "serial": "#d65f5f",
            "Hillis-Steele": "#e0a458",
            "Blelloch": "#4c9f70",
        }

        # Current-N values for the markers.
        _work_now = {"serial": _N, "Hillis-Steele": _N * _log2n,
                     "Blelloch": 2 * _N}
        _depth_now = {"serial": _N, "Hillis-Steele": _log2n,
                      "Blelloch": 2 * _log2n}

        _fig, (_ax_w, _ax_d) = plt.subplots(1, 2, figsize=(9.6, 4.0))

        for _name in _work:
            _ax_w.plot(_Ns, _work[_name], color=_colors[_name],
                       linewidth=2, label=_name)
            _ax_w.scatter([_N], [_work_now[_name]], color=_colors[_name],
                          s=45, zorder=5, edgecolor="white", linewidth=0.8)
        _ax_w.axvline(_N, color="#999", linestyle=":", linewidth=1.1)
        _ax_w.set_xscale("log", base=2)
        _ax_w.set_yscale("log")
        _ax_w.set_xlabel("array size  N")
        _ax_w.set_ylabel("total work (operations)")
        _ax_w.set_title("WORK: serial & Blelloch are O(N); HS is O(N log N)")
        _ax_w.legend(loc="upper left", fontsize=8)
        _ax_w.grid(True, which="both", alpha=0.15)

        for _name in _depth:
            _ax_d.plot(_Ns, _depth[_name], color=_colors[_name],
                       linewidth=2, label=_name)
            _ax_d.scatter([_N], [_depth_now[_name]], color=_colors[_name],
                          s=45, zorder=5, edgecolor="white", linewidth=0.8)
        _ax_d.axvline(_N, color="#999", linestyle=":", linewidth=1.1)
        _ax_d.set_xscale("log", base=2)
        _ax_d.set_yscale("log")
        _ax_d.set_xlabel("array size  N")
        _ax_d.set_ylabel("depth (critical-path steps)")
        _ax_d.set_title("DEPTH: serial is O(N); HS & Blelloch are O(log N)")
        _ax_d.legend(loc="upper left", fontsize=8)
        _ax_d.grid(True, which="both", alpha=0.15)

        _fig.suptitle(
            f"N = 2^{_log2n} = {_N:,}    |    "
            f"work:  serial {_N:,}  ·  HS {_N * _log2n:,}  ·  Blelloch {2 * _N:,}    |    "
            f"depth:  serial {_N:,}  ·  HS {_log2n}  ·  Blelloch {2 * _log2n}",
            y=1.02, fontsize=8.5)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    - **Scan is the canonical hard parallel pattern.** Unlike vector-add or even a
      reduction, it has a *real* data dependency — output $i$ needs the running result
      up to $i$. It is not embarrassingly parallel, and learning to break the
      left-to-right chain is a transferable skill.
    - **Work vs depth is a recurring GPU design axis.** The Brent's-theorem-flavored
      trade — minimize the critical path (depth) without blowing up total operations
      (work) — shows up again in tree reductions, sorts, and segmented scans. Scan is
      where it's clearest: serial, Hillis-Steele, and Blelloch are three points on that
      plane.
    - **Associative scan is everywhere.** Cumsum is one instance; the same machinery
      powers running statistics, stream compaction (exclusive scan of a flag array gives
      output offsets), histogramming, and — most relevant to modern ML — **linear
      attention and state-space models (SSMs)**, where the recurrence is reframed as an
      associative scan to make it parallel.
    - **Block-local scan + cross-block carry is the standard decomposition.** It mirrors
      the block reductions of `1c`: each block scans its own tile, you scan the
      per-block totals, then add each block's exclusive carry to its tile. One in-block
      scan kernel composes into a full-array scan — the same hierarchy you've used all
      along.

    Master the dependency-breaking idea here and recurrences stop looking inherently
    serial.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Now go write it

    Time to make a recurrence parallel yourself. The exercise is a **cumsum** kernel,
    scored on achieved **bandwidth** (scan is memory-bound — place it on `0d`'s roofline
    and you'll see it pinned to the slanted memory roof):

    ```bash
    python -m harness.runner e09 --watch
    ```

    `e09` is the crux of this lecture: do a **block-local scan** (Triton ships
    `tl.cumsum` as a building block for the in-block part) and then stitch blocks
    together with a **cross-block carry** — the block-local + carry decomposition from
    the takeaways. Choosing inclusive vs exclusive, picking Hillis-Steele vs Blelloch for
    the in-block scan, and getting the carry right are yours to write. (If the stub isn't
    on disk yet, this is your forward pointer — it's the cumsum exercise.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [1F: Fused Norms](../1f_fused_norms/) &nbsp;|&nbsp; Next: [2A: Autotuning](../2a_autotuning/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
