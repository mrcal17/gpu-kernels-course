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
    # 1C: Reductions

    > *"A map keeps the shape. A reduction collapses it — and collapsing in parallel is
    > where the GPU earns its keep."*

    So far every kernel has been a **map**: one output per input, lanes working in happy
    isolation (`1a`), each warp reading its own contiguous tile (`1b`). Reductions break
    that independence. A **reduction** combines many values into few — sum a vector to a
    scalar, take the max of each row of a matrix — and now lanes must *cooperate*, because
    the answer depends on data spread across the whole block.

    This is the second fundamental parallel pattern, and it underlies almost everything
    interesting that follows: softmax (`1d`) is a max-reduce then a sum-reduce; layernorm
    (`1f`) is a mean and a variance reduce; attention (`2b`) is a reduction over keys.
    Get the reduction pattern right and those become variations on a theme. We'll build
    the **tree reduction**, see why it's log-depth, do it per-block with `tl.sum`/`tl.max`
    along an axis, and watch the numerical traps that come with summing many numbers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. The reduction pattern

    A reduction folds a collection $\{x_0, x_1, \dots, x_{n-1}\}$ down to a single value
    using an **associative** binary operator $\oplus$:

    $$\text{reduce}_\oplus(x) = x_0 \oplus x_1 \oplus \cdots \oplus x_{n-1}.$$

    The usual operators — $+$ (sum), $\max$, $\min$, $\times$ (product) — are all
    associative and commutative, which is exactly what lets us reorder the work for
    parallelism. The naive serial version is a single accumulator swept left to right:

    $$\text{acc} \leftarrow \text{acc} \oplus x_i \quad \text{for } i = 0 \dots n-1,$$

    which takes $n-1$ steps, each depending on the last — a **chain of length $n$**. On a
    CPU that's fine. On a GPU it's a disaster: the entire warp would serialize on one
    accumulator, the other 31 lanes idle. The whole machine wants to do many of these
    folds *at once*. Associativity is the permission slip: because
    $(a\oplus b)\oplus c = a\oplus(b\oplus c)$, we may fold pairs in parallel and combine
    the partials in any grouping we like.

    Two flavors you'll write:

    - **Full reduce:** a whole array → one scalar (e.g. the L2 norm's $\sum x_i^2$).
    - **Row reduce:** an $R \times C$ matrix → an $R$-vector, reducing along the rows
      (each program handles one row). This is `e04`, and the building block of softmax and
      the norms.

    > [PMPP](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)
    > Ch. 10 ("Reduction") is the canonical treatment of the parallel reduction pattern.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Tree reduction: why it's log-depth

    Instead of one long chain, pair up the elements and fold each pair in parallel, then
    pair the results, and so on. Each step **halves** the number of live values:

    $$n \;\to\; \tfrac{n}{2} \;\to\; \tfrac{n}{4} \;\to\; \cdots \;\to\; 1.$$

    The number of *steps* (the **depth**) is therefore $\lceil \log_2 n \rceil$, while the
    total *work* (number of $\oplus$ operations) is still $n-1$ — the same as serial. Two
    quantities, two different costs:

    $$\underbrace{\text{work} = n - 1}_{\text{operations, unchanged}}
      \qquad
      \underbrace{\text{depth} = \lceil \log_2 n \rceil}_{\text{sequential steps, collapsed}}.$$

    On a parallel machine with enough lanes, the depth is what you wait for. Summing
    $n = 1024$ elements: serial is **1023** dependent steps; the tree is **10**. That gap
    — $n$ vs. $\log_2 n$ — is the entire reason reductions belong on a GPU.

    The catch is the **tail of the tree**: near the root only a few lanes are still active
    (then 2, then 1), so the warp is mostly idle in the last steps. That's fine for one
    block's worth of data, but for *huge* arrays you don't reduce all the way down in one
    block — you do per-block partials and combine them (§4).
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        print("=== Tree reduction: depth vs. work ===\n")
        print(f"  {'n':>8s} {'serial steps':>13s} {'tree depth':>11s} {'speedup':>9s}")
        print("  " + "-" * 46)
        for _n in [16, 64, 256, 1024, 4096, 65536]:
            _serial = _n - 1
            _depth = int(np.ceil(np.log2(_n)))
            print(f"  {_n:>8d} {_serial:>13d} {_depth:>11d} {_serial / _depth:>8.0f}x")
        print("\n  Work is the same (~n ops); depth collapses from n to log2(n).")
        print("  On a parallel machine you pay the DEPTH, not the work.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The tree, drawn

    Below is a sum-reduction of 8 values as a binary tree. Read it **bottom-up**: the
    leaves are the inputs, each level pairs and folds, and after $\log_2 8 = 3$ levels a
    single value remains at the root. Every horizontal level is one parallel step; the
    number of active lanes halves as you climb.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        leaves = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        n = len(leaves)
        depth = int(np.log2(n))   # n is a power of two here

        # Build the levels of partial sums (bottom -> top).
        levels = [leaves.astype(int)]
        while len(levels[-1]) > 1:
            _prev = levels[-1]
            levels.append(np.array([_prev[i] + _prev[i + 1]
                                    for i in range(0, len(_prev), 2)]))

        _fig, _ax = plt.subplots(figsize=(8.5, 4.2))
        _ax.set_xlim(-0.5, n - 0.5)
        _ax.set_ylim(-0.5, depth + 0.6)
        _ax.axis("off")
        _ax.set_title("Tree reduction of 8 values: depth = log2(8) = 3 steps")

        _colors = ["#4c9f70", "#5b8def", "#e0a458", "#d65f5f"]

        # node x-positions per level (centered)
        def _xpos(level, count):
            span = n - 1
            return np.linspace(0, span, count) if count > 1 else np.array([span / 2.0])

        _positions = [(_xpos(_lv, len(_vals)), _lv) for _lv, _vals in enumerate(levels)]

        # edges first (so nodes draw on top)
        for _lv in range(len(levels) - 1):
            _xs_lo, _ = _positions[_lv]
            _xs_hi, _ = _positions[_lv + 1]
            for _j, _xhi in enumerate(_xs_hi):
                for _child in (2 * _j, 2 * _j + 1):
                    if _child < len(_xs_lo):
                        _ax.plot([_xs_lo[_child], _xhi], [_lv + 0.18, _lv + 1 - 0.18],
                                 color="#bbb", linewidth=1.2, zorder=1)

        # nodes
        for _lv, _vals in enumerate(levels):
            _xs, _ = _positions[_lv]
            for _x, _v in zip(_xs, _vals):
                _ax.add_patch(plt.Circle((_x, _lv), 0.22, facecolor=_colors[_lv % 4],
                                         edgecolor="white", zorder=2))
                _ax.text(_x, _lv, f"{int(_v)}", ha="center", va="center",
                         color="white", fontsize=9, weight="bold", zorder=3)
            _ax.text(-0.55, _lv, f"step {_lv}" if _lv > 0 else "leaves",
                     ha="right", va="center", fontsize=8, color="#666")

        _ax.text(n / 2.0 - 0.5, depth + 0.45,
                 f"root = sum = {int(levels[-1][0])}", ha="center",
                 fontsize=9, color="#333", weight="bold")
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Steps vs. array size

    Drag the array size. The plot contrasts the **serial** step count ($n-1$, the long
    dependent chain) against the **tree depth** ($\lceil\log_2 n\rceil$, what a parallel
    machine actually waits for). Note the y-axis is log-scaled: the serial line shoots up
    linearly while the tree depth crawls — that widening gap is the parallel reduction's
    payoff, and it's why doubling the array adds only *one* step to the tree.
    """)
    return


@app.cell
def _(mo):
    size_slider = mo.ui.slider(start=4, stop=20, step=1, value=10,
                               label="log2(array size)  ->  n = 2^value")
    size_slider
    return (size_slider,)


@app.cell
def _(size_slider):
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        log_n = int(size_slider.value)
        n = 2 ** log_n

        logs = np.arange(2, 21)
        ns = 2.0 ** logs
        serial = ns - 1
        depth = logs.astype(float)   # log2(2^k) = k

        _fig, _ax = plt.subplots(figsize=(8.0, 3.8))
        _ax.plot(ns, serial, color="#d65f5f", linewidth=2, marker="o",
                 markersize=3, label="serial: n-1 steps")
        _ax.plot(ns, depth, color="#4c9f70", linewidth=2, marker="o",
                 markersize=3, label="tree: log2(n) steps")
        _ax.axvline(n, color="#5b8def", linestyle="--", linewidth=1.3)
        _ax.scatter([n], [n - 1], color="#d65f5f", zorder=5)
        _ax.scatter([n], [log_n], color="#4c9f70", zorder=5)
        _ax.set_xscale("log")
        _ax.set_yscale("log")
        _ax.set_xlabel("array size n (log scale)")
        _ax.set_ylabel("sequential steps (log scale)")
        _ax.set_title(
            f"n = 2^{log_n} = {n:,}:  serial {n - 1:,} steps  vs.  tree {log_n} steps"
        )
        _ax.legend(loc="upper left", fontsize=9)
        _ax.grid(True, which="both", alpha=0.15)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. `tl.sum` / `tl.max` along an axis

    Here is Triton's gift: you almost never hand-code the tree. Triton provides
    **reduction primitives** that take a tile and collapse it along an axis, emitting the
    log-depth tree (and the warp-shuffle tricks underneath) for you:

    - `tl.sum(x, axis=0)` — sum the tile along an axis.
    - `tl.max(x, axis=0)` / `tl.min(x, axis=0)` — extrema along an axis.

    For a **row reduce** (`e04`), the shape of the kernel is: one program per row, load the
    row as a tile, reduce it to one number, store it. In skeleton form (the `e04` stub
    style — structure shown, body left to you):

    ```python
    @triton.jit
    def row_sum_kernel(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)                       # one program per row
        offs = tl.arange(0, BLOCK_SIZE)              # lanes along the row
        mask = offs < n_cols                         # guard the ragged row tail
        x = tl.load(x_ptr + row * n_cols + offs,     # load this row's tile
                    mask=mask, other=0.0)            # other=0.0 is the SUM identity
        # row_total = tl.sum(x, axis=0)              # <- the reduction (you write it)
        # tl.store(out_ptr + row, row_total)
    ```

    The single most important detail is the **`other` value: it must be the identity of
    your operator.** For `tl.sum`, masked lanes must read `0.0` (adding zero changes
    nothing). For `tl.max`, masked lanes must read $-\infty$ (`float("-inf")`), so the
    overhang never wins the max. Get the identity wrong and the ragged tail silently
    corrupts the result — the reduction analogue of the missing-mask bug from `1a`.

    If a row is wider than one tile (`n_cols > BLOCK_SIZE`), you loop: accumulate a running
    partial across several tiles, then reduce. That's the two-level structure of §4.

    > [Triton reduction ops](https://triton-lang.org/main/python-api/triton.language.html#reduction-ops)
    > (`tl.sum`, `tl.max`, `tl.min`, …); the
    > [layernorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
    > shows row reductions used in anger.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Per-block partial reductions

    A single program (block) can only hold so much in registers/shared memory, and a
    reduction over *millions* of elements won't fit in one tile. The standard answer is
    **two levels**:

    1. **Partial reductions.** Launch many programs; each reduces its own chunk to one
       partial result. With $G$ programs you produce $G$ partials. These run fully in
       parallel — no cross-block communication, which `0b` told you blocks can't do
       cheaply anyway.
    2. **Combine the partials.** Reduce the $G$ partials to the final answer — either a
       second small kernel, an atomic accumulate, or just a `torch` reduce on the $G$-vector
       (when $G$ is small, the host can finish it).

    $$\underbrace{x \in \mathbb{R}^{n}}_{\text{input}}
      \;\xrightarrow{\;G\text{ programs, in parallel}\;}\;
      \underbrace{p \in \mathbb{R}^{G}}_{\text{partials}}
      \;\xrightarrow{\;\text{combine}\;}\;
      \underbrace{\text{scalar}}_{\text{result}}.$$

    For the **row reduce** of `e04` this is the easy case: each row is independent, so one
    program per row needs **no combine step** at all — the partial *is* the answer. You
    only reach for two-level combining when a single reduction axis is too long for one
    program's tile (e.g. reducing a 10-million-element vector, or a very wide row).

    The reason this matters: it keeps every program doing useful, coalesced work in
    parallel (the `1b` lesson) and confines the awkward "few-active-lanes" tail of the tree
    to a tiny final combine over $G$ values, instead of paying it across the whole array.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Numerical considerations

    Reductions are where floating-point stops being exact. Three things to carry into the
    kernels you write:

    **(a) Order changes the answer.** Float addition is *not* associative:
    $(a + b) + c \neq a + (b + c)$ in general, because each `+` rounds. A parallel tree
    reduction sums in a different order than a serial loop, so your kernel's result will
    differ from `torch`'s in the last bits. This is **expected** — it's why the harness
    compares with a tolerance (`TOL`), not for bit-exact equality.

    **(b) Tree order is often *more* accurate.** Summing many numbers serially lets a
    large running accumulator swamp small addends (the small ones round away entirely).
    The tree pairs like-magnitude partials, so it typically has *lower* error than the
    naive serial sum — a rare case where the parallel-friendly algorithm is also the
    numerically nicer one.

    **(c) Overflow and the max-shift trick.** Summing squares ($\sum x_i^2$, for an L2
    norm) or exponentials ($\sum e^{x_i}$, for softmax) can overflow float32. The standard
    guard is to **reduce for the max first, subtract it, then reduce the sum** — so the
    largest term becomes $e^0 = 1$ and nothing overflows. That "max-reduce, then
    shifted-sum-reduce" is precisely the structure of softmax, which is why `1d` is the
    direct sequel to this lecture. Hold the pattern: *a reduction for the shift, then a
    reduction for the total.*

    The cell below shows order-dependence and the accuracy gap on a deliberately nasty sum.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        # A nasty sum: one big value among many tiny ones, in float32.
        rng = np.random.default_rng(0)
        big = np.array([1e8], dtype=np.float32)
        tiny = np.ones(1_000_000, dtype=np.float32)          # each 1.0
        x = np.concatenate([big, tiny])                       # true sum = 1e8 + 1e6

        true_sum = 1e8 + 1e6   # 101,000,000 exactly

        # Serial float32 accumulation (big first -> tiny ones get swamped).
        _serial = np.float32(0.0)
        for _v in x:
            _serial = np.float32(_serial + _v)

        # Pairwise / tree-style sum (numpy's default reduces pairwise).
        _pairwise = np.float32(np.add.reduce(x.astype(np.float32)))

        # float64 reference.
        _f64 = float(np.sum(x.astype(np.float64)))

        print("=== Summing 1e8 + (1,000,000 x 1.0) in float32 ===")
        print(f"  true sum                : {true_sum:,.1f}")
        print(f"  float64 reference       : {_f64:,.1f}")
        print(f"  serial float32 (big 1st): {float(_serial):,.1f}   "
              f"err {abs(float(_serial) - true_sum):,.0f}")
        print(f"  pairwise/tree float32   : {float(_pairwise):,.1f}   "
              f"err {abs(float(_pairwise) - true_sum):,.0f}")
        print("\n  Serial lets the 1e8 accumulator swallow the 1.0s (they round away).")
        print("  The pairwise/tree order keeps like magnitudes together -> less error.")
        print("  Either way you won't match torch bit-for-bit: compare with a TOLERANCE.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    - **Reductions cooperate; maps don't.** The moment lanes must share to produce one
      answer, you're in reduction-land. Reach for `tl.sum`/`tl.max` along an axis rather
      than hand-rolling the tree — Triton emits the log-depth version for you.
    - **The `other` value is the operator's identity.** `0.0` for sum, `-inf` for max.
      This is the reduction's version of the mask: it's what makes the ragged tail
      correct. Wrong identity = silently wrong result.
    - **Two levels for long axes.** Per-block partials in parallel, then a tiny combine.
      Row reduce (`e04`) is the lucky case where one-program-per-row needs no combine at
      all — start there.
    - **Expect last-bit differences, guard overflow.** Float sums depend on order, so
      compare against a tolerance, not for equality. When you sum squares or exponentials,
      do a max-reduce first and shift — the trick that becomes softmax.

    Reductions plus maps are the two atoms of Part 1. Softmax (`1d`) is your first kernel
    that *fuses* them: a max-reduce, a shifted exp-map, a sum-reduce, and a normalize —
    all in one pass over the row.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Now go write it

    Open the harness and write the row-reduction kernel:

    ```bash
    python -m harness.runner e04 --watch
    ```

    `e04` asks you to reduce each row of a matrix to its sum (and/or max) — one program per
    row, load the row's tile, `tl.sum`/`tl.max` along the axis, store the scalar. The two
    things that make it pass: the **mask** on the ragged row tail and the **right `other`
    identity** (`0.0` for sum, `-inf` for max) so that tail never corrupts the reduction.
    It's scored in GB/s — the read traffic of the matrix dominates — so coalesce the row
    loads (`1b`) while you're at it.

    Nail this and you're one fused step from softmax (`1d`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [1B: Memory & Coalescing](../1b_memory_coalescing/) &nbsp;|&nbsp; Next: [1D: Softmax](../1d_softmax/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
