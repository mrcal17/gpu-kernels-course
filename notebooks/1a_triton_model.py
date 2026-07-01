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
    # 1A: The Triton Programming Model

    > *"In CUDA you write the story of one thread. In Triton you write the story of one
    > block — and let the compiler narrate the 1024 lanes."*

    Part 0 gave you the hardware's mental model: a grid of blocks, each block a bundle of
    warps, warps of 32 lockstep lanes, all hiding latency by oversubscribing 70 SMs. Now
    we pick up the first tool that lets you *write* for that machine. Triton keeps the
    same execution model from `0b` but raises the **unit you program** from one thread to
    one **block** (Triton calls it a *program*). You write code as if for a single
    program operating on a whole vector of `BLOCK_SIZE` elements; Triton lowers it to the
    warps and lanes underneath, handles coalescing, and picks register allocation.

    This is a different *altitude*, not a different machine. Everything you learned still
    holds — you just express it one level up. By the end of this lecture you'll be able to
    read and reason about every line of a Triton kernel, and you'll write your first one
    (`e01`, vector-add) in the terminal harness.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. Two altitudes: the thread model vs. the block model

    In the **CUDA / SIMT** model (`0b`) you write code for **one thread**. The thread asks
    "which element am I?" via `threadIdx`/`blockIdx`, loads its single scalar, does its
    scalar math, writes its single result. The hardware runs 32 such threads as a warp in
    lockstep — but *you* wrote the scalar program, and the 32-wide vectorization is
    implicit in how the warp executes it.

    In the **Triton** model you write code for **one program** — Triton's word for one
    block. The program asks "which *block* am I?" via `tl.program_id`, then works on a
    whole **array** of `BLOCK_SIZE` elements at once. The operations are written as if on
    vectors:

    $$\underbrace{c_i = a_i + b_i}_{\text{CUDA: one } i \text{ per thread}}
      \qquad\longrightarrow\qquad
      \underbrace{\mathbf{c} = \mathbf{a} + \mathbf{b}}_{\text{Triton: a whole tile per program}}$$

    Same arithmetic, same hardware, one level up. Triton's compiler takes your block-level
    program and splits the `BLOCK_SIZE` elements across the block's warps and lanes for
    you. The coalesced loads, the warp scheduling, the lane masking — handled. What you
    own is the **decomposition**: how to map the data onto programs, and how each program
    indexes its slice.

    > [Triton: vector-add tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
    > introduces this block-program model; the [Triton intro](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)
    > contrasts it with CUDA's thread model directly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The same kernel, side by side

    Two ways to say "add two vectors". On the left, the CUDA scalar program (one thread,
    one element). On the right, the Triton block program (one program, `BLOCK_SIZE`
    elements). Read them as the *same* decomposition seen from two altitudes — both
    compute `pid * BLOCK + lane` as the global index; CUDA names the lane `threadIdx.x`,
    Triton folds all the lanes into `tl.arange`.

    ```text
    CUDA C++ (per-thread)                  Triton (per-program / per-block)
    ----------------------------------     ----------------------------------------
    i = blockIdx.x * blockDim.x            pid  = tl.program_id(0)
        + threadIdx.x;                     offs = pid * BLOCK + tl.arange(0, BLOCK)
    if (i < n)                             mask = offs < n
        c[i] = a[i] + b[i];                a = tl.load(a_ptr + offs, mask=mask)
                                           b = tl.load(b_ptr + offs, mask=mask)
                                           tl.store(c_ptr + offs, a + b, mask=mask)
    ```

    The CUDA `if (i < n)` guard and Triton's `mask` are the *same* idea — keep
    out-of-range lanes from touching memory. The difference is altitude: CUDA writes the
    guard for one lane; Triton writes it once for all `BLOCK_SIZE` lanes as a boolean
    vector. We unpack each Triton primitive next.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. `tl.program_id`: which block am I?

    A Triton kernel is launched over a **grid** of programs, exactly like a CUDA grid of
    blocks. Inside the kernel, `tl.program_id(axis)` returns this program's coordinate
    along a grid axis — the direct analogue of CUDA's `blockIdx.x`:

    $$\texttt{pid} = \texttt{tl.program\_id(0)} \;\in\; \{0, 1, \dots, \text{num\_programs}-1\}.$$

    There is **no `threadIdx`** in Triton. The lanes within a program are not addressed
    individually in your code — they are summoned all at once by `tl.arange` (next
    section). For a 1-D problem you use axis `0`; for a 2-D tiled problem (matmul, `1e`)
    you'll use `tl.program_id(0)` and `tl.program_id(1)` for the row-tile and column-tile
    coordinates.

    The mapping to `0b` is one-to-one: **one Triton program = one CUDA block = one unit
    that lands on one SM.** Programs are independent and may run in any order, on any of
    the 70 SMs — the same fire-and-forget contract for blocks you learned earlier.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. `tl.arange` + `pid`: building the index range

    A program needs the global indices of *its* slice of the array. You build them by
    offsetting a local lane-range by the program's start:

    $$\texttt{offs} = \underbrace{\texttt{pid} \cdot \texttt{BLOCK\_SIZE}}_{\text{where my block starts}}
      \;+\; \underbrace{\texttt{tl.arange(0, BLOCK\_SIZE)}}_{[0,1,\dots,\text{BLOCK\_SIZE}-1]}.$$

    `tl.arange(0, BLOCK_SIZE)` is a **vector** $[0,1,\dots,\text{BLOCK\_SIZE}-1]$ living
    inside one program — it *is* the per-lane index, but expressed as one array instead of
    a scalar per thread. Adding the scalar `pid * BLOCK_SIZE` broadcasts across the whole
    vector, giving program `pid` the contiguous global indices

    $$[\,\texttt{pid}\cdot B,\; \texttt{pid}\cdot B + 1,\; \dots,\; \texttt{pid}\cdot B + B - 1\,].$$

    Because consecutive lanes get consecutive addresses, this is exactly the **coalesced**
    layout from `0c` — a warp's 32 lanes will hit 32 contiguous, aligned elements. The
    numpy cell below reconstructs `offs` for a few programs so you can see the tiling
    explicitly (this is the *concept* in numpy, not the Triton API).
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        BLOCK_SIZE = 8       # tiny, so the whole picture fits
        n = 20               # ragged on purpose: not a multiple of BLOCK_SIZE
        num_programs = -(-n // BLOCK_SIZE)   # ceil div

        print("=== Reconstructing offs = pid*BLOCK + arange(0, BLOCK) ===")
        print(f"  n={n}, BLOCK_SIZE={BLOCK_SIZE}  ->  {num_programs} programs\n")
        for _pid in range(num_programs):
            _arange = np.arange(BLOCK_SIZE)            # tl.arange(0, BLOCK_SIZE)
            _offs = _pid * BLOCK_SIZE + _arange        # global indices for this program
            _mask = _offs < n                          # guard the tail (next section)
            _shown = [f"{o}" if m else f"({o})" for o, m in zip(_offs, _mask)]
            print(f"  pid={_pid}:  offs = [{', '.join(_shown)}]")
        print("\n  (indices in parens are >= n: masked off, never touched).")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. The `mask` idiom: handling the ragged tail

    Your array length $n$ is almost never a clean multiple of `BLOCK_SIZE`. The grid is
    sized with a **ceiling division**, so the *last* program's `offs` run past the end of
    the array. Touching those addresses would read or write out of bounds. The fix is a
    boolean **mask**:

    $$\texttt{mask} = \texttt{offs} < n
      \qquad(\text{True for valid lanes, False for the overhang}).$$

    `mask` is a vector the same shape as `offs`. You hand it to every memory operation:
    `tl.load(ptr + offs, mask=mask)` loads only where `mask` is True and returns a safe
    default (e.g. `other=0.0`) elsewhere; `tl.store(ptr + offs, val, mask=mask)` writes
    only the valid lanes. The masked-off lanes still *execute* (SIMT lockstep — `0b`), but
    their memory effect is suppressed.

    This is the exact analogue of CUDA's `if (i < n)` guard, lifted to operate on the
    whole block's lane-vector at once. **Every** elementwise Triton kernel you write will
    have this mask; forgetting it is the classic first bug (silent out-of-bounds).

    > [Triton `tl.load` / `tl.store`](https://triton-lang.org/main/python-api/triton.language.html)
    > document the `mask` and `other` arguments. The
    > [vector-add tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
    > shows the canonical masking pattern.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. `tl.load` / `tl.store`: the masked memory boundary

    These two are the only way data crosses between DRAM and a program's registers. They
    take a **vector of addresses** (`base_ptr + offs`) and move a whole tile at once:

    - `x = tl.load(a_ptr + offs, mask=mask, other=0.0)` — gather `BLOCK_SIZE` elements
      into a register tile. Where `mask` is False, no DRAM access happens and `other`
      fills the slot. Triton coalesces the contiguous `offs` into the fewest memory
      transactions — 32-byte sectors at L2/DRAM (the `0c` story; `1b` makes it precise) —
      so you get near-peak bandwidth *for free* when `offs` is contiguous.
    - `tl.store(c_ptr + offs, x, mask=mask)` — scatter the tile back to DRAM, writing
      only masked-True lanes.

    Between a load and a store, the tile `x` lives in registers/shared memory and you do
    pure on-chip arithmetic on it (`x + y`, `tl.exp(x)`, `x * scale`, …). The discipline
    is the same one `0c` drilled: **load once, do all your math on-chip, store once.**
    Each extra trip to DRAM is bandwidth you cannot get back.

    A subtlety to file away: `tl.load`/`tl.store` are where coalescing is won or lost.
    Contiguous `offs` → one coalesced burst. A permuted or strided `offs` → scattered
    transactions and a fraction of peak bandwidth (you'll measure this in `1b`/`e03`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6. `tl.constexpr` and the launch grid

    `BLOCK_SIZE` is special. It is a **compile-time constant**, declared in the kernel
    signature as `BLOCK_SIZE: tl.constexpr`. Marking it `constexpr` lets Triton's compiler
    bake the value into the generated code — it can unroll loops and size register tiles.
    (The number of warps per program is a *separate* knob, `num_warps` (default 4), set by
    the programmer or autotuner; `BLOCK_SIZE` only sets how many elements those warps must
    cover, not the warp count.) `constexpr` is also what `tl.arange(0, BLOCK_SIZE)` needs:
    the range length must be known at compile time. Different
    `BLOCK_SIZE` values compile to *different* kernels, which is exactly what autotuning
    (`2a`) exploits.

    On the host side you compute the **grid** — how many programs to launch — with the
    same ceiling division as CUDA:

    $$\text{num\_programs} = \left\lceil \frac{n}{\texttt{BLOCK\_SIZE}} \right\rceil
      = \texttt{triton.cdiv}(n, \texttt{BLOCK\_SIZE}).$$

    In Triton the grid is usually a callable: `grid = lambda meta: (triton.cdiv(n,
    meta["BLOCK_SIZE"]),)`. The lambda receives the chosen `BLOCK_SIZE` (`meta`) and
    returns a tuple of grid dimensions — here 1-D, so a single-element tuple. The ceiling
    division guarantees the grid covers *every* element, and the mask (§4) cleans up the
    overhang the ceiling created. Grid and mask are two halves of one idea: **cover
    everything, then guard the excess.**

    > Ceiling division in integer arithmetic: $\lceil n/b \rceil = \lfloor (n+b-1)/b
    > \rfloor$, which in Python is the `-(-n // b)` idiom — and is exactly what
    > `triton.cdiv` computes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 7. Visualizing the decomposition

    The picture below shows a 1-D array of length $n$ carved into `BLOCK_SIZE`-wide chunks,
    one per program. Each program owns a contiguous run of indices (`pid*BLOCK + arange`).
    The final program is **ragged**: the ceiling-division grid over-covers, so its tail
    lanes (hatched red) are masked off and never touch memory. This is the whole §3–§6
    story in one image.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        BLOCK = 8
        n = 27                       # ragged: 27 = 3*8 + 3
        num_prog = -(-n // BLOCK)    # ceil div -> 4 programs
        covered = num_prog * BLOCK   # 32 lanes total; 32-27 = 5 masked

        _fig, _ax = plt.subplots(figsize=(9.5, 2.8))
        _ax.set_xlim(-0.5, covered + 0.5)
        _ax.set_ylim(0, 3.2)
        _ax.axis("off")
        _ax.set_title(
            f"n={n}, BLOCK_SIZE={BLOCK}  ->  cdiv = {num_prog} programs "
            f"({covered} lanes, {covered - n} masked)"
        )

        _prog_colors = ["#eef3ff", "#dff0e4", "#fdf0e0", "#f3e8fb"]
        _edge_colors = ["#5b8def", "#4c9f70", "#e0a458", "#a06fd0"]

        for _pid in range(num_prog):
            _x0 = _pid * BLOCK
            # program band
            _ax.add_patch(mpatches.Rectangle(
                (_x0 - 0.45, 1.0), BLOCK - 0.1, 1.3,
                facecolor=_prog_colors[_pid % 4],
                edgecolor=_edge_colors[_pid % 4], linewidth=1.8))
            _ax.text(_x0 + BLOCK / 2 - 0.5, 2.45, f"program {_pid}",
                     color=_edge_colors[_pid % 4], fontsize=9, weight="bold")
            # per-lane cells
            for _k in range(BLOCK):
                _idx = _x0 + _k
                _valid = _idx < n
                _fc = _edge_colors[_pid % 4] if _valid else "#f7d6d6"
                _hatch = None if _valid else "////"
                _ax.add_patch(mpatches.Rectangle(
                    (_idx - 0.4, 1.1), 0.8, 0.55,
                    facecolor=_fc, edgecolor="#ffffff",
                    hatch=_hatch, linewidth=0.5))
                _ax.text(_idx, 0.75, f"{_idx}", ha="center", fontsize=6.5,
                         color="#333" if _valid else "#c0392b")

        # legend
        _ax.add_patch(mpatches.Rectangle((0.0, 0.0), 0.8, 0.0))  # spacer
        _valid_patch = mpatches.Patch(facecolor="#5b8def", label="valid lane (mask=True)")
        _mask_patch = mpatches.Patch(facecolor="#f7d6d6", hatch="////",
                                     label="overhang lane (mask=False)")
        _ax.legend(handles=[_valid_patch, _mask_patch], loc="upper right",
                   fontsize=7.5, framealpha=0.9)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Slide `BLOCK_SIZE` and watch the grid

    Fix the array length $n$ and vary `BLOCK_SIZE`. The readout shows the number of
    programs Triton launches (`cdiv(n, BLOCK_SIZE)`) and how many lanes the last program
    has to mask off. Two reflexes to build:

    - **Bigger `BLOCK_SIZE` → fewer programs**, each doing more work (more lanes,
      typically more warps per program). Smaller `BLOCK_SIZE` → more programs.
    - **The masked tail** is `cdiv(n, B)*B - n` lanes — wasted only on the final program.
      A `BLOCK_SIZE` that divides $n$ evenly leaves zero waste, but that's rarely worth
      contorting your launch for; the mask makes any size correct.
    """)
    return


@app.cell
def _(mo):
    blk_slider = mo.ui.slider(start=16, stop=512, step=16, value=128,
                              label="BLOCK_SIZE (elements per program)")
    blk_slider
    return (blk_slider,)


@app.cell
def _(blk_slider):
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        n = 1000                       # fixed, deliberately ragged
        B = int(blk_slider.value)

        block_sizes = np.arange(16, 513, 16)
        n_progs = np.array([-(-n // int(b)) for b in block_sizes])
        masked = np.array([int(b) * (-(-n // int(b))) - n for b in block_sizes])

        cur_progs = -(-n // B)
        cur_masked = B * cur_progs - n

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4))

        _ax1.plot(block_sizes, n_progs, color="#5b8def", linewidth=2)
        _ax1.axvline(B, color="#d65f5f", linestyle="--")
        _ax1.scatter([B], [cur_progs], color="#d65f5f", zorder=5)
        _ax1.set_xlabel("BLOCK_SIZE")
        _ax1.set_ylabel("number of programs = cdiv(n, B)")
        _ax1.set_title(f"n={n}: {cur_progs} programs at B={B}")

        _ax2.plot(block_sizes, masked, color="#4c9f70", linewidth=2)
        _ax2.axvline(B, color="#d65f5f", linestyle="--")
        _ax2.scatter([B], [cur_masked], color="#d65f5f", zorder=5)
        _ax2.set_xlabel("BLOCK_SIZE")
        _ax2.set_ylabel("masked lanes in last program")
        _ax2.set_title(f"{cur_masked} of {B} tail lanes masked")

        _fig.suptitle(
            f"BLOCK_SIZE={B}  ->  {cur_progs} programs, last one masks {cur_masked} lanes",
            y=1.03)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 8. Putting it together: the vector-add skeleton

    Here is the **shape** of the kernel you're about to write — signature and structure,
    with the crux left as comments for you to fill in. This is the `e01` stub style: it
    shows you *where* each primitive from §2–§6 goes, without handing you the body.

    ```python
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
        # 1. which program am I?                 -> tl.program_id(0)
        pid = ...
        # 2. the global indices this program owns -> pid*BLOCK_SIZE + tl.arange(...)
        offs = ...
        # 3. guard the ragged tail               -> offs < n
        mask = ...
        # 4. load a and b tiles (masked)         -> tl.load(..., mask=mask)
        # 5. compute c = a + b on-chip
        # 6. store c (masked)                    -> tl.store(..., mask=mask)

    def add(a, b):
        c = torch.empty_like(a)
        n = a.numel()
        # grid: enough programs to cover all n elements
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        add_kernel[grid](a, b, c, n, BLOCK_SIZE=1024)
        return c
    ```

    The six numbered lines are §2 through §6 in order. Filling them in *is* the exercise —
    do not look for a completed version here; the whole point is that you write the body
    yourself in the harness, where it runs on your real GPU and gets scored in GB/s.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 9. Two more questions: is it *correct*, and is it *fast*?

    Writing the kernel is half the job. The other half — the half you'll repeat on **every
    exercise in this course** — is proving it's right and measuring how fast it is, against a
    reference you trust. The official [Triton vector-add
    tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
    folds exactly this into its first kernel, and so do we. Two questions, two tools.

    **Is it correct?** Compute the answer a trusted way (torch), then compare — with
    `torch.testing.assert_close`, not `==`. For *this* kernel the comparison is the easy
    case: an elementwise add does the same one rounding per element no matter how the work
    is split, so your result should match torch **bit-for-bit** and you can demand zero
    tolerance. The reason to build the `assert_close` habit now is the ops coming next:
    reductions (`1c`) and matmul (`1e`) sum in a *different order* than torch, so their
    results differ in the last bits and are still correct — there you'll loosen the
    tolerance instead. Either way, compute the reference **first**, so a buggy in-place
    kernel can't overwrite your golden answer:

    ```python
    ref = a + b                 # trusted answer, captured BEFORE your kernel runs
    out = vector_add(a, b)      # your kernel
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)   # add is EXACT -> demand 0/0
    ```

    `assert_close` raises with a diagnostic (max abs/rel error, how many elements mismatched)
    when it fails — far more useful than `torch.allclose`'s bare `True/False`. The tolerance
    is **op-specific**: a plain add is exact (`0/0`), but a long reduction or a matmul must
    loosen, because more summation means more reordering. (The full tolerance table and the
    *why* live in the reference card, `7b`.)

    **Is it fast?** Time it honestly — warm up, run many times, take the **median** — then
    turn milliseconds into a **rate** you can hold against the roof. `triton.testing.do_bench`
    does the warmup and median for you; the GB/s formula is just *bytes moved ÷ time*:

    ```python
    import triton
    ms   = triton.testing.do_bench(lambda: vector_add(a, b), warmup=25, rep=100, return_mode="median")
    gbps = 3 * a.numel() * a.element_size() / (ms * 1e-3) / 1e9       # read a, read b, write out
    ref_ms = triton.testing.do_bench(lambda: a + b, warmup=25, rep=100, return_mode="median")  # the torch bar
    ```

    Then **judge** the number: a contiguous add should land near **896 GB/s** (your card's
    DRAM roof from `0d`) — 90%+ means you're saturating memory and there's nothing left to
    win; 30% at full occupancy is the fingerprint of uncoalesced loads (`1b`). And
    `ref_ms / ms` tells you whether you beat torch's own kernel.

    This is the loop for **every** kernel you'll write: **write -> `assert_close` ->
    `do_bench` + roofline -> (profile) -> repeat.** The terminal harness runs exactly this
    for you — the `[PASS]`, `[PERF] ... GB/s`, and `[REF] torch ... (N.NNx your time)` lines
    you're about to see *are* `assert_close` + `do_bench` + this GB/s math. Every exercise's
    README has a *"Validate & benchmark it yourself"* section so you can run the loop by hand;
    `7b` is the full reference card.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    - **Think in tiles, not lanes.** Every Part-1 kernel is "load a tile, compute on it,
      store a tile." Once `program_id → offs → mask → load → compute → store` is reflex,
      you'll recognize it inside softmax, layernorm, matmul — they're all this skeleton
      with a richer middle.
    - **Mask everything.** The ceiling-division grid always over-covers; the mask is what
      makes that correct. A missing mask is a silent out-of-bounds, the most common
      first-kernel bug.
    - **Keep `offs` contiguous.** Contiguous indices are what let `tl.load`/`tl.store`
      coalesce into the fewest memory sectors — the bandwidth lesson from `0c`, now
      something you control directly. `1b` makes this measurable.
    - **`BLOCK_SIZE` is a knob, not a law.** It's a `tl.constexpr` the compiler builds
      around; trying a few values is the seed of autotuning (`2a`). Correctness doesn't
      depend on it — the mask handles any size — but performance does.

    You now have the full vocabulary to read any Triton kernel in this course. The rest of
    Part 1 adds new *middles* (reductions, online softmax, tiling) onto this same frame.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Now go write it

    This lecture unlocks your first two kernels. Open the terminal harness and start with
    vector-add — the canonical use of `program_id → offs → mask → load/store`:

    ```bash
    python -m harness.runner e01 --watch
    ```

    `e01` is the skeleton from §8 with the six lines left to you. `--watch` re-runs the
    spec every time you save, so you get instant pass/fail and a GB/s score against the
    896 GB/s ceiling.

    When it's green, do the follow-up — a *fused* elementwise kernel that does several ops
    per loaded tile, so you pay the DRAM traffic once instead of once per op:

    ```bash
    python -m harness.runner e02 --watch
    ```

    `e02` is where "fuse to amortize traffic" (from `0b`/`0c`) stops being advice and
    becomes a measurable win.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [0D: Occupancy & the Roofline](../0d_occupancy_and_roofline/) &nbsp;|&nbsp; Next: [1B: Memory & Coalescing](../1b_memory_coalescing/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
