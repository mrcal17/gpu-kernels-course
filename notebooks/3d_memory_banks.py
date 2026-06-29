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
    # 3D: Coalescing & Bank Conflicts

    > *"Bandwidth is never the number on the spec sheet. It is that number times how well
    > your 32 lanes agree on which bytes they want."*

    In `1b` you learned that a kernel's real budget is **bandwidth**, and that *how* a
    warp's threads address memory decides how much of the 896 GB/s you actually get.
    Triton's block-wise loads were coalesced for you when your offsets were contiguous.
    Here you re-derive the rules at the CUDA level, where the access pattern is yours to
    spell out — and you meet a *second* alignment rule that lives one level up, in shared
    memory: **bank conflicts**.

    Two structures, two rules, same underlying idea (parallel access to parallel
    hardware):

    1. **Global memory — coalescing.** The 32 lanes of a warp should touch one contiguous,
       aligned chunk so the hardware serves them in the fewest memory transactions. This
       is the CUDA-level version of `1b`.
    2. **Shared memory — bank conflicts.** Shared memory is split into 32 **banks**; if
       two lanes hit the same bank (different address) their accesses *serialize*. The
       fix is often a one-element padding trick.

    Plus the tools that push effective bandwidth further: vectorized `float4` loads and
    the read-only path (`__ldg`). By the end you'll read an access pattern and predict its
    transaction count and conflict degree — then write a conflict-free transpose.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. Global-memory coalescing (the CUDA view of `1b`)

    The GPU does not fetch one float at a time. It moves global memory in **aligned
    transactions** — think 32-, 64-, or 128-byte segments. When a warp issues a load, the
    hardware looks at the 32 addresses its lanes requested and counts *how many distinct
    segments* they fall into. That segment count is the number of memory transactions; the
    bytes you actually wanted divided by the bytes the transactions moved is your
    **efficiency**.

    The ideal — a **coalesced** access — is when lane $i$ reads element $i$ of a
    contiguous, aligned array:

    ```cpp
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = in[i];      // lane 0->in[0], lane 1->in[1], ...  one 128-byte transaction
    ```

    The 32 lanes of a warp request 32 consecutive floats = 128 contiguous bytes = **one
    transaction**, 100% efficient. This is exactly the index idiom from `3a`, and *why*
    that idiom is the default: it coalesces by construction.

    Now break it. A **strided** access — lane $i$ reads `in[i * stride]` — scatters the 32
    requests across many segments:

    ```cpp
    float x = in[i * stride];   // stride=2 -> lanes span 2x the bytes -> ~2x transactions
    ```

    With stride $s$, a warp's requests span ~$s\times$ the address range, so the hardware
    issues up to ~$s\times$ as many transactions, and efficiency falls toward $1/s$. The
    worst case — every lane in a different segment — turns one ideal transaction into 32.
    The operational-intensity math of `0c`/`0d` assumed you *got* the bandwidth; this is
    the discount you pay when the warp's addresses don't agree.

    **The lesson, restated from `1b`:** make consecutive lanes touch consecutive
    addresses. Lay out data so the warp's natural index (`threadIdx.x`) walks contiguous
    memory. This is why row-major vs column-major access matters so much in the transpose
    you'll write.

    > [CUDA C++ Best Practices Guide §9.2.1, "Coalesced Access to Global Memory"](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
    > and PMPP Ch. 6 ("Performance considerations").
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactive: access stride vs transactions

    Slide the access **stride**. For a 32-lane warp reading `in[lane * stride]` (4-byte
    floats, 128-byte segments), the plot counts how many distinct 128-byte segments the
    warp touches — the transaction count — and the resulting efficiency vs the coalesced
    ideal. Stride 1 is one transaction at ~100%; each increase scatters the warp wider and
    drives efficiency toward $1/\text{stride}$.
    """)
    return


@app.cell
def _(mo):
    stride_slider = mo.ui.slider(start=1, stop=16, step=1, value=1,
                                 label="access stride (elements between lanes)")
    stride_slider
    return (stride_slider,)


@app.cell
def _(stride_slider):
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        stride = int(stride_slider.value)
        WARP = 32
        BYTES = 4                 # float
        SEG = 128                 # bytes per transaction segment

        lanes = np.arange(WARP)
        addrs = lanes * stride * BYTES                  # byte address per lane
        segments = np.unique(addrs // SEG)              # distinct 128-byte segments
        n_tx = len(segments)
        useful = WARP * BYTES                            # bytes the warp wanted
        moved = n_tx * SEG                              # bytes the transactions moved
        eff = useful / moved

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9.8, 3.6),
                                          gridspec_kw={"width_ratios": [1.3, 1]})

        # left: which segment each lane lands in
        seg_of_lane = (addrs // SEG)
        uniq = {s: k for k, s in enumerate(sorted(set(seg_of_lane)))}
        palette = plt.cm.tab20(np.linspace(0, 1, max(len(uniq), 1)))
        cols = [palette[uniq[s]] for s in seg_of_lane]
        _ax1.scatter(lanes, seg_of_lane, c=cols, s=40)
        _ax1.set_xlabel("lane (0..31)")
        _ax1.set_ylabel("128-byte segment index")
        _ax1.set_title(f"stride={stride}: warp touches {n_tx} segment(s)")
        _ax1.grid(True, alpha=0.15)

        # right: transactions and efficiency vs a sweep of strides
        ss = np.arange(1, 17)
        txs = []
        effs = []
        for _s in ss:
            _a = lanes * _s * BYTES
            _n = len(np.unique(_a // SEG))
            txs.append(_n)
            effs.append((WARP * BYTES) / (_n * SEG))
        _ax2.plot(ss, txs, color="#5b8def", linewidth=2, label="transactions")
        _ax2.scatter([stride], [n_tx], color="#d65f5f", s=70, zorder=5)
        _ax2.set_xlabel("stride")
        _ax2.set_ylabel("transactions / warp", color="#5b8def")
        _ax2.set_title(f"efficiency = {eff:.0%}  ({n_tx} tx)")
        _ax2.grid(True, alpha=0.15)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Shared memory's 32 banks

    Shared memory (the `3b` scratchpad) has its own parallel-access rule. It is physically
    divided into **32 banks**, interleaved by 4-byte word: address $a$ (in 4-byte words)
    lives in bank $a \bmod 32$. The 32 banks can each serve one request
    simultaneously — 32 distinct addresses (one per bank) in a single cycle —
    that's what makes shared memory fast.

    $$\text{bank}(a) \;=\; a \bmod 32 \qquad (a \text{ in 4-byte words}).$$

    The rule for a warp's shared-memory access:

    - **No conflict** — the 32 lanes hit 32 *distinct* banks (or all read the *same*
      address, which the hardware **broadcasts**). One cycle. Ideal.
    - **$k$-way bank conflict** — $k$ lanes hit the *same bank* at *different* addresses.
      The hardware can't serve them together, so it **serializes**: the access takes $k$
      cycles. A 32-way conflict is 32× slower than conflict-free.

    The trap is in 2-D shared arrays. Take `__shared__ float tile[32][32]`. A warp reading
    a **row** `tile[ty][0..31]` walks consecutive words → 32 distinct banks → conflict-free.
    But a warp reading a **column** `tile[0..31][tx]` strides by 32 words between lanes —
    and since the stride (32) is a multiple of the bank count (32), **every lane lands in
    the same bank** → a 32-way conflict, the worst case. Column access of a 32-wide tile
    is the textbook bank-conflict disaster, and it's exactly what a transpose does.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. The +1 padding trick

    The fix is delightfully cheap: **pad the inner dimension by one**. Declare the tile
    one column wider than you use:

    ```cpp
    __shared__ float tile[32][32];     // BAD: column access -> 32-way conflict
    __shared__ float tile[32][33];     // GOOD: pad inner dim by 1 -> conflict-free
    //                      ^^ the +1 shears the bank mapping
    ```

    Why it works: with width 33, moving down one row advances the linear address by 33
    words, not 32. So column neighbors land in banks that differ by $33 \bmod 32 = 1$ —
    consecutive banks, all distinct. The padding column is never read; it exists only to
    **shear the row-to-bank alignment** so a column no longer collapses onto one bank. You
    pay one wasted column of shared memory (here $32\times4 = 128$ bytes/block) to turn a
    32× serialization into a single cycle. It is the highest-leverage one-character fix in
    CUDA.

    The general rule: if a warp accesses a 2-D shared tile with a stride that shares a
    factor with 32, pad the inner dimension so the effective stride becomes coprime with
    32 (often just `+1`). You'll apply this directly in the transpose — load a tile, then
    read it transposed from shared memory, where the padding is what keeps the transposed
    read conflict-free.

    > [CUDA C++ Best Practices Guide §9.2.3, "Shared Memory"](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory)
    > works the matrix-transpose bank-conflict example in full; PMPP Ch. 6 covers banks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Picture: 32 banks, a column conflict, and the +1 fix

    Left: a warp reads a **column** of an unpadded `[8][8]` tile (shrunk from 32 for
    legibility, but the collapse is the same) — every lane maps to **one bank**, an
    8-way (→ 32-way at full size) conflict, drawn as all lanes piling onto a single bank
    column. Right: the **+1 padded** `[8][9]` tile — the same column read now lands each
    lane on a *distinct* bank, conflict-free. The padding column (hatched) is never read;
    it only shifts the mapping.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        N = 8           # illustrative tile dim (real banks = 32)
        BANKS = N       # banks scaled to match the picture

        def draw(ax, width, title, col_read=2):
            ax.set_xlim(-0.5, BANKS + 0.5)
            ax.set_ylim(-1.4, N + 0.6)
            ax.axis("off")
            ax.set_title(title, fontsize=9.5)
            # bank axis labels at bottom
            for _bk in range(BANKS):
                ax.text(_bk + 0.5, -0.55, f"b{_bk}", ha="center", fontsize=6.5,
                        color="#888")
            ax.text(BANKS / 2, -1.15, "shared-memory bank", ha="center",
                    fontsize=8, color="#666")
            # place each tile element at (bank, row); bank = (row*width + col) % BANKS
            for _r in range(N):
                for _c in range(width):
                    _lin = _r * width + _c
                    _bk = _lin % BANKS
                    _pad = _c >= N
                    _read = (_c == col_read) and not _pad
                    _face = ("#fde9e9" if _read else
                             ("#f0f0f0" if _pad else "#eef3ff"))
                    _edge = ("#d65f5f" if _read else
                             ("#bbb" if _pad else "#9bb7e8"))
                    ax.add_patch(mpatches.Rectangle(
                        (_bk + 0.08, _r + 0.08), 0.84, 0.84,
                        facecolor=_face, edgecolor=_edge, linewidth=1.0,
                        hatch="////" if _pad else None))
            return

        _fig, (_axL, _axR) = plt.subplots(1, 2, figsize=(10.0, 4.0))
        draw(_axL, N, f"tile[{N}][{N}]  (no pad): column read -> 1 bank, conflict")
        draw(_axR, N + 1, f"tile[{N}][{N}+1] (+1 pad): column read -> distinct banks")

        # annotate the conflict pile on the left: all red cells share a bank column
        _axL.text(N / 2, N + 0.25,
                  "all read cells -> same bank (serialized)",
                  ha="center", fontsize=7.5, color="#d65f5f")
        _axR.text((N + 1) / 2, N + 0.25,
                  "read cells -> different banks (1 cycle)",
                  ha="center", fontsize=7.5, color="#4c9f70")
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Vectorized loads (`float4`) and the read-only path (`__ldg`)

    Two more levers raise *effective* bandwidth beyond plain coalescing.

    **Vectorized loads — `float4`.** Each lane can load 16 bytes in *one* instruction by
    reading a `float4` (4 packed floats) instead of a `float`. A warp then moves
    $32\times16 = 512$ bytes per instruction with fewer, wider transactions and fewer
    address calculations — often the difference between bandwidth-bound-but-slow and
    bandwidth-bound-at-peak:

    ```cpp
    // reinterpret as float4 and load 4 contiguous floats per lane in one go
    const float4* in4 = reinterpret_cast<const float4*>(in);
    float4 v = in4[i];          // one 16-byte load; v.x v.y v.z v.w
    ```

    The catch is **alignment**: a `float4` load requires a 16-byte-aligned address, so the
    base pointer and your indexing must respect that (the count must be a multiple of 4,
    etc.). When it fits, it's nearly free throughput.

    **Read-only data cache — `__ldg`.** For data a kernel only reads (never writes), you
    can route the load through the read-only / texture cache, which is optimized for
    broadcast and spatial reuse:

    ```cpp
    float x = __ldg(&in[i]);    // load via the read-only cache
    ```

    On modern architectures the compiler often does this automatically when it can prove a
    pointer is read-only (e.g. `const __restrict__`), but `__ldg` makes the intent explicit
    and is a reliable hint. It doesn't change *what* you read — it changes *which cache*
    serves it, helping when many threads re-read the same read-only data (like a tile of
    weights).

    > [CUDA C++ Best Practices Guide §9.2, "Device Memory Spaces"](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#device-memory-spaces)
    > covers vectorized access; `__ldg` and read-only caching are in the Programming Guide
    > §7.8 / the [Kepler read-only cache notes](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#read-only-data-cache-load-function).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    Coalescing and bank conflicts are the two places a "bandwidth-bound" kernel quietly
    leaves half its bandwidth on the table.

    - **Coalesce global access.** Consecutive lanes → consecutive addresses, one
      transaction per warp. Strided/scattered access multiplies transactions and divides
      efficiency by the stride. The `3a` index idiom coalesces by construction; preserve
      it through your data layout (`1b`, now to the metal).
    - **Watch shared-memory banks.** A warp wants 32 distinct banks (or a broadcast). A
      2-D tile accessed by column with a width that's a multiple of 32 collapses to a
      32-way conflict — pad the inner dimension by 1 to shear the mapping. One character,
      32× faster.
    - **Widen the loads.** `float4` moves 16 bytes/lane/instruction when alignment allows
      — fewer, fatter transactions toward the bandwidth roof of `0d`.
    - **Hint read-only data.** `__ldg` / `const __restrict__` routes read-only loads
      through the read-only cache, helping broadcast and reuse patterns.

    Every Part-3 memory kernel — copy, transpose, the loads inside tiled matmul — is won
    or lost on these four points.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Now go write it

    Time to make both rules pay off at once. The transpose is the perfect crucible: a
    naive transpose is coalesced reading but **strided (uncoalesced) writing** — so you
    stage a tile in shared memory to fix the global pattern, and then the *transposed*
    shared read hits the column-conflict trap unless you pad:

    ```bash
    python -m harness.runner c04 --watch
    ```

    `c04` is the conflict-free transpose — the CUDA counterpart of Triton's `e06`. You'll
    coalesce both the global read and write by going through a `__shared__` tile, and add
    the `+1` padding so the transposed shared access is conflict-free. The harness builds
    it with `nvcc -arch=sm_120` and reports **bandwidth**; watch the number jump when the
    padding removes the bank conflict, and again if you vectorize the copy. Both rules from
    this lecture, in one kernel you write yourself.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [3C: Warp-Level Primitives](../3c_warp_primitives/) &nbsp;|&nbsp; Next: [3E: Occupancy Tuning](../3e_occupancy_tuning/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
