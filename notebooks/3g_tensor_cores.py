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
    # 3G: Tensor Cores

    > *"A CUDA core multiplies two numbers. A tensor core multiplies two matrices —
    > in one instruction."*

    > **Advanced / off the critical path.** This is the last lecture of Part 3 and the
    > capstone of the CUDA track. It builds on the tiled matmul (`3b`) and warp
    > primitives (`3c`), forward-references Blackwell's FP4 (`4a`), and unlocks the WMMA
    > matmul exercise `c06`.

    Every matmul you have written so far — Triton and CUDA — ran on the **CUDA cores**:
    scalar lanes, each doing one fused multiply-add per cycle. That is how the roofline's
    ~tens-of-TFLOP/s FP32 ceiling (`0d`) is built. But a modern GPU has a *second* set of
    math units that do not appear in that number at all: **tensor cores**, units that
    perform an entire small **matrix-multiply-accumulate** (MMA) per instruction. They are
    why the GPU's *real* peak for ML — in fp16/bf16/fp8/fp4 — is an order of magnitude
    above the FP32 CUDA-core figure.

    This lecture is the *what* and the *why*, not a full hand-written kernel: what a
    tensor core computes, the **WMMA** API you reach it through from CUDA C++, the dtypes
    it speaks, and a glimpse of the `mma.sync` PTX underneath — and why libraries like
    **CUTLASS / CuTe** exist to tame the complexity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. What a tensor core is: one instruction, one tile-MMA

    A CUDA core executes a scalar FMA: $d = a\cdot b + c$, one multiply-add. A **tensor
    core** executes a small **matrix** multiply-accumulate in a single hardware
    operation:

    $$D = A \cdot B + C,$$

    where $A, B, C, D$ are small *tiles* — e.g. $16\times16$. That one instruction does
    $16\times16\times16 = 4096$ multiply-adds (= 8192 FLOP). The same work on CUDA cores
    is thousands of separate FMA instructions. **That density is the entire point**: by
    folding a whole tile-matmul into one op, the tensor core hits a throughput the scalar
    lanes cannot approach.

    Two consequences flow from this:

    1. **It is a *warp-wide* operation.** A tensor core is fed by all **32 lanes of a
       warp acting together** — the operand tiles are *distributed across the warp's
       registers*, not owned by one thread. You do not call it per-thread; the warp calls
       it collectively. (This is why `3c`'s warp-level thinking is the prerequisite.)
    2. **It accumulates.** Like a tiled matmul's K-loop, you issue many tile-MMAs into the
       *same* accumulator $C$, walking the K dimension. The accumulator typically stays in
       a wider type (fp32) for precision even when the inputs are fp16.

    A tiled matmul on tensor cores is therefore the `3b` structure with the **inner
    product replaced by a sequence of tile-MMAs**: load a tile of A and a tile of B, do
    one `mma`, accumulate, advance K. The tiling, shared memory, and pipelining (`3f`) all
    still apply — only the innermost multiply changes.

    > [CUDA C++ Programming Guide — Warp Matrix Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions)
    > and PMPP's tensor-core chapter develop this tile-MMA model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. The WMMA API: fragments, load / mma / store

    CUDA C++ exposes tensor cores through the **WMMA** (Warp Matrix Multiply-Accumulate)
    API in `nvcuda::wmma`. The central abstraction is the **fragment**: an opaque,
    warp-distributed register tile. You never index a fragment by hand — its layout
    across the 32 lanes is hardware-defined and deliberately hidden. You only do three
    things to it: **load**, **mma**, **store**.

    ```cuda
    #include <mma.h>
    using namespace nvcuda::wmma;

    // declare fragments for a 16x16x16 MMA (M x N x K):
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float>        acc_frag;  // fp32 accumulate

    fill_fragment(acc_frag, 0.0f);                 // zero the accumulator

    for (int k = 0; k < K; k += 16) {              // walk the K dimension
        load_matrix_sync(a_frag, A + ..., lda);    // global/shared -> fragment
        load_matrix_sync(b_frag, B + ..., ldb);
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);   // D = A*B + C, one tile-MMA
    }
    store_matrix_sync(C + ..., acc_frag, ldc, mem_row_major);  // fragment -> memory
    ```

    The `_sync` suffix is the tell: **every one of these is a warp-collective operation**
    — all 32 lanes must call it together, with converged control flow, exactly like the
    `__shfl_sync` primitives of `3c`. The three verbs:

    | WMMA call | What it does |
    |---|---|
    | `load_matrix_sync(frag, ptr, ld)` | cooperatively load a tile from memory into the warp's fragment, with leading dimension `ld` |
    | `mma_sync(d, a, b, c)` | the tile-MMA: `d = a*b + c`, one tensor-core instruction |
    | `store_matrix_sync(ptr, frag, ld, layout)` | write the accumulator fragment back to memory |
    | `fill_fragment(frag, v)` | broadcast-initialize a fragment (zero the accumulator) |

    The fragment shapes are a **template triple `<M, N, K>`** (the legal tile sizes — e.g.
    `16,16,16` or `32,8,16` for fp16) plus the element type and memory layout. The compiler
    maps the chosen shape and dtype to the right `mma.sync` instruction underneath.

    > [Programming Guide — WMMA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
    > lists the supported `<M,N,K>` shapes per dtype. The opacity of fragments is
    > deliberate: NVIDIA changes the internal layout across architectures.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Supported dtypes: precision for throughput

    Tensor cores are *typed* hardware: each generation adds lower-precision input formats,
    and **lower precision means more MMAs per cycle**. The accumulator usually stays in
    fp32 to hold the sum, so you trade *input* precision for throughput while keeping the
    *running total* accurate. The lineage your 5070 Ti (`sm_120`, Blackwell, 5th-gen
    tensor cores) inherits:

    | Input dtype | Bits | Introduced | Note |
    |---|---|---|---|
    | **fp16** | 16 | Volta | the original tensor-core format; fp32 accumulate |
    | **bf16** | 16 | Ampere | same 8-bit exponent as fp32 → friendlier dynamic range |
    | **tf32** | 19 (stored as 32) | Ampere | a "drop-in" for fp32 matmul: 10-bit mantissa, ~free accuracy hit |
    | **int8** | 8 | Turing | quantized inference (`2c`), int32 accumulate |
    | **fp8** (e4m3 / e5m2) | 8 | Hopper | training/inference at 8-bit float |
    | **fp4** (e2m1) | 4 | **Blackwell** | new on your card — covered in `4a` |

    The rule of thumb: **halving the input bit-width roughly doubles tensor-core
    throughput** (fewer bits to move and multiply). fp16/bf16 → fp8 → fp4 is a ladder of
    ~2× steps, which is why so much of modern ML kernel work (`2c` quantization,
    flash-attention in fp16/fp8) is about *getting onto the lowest precision the model
    tolerates*. The accuracy cost is real and model-dependent — that is the quantization
    tradeoff from `2c`, now grounded in *why* the hardware rewards it.

    > [CUDA C++ Programming Guide — supported WMMA types](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#element-types-and-matrix-sizes).
    > Blackwell fp4 and the tensor-memory accelerator are detailed in `4a`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Throughput by dtype (illustrative)

    The bars below sketch the relative matmul throughput of CUDA cores vs tensor cores
    across the dtype ladder. The numbers are **illustrative multipliers**, not measured
    5070 Ti figures (real peaks are measured in the capstone) — the *shape* is the point:
    a big jump from CUDA cores to tensor cores, then a steady ~2× per halving of input
    precision. Use the dropdown to highlight a dtype and read its multiplier over the
    FP32 CUDA-core baseline.
    """)
    return


@app.cell
def _(mo):
    dtype_dropdown = mo.ui.dropdown(
        options=["fp32 (CUDA core)", "tf32", "fp16", "bf16", "int8", "fp8", "fp4"],
        value="fp16",
        label="highlight dtype")
    dtype_dropdown
    return (dtype_dropdown,)


@app.cell
def _(dtype_dropdown):
    def _run():
        import matplotlib.pyplot as plt

        # Illustrative throughput multipliers vs FP32 CUDA-core baseline (=1).
        # NOT measured 5070 Ti peaks — relative shape only.
        data = [
            ("fp32 (CUDA core)", 1.0,  "#9aa7c7"),
            ("tf32",             8.0,  "#5b8def"),
            ("fp16",            16.0,  "#4c9f70"),
            ("bf16",            16.0,  "#4c9f70"),
            ("int8",            32.0,  "#e0a458"),
            ("fp8",             32.0,  "#e0a458"),
            ("fp4",             64.0,  "#d65f5f"),
        ]
        names = [d[0] for d in data]
        vals = [d[1] for d in data]
        sel = dtype_dropdown.value
        colors = []
        for _n, _v, _c in data:
            colors.append(_c if _n == sel else "#dfe4ee")

        _fig, _ax = plt.subplots(figsize=(8.6, 3.8))
        _bars = _ax.bar(names, vals, color=colors, edgecolor="none")
        for _b, _v in zip(_bars, vals):
            _ax.text(_b.get_x() + _b.get_width() / 2, _v + 0.8,
                     f"{_v:.0f}x", ha="center", fontsize=8, color="#444")
        _ax.set_yscale("log", base=2)
        _ax.set_ylabel("matmul throughput vs FP32 CUDA core (x, log2)")
        _ax.set_title(
            f"tensor cores vs CUDA cores — '{sel}' = "
            f"{dict((d[0], d[1]) for d in data)[sel]:.0f}x (illustrative)")
        _ax.tick_params(axis="x", labelrotation=15)
        _ax.grid(True, axis="y", which="both", alpha=0.15)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Fragment tiling: how a 16x16x16 MMA covers a big matmul

    A single tile-MMA is tiny ($16\times16\times16$). A real matmul is $M\times N\times K$
    with all three in the thousands. So — exactly as in `3b` — you **tile**: the output
    $C$ is a grid of $16\times16$ fragment-tiles, and each output tile is the sum of a
    *row of K-tiles of A* times a *column of K-tiles of B*. One warp owns one (or a few)
    output tile(s) and walks K, issuing one `mma_sync` per K-step into its accumulator
    fragment.

    The diagram below shows the geometry: the highlighted $16\times16$ output tile is
    produced by sweeping the matching **band of A** (16 rows, all of K) against the
    **band of B** (all of K, 16 columns), one $16\times16\times16$ MMA per K-step. The
    accumulator stays resident in registers across the whole sweep.
    """)
    return


@app.cell
def _():
    def _run():
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        T = 16  # fragment tile dim
        nM, nN, nK = 3, 3, 4   # number of tiles along M, N, K (illustrative)

        _fig, _ax = plt.subplots(figsize=(8.2, 4.6))
        _ax.set_xlim(-1, nK + nN + 3)
        _ax.set_ylim(-1, max(nM, nK) + 2)
        _ax.axis("off")
        _ax.set_title("16x16x16 fragment tiling: one output tile = A-band x B-band")

        # --- A matrix (M x K), left ---
        ax0 = 0
        ay0 = 0
        for _i in range(nM):
            for _k in range(nK):
                _hot = (_i == 1)  # highlight the band feeding output tile (1,1)
                _ax.add_patch(mpatches.Rectangle(
                    (ax0 + _k, ay0 + (nM - 1 - _i)), 1, 1,
                    facecolor="#5b8def" if _hot else "#e8edf8",
                    edgecolor="white", linewidth=1.5))
        _ax.text(ax0 + nK / 2, ay0 + nM + 0.2, "A  (M x K)",
                 ha="center", fontsize=9, color="#3060c0")
        _ax.text(ax0 + nK / 2, ay0 - 0.5, f"each cell = {T}x{T} tile",
                 ha="center", fontsize=7, color="#888")

        # --- B matrix (K x N), upper right ---
        bx0 = nK + 2
        by0 = nM + 0.0
        for _k in range(nK):
            for _j in range(nN):
                _hot = (_j == 1)
                _ax.add_patch(mpatches.Rectangle(
                    (bx0 + _j, by0 - _k), 1, 1,
                    facecolor="#4c9f70" if _hot else "#e6f1ea",
                    edgecolor="white", linewidth=1.5))
        _ax.text(bx0 + nN / 2, by0 + 1.2, "B  (K x N)",
                 ha="center", fontsize=9, color="#2e6b48")

        # --- C matrix (M x N), lower right ---
        cx0 = nK + 2
        cy0 = 0
        for _i in range(nM):
            for _j in range(nN):
                _hot = (_i == 1 and _j == 1)
                _ax.add_patch(mpatches.Rectangle(
                    (cx0 + _j, cy0 + (nM - 1 - _i)), 1, 1,
                    facecolor="#d65f5f" if _hot else "#f7e6e6",
                    edgecolor="white", linewidth=1.5))
        _ax.text(cx0 + nN / 2, cy0 - 0.5, "C = A x B  (M x N)",
                 ha="center", fontsize=9, color="#a33")
        _ax.text(cx0 + 1.5, cy0 + (nM - 1 - 1) + 0.5, "acc",
                 ha="center", va="center", fontsize=7, color="white")

        # arrow: K sweep
        _ax.annotate("", xy=(ax0 + nK - 0.1, ay0 + nM + 0.6),
                     xytext=(ax0 + 0.1, ay0 + nM + 0.6),
                     arrowprops=dict(arrowstyle="->", color="#888"))
        _ax.text(ax0 + nK / 2, ay0 + nM + 0.75, "K sweep: one mma_sync per step",
                 ha="center", fontsize=7, color="#888")
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Underneath WMMA: a glimpse of `mma.sync` PTX (and why CUTLASS exists)

    WMMA is the *friendly* face of tensor cores. One level down is the raw PTX
    instruction the compiler emits — `mma.sync` (and the warpgroup-wide `wgmma`
    on Hopper, the Tensor-Memory-based `tcgen05` family on Blackwell). A single
    `mma.sync` looks like:

    ```ptx
    // m16n8k16, fp16 inputs, fp32 accumulate — one warp-wide tile-MMA
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {d0, d1, d2, d3},          // D accumulator fragment (4 fp32 regs/lane)
        {a0, a1, a2, a3},          // A fragment  (registers, this lane's share)
        {b0, b1},                  // B fragment
        {c0, c1, c2, c3};          // C accumulator in
    ```

    Notice what is exposed: the **exact register operands each lane contributes**, the
    **precise `m16n8k16` shape**, the operand **layouts** (`row`/`col`), and the **type
    quad** (`f32.f16.f16.f32` = D, A, B, C types). To use this directly you must:

    - hand-place the A/B/C tiles into the *exact* registers each of the 32 lanes must
      hold (the layout WMMA hid from you),
    - feed the tensor cores fast enough — which on recent GPUs means **`cp.async`
      multi-buffered shared-memory pipelines** (`3f`) and, on Hopper/Blackwell, the
      **TMA** bulk-copy engine (`4a`),
    - and choose tile shapes, swizzles, and pipeline depths per problem size.

    This is *enormous* low-level complexity, and getting it wrong leaves most of the
    tensor-core throughput on the floor. That is precisely the gap **CUTLASS** (and its
    modern layout algebra, **CuTe**) fills: a C++ template library that composes the
    fragment layouts, the `cp.async`/TMA pipelines, the swizzles, and the `mma`/`wgmma`
    instructions into production-grade GEMMs you parameterize rather than hand-write.
    WMMA is enough to *learn* tensor cores and write a correct kernel; CUTLASS/CuTe is
    what you reach for to actually *match* cuBLAS. (Triton, meanwhile, generates this
    layer for you from `tl.dot` — the same instructions, a different abstraction.)

    > [CUTLASS](https://github.com/NVIDIA/cutlass) and the
    > [CuTe layout docs](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/00_quickstart.html);
    > the [PTX ISA `mma.sync` reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    - **Tensor cores are where ML FLOPs live.** The FP32 CUDA-core roofline ceiling
      understates the GPU by ~10×. For matmul and attention, the real peak is on the
      tensor cores, in fp16/bf16/fp8/fp4 — and reaching it is the whole point of `2c`,
      `2b`, and the capstone.
    - **Think warp-collective.** A tensor-core op is issued by all 32 lanes together over
      warp-distributed fragments. The `_sync` primitives of `3c` were the warm-up; WMMA
      is the same discipline applied to a tile-MMA.
    - **Precision is a throughput knob.** Each halving of input bits ~doubles tensor-core
      throughput. Getting a model onto the lowest precision it tolerates (`2c`) is often
      the biggest single win available.
    - **Feeding them is the hard part.** A tensor core starves without a good
      shared-memory pipeline (`3f`). The MMA is one instruction; keeping it fed across the
      K-loop is the engineering — which is why CUTLASS/CuTe and Triton exist.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Now go write it

    The final CUDA exercise puts a tensor core under your own hands. Exercise `c06` asks
    you to write a **WMMA matmul**: declare the fragments, tile the output, walk K with
    `load_matrix_sync` / `mma_sync`, and `store_matrix_sync` the result.

    ```bash
    python -m harness.runner c06
    ```

    Start with the simplest legal configuration — fp16 inputs, fp32 accumulate, a single
    `16x16x16` tile per warp — get it *correct* against the reference, then add tiling
    over a larger matrix. The §2 skeleton is the shape of the answer; choosing the tile
    decomposition, mapping warps to output tiles, and handling the leading dimensions and
    the K-tail are yours. Once it's correct, the natural next step is to feed it with the
    `cp.async` pipeline from `3f` — and you'll have, by hand, the kernel CUTLASS
    generates.

    *(If `c06` has no on-disk stub yet, the command is still your forward pointer — it's
    the capstone CUDA exercise after the pipelined matmul `c05`.)*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [3F: Async Copy & Pipelining](../3f_async_pipelining/) &nbsp;|&nbsp; Next: [4A: Blackwell (sm_120)](../4a_blackwell/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
