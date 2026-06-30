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
    # 7A: Study Guide & Reference

    > *"You don't need to remember it. You need to know where it is, and what question
    > it answers."*

    This is the reference card for the whole course — the page you keep open while you
    write kernels, not a lecture you read once. It collects the **API cheat-sheets**
    (Triton, CUDA launch/memory/sync), the **occupancy math**, a **profiling guide**, the
    **5070 Ti roofline numbers** in one table, a **glossary** of every term, a
    **debugging checklist** for "why is my kernel slow?", and the **full reference link
    list**.

    Everything here is a recall aid — the derivations live in the lectures it points back
    to. Skim the table of contents, then dip in.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. The 5070 Ti in one table

    Your target hardware, queried (`python -m harness.device_info`), not guessed. These
    are the numbers every estimate in the course plugs into.

    | Quantity | Value | Where it bites |
    |---|---|---|
    | GPU / arch | RTX 5070 Ti, Blackwell | `4a` |
    | Compute capability | **`sm_120`** | feature gating (`4a`) |
    | SMs | **70** | grid sizing, total parallelism (`0b`) |
    | Warp size | **32** | everything — coalescing, divergence (`0b`,`3c`,`3d`) |
    | Max resident threads / SM | **1536** (= **48 warps**) | occupancy ceiling (`0d`) |
    | Max threads / block | **1024** (= 32 warps) | launch config (`0b`) |
    | Registers / SM | **65,536** | reg-bound occupancy; ~42 regs/thd for 100% (`0d`,`3e`) |
    | Shared memory / SM | **~100 KB** (48 KB/block default, opt-in ~99 KB) | smem-bound occupancy, tile size (`0c`,`0d`,`3b`) |
    | L2 cache | **48 MB** | reuse across blocks (`0c`) |
    | Memory bus | **256-bit** | — |
    | DRAM bandwidth | **~896 GB/s** | the memory roof (`0c`,`0d`) |
    | FP32 peak | ~tens of TFLOP/s | the compute roof for FP32 (`0d`) |
    | Toolchain | CUDA 13.1, PyTorch 2.10 (cu128), Triton 3.6.0 | builds (`CLAUDE.md`) |

    **The two roofs.** Memory roof = **896 GB/s** (the slope). Compute roof = your
    *measured* peak for the dtype you're using — FP32 is tens of TFLOP/s; tensor-core
    FP16/FP8/FP4 are far higher and must be **measured** (`4a` warns: don't quote
    datasheet narrow-precision numbers for consumer `sm_120`). Ridge point
    $I_{\text{ridge}} = P_{\text{peak}} / B$ separates memory- from compute-bound.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Occupancy math (recall card)

    $$\text{occupancy} = \frac{\text{resident warps}}{48},
      \qquad
      \text{resident blocks} = \min\!\big(b_{\text{thr}},\, b_{\text{reg}},\,
      b_{\text{smem}},\, b_{\text{cap}}\big).$$

    For a block of $T$ threads using $R$ registers/thread and $S$ bytes of shared memory:

    $$b_{\text{thr}} = \left\lfloor \frac{1536}{T} \right\rfloor,\quad
      b_{\text{reg}} = \left\lfloor \frac{65536}{R\,T} \right\rfloor,\quad
      b_{\text{smem}} = \left\lfloor \frac{100\,\text{KB}}{S} \right\rfloor.$$

    Resident warps $= \text{resident blocks} \times (T/32)$. The **smallest** of the four
    rules is the binding limiter. Rules of thumb: $R \lesssim 42$ for 100% occupancy; a
    block grabbing 50 KB smem caps at ~1 block/SM; block size should be a multiple of 32.
    **Occupancy is a means, not a goal** — enough warps to hide latency, then stop.
    (Full derivation: `0d`.) Use the small calculator below to spot-check a config.
    """)
    return


@app.cell
def _():
    def _run():
        # Quick occupancy spot-check for a few common configs on the 5070 Ti.
        MAX_THREADS, MAX_REGS, SMEM_KB, CAP = 1536, 65536, 100.0, 32

        def occ(tpb, regs, smem_kb):
            bt = MAX_THREADS // tpb
            br = MAX_REGS // (regs * tpb)
            bs = int(SMEM_KB // smem_kb) if smem_kb > 0 else 999
            blocks = min(bt, br, bs, CAP)
            warps = blocks * (tpb // 32)
            binder = min(
                [("thr", bt), ("reg", br), ("smem", bs), ("cap", CAP)],
                key=lambda kv: kv[1],
            )[0]
            return blocks, warps, warps / 48.0, binder

        print("=== Occupancy spot-check (5070 Ti) ===\n")
        print(f"  {'tpb':>5s} {'regs':>5s} {'smem':>6s} | {'blks':>4s} {'warps':>5s} "
              f"{'occ':>5s}  binder")
        print("  " + "-" * 50)
        for _t, _r, _s in [(256, 32, 8), (256, 64, 8), (512, 32, 16),
                           (128, 40, 48), (1024, 32, 8)]:
            _b, _w, _o, _bind = occ(_t, _r, _s)
            print(f"  {_t:>5d} {_r:>5d} {_s:>5.0f}K | {_b:>4d} {_w:>5d} "
                  f"{_o:>4.0%}  {_bind}")
        print("\n  binder = which budget is smallest (the one to spend on).")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Cheat-sheets (pick a topic below)

    The dropdown after this cell surfaces the cheat-sheet for whichever topic you choose —
    Triton API, CUDA launch, CUDA memory & sync, or the occupancy/roofline formulas.
    They're the same references printed in §§3.1–3.4 below for scrolling; the dropdown is
    for "I just need the *one* I'm using right now."
    """)
    return


@app.cell
def _(mo):
    cheat_dropdown = mo.ui.dropdown(
        options=[
            "Triton API",
            "CUDA launch & indexing",
            "CUDA memory & sync",
            "Occupancy & roofline formulas",
        ],
        value="Triton API",
        label="cheat-sheet topic",
    )
    cheat_dropdown
    return (cheat_dropdown,)


@app.cell
def _(cheat_dropdown, mo):
    def _pick():
        _triton = r"""
    ### Triton API cheat-sheet

    ```python
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(x_ptr, ..., out_ptr, n, BLOCK: tl.constexpr):
        # pid  = tl.program_id(axis=0)              # which program (block) am I
        # offs = pid * BLOCK + tl.arange(0, BLOCK)  # this program's element indices
        # mask = offs < n                           # guard the ragged tail
        # tl.load(ptr + offs, mask=mask, other=0.0) # masked global load -> a tile
        # ... compute on the loaded tiles ...       # TODO: your op here
        # tl.store(out_ptr + offs, result, mask=mask)  # masked global store
        ...

    # launch: grid is a function of meta-params (e.g. BLOCK)
    # grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    # kernel[grid](x, ..., out, n, BLOCK=1024)
    ```
    *(shapes only — fill the body in the matching `eNN` exercise, not here.)*

    | Need | Call |
    |---|---|
    | which program | `tl.program_id(axis=0/1/2)` |
    | number of programs | `tl.num_programs(axis)` |
    | index range | `tl.arange(0, BLOCK)` |
    | masked load / store | `tl.load(ptr+offs, mask=, other=)`, `tl.store(ptr+offs, val, mask=)` |
    | 2-D tile pointer | `tl.make_block_ptr(...)` / explicit `row[:,None]*stride + col[None,:]` |
    | matmul (tensor cores) | `tl.dot(a, b)` (accumulate in fp32) |
    | reductions | `tl.sum/max/min(x, axis=)`, `tl.cumsum`, `tl.associative_scan` |
    | math | `tl.exp/log/sqrt/maximum/where/sigmoid` |
    | compile-time const | `BLOCK: tl.constexpr` |
    | autotune | `@triton.autotune(configs=[triton.Config({...}, num_warps=, num_stages=)], key=[...])` |
    | grid ceil-div | `triton.cdiv(n, BLOCK)` |
    | benchmark | `triton.testing.do_bench(lambda: kernel[grid](...))` |

    *Cross-refs:* `1a` (model), `1c`/`1d` (reduce/softmax), `1e`/`2a` (matmul/autotune),
    `1g` (scan), `2b` (flash-attn), `2d` (autograd).
    """

        _launch = r"""
    ### CUDA launch & indexing cheat-sheet

    ```cpp
    __global__ void kernel(const float* x, float* out, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;   // global 1-D index
        if (i < n) out[i] = x[i] * 2.0f;                 // guard the tail
    }

    // host launch: <<<grid, block>>>
    int block = 256;
    int grid  = (n + block - 1) / block;                 // ceil-div
    kernel<<<grid, block>>>(d_x, d_out, n);
    cudaError_t err = cudaGetLastError();                // ALWAYS check
    cudaDeviceSynchronize();                             // kernels are async
    ```

    | Need | Symbol |
    |---|---|
    | thread index in block | `threadIdx.x/.y/.z` |
    | block index in grid | `blockIdx.x/.y/.z` |
    | block / grid dims | `blockDim`, `gridDim` |
    | global 1-D id | `blockIdx.x*blockDim.x + threadIdx.x` |
    | 2-D row/col | `row = blockIdx.y*blockDim.y + threadIdx.y`, sym. for col |
    | launch | `kernel<<<grid, block, smem_bytes, stream>>>(args)` |
    | cap registers | `__launch_bounds__(maxThreads, minBlocks)` |
    | error check | `cudaGetLastError()`, wrap launches in a `CUDA_CHECK` macro |

    *Cross-refs:* `3a` (model + build loop), `3e` (`__launch_bounds__`/occupancy).
    """

        _mem = r"""
    ### CUDA memory & sync cheat-sheet

    ```cpp
    __global__ void tiled(const float* A, const float* B, float* C, int N) {
        __shared__ float As[T][T];          // per-block shared memory
        __shared__ float Bs[T][T];
        // ... cooperative load of a tile into As/Bs ...
        __syncthreads();                    // all threads finished loading
        // ... compute on the shared tile ...
        __syncthreads();                    // before overwriting As/Bs next iter
    }
    ```

    | Need | Call |
    |---|---|
    | static shared mem | `__shared__ float buf[N];` |
    | dynamic shared mem | `extern __shared__ float buf[];` + smem arg in `<<<>>>` |
    | block barrier | `__syncthreads()` |
    | warp shuffle | `__shfl_down_sync(mask, val, delta)`, `__shfl_xor_sync(...)` |
    | warp vote | `__ballot_sync`, `__any_sync`, `__all_sync` |
    | warp mask | `__activemask()`, full = `0xffffffff` |
    | read-only cache | `__ldg(ptr)` |
    | vectorized load | `float4 v = *reinterpret_cast<const float4*>(ptr);` |
    | atomics | `atomicAdd/Max/CAS(addr, val)` |
    | async copy (Ampere+) | `cp.async` (PTX) / `memcpy_async` (cooperative groups) |
    | bulk async (Hopper+) | TMA `cp.async.bulk.tensor` (see `4a`) |
    | h<->d copy | `cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice)` |

    *Cross-refs:* `3b` (shared tiling), `3c` (warp primitives), `3d` (banks/coalescing),
    `3f` (async/pipelining), `4a` (TMA).
    """

        _occ = r"""
    ### Occupancy & roofline formulas

    **Occupancy** (5070 Ti: 1536 thr, 65536 regs, ~100 KB smem per SM):
    $$\text{occ} = \frac{\text{resident warps}}{48},\quad
      \text{blocks} = \min\!\Big(\tfrac{1536}{T},\, \tfrac{65536}{RT},\,
      \tfrac{100\text{KB}}{S},\, b_{\text{cap}}\Big).$$
    Resident warps $= \text{blocks}\times(T/32)$. Want 100%? Keep $R \lesssim 42$.

    **Roofline:**
    $$P(I) = \min(P_{\text{peak}},\; B\cdot I),\qquad
      I = \frac{\text{FLOPs}}{\text{bytes moved}},\qquad
      I_{\text{ridge}} = \frac{P_{\text{peak}}}{B}.$$
    $I < I_{\text{ridge}}$ → **memory-bound** (chase bandwidth: coalesce, fuse, reuse).
    $I > I_{\text{ridge}}$ → **compute-bound** (chase FLOP/s: tensor cores, lower precision).
    $B \approx 896$ GB/s. Achieved GB/s $=$ bytes / median latency; TFLOP/s $=$ FLOPs / latency.

    **Little's Law** (why occupancy buys latency hiding):
    $$\text{required concurrency} = \text{latency} \times \text{throughput}.$$

    *Cross-refs:* `0c` (intensity), `0d` (occupancy + roofline), `3e` (reg/occ tuning).
    """

        _map = {
            "Triton API": _triton,
            "CUDA launch & indexing": _launch,
            "CUDA memory & sync": _mem,
            "Occupancy & roofline formulas": _occ,
        }
        return mo.md(_map[cheat_dropdown.value])

    _pick()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### 3.1 Triton API (full)

    ```python
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(x_ptr, ..., out_ptr, n, BLOCK: tl.constexpr):
        # pid  = tl.program_id(axis=0)               # who am I
        # offs = pid * BLOCK + tl.arange(0, BLOCK)   # my element indices
        # mask = offs < n                            # guard the tail
        # tl.load(ptr + offs, mask=mask)             # masked load -> tile
        # ... compute ...                            # TODO: the actual op
        # tl.store(out_ptr + offs, result, mask=mask)
        ...

    # grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # launch shape
    # kernel[grid](x, ..., out, n, BLOCK=1024)
    ```

    Key calls: `program_id` / `num_programs` (who am I), `arange` (index range),
    `load`/`store` with `mask=`/`other=` (guarded global access), `dot` (tensor-core
    matmul, fp32 accumulate), `sum`/`max`/`cumsum`/`associative_scan` (reductions),
    `make_block_ptr` (2-D tiles), `constexpr` (compile-time), `@triton.autotune` +
    `triton.Config(..., num_warps=, num_stages=)` (tuning), `triton.cdiv` (ceil-div),
    `triton.testing.do_bench` (timing).

    ### 3.2 CUDA launch & indexing (full)

    ```cpp
    __global__ void scale(const float* x, float* out, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) out[i] = 2.0f * x[i];
    }
    int block = 256, grid = (n + block - 1) / block;
    scale<<<grid, block>>>(d_x, d_out, n);
    cudaError_t e = cudaGetLastError();    // check EVERY launch
    cudaDeviceSynchronize();
    ```

    `threadIdx`/`blockIdx`/`blockDim`/`gridDim`; 2-D via `.y`; `<<<grid, block, smem,
    stream>>>`; `__launch_bounds__(maxThreads, minBlocks)` to cap registers.

    ### 3.3 CUDA memory & sync (full)

    `__shared__` / `extern __shared__` (shared mem), `__syncthreads()` (block barrier),
    `__shfl_down_sync` / `__shfl_xor_sync` (warp shuffle), `__ballot_sync` / `__any_sync`
    / `__all_sync` (warp vote), `__ldg` (read-only cache), `float4` reinterpret
    (vectorized load), `atomicAdd` (atomics), `cp.async` (Ampere async copy), TMA
    `cp.async.bulk.tensor` (Hopper/Blackwell bulk copy, `4a`), `cudaMemcpy` (host↔device).

    ### 3.4 Occupancy & roofline (full)

    See §2 for occupancy and §1 for the roofline numbers; the formula card is in the
    dropdown above. The one-liners: occupancy = resident warps / 48, binding limiter =
    min of the four budgets; roofline $P=\min(P_{\text{peak}}, B\cdot I)$, ridge =
    $P_{\text{peak}}/B$, left = memory-bound, right = compute-bound.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Profiling guide

    **`triton.testing.do_bench`** — the everyday timer. Warms up, runs many iters, returns
    a robust **median** latency (ms). Pass `quantiles=[0.5, 0.2, 0.8]` for a spread.
    Convert: GB/s = bytes / (ms·1e-3); TFLOP/s = FLOPs / (ms·1e-3) / 1e12. *Always* bench
    the torch baseline through the same harness.

    **Nsight Compute (`ncu`)** — the microscope, when you're below roof and don't know why.
    Run `ncu --set full -o report ./prog` (or target a Python/Triton launch). The metrics
    that diagnose the most common problems:

    | Metric (ncu name) | Tells you | If bad → check |
    |---|---|---|
    | `sm__throughput.avg.pct_of_peak` | overall SM utilization | the rest of this table |
    | `gpu__dram_throughput.avg.pct_of_peak` | DRAM bandwidth used | coalescing (`3d`), fusion |
    | `sm__warps_active.avg.pct_of_peak` (achieved occupancy) | warps resident vs max | occupancy limiters (`0d`,`3e`) |
    | `l1tex__t_sector_hit_rate` | L1/tex cache hits | access pattern, reuse |
    | `smsp__sass_average_branch_targets...` (divergence) | warp divergence | uniform branches (`0b`) |
    | shared-mem bank-conflict counters | bank conflicts | padding/swizzle (`3d`) |
    | stall reasons (`smsp__warp_issue_stalled_*`) | why warps aren't issuing | the dominant stall |

    The loop: `do_bench` to see *that* you're slow and place the dot on the roofline; `ncu`
    to see *why* and which knob to turn.

    > [`do_bench` docs](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html),
    > [Nsight Compute docs](https://docs.nvidia.com/nsight-compute/),
    > [Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Debugging checklist — "my kernel is slow"

    Walk this in order. Each branch points at the lecture with the fix. The flowchart
    below renders the same logic as a picture.

    1. **Did you measure right?** `do_bench` median, warmed up, large shape, vs torch
       baseline same harness. (§4) Many "slow kernels" are launch-bound tiny shapes.
    2. **Where is it on the roofline?** Compute $I$, place the dot. Memory- or
       compute-bound? That decides every knob below. (`0d`)
    3. **Memory-bound? →** Check **coalescing** (contiguous per warp, `3d`/`1b`), **fusion**
       (too many passes over DRAM? fuse, `1f`), **reuse** (tile into shared mem, `3b`/`1e`),
       vectorized loads (`float4`, `3d`).
    4. **Compute-bound? →** Are you using **tensor cores** (`tl.dot` / WMMA, `3g`)? Can you
       drop **precision** (FP16→FP8→FP4, `4a`/`2c`)? Is the inner loop tight?
    5. **Either way: occupancy.** Achieved occupancy low? Find the **binding limiter**
       (threads / registers / shared mem, `0d`/`3e`) and spend that resource. Enough warps
       to hide latency — not blindly 100%.
    6. **Divergence?** Data-dependent branches in the warp's hot path → make them
       warp-uniform or hoist them out (`0b`).
    7. **Bank conflicts?** Shared-memory accesses where lanes hit the same bank → pad or
       swizzle (`3d`).

    If all seven are clean and you're near the roof: **you're done** — the roofline says
    so. The most common single miss is **#1** (measuring noise) and **#3-coalescing**.
    """)
    return


@app.cell
def _():
    def _run():
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        _fig, _ax = plt.subplots(figsize=(8.8, 6.2))
        _ax.set_xlim(0, 10)
        _ax.set_ylim(0, 12)
        _ax.axis("off")
        _ax.set_title("Debugging checklist: why is my kernel slow?", fontsize=12)

        def box(x, y, w, h, text, fc, fs=8.5):
            _ax.add_patch(mpatches.FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.04",
                facecolor=fc, edgecolor="#444", linewidth=1.1))
            _ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                     fontsize=fs, wrap=True)

        def arrow(x1, y1, x2, y2, label=None, col="#444"):
            _ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                         arrowprops=dict(arrowstyle="->", color=col, lw=1.3))
            if label:
                _ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2, label,
                         fontsize=7.5, color=col)

        # top: measure
        box(3.2, 10.7, 3.6, 1.0, "1. Measured right?\n(do_bench, warm, large, vs torch)", "#cbd6ee")
        # roofline split
        box(3.2, 8.9, 3.6, 1.0, "2. Place on roofline\n(compute I, find the bound)", "#cbd6ee")
        arrow(5.0, 10.7, 5.0, 9.9)

        # memory-bound branch (left)
        box(0.3, 6.7, 3.4, 1.2,
            "3. MEMORY-bound:\ncoalescing? fusion?\nreuse (tile)? float4?", "#dff0e4")
        # compute-bound branch (right)
        box(6.3, 6.7, 3.4, 1.2,
            "4. COMPUTE-bound:\ntensor cores? lower\nprecision? tight loop?", "#fdebd0")
        arrow(4.2, 8.9, 2.0, 7.9, "I < ridge", "#4c9f70")
        arrow(5.8, 8.9, 8.0, 7.9, "I > ridge", "#e0a458")

        # common: occupancy
        box(3.2, 4.7, 3.6, 1.1,
            "5. Occupancy:\nfind binding limiter\n(thr / reg / smem)", "#cbd6ee")
        arrow(2.0, 6.7, 4.0, 5.8)
        arrow(8.0, 6.7, 6.0, 5.8)

        # divergence / banks
        box(0.6, 2.6, 3.6, 1.0, "6. Divergence?\nmake warp-uniform", "#f4cccc")
        box(5.8, 2.6, 3.6, 1.0, "7. Bank conflicts?\npad / swizzle smem", "#f4cccc")
        arrow(4.4, 4.7, 2.6, 3.6)
        arrow(5.6, 4.7, 7.4, 3.6)

        # done
        box(3.4, 0.5, 3.2, 1.0, "All clean & near roof?\nDONE.", "#d5e8d4", fs=9.5)
        arrow(2.4, 2.6, 4.4, 1.5)
        arrow(7.6, 2.6, 5.6, 1.5)

        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6. Glossary

    | Term | One-line meaning |
    |---|---|
    | **thread** | one lane of execution; runs your kernel body on its own data |
    | **warp** | 32 threads issuing the same instruction in lockstep (SIMT unit) |
    | **block (thread block)** | group of threads (≤1024) on one SM, sharing smem + barriers |
    | **grid** | the whole launch — all blocks of a kernel |
    | **cluster** | (Hopper/Blackwell) co-resident group of blocks; check `sm_120` support (`4a`) |
    | **SM (streaming multiprocessor)** | a GPU core that runs resident blocks' warps; you have 70 |
    | **SIMT** | Single Instruction, Multiple Thread — the warp execution model |
    | **occupancy** | resident warps ÷ max warps per SM (48); a means to latency hiding |
    | **latency hiding** | covering a stall by issuing from another ready warp |
    | **Little's Law** | required concurrency = latency × throughput |
    | **coalescing** | a warp's 32 threads hitting contiguous addresses → one wide transaction |
    | **bank conflict** | ≥2 lanes hitting the same shared-memory bank → serialized access |
    | **divergence** | a warp taking both sides of a data-dependent branch → both run, masked |
    | **shared memory (smem)** | fast on-SM scratchpad, shared within a block (~100 KB/SM) |
    | **register** | fastest per-thread storage; 65,536/SM, partitioned across threads |
    | **global / DRAM** | the 16 GB main memory; ~896 GB/s, hundreds of cycles latency |
    | **L2 cache** | 48 MB cache shared across SMs |
    | **bandwidth** | bytes/second movable from DRAM (the memory roof, ~896 GB/s) |
    | **arithmetic / operational intensity** | FLOPs ÷ bytes moved (the roofline x-axis) |
    | **roofline** | $P=\min(P_{\text{peak}}, B\cdot I)$ — the performance ceiling vs intensity |
    | **ridge point** | $P_{\text{peak}}/B$ — where memory- and compute-bound meet |
    | **memory-bound** | left of ridge; bandwidth limits you (fuse/coalesce/reuse) |
    | **compute-bound** | right of ridge; FLOP/s limits you (tensor cores, lower precision) |
    | **tiling / blocking** | loading a sub-block into smem to reuse it (raises intensity) |
    | **fusion** | merging ops into one kernel to avoid extra DRAM round-trips |
    | **tensor core** | fixed-function MMA unit; dense low-precision matmul (`3g`,`4a`) |
    | **WMMA / mma.sync / wgmma** | the instructions that drive tensor cores |
    | **TMA (Tensor Memory Accelerator)** | hardware bulk async global↔smem tensor copy (`4a`) |
    | **cp.async** | (Ampere+) per-thread async global→smem copy for pipelining (`3f`) |
    | **DSMEM (distributed shared memory)** | cross-block smem within a cluster (`4a`) |
    | **FP8 / FP4** | 8-/4-bit floats (E4M3, E5M2, E2M1) for narrow-precision matmul (`4a`) |
    | **microscaling (MX)** | a shared power-of-two scale per small block of narrow floats (`4a`) |
    | **quantization** | storing tensors in a narrow type with scales/zero-points (`2c`) |
    | **online softmax** | streaming softmax with a running max/sum (no full row) (`1d`,`2b`) |
    | **autotuning** | searching block sizes / num_warps / num_stages for the best config (`2a`) |
    | **`program_id`** | Triton: which program (block) the instance is (`1a`) |
    | **`do_bench`** | `triton.testing` median-latency benchmark (§4) |
    | **Nsight Compute (`ncu`)** | NVIDIA kernel profiler — occupancy, throughput, stalls (§4) |
    | **launch config** | the `<<<grid, block>>>` / Triton grid that maps data to threads |
    | **`__syncthreads`** | block-wide barrier; all threads must reach it |
    | **`__launch_bounds__`** | cap registers/thread to control occupancy (`3e`) |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 7. Reference links

    **Primary texts**

    - **PMPP** — *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj), 4th ed.
      — the fundamentals spine.
      [Publisher](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)
    - **Triton tutorials** — vector-add → softmax → matmul → layernorm → flash-attn → fp8.
      [triton-lang.org/main/getting-started/tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
    - **CUDA C++ Programming Guide** —
      [docs.nvidia.com/cuda/cuda-c-programming-guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
    - **CUDA C++ Best Practices Guide** —
      [docs.nvidia.com/cuda/cuda-c-best-practices-guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
    - **PTX ISA** —
      [docs.nvidia.com/cuda/parallel-thread-execution](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

    **Matmul & optimization**

    - **Simon Boehm — "How to optimize a CUDA matmul kernel"** —
      [siboehm.com/articles/22/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM)
    - **CUTLASS / CuTe** —
      [github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)

    **Attention & kernels**

    - **FlashAttention** (Dao et al.) —
      [arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135) (v1),
      [arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691) (v2)
    - **GPU MODE** (lectures + problem sets) —
      [github.com/gpu-mode](https://github.com/gpu-mode) ·
      [youtube.com/@GPUMODE](https://www.youtube.com/@GPUMODE)

    **Blackwell / narrow precision**

    - **NVIDIA Blackwell architecture whitepaper** —
      [resources.nvidia.com/en-us-blackwell-architecture](https://resources.nvidia.com/en-us-blackwell-architecture)
    - **OCP Microscaling (MX) Formats spec** —
      [opencompute.org](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)

    **Profiling**

    - **`triton.testing.do_bench`** —
      [triton-lang.org/main/python-api/generated/triton.testing.do_bench.html](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)
    - **Nsight Compute** —
      [docs.nvidia.com/nsight-compute](https://docs.nvidia.com/nsight-compute/) ·
      [Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters

    The skill the course is really teaching isn't any one kernel — it's the **loop**:
    place the work on the roofline, identify the bound, apply the matching pattern, tune
    against a measured roof, verify, repeat. This page is the index to that loop. When a
    kernel underperforms, you don't start from scratch; you start at §5, walk the
    checklist, and each branch points back to the lecture with the fix.

    Keep it open. The best kernel engineers aren't the ones who memorized the API — they're
    the ones who reach for the right diagnostic fast.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Where this is used

    Everywhere, from here on. This is the module you return to while writing every
    exercise (`e01`–`e13`, `c01`–`c06`) and the capstone (`4b`):

    - Stuck on an API call? §3 cheat-sheets (or the dropdown).
    - Kernel slow? §5 checklist + the flowchart.
    - Need a number for an estimate? §1 hardware table, §2 occupancy math.
    - Benchmarking? §4 profiling guide.
    - Forgot a term? §6 glossary.
    - Want to go deeper? §7 links.

    You've reached the end of the course. Go write kernels — and keep this page bookmarked.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [4B: Capstone](../4b_capstone/) &nbsp;|&nbsp; Next: [7B: Validating & Benchmarking](../7b_validation_and_benchmarking/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
