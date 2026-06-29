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
    # 0A: Orientation — From Running CUDA to Writing It

    > *"The free lunch is over."* — Herb Sutter
    >
    > You already drive the GPU every day through PyTorch. This course is about
    > climbing under the hood and writing the kernels yourself — first in **Triton**
    > (fast feedback, Python-level), then in **CUDA C++** (all the way to the metal).

    You know how to *run* CUDA. You have never written a `__global__` function or a
    Triton `@jit` kernel. By the end you will have written both — vector-add up to a
    fused attention kernel — and you will be able to look at a kernel and say *why*
    it is fast or slow.

    The whole course rests on one question you will ask of every kernel:

    $$\text{Am I limited by } \textbf{memory bandwidth} \text{ or by } \textbf{compute}?$$

    Everything else — coalescing, tiling, occupancy, tensor cores — is in service of
    pushing one of those two ceilings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. How this course is built

    There are **two tracks**, and they are different tools on purpose:

    - **Lectures** (these marimo notebooks) build *intuition*: the execution model,
      the memory hierarchy, the patterns, the math, the pictures. They run in your
      browser — pure `numpy`/`matplotlib`. They *show* kernel code but never run it.
    - **Exercises** (a terminal harness) are where you *write kernels* on the real
      GPU. marimo is the wrong place to iterate on a kernel; a watch-on-save runner
      that checks correctness **and** reports GB/s is the right place.

    A lecture ends by pointing you at the exercises it unlocks. The rule for the
    exercises: **you write every kernel.** The harness gives stubs and hints, never
    the answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. How to run it

    **Lectures** (from the repo root):
    ```bash
    marimo edit notebooks/0b_execution_model.py    # or any module
    ./launch.ps1 0b                                 # Windows helper
    ```

    **Exercises** — the real work:
    ```bash
    python -m harness.device_info        # print YOUR gpu's properties
    python -m harness.runner e01 --watch # write exercises/e01.../kernel.py, save, watch it check
    python -m harness.runner --all       # see your whole progress board
    ```

    The `--watch` runner re-runs the moment you save `kernel.py`: it verifies your
    output against a torch reference, then prints latency and achieved bandwidth (or
    TFLOP/s) against your card's ceiling. That number is the point.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Your hardware — the numbers that decide everything

    Kernel performance is a negotiation with *specific* limits. These were queried
    from the card (`python -m harness.device_info`), not copied from a spec sheet —
    rerun it any time. Memorize the shape of them; we will use them constantly.
    """)
    return


@app.cell
def _():
    def _run():
        # Queried from the RTX 5070 Ti (sm_120). Hardcoded here because notebooks
        # run in-browser (no torch); run `python -m harness.device_info` for live.
        rows = [
            ("Architecture",            "Blackwell, sm_120"),
            ("SMs",                     "70"),
            ("Warp size",               "32 threads"),
            ("Max resident threads/SM", "1536  (= 48 warps)"),
            ("Max threads/block",       "1024"),
            ("Registers/SM",            "65,536  (~42 regs/thread for full occupancy)"),
            ("Shared memory/SM",        "100 KB  (48 KB/block default, ~99 KB opt-in)"),
            ("L2 cache",                "48 MB"),
            ("DRAM",                    "16 GB GDDR7, 256-bit"),
            ("Peak DRAM bandwidth",     "~896 GB/s"),
        ]
        print("=== RTX 5070 Ti — the budget ===")
        for _k, _v in rows:
            print(f"  {_k:26s}: {_v}")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The latency you are hiding

    The reason occupancy matters: the memory hierarchy spans **three orders of
    magnitude** in latency. A kernel that stalls on DRAM without enough other work
    in flight is throwing away the chip. The bar chart below (log scale) is the
    single most important picture in the course.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        # Order-of-magnitude latencies (cycles) -- illustrative, typical GPU values.
        levels = ["Register", "Shared/L1", "L2", "DRAM (HBM/GDDR)"]
        cycles = [1, 30, 200, 500]
        colors = ["#4c9f70", "#5b8def", "#e0a458", "#d65f5f"]

        _fig, _ax = plt.subplots(figsize=(7, 3.2))
        _ax.barh(levels, cycles, color=colors)
        _ax.set_xscale("log")
        _ax.set_xlabel("approx. access latency (cycles, log scale)")
        _ax.set_title("Memory hierarchy: each level ~10x slower than the last")
        for _i, _c in enumerate(cycles):
            _ax.text(_c, _i, f"  ~{_c}", va="center")
        _ax.invert_yaxis()
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. The map

    Parts build on each other. Part 0 is the hard prerequisite for everything; Part 3
    re-derives Part 1's patterns in raw CUDA.

    ```
    0a ─ 0b ─ 0c ─ 0d           Orientation & execution model  (you are here)
                │
                ▼
       1a ─ 1b ─ 1c ─ 1d ─ 1e ─ 1f ─ 1g    Parallel patterns in Triton
                            │
                            ▼
                  2a ─ 2b ─ 2c ─ 2d         ML kernels in Triton
                            │
       (Triton mastery)     ▼
       3a ─ 3b ─ 3c ─ 3d ─ 3e ─ 3f ─ 3g    CUDA C++, to the metal
                            │
                            ▼
                      4a ─ 4b ──► 7a        Capstone, Blackwell, Reference
    ```

    **Prerequisite:** comfortable Python + PyTorch. Part 3 wants some C/C++ comfort.
    No kernel experience assumed — we start from the execution model next.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    > **References to keep open.** *Programming Massively Parallel Processors* (Hwu,
    > Kirk, El Hajj) is our spine. The [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
    > and the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
    > are the two primary docs. [GPU MODE](https://github.com/gpu-mode) lectures are
    > excellent supplementary material.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; Next: [0B: The Execution Model](../0b_execution_model/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
