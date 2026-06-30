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
    # GPU Kernels — From Running CUDA to Writing It

    You already drive the GPU through PyTorch every day. This course takes you under
    the hood to **write the kernels yourself** — first in **Triton** (Python-level,
    fast feedback), then in **CUDA C++** (all the way to the metal). You build the
    parallel-patterns foundation, then apply it to real ML kernels: matmul, softmax,
    fused norms, flash attention, quantization.

    Every kernel is judged by one question: **am I limited by memory bandwidth or by
    compute?** Coalescing, tiling, occupancy, and tensor cores are all in service of
    pushing one of those two ceilings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this course

    There are **two tracks**, by design:

    - **Lectures** (these notebooks) build intuition — the execution model, the memory
      hierarchy, the patterns, the math, the pictures. They run in your browser
      (`numpy`/`matplotlib`) and *show* kernel code without running it.
    - **Exercises** (a terminal harness) are where you **write kernels** on the real
      GPU. A watch-on-save runner checks correctness and reports GB/s or TFLOP/s.

    ```bash
    # lectures
    marimo edit notebooks/0b_execution_model.py     # or ./launch.ps1 0b

    # exercises -- the real work
    python -m harness.device_info                    # your GPU's properties
    python -m harness.runner e01 --watch             # write a kernel, watch it check
    python -m harness.runner --all                   # progress board
    ```

    Work the lectures and exercises together: each lecture ends by pointing at the
    exercises it unlocks. **You write every kernel** — the harness gives hints, never
    solutions.

    > **Prerequisite:** a comfortable Python + PyTorch user. Part 3 (CUDA C++) wants
    > some C/C++ familiarity. **No kernel experience assumed** — we start from the
    > execution model. This course does not teach PyTorch or linear algebra.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 0 — Orientation & the execution model

    | Module | What you'll learn |
    |---|---|
    | [0A: Orientation](0a_orientation/) | how the course works; the two tracks; your hardware; the roofline mindset |
    | [0B: The Execution Model](0b_execution_model/) | SIMT; thread → warp → block → grid; mapping to SMs; divergence; the launch boundary |
    | [0C: Memory Hierarchy](0c_memory_hierarchy/) | registers → shared → L2 → DRAM; latency & bandwidth; coalescing preview; the memory wall |
    | [0D: Occupancy & the Roofline](0d_occupancy_and_roofline/) | occupancy & its three limiters; latency hiding; the roofline; compute- vs memory-bound |

    ## Part 1 — Parallel patterns in Triton

    | Module | What you'll learn | Exercise |
    |---|---|---|
    | [1A: The Triton Programming Model](1a_triton_model/) | `program_id`, blocks, masking, `load`/`store`, launch grids | `e01`, `e02` |
    | [1B: Memory & Coalescing](1b_memory_coalescing/) | bandwidth as the budget; coalesced vs strided | `e03` |
    | [1C: Reductions](1c_reductions/) | tree reduction; row reduce; `tl.sum`/`tl.max` | `e04` |
    | [1D: Softmax](1d_softmax/) | the online-max trick; numerical stability | `e05` |
    | [1E: Tiling & Matmul](1e_tiling_matmul/) | 2-D tiling; the tiled GEMM; arithmetic intensity | `e06`, `e07` |
    | [1F: Fused Norms](1f_fused_norms/) | LayerNorm/RMSNorm; fusing to cut memory traffic | `e08` |
    | [1G: Scan / Prefix-Sum](1g_scan/) *(advanced)* | the associative-scan pattern | `e09` |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2 — ML kernels in Triton

    | Module | What you'll learn | Exercise |
    |---|---|---|
    | [2A: Autotuning](2a_autotuning/) | `@triton.autotune`; masking arbitrary shapes; roofline-guided tuning | `e10` |
    | [2B: Flash Attention](2b_flash_attention/) | blocked QK·V; online softmax over KV blocks | `e11` |
    | [2C: Quantization](2c_quantization/) | int8/fp8 dequant + matmul; scales & zero-points | `e12` |
    | [2D: Autograd Integration](2d_autograd/) | `torch.autograd.Function`; custom ops; fused backward | `e13` |

    ## Part 3 — CUDA C++, to the metal

    | Module | What you'll learn | Exercise |
    |---|---|---|
    | [3A: The CUDA C++ Execution Model](3a_cuda_model/) | `__global__`, indexing, launch, the `nvcc` loop | `c01` |
    | [3B: Shared-Memory Tiling](3b_shared_tiling/) | shared memory by hand; the manual tiled matmul | `c02` |
    | [3C: Warp-Level Primitives](3c_warp_primitives/) | `__shfl`, warp reductions, ballot | `c03` |
    | [3D: Coalescing & Bank Conflicts](3d_memory_banks/) | coalescing rules; bank conflicts; `float4` | `c04` |
    | [3E: Occupancy Tuning](3e_occupancy_tuning/) *(advanced)* | registers vs occupancy; `__launch_bounds__`; `ncu` | analysis |
    | [3F: Async Copy & Pipelining](3f_async_pipelining/) *(advanced)* | `cp.async`; double-buffering | `c05` |
    | [3G: Tensor Cores](3g_tensor_cores/) *(advanced)* | WMMA → `mma`; a glimpse of CUTLASS/CuTe | `c06` |

    ## Part 4 — Capstone & Blackwell · Part 7 — Reference

    | Module | What you'll learn |
    |---|---|
    | [4A: Blackwell (sm_120)](4a_blackwell/) | 5th-gen tensor cores, FP4/FP8, TMA — what's new on your card |
    | [4B: Capstone](4b_capstone/) | fused attention *or* quantized GEMM from scratch, benchmarked vs torch |
    | [7A: Study Guide & Reference](7a_study_guide/) | cheat-sheets, profiling, glossary, the roofline numbers, links |
    | [7B: Validating & Benchmarking](7b_validation_and_benchmarking/) | proving correctness with `assert_close`; timing with `do_bench`; GB/s & TFLOP/s vs the roofline; the `spec.py` contract |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Dependency graph

    ```
    0a ─ 0b ─ 0c ─ 0d           Orientation & execution model
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

    Part 0 is the hard prerequisite for everything. Part 3 re-derives Part 1's patterns
    in raw CUDA, so it depends on Part 1, not on Part 2. Dip into 7A any time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## References

    | Text | Use |
    |---|---|
    | [Programming Massively Parallel Processors](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) (Hwu, Kirk, El Hajj) | the fundamentals spine |
    | [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) | Parts 1–2 |
    | [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) · [Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | Part 3 |
    | [How to optimize a CUDA matmul](https://siboehm.com/articles/22/CUDA-MMM) (Simon Boehm) | Part 3 matmul |
    | [GPU MODE](https://github.com/gpu-mode) lectures & problem sets | cross-cutting |
    | [FlashAttention](https://arxiv.org/abs/2205.14135) | Part 2 attention |
    | [CUTLASS / CuTe](https://github.com/NVIDIA/cutlass) | Part 3 tensor cores |

    *Built with [marimo](https://marimo.io). Source layout and conventions in
    `SEGMENTATION.md` and `CLAUDE.md`.*
    """)
    return


if __name__ == "__main__":
    app.run()
