# SEGMENTATION.md — GPU Kernels course plan

> Mirrors the structure/conventions of `ml-course` and `info-theory-course`.
> Single source of truth for the syllabus, the lecture↔exercise map, the
> dependency graph, and build notes.

## Owner decisions
1. **Stack order:** Triton first (fast feedback, ML-relevant), then CUDA C++ (re-derive
   each pattern by hand to the metal). Both target the learner's real hardware.
2. **Fundamentals → ML:** build the parallel-patterns foundation, then apply it to
   matmul / attention / quantization.
3. **Split delivery:** *lectures* are pyodide-safe marimo notebooks (intuition +
   visualization + shown code); *doing* is the terminal harness (real kernels).
4. **Intuition-and-code-first** (PMPP / GPU-MODE flavor): derive the idea, visualize
   it, then send the reader to the harness to write the kernel.
5. **Prerequisite:** comfortable Python + PyTorch user; some C/C++ familiarity for
   Part 3. No prior kernel experience assumed — we start from the execution model.

## Target hardware (queried, not guessed — `python -m harness.device_info`)
- **NVIDIA RTX 5070 Ti**, Blackwell, **`sm_120`**, 16 GB GDDR7
- **70 SMs**; warp = 32; **1536 max resident threads/SM** (→ 48 warps); 1024 threads/block
- **65,536 registers/SM** (≈42 regs/thread for full occupancy)
- **100 KB shared mem/SM** (48 KB per block default, opt-in ≈99 KB); **48 MB L2**; 256-bit bus
- **≈896 GB/s** theoretical DRAM bandwidth; FP32 ≈ tens of TFLOP/s (tensor cores much higher — measured in Parts 2–3)
- Toolchain: CUDA Toolkit 13.1 (`nvcc`), PyTorch 2.10 (cu128), Triton 3.6.0

## Reference texts (⭐ = primary anchor)
| Ref | Use |
|---|---|
| ⭐ *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj), 4th ed. | the fundamentals spine (Parts 0,1,3) |
| ⭐ Triton tutorials (triton-lang.org) | vector-add → softmax → matmul → layernorm → flash-attn → fp8 (Parts 1,2) |
| ⭐ CUDA C++ Programming Guide / Best Practices Guide | Part 3 |
| Simon Boehm, "How to optimize a CUDA matmul kernel" (siboehm.com) | Part 3 matmul tuning |
| GPU MODE lectures + problem sets (youtube / github) | cross-cutting |
| FlashAttention papers (Dao et al.) | Part 2 attention |
| CUTLASS / CuTe docs | Part 3 tensor cores |

---

## Syllabus

### Part 0 — Orientation & the execution model  *(thorough grounding — do not rush)*
| Module | Topics | Exercises |
|---|---|---|
| `0a_orientation` | how the course works; lectures vs harness; toolchain check; the roofline mindset; roadmap | `device_info` |
| `0b_execution_model` *(template)* | throughput vs latency; SIMT; thread→warp→block→grid; mapping to SMs; warp scheduling; divergence; the kernel-launch boundary | — |
| `0c_memory_hierarchy` | registers / shared / L1 / L2 / global / HBM; latency & bandwidth numbers; coalescing preview; the memory-wall | `e03` (preview) |
| `0d_occupancy_and_roofline` | occupancy & latency hiding (the central idea); the roofline model; compute- vs memory-bound; how to read a kernel's ceiling | — |

### Part 1 — Parallel patterns in Triton  *(fundamentals)*
| Module | Topics | Exercises |
|---|---|---|
| `1a_triton_model` | `program_id`, blocks, `tl.arange`, masking, `tl.load/store`; how Triton maps to the SIMT model; launch grids & `tl.constexpr` | `e01` vector-add, `e02` fused elementwise |
| `1b_memory_coalescing` | bandwidth as the budget; coalesced vs strided; measuring GB/s vs the 896 ceiling | `e03` copy/bandwidth |
| `1c_reductions` | tree reduction; row reduce; `tl.sum`/`tl.max`; numerical care | `e04` row sum/max |
| `1d_softmax` | the online-max trick; numerical stability; reduction+map fusion | `e05` softmax |
| `1e_tiling_matmul` | 2-D program ids; blocking/tiling; the tiled GEMM; reuse arithmetic | `e06` transpose, `e07` tiled matmul |
| `1f_fused_norms` | LayerNorm / RMSNorm; fusing reduce+normalize+affine; welford note | `e08` layernorm/rmsnorm |
| `1g_scan` | prefix-sum / cumsum; the associative-scan pattern (the hard one) | `e09` cumsum |

### Part 2 — ML kernels in Triton
| Module | Topics | Exercises |
|---|---|---|
| `2a_autotuning` | `@triton.autotune`, configs, masking arbitrary shapes; roofline-guided tuning; `triton.testing.do_bench` | `e10` autotuned matmul |
| `2b_flash_attention` | blocked QK·V; online softmax over KV blocks; why it's memory-bound-friendly | `e11` flash-attn forward |
| `2c_quantization` | int8 / fp8 dequant + matmul; scales & zero-points; perf vs accuracy | `e12` quantized matmul |
| `2d_autograd` | wiring kernels into `torch.autograd.Function`; custom ops; a fused backward | `e13` fused op + backward |

### Part 3 — CUDA C++  *(re-derive every pattern, to the metal)*
| Module | Topics | Exercises |
|---|---|---|
| `3a_cuda_model` | `__global__`, `blockIdx/threadIdx`, launch config, error-checking macro, the `nvcc` build loop | `c01` vector-add |
| `3b_shared_tiling` | shared memory by hand; the tiled matmul manually; `__syncthreads` | `c02` tiled matmul |
| `3c_warp_primitives` | `__shfl_*`, warp reductions, `__ballot`, warp-synchronous programming | `c03` warp reduce |
| `3d_memory_banks` | coalescing rules; shared-memory bank conflicts; vectorized `float4` loads; `__ldg` | `c04` conflict-free transpose |
| `3e_occupancy_tuning` | registers vs occupancy; `__launch_bounds__`; occupancy calculator; reading `ncu` | analysis exercise |
| `3f_async_pipelining` | `cp.async`; double-buffering; software pipelining | `c05` pipelined matmul |
| `3g_tensor_cores` | WMMA → `mma.sync`; fragments; a glimpse of CUTLASS / CuTe | `c06` WMMA matmul |

### Part 4 — Capstone & Blackwell
| Module | Topics | Exercises |
|---|---|---|
| `4a_blackwell` | `sm_120` specifics: 5th-gen tensor cores, FP4/FP8, TMA / tensor memory accelerator; what's new vs Ada/Hopper | — |
| `4b_capstone` | fused attention *or* quantized GEMM from scratch; benchmark vs `torch.scaled_dot_product_attention`; integrate into a real model | `capstone` |

### Part 7 — Reference
| Module | Topics |
|---|---|
| `7a_study_guide` | cheat-sheets (Triton API, CUDA launch, occupancy math); profiling (Nsight Compute / `do_bench`); the 5070 Ti roofline numbers; glossary; debugging checklist; reference links |

---

## Dependency graph (ASCII)
```
0a ─ 0b ─ 0c ─ 0d
            │
            ▼
   1a ─ 1b ─ 1c ─ 1d ─ 1e ─ 1f ─ 1g
                        │
                        ▼
              2a ─ 2b ─ 2c ─ 2d
                        │
   (Triton mastery)     ▼
   3a ─ 3b ─ 3c ─ 3d ─ 3e ─ 3f ─ 3g
                        │
                        ▼
                  4a ─ 4b ──► 7a (reference, dip in anytime)
```
Part 3 depends conceptually on Part 1 (same patterns, lower level), not on Part 2.
Part 0 is the hard prerequisite for everything.

## Exercise ↔ lecture map
| Exercise | Unlocked by | Metric |
|---|---|---|
| `e01` vector-add | 1a | bandwidth |
| `e02` fused elementwise | 1a | bandwidth |
| `e03` copy / bandwidth | 0c, 1b | bandwidth |
| `e04` row reduce | 1c | bandwidth |
| `e05` softmax | 1d | bandwidth |
| `e06` transpose | 1e | bandwidth |
| `e07` tiled matmul | 1e | flops |
| `e08` layernorm | 1f | bandwidth |
| `e09` cumsum | 1g | bandwidth |
| `e10` autotuned matmul | 2a | flops |
| `e11` flash-attn fwd | 2b | flops |
| `e12` quantized matmul | 2c | flops |
| `e13` fused op + backward | 2d | bandwidth |
| `c01`–`c06` | Part 3 | per-exercise |
| `capstone` | 4b | flops |

## Build notes
- Lecture notebooks must stay **pyodide-safe** (numpy/scipy/matplotlib only) so
  `python build_site.py` can export them to WASM. Kernel code is *shown*, not run,
  in notebooks; it is *run* in the harness.
- Modules marked advanced/optional off the critical path: `1g`, `3e`–`3g`, `4a`.
- Keep `home.py` syllabus tables and this file in sync with `notebooks/` file order.

## Authoring status
- ✅ infra (`harness/`, `build_site.py`, `launch.ps1`, requirements), conventions (`CLAUDE.md`)
- ✅ all 26 lecture notebooks authored (`home` + `0a`–`7a`); every one passes `marimo check`
- ✅ cross-model adversarial review run (Claude + Codex critic, binary-rubric verify); see `REVIEW.md`.
      Fixed: solution-leaks in `1d`/`1e`/`1f`/`7a` (lectures had shipped full kernels — now blanked to
      `...`-style skeletons), harness `--watch` crash + mean/median + reference-ordering bugs, the
      per-SM block cap (now 32 across `0b`/`0d`/`3e`/`7a`), and the `e07` TF32 precision note.
- ✅ exercises `e01`–`e13` scaffolded (stubs + specs + hints; all report `[TODO]`, references verified on-GPU); `c01`–`c06` outlined
