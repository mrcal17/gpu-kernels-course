# GPU Kernels — a course

Learn to write GPU kernels from scratch: **Triton first** (Python-level, fast
feedback), then **CUDA C++** (to the metal). Fundamentals → ML kernels. Built for, and
tuned to, an **RTX 5070 Ti (Blackwell, sm_120)**.

**▶ Live course site:** https://mrcal17.github.io/gpu-kernels-course/ — the lectures
run in your browser (WASM); the exercises run locally on your GPU.

Two tracks, by design:
- **Lectures** — marimo notebooks in [`notebooks/`](notebooks/) that build intuition
  and visualize the ideas (run in-browser; they show kernel code without running it).
- **Exercises** — a terminal harness in [`exercises/`](exercises/) + [`harness/`](harness/)
  where you write kernels on the real GPU and measure GB/s and TFLOP/s.

## Quick start
```bash
# 1. see your GPU's real limits
python -m harness.device_info

# 2. read a lecture
pip install -r requirements.txt          # marimo, numpy, scipy, matplotlib
marimo edit notebooks/0a_orientation.py  # or  ./launch.ps1 0a

# 3. write your first kernel (needs torch + triton)
python -m harness.runner e01 --watch
```

## Layout
| Path | What |
|---|---|
| `notebooks/` | lecture notebooks (`home.py` is the index) |
| `exercises/` | the kernel-writing exercises (stubs + specs + hints) |
| `harness/` | the runner (`runner.py`) and `device_info.py` |
| `SEGMENTATION.md` | the full course plan / syllabus / dependency graph |
| `CLAUDE.md` | authoring & build conventions |
| `build_site.py` | export the lectures to a WASM site in `docs/` (GitHub Pages) |

## Requirements
- **Lectures / site:** `requirements.txt` (marimo + numpy/scipy/matplotlib — pyodide-safe).
- **Exercises:** a CUDA-enabled **PyTorch** + **Triton** (already installed here:
  torch 2.10 cu128, triton 3.6). Part 3 also needs the **CUDA Toolkit** (`nvcc`) and,
  on Windows, an MSVC host compiler. See `requirements-exercises.txt`.

Start at [`notebooks/home.py`](notebooks/home.py) or just run `./launch.ps1`.
