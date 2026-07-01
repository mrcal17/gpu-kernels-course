# CLAUDE.md — authoring & build conventions for `gpu-kernels`

This course mirrors the structure and conventions of `info-theory-course` and
`ml-course`, with **one deliberate split**:

- **Lectures = marimo notebooks** in `notebooks/` (pyodide-safe: numpy / scipy /
  matplotlib only). They build intuition, derive the math, and *show* kernel code
  in fenced markdown — they do **not** run kernels.
- **Exercises = a terminal harness** in `exercises/` + `harness/` (real GPU:
  torch / triton / nvcc). marimo is the wrong tool for writing kernels, so the
  doing happens here. Each lecture ends by pointing at the exercises it unlocks.

> **Teaching-mode rule (non-negotiable).** Exercise files are stubs the learner
> fills in. **Never** commit a worked kernel solution into `exercises/`. Lectures
> may show *illustrative* snippets of an idea, but the exercises themselves stay
> solution-free — hints, formulas (with variables left to map), and conceptual
> nudges only. See the global teaching-mode instructions.

---

## Notebook conventions (copy these exactly)

**Standard header** — every notebook starts with:
```python
import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)
```
and ends with:
```python
if __name__ == "__main__":
    app.run()
```

**Rules of the road:**
1. **First cell** is the bare `import marimo as mo`. Import `numpy`/`matplotlib`
   **locally inside each code cell that uses them**, not globally (pyodide + marimo
   DAG hygiene).
2. **Markdown cells** use `mo.md(r"""...""")` — always raw triple-quoted strings,
   never f-strings (no variable interpolation in prose). Use `@app.cell(hide_code=True)`
   for pure-prose cells.
3. **Math** is LaTeX: inline `$H(X)$`, display `$$ ... $$`. The `r"""` is what keeps
   backslashes alive. Box key results with `$$\boxed{...}$$`.
4. **Demo code cells** wrap everything in a local `def _run(): ...; _run()` and
   prefix locals with `_` so nothing leaks into marimo's global DAG. Never redeclare
   a variable across cells; never use `global`.
5. **Plots:** matplotlib only. Build `_fig, _ax = plt.subplots(...)`, return `_fig`
   as the last expression — never `plt.show()`.
6. **Interactivity:** define a `mo.ui.slider(...)` (etc.) in one cell and display it;
   read `.value` in a **separate, later** cell (marimo reactivity rule).
7. **Showing kernel code:** GPU kernels (Triton/CUDA) that the reader will actually
   run go in **fenced code blocks inside `mo.md`**, with a pointer to the matching
   terminal exercise. Do not `import triton`/`torch` in a notebook — it breaks the
   WASM build.
8. **Last cell** is a `hide_code=True` nav footer (see template) with
   `../<module>/` site-relative links and HTML arrow entities (`&#8592;`, `&#8594;`,
   `&#8593;`).
9. After authoring, run `marimo check --fix notebooks/<file>.py`.

**Lesson skeleton** (mirror `0b_execution_model.py`, the template):
title + epigraph → numbered `## N.` concept sections (prose + LaTeX + worked
example + a `>` reference blockquote) → matplotlib visualization → `mo.ui`
slider+plot → "Why this matters for kernels" → **"Now go write it"** pointer to
exercises → nav footer.

**Naming:** `notebooks/<part><letter>_<snake_slug>.py`, lowercase. Part digit +
letter orders within the part. `home.py` is the index. Keep `home.py` and
`SEGMENTATION.md` in sync with file order.

---

## Exercise conventions

```
exercises/<id>_<slug>/
  kernel.py   # learner writes this — stub with TODOs, raises NotImplementedError
  spec.py     # harness contract (reference, inputs, metric). Do not edit.
  README.md   # the brief + layered hints (NO solution)
```
- Triton exercises are `eNN_*`; CUDA C++ exercises are `cNN_*`.
- `spec.py` defines `TITLE, ENTRYPOINT, make_inputs(), reference(*), METRIC,
  bytes_moved(*)/flops(*), TOL`. See `harness/runner.py` for the full contract.
- Run: `python -m harness.runner e01 --watch` (from repo root).

---

## Build & run

- **Author/run a lecture:** `marimo edit notebooks/0b_execution_model.py`
- **Launch helper:** `pwsh ./launch.ps1 0b`  (or `bash launch.sh 0b`)
- **Query your GPU:** `python -m harness.device_info`
- **Run an exercise:** `python -m harness.runner e01 --watch`
- **Build the WASM site:** `python build_site.py` → `docs/` (GitHub Pages).
  Only lecture notebooks are exported; the exercise harness is local-only.

Deps: `requirements.txt` is the pyodide-safe set for notebooks/site. The exercise
harness uses your local GPU stack (`requirements-exercises.txt`: torch + triton —
tested with torch 2.10 cu128 + triton 3.6 on an RTX 5070 Ti).

---

## Personal solutions (keep them out of git)

- The author's worked solutions live in `solutions/` (gitignored).
- The solved exercise `kernel.py` files (`e01`–`e07`) are marked
  **`skip-worktree`** in git, so `git add -A` cannot stage the worked code —
  the committed blobs stay the clean stubs.
- Inspect / clear the flags:
  ```bash
  git ls-files -v | grep ^S                       # list skip-worktree'd files
  git update-index --no-skip-worktree <file>      # clear (to commit a stub change)
  git update-index --skip-worktree <file>         # re-apply
  ```
- **When a new exercise gets solved, apply the same treatment** before working
  in its `kernel.py`: `git update-index --skip-worktree exercises/<id>/kernel.py`.
