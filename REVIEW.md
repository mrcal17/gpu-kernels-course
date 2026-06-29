# gpu-kernels — Course Review

*Cross-model adversarial review: Claude dimension reviewers (teaching, pedagogy, harness)
plus an agent-swarm Codex critic. Every finding below survived a binary-rubric refute panel.
17 unique candidates were raised; 13 are confirmed here. Counts in this report dedupe findings
that multiple reviewers raised about the same location.*

---

## 1. Executive summary

**Verdict: fix-first.**

The course is structurally sound and pedagogically thoughtful — the lecture prose, numpy/Welford
demos, roofline reasoning, and the `--watch` exercise harness are genuinely good teaching
infrastructure. But it ships with a **systematic solution-leak problem**: four lectures (1d, 1e,
1f) and the 7a study guide hand over complete, copy-pasteable Triton kernels for the very exercises
they unlock (e05 softmax, e07 matmul, e08 LayerNorm, e01 vector-add), each behind a label that
*claims* the solution is withheld ("skeleton — structure only", "NOT the e07 solution",
"ILLUSTRATIVE SKELETON — not the solution"). These labels are false and directly violate the
course's own no-complete-solution rule (CLAUDE.md:14-17). A learner can lift any of these blocks
verbatim and pass. This guts the exercise-driven design that is the course's whole premise, so it
must be fixed before the course ships. A secondary cluster of harness robustness and
internal-consistency bugs (watch-loop crash on save, mean-vs-median mislabeling, occupancy-cap
disagreement, a TF32 precision inconsistency in e07) are real but non-blocking. None of the leaked
code is a safety issue — every block is inside fenced markdown and is pyodide-safe.

---

## 2. Findings by severity

### Critical

The pattern in C1–C3 is identical: a fenced block in a lecture is labelled as a
withheld skeleton, but the kernel body is fully written out and matches the target exercise's
contract line-for-line. The correct model already exists in the repo — `1a_triton_model.py:433-456`
and `3b_shared_tiling.py:360-383` leave the load/index expressions as `...`, and that is the style
to copy.

**C1 — `notebooks/1d_softmax.py:251-276` — 1d ships the complete e05 softmax kernel.**
The block labelled "skeleton — structure only" (line 249) with the claim "Notice what is *not*
spelled out" (line 278) is a fully runnable `@triton.jit softmax_kernel`: masked load with
`other=-inf`, row max `tl.max`, stable `tl.exp(x-m)`, `tl.sum`, normalize `y=e/z`, and the masked
store. Only the launch grid / `BLOCK_SIZE` / `num_warps` are withheld — and those are trivial
(e05's README hint says "pick a power of two >= N"). The hard part, the kernel body, is the entire
e05 contract, handed over.
*Fix:* blank the body to comments/`...` in the 1a §8 style, leaving only the signature and section
markers. Keep the numerical-stability math in prose/numpy (that is the lecture's real content).
Also correct the false "what is not spelled out" claim at line 278.

**C2 — `notebooks/1e_tiling_matmul.py:268-292` — 1e ships nearly the entire e07 tiled GEMM.**
The block labelled "STRUCTURE (illustrative pseudocode — NOT the e07 solution)" is valid,
near-complete Triton: 2-D program ids, both offset broadcasts, `acc = tl.zeros(...)`, the
`a_ptrs`/`b_ptrs` construction, the full K-loop body (`tl.load`/`tl.load`/`acc += tl.dot`), and
both pointer advances. The *only* thing missing is the `mask=...` arguments and the final store —
and e07's README confirms masking + the masked store *are* "the crux." So the lecture gives away
everything except the crux's mask expressions, and the "NOT the e07 solution" label is inaccurate.
(The transpose skeleton at lines 70-83 has the same only-missing-one-line shape but is lower
stakes.)
*Fix:* strip the K-loop body and accumulator construction to comments, or remove enough lines that
the block is structurally incomplete rather than merely mask-incomplete. Correct the label.

**C3 — `notebooks/1f_fused_norms.py:244-266` — 1f ships the complete forward e08 LayerNorm, and the "crux is yours" claim is false.**
The block labelled "ILLUSTRATIVE SKELETON — not the solution" (line 245) with prose "the crux …
is yours to write" (lines 241-242) is a complete, runnable forward LayerNorm `@triton.jit`:
per-row offset, masked load, mean, centered `xc` via `tl.where`, variance `tl.sum(xc*xc)/N`,
`rstd = 1/tl.sqrt(var+eps)`, gamma/beta loads, `y = xc*rstd*g + b`, and the masked store. Lines
268-269 go further and hand over the RMSNorm variant in prose. The header claims the masking and
"the backward pass" are yours — but e08 (`exercises/e08_layernorm/README.md:11`, targets
`F.layer_norm`) is **forward-only with no backward**, and the masking is already shown in the
block. So the lecture gives away the entire forward kernel e08 requires while claiming it doesn't.
*Fix:* cut the reduction/normalize lines to comments, drop the RMSNorm hand-off, leave only the
load/store frame. Correct the "crux is yours / backward pass" claim — e08 has no backward
component.

### Warning

**W1 — `harness/runner.py:157-167` — `--watch` loop crashes on transient `stat()` errors during editor saves (Windows).**
The watch loop calls `target.stat()` every 0.4s inside a `try/except` that catches **only**
`KeyboardInterrupt`. On Windows, when an editor (VS Code, etc.) saves `kernel.py`, the file is
briefly locked or atomically replaced, so `stat()` can raise `PermissionError` /
`FileNotFoundError`. Those propagate and crash the watcher — the primary advertised workflow
(`python -m harness.runner e01 --watch`) — at exactly the moment the learner saves, which is the
failure mode `--watch` exists to handle.
*Fix:* add `except (FileNotFoundError, PermissionError, OSError): continue` alongside the
`KeyboardInterrupt` handler so a transient save error is skipped and retried on the next poll.

**W2 — `notebooks/7a_study_guide.py:167-182` (and a near-identical block ~311-324) — study guide leaks a complete e01 vector-add solution.**
The 7a cheat-sheet includes a complete, copy-pasteable vector-add kernel (program_id, offs, mask,
two masked loads, store of `x+y`) plus the `grid` lambda and the `kernel[grid](...)` launch — i.e.
effectively the entire e01 solution. A second near-identical block appears around lines 311-324
(the "Triton API (full)" `add_kernel`). Confirmed by direct read. (Treated as Warning rather than
Critical because e01 is the trivial first exercise and a study guide is a more defensible place for
a worked example than a lecture — but it is still a leak.)
*Fix:* replace the full kernel/launch with a signature plus TODO comments, or gate complete worked
solutions behind a clearly-marked post-completion section.

**W3 — `exercises/e07_matmul/spec.py:4-7` — TF32 disabled on the reference but `tl.dot` guidance defaults to TF32.**
`spec.py` sets `allow_tf32 = False` so the torch reference is true fp32, but the README (line 31,
"tl.dot maps to tensor cores") and the kernel stub steer learners to plain `tl.dot` with no mention
of `input_precision`. On fp32 inputs Triton's `tl.dot` defaults to TF32 tensor-core math, so a
learner following the hints multiplies in TF32 against a true-fp32 reference — an internally
inconsistent precision target. (The tolerances `atol=1e-1`/`rtol=1e-2` at K=1024 are loose enough
that it likely still passes, which is why this is a Warning, but the guidance contradicts itself.)
*Fix:* either tell learners to use `tl.dot(..., input_precision='ieee')` in the README/stub, or
explicitly state that a TF32-accumulating kernel is acceptable and that the tolerances are set for
it.

### Minor

**M1 — `harness/runner.py:57-62` — `_bench` claims median ms but `do_bench` returns mean.**
The docstring/comment say "Return median milliseconds per call," but `_bench` calls
`triton.testing.do_bench(fn, warmup=25, rep=100)` with no `return_mode`/`quantiles`. In triton 3.6
`do_bench` defaults to `return_mode='mean'`, and the CUDA-event fallback also computes a mean. So
the reported GB/s and TFLOP/s are mean-based, contradicting the documented contract. Not
correctness-affecting, but the stated metric is wrong.
*Fix:* pass `quantiles=[0.5]` / `return_mode='median'` (and compute median in the fallback), or fix
the docstring/comment to say "mean milliseconds per call."

**M2 — `harness/runner.py:109-117` — `reference()` is computed after the learner kernel runs on the same tensors.**
`make_inputs()` is called once (line 105); `entry(*inputs)` runs at 109 and `spec.reference(*inputs)`
at 117 on the same tensors. If a buggy kernel mutates an input in place — plausible for the
stride-heavy e06 transpose / e07 matmul, where a learner could mis-target a store at the input
pointer — the reference is then computed on corrupted data, so a wrong kernel could spuriously
*pass* or fail confusingly.
*Fix:* compute `ref = spec.reference(*inputs)` **before** calling `entry(*inputs)`, or run the
kernel on cloned inputs, so the reference is never derived from kernel-mutated data.

**M3 — per-SM block cap is inconsistent across the occupancy notebooks (16 vs 24).**
`notebooks/0b_execution_model.py:276` uses `MAX_BLOCKS_SM = 16` (with prose at line 251 stating a
"typically 16-block-per-SM" limit), but every later occupancy calculator uses 24:
`0d_occupancy_and_roofline.py:195`, `3e_occupancy_tuning.py:364`, and `7a_study_guide.py:101` all
use a cap of 24. The course's occupancy math therefore disagrees with itself for small block sizes.
(Aside: the real Blackwell sm_120 per-SM resident-block cap is 32, so neither value matches the
stated target GPU.)
*Fix:* use a single documented per-SM block cap consistently across 0b/0d/3e/7a — ideally 32 to
match the sm_120 target, or one clearly-labelled illustrative value — and update 0b's "16-block"
prose to match.

---

## 3. What's solid

- **The lecture prose and conceptual content are strong.** The numerical-stability explanation in
  1d, the reduce→map framing and Welford numpy demo in 1f, and the DRAM-traffic / fused one-load-one-store
  argument are the real teaching value and hold up well — the leak fixes can preserve all of it.
- **The course already demonstrates the correct skeleton style it should apply everywhere.**
  `1a_triton_model.py:433-456` and `3b_shared_tiling.py:360-383` leave load/index expressions as
  `...` — a ready-made template for fixing C1–C3.
- **The `--watch` exercise harness is a good exercise-driven workflow** (`python -m harness.runner
  e01 --watch`), with a clean spec/reference/benchmark structure per exercise.
- **Exercises ship real contracts:** per-exercise stubs with TODOs, README hint sequences, and
  reference-backed correctness plus GB/s / TFLOP/s benchmarking give learners concrete, measurable
  targets.
- **The occupancy/roofline thread is a genuine through-line** across 0b/0d/3e/7a — the only problem
  is the inconsistent constant (M3), not the pedagogy.
