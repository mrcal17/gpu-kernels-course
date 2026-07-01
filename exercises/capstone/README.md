# capstone — your kernel, end to end

**Goal:** design, implement, verify, benchmark, and integrate one real kernel —
**fused attention** (FlashAttention-style forward) or **quantized GEMM** — with no
prescribed steps. Lecture `4b` is the full project brief: the two options, the
milestones (M0–M4), the acceptance criteria, and the roofline targets. Read it first.

Unlocked by: everything. This is the integration test for *you*.

## How this exercise is different

Every other exercise gives you a finished `spec.py` and says *don't edit it*. Here the
spec **is the first deliverable**:

1. **M0 — write `spec.py`.** Choose your op, shapes, dtypes, metric, and tolerance, and
   encode them as the harness contract (reference, `make_inputs`, `flops`/`bytes_moved`).
   The file ships as a small runnable fused-attention placeholder with every field
   explained — replace it with *your* contract and be ready to defend each choice.
2. **M1–M3 — write `kernel.py`.** Naive-correct first, then the structural win
   (online-softmax streaming / dequant-in-loop), then tune. The runner re-checks
   correctness on every save.
3. **M4 — integrate.** Swap the kernel into a real model and confirm parity + a measured
   speedup. That part lives outside the harness — `4b` §2 tells you what "done" means.

## Run it
```bash
python -m harness.runner capstone --watch
```
Until you implement the entrypoint it reports `[TODO]`; once it passes you get
`[PASS]` / `[PERF]` / `[REF]` like every other exercise.

## The validate/benchmark contract

The runner is doing nothing you can't do yourself: reference-before-kernel,
`torch.testing.assert_close` with *your* tolerance, `do_bench` median latency, then
FLOP or byte counts → TFLOP/s or GB/s against the roofline. Lecture `7b` is the
reference card for that whole loop — tolerances, timing traps, throughput math, and
the `spec.py` contract decoded. Hold yourself to it: a claim needs a measurement.

## Hints
There are none. That's the exercise.
