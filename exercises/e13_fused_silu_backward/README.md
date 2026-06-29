# e13 — Fused SiLU with a hand-written backward

**Goal:** a fused `SiLU(x) = x * sigmoid(x)` forward kernel AND a hand-written
backward kernel, glued together with `torch.autograd.Function` so PyTorch calls
your backward inside `.backward()`. The point (lecture 2d): you own the
gradient, and you *recompute* instead of stashing intermediates.

Unlocked by: `2d` (custom autograd Functions).

## The spec
- Input: one flat float32 vector `x`. Output: a **tuple** `(out, grad_input)`.
- `out = SiLU(x)`; `grad_input` = gradient of `out.sum()` w.r.t. `x` (so the
  incoming `grad_output` is all-ones). The harness grades **both** at once.
- Metric: **bandwidth** = `4*N*element_size / time` (forward reads x + writes out,
  backward reads x + writes grad_input).

## What to write (`kernel.py`)
- `silu_fwd_kernel` + `silu_bwd_kernel` (two `@triton.jit` kernels).
- thin launch wrappers for each.
- a `torch.autograd.Function` subclass wiring them together.
- `silu_fwd_bwd(x)` — runs the forward, triggers the backward, returns the tuple.

## Hints — one at a time
1. **Two kernels and one wrapper.** Kernel A is the fused SiLU forward
   (`out = x * sigmoid(x)`, elementwise). Kernel B is the fused backward (given
   `grad_output`, produce `grad_input`). The wrapper is a
   `torch.autograd.Function` that glues them so PyTorch can call your backward.
2. **Forward is a 1-D elementwise map** over a flat vector — the same shape as
   your very first vector-add/elementwise kernels: one program per chunk, a tail
   mask, load-compute-store.
3. **The autograd contract (lecture 2d):** `backward` receives
   `grad_output = dL/d(out)` and must return `dL/dx = grad_output * (d out / d x)`,
   elementwise here. You need the derivative of SiLU. Derive it from
   `out = x*sigmoid(x)` using the product rule and the fact that
   `sigmoid'(x) = sigmoid(x)*(1 - sigmoid(x))`. The formula's symbols will not be
   your code's variable names — map them yourself.
4. **Save the minimum.** Your backward needs `x` to recompute `sigmoid(x)`; save
   `x` in the forward via `ctx.save_for_backward` and recompute `sigmoid` in the
   backward rather than stashing the sigmoid tensor too. That
   recompute-don't-store choice is the whole fused-backward lesson from 2d.
5. **Wiring it up:** subclass `torch.autograd.Function` with staticmethod
   `forward(ctx, x)` and `backward(ctx, grad_output)`; call it with `.apply`
   (never the constructor); backward returns ONE gradient because forward took
   one tensor input.
6. **Why the entrypoint returns a tuple:** the harness grades both passes at once.
   Your `silu_fwd_bwd` must run the forward AND trigger the backward
   (`out.sum().backward()`) and hand back `(forward_out, grad_input)`. The
   reference returns torch's own `(silu, grad)`, and the runner compares
   element-wise — so a correct forward with a wrong derivative still fails.
7. **Guard against mutating inputs:** the runner computes the reference from the
   same `x` *before* calling you, so make the tensor you attach `requires_grad`
   to a **clone**, and **detach** the forward output you return so it carries no
   stray graph. Otherwise you corrupt the run or the grader complains.

## Going for performance
- Both kernels are pure bandwidth-bound elementwise maps — the work is one
  `sigmoid` plus a couple of mults per element, so you should land near the
  memory roofline (the `0d` story). `tl.sigmoid` is one transcendental; building
  it from `tl.exp` costs the same.
- `BLOCK_SIZE` of 1024–4096 is the usual sweet spot for a flat 1-D map; bigger
  blocks mean fewer programs but more registers per program.
- The fused win vs. plain autograd: PyTorch's unfused chain would
  materialize and re-read intermediate tensors. Fusing forward and backward each
  into a single kernel pass is exactly why `bytes_moved` is only `4*N` here.
