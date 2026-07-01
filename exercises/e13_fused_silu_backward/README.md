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
- Metric: **bandwidth** = `5*N*element_size / time` (fwd: read x, write out;
  bwd: read x, read grad_output, write grad_input).

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
8. **The `grad_output` autograd hands you is not what you might assume.** Before
   doing raw pointer math on it in the backward, ask what its strides are —
   autograd makes no contiguity promise, and `y.sum().backward()` in particular
   delivers an *expanded* all-ones tensor with stride 0. Raw pointer arithmetic
   on that reads garbage. Check the layout and normalize it if needed.

## Validate & benchmark it yourself
The runner's `[PASS]` / `[PERF]` / `[REF]` lines are just the `1a` correctness-and-speed
loop. Here it is for this kernel, to run yourself in a scratch script:

```python
import torch, triton
import torch.nn.functional as F

torch.manual_seed(0); x = torch.randn(1 << 24, device="cuda", dtype=torch.float32) * 4.0   # as spec.py builds it

# reference: forward + the gradient via autograd (VJP with an upstream grad of ones)
xr = x.detach().clone().requires_grad_(True)
y_ref = F.silu(xr)
y_ref.sum().backward()                       # fills xr.grad
y, gx = silu_fwd_bwd(x)                       # your fused kernel returns BOTH outputs
torch.testing.assert_close(y,  y_ref,   atol=1e-5, rtol=1e-5)   # elementwise fp32 -> tight
torch.testing.assert_close(gx, xr.grad, atol=1e-5, rtol=1e-5)

ms   = triton.testing.do_bench(lambda: silu_fwd_bwd(x), warmup=25, rep=100, return_mode="median")
gbps = 5 * x.numel() * x.element_size() / (ms * 1e-3) / 1e9   # fwd: read x, write y; bwd: read x, read grad_output, write grad_input
print(f"{gbps:.0f} GB/s")
```

One honest caveat: the timed call is the whole `silu_fwd_bwd` — clone, graph build,
`.sum()`, and autograd dispatch included — not just your two kernels. The large `N`
keeps kernel time dominant, but when tuning block sizes, benchmark the raw kernel
launches separately.

Bandwidth-bound, and unusually **tight** (`1e-5`): SiLU and its derivative are elementwise
fp32 with no reduction, so a correct kernel matches almost exactly — slack this tight is what
catches a wrong derivative formula. Fusing forward + backward means five buffers of traffic in
two passes. Full tolerance table and timing traps: `7b`.

## Going for performance
- Both kernels are pure bandwidth-bound elementwise maps — the work is one
  `sigmoid` plus a couple of mults per element, so you should land near the
  memory roofline (the `0d` story). `tl.sigmoid` is one transcendental; building
  it from `tl.exp` costs the same.
- Block size is a tuning knob for a flat 1-D map: too small and you launch more
  programs than needed; too large and each program burns more registers. Try a
  few powers of two and watch the bandwidth number to find the sweet spot.
- The fused win vs. plain autograd: PyTorch's unfused chain would
  materialize and re-read intermediate tensors. Fusing forward and backward each
  into a single kernel pass is exactly why `bytes_moved` is only `5*N` here.
