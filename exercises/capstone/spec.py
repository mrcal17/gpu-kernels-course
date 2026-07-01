"""Capstone spec -- the ONE spec.py you are allowed (and required) to edit.

Every other exercise hands you this file and says "don't touch". The capstone
inverts that: designing the contract -- reference, inputs, metric, tolerance --
IS milestone M0 of lecture 4b. Fill this in for the target you chose there
(Option A: fused attention, Option B: quantized GEMM), then write kernel.py
against it.

The harness contract (see harness/runner.py, decoded line-by-line in lecture 7b):
  TITLE        : str                     shown by the runner
  ENTRYPOINT   : str                     the function in kernel.py the runner calls
  METRIC       : "flops" | "bandwidth" | "none"
  TOL          : dict                    kwargs for torch.testing.assert_close
  make_inputs(): tuple of cuda tensors   passed positionally to your kernel
  reference(*inputs)                     the trusted torch answer (the runner
                                         computes it BEFORE calling your kernel,
                                         so an in-place bug can't corrupt it)
  flops(*inputs) -> int                  required if METRIC == "flops"
  bytes_moved(*inputs) -> int            required if METRIC == "bandwidth"

Everything below is a small, runnable Option-A (fused attention) placeholder so
`python -m harness.runner capstone` works out of the box. Replace it with YOUR
shapes, dtypes, tolerance, and metric -- and be ready to defend each choice
against 4b's acceptance criteria (Section 3).
"""
import torch

TITLE = "Capstone -- your kernel, your contract (edit this spec first)"

# The function in kernel.py the runner calls. Rename it if your op deserves a
# better name -- just keep spec.py and kernel.py in sync.
ENTRYPOINT = "capstone"

# Which roof are you chasing? Attention and GEMM are compute-bound at large
# shapes -> "flops" (TFLOP/s). If your variant is memory-bound, switch to
# "bandwidth" (GB/s) and define bytes_moved() instead of flops(). Naming your
# roof BEFORE you tune is the point of 4b Section 4.
METRIC = "flops"

# M0 says: fix the tolerance up front, and justify it. fp16 attention wants
# roughly atol=rtol=1e-2; a quantized GEMM wants a bound derived from the
# operand format (int8/fp8 error is quantization error, not fp16 rounding).
# 7b's tolerance table is the reference.
TOL = {"atol": 1e-2, "rtol": 1e-2}

# Placeholder shapes: small, so the not-implemented loop is instant. Your REAL
# benchmark shapes must be large enough that the math dominates the launch
# overhead (4b Section 6) -- and you should bench a curve of sizes, not one point.
B, H, N, D = 2, 4, 512, 64


def make_inputs():
    """Return the tensors the runner passes to your kernel, already on cuda.
    Fix the seed so a failure reproduces exactly."""
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    return q, k, v


def reference(q, k, v):
    """The trusted answer, in torch. Option A: torch's fused SDPA (also your
    speed baseline). Option B: a full-precision matmul of the dequantized
    operands (accuracy reference), with a tuned int8/fp8 path as the speed bar
    if you have one."""
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def flops(q, k, v):
    """FLOPs of ONE call -- the runner divides by measured seconds for TFLOP/s.
    Attention forward is two GEMMs per (batch, head): QK^T costs 2*N*N*D and
    P@V another 2*N*N*D. A plain GEMM (Option B) is the usual 2*M*N*K."""
    B_, H_, N_, D_ = q.shape
    return B_ * H_ * (4 * N_ * N_ * D_)


# def bytes_moved(*inputs) -> int:
#     """Define this INSTEAD of flops() if METRIC == "bandwidth": count the
#     bytes your kernel must move at minimum (each buffer read once + each
#     output written once), as in e01-e09's specs."""
