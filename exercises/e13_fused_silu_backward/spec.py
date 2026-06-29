"""Harness contract for e13: fused SiLU with a hand-written backward. Do not edit."""
import torch
import torch.nn.functional as F

TITLE = "Fused SiLU with a hand-written backward"
ENTRYPOINT = "silu_fwd_bwd"
METRIC = "bandwidth"
TOL = {"atol": 1e-5, "rtol": 1e-5}   # pure fp32 elementwise, no reduction

N = 1 << 20   # 1,048,576 elements, a flat fp32 vector


def make_inputs():
    torch.manual_seed(0)
    # *4.0 spreads values into the saturating tails of sigmoid, so the
    # derivative is exercised across its full range, not just near 0.
    x = torch.randn(N, device="cuda", dtype=torch.float32) * 4.0
    return (x,)


def reference(x):
    # forward: SiLU(x) = x * sigmoid(x), elementwise.
    forward_out = F.silu(x)

    # backward: torch's TRUE VJP of out.sum() w.r.t. x (grad_output all-ones).
    # Graded against autograd, not a closed form, so a transcribed derivative
    # that is subtly wrong fails here.
    x2 = x.detach().clone().requires_grad_(True)
    y = F.silu(x2)
    y.sum().backward()
    grad_input = x2.grad

    return (forward_out, grad_input)


def bytes_moved(x):
    # fused forward reads x + writes out (2*N); fused backward reads x + writes
    # grad_input (2*N). Total 4*N elements. An unfused autograd chain moves more.
    return 4 * x.numel() * x.element_size()
