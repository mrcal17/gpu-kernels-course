"""Harness contract for e02: fused elementwise (SiLU). Do not edit."""
import torch

TITLE = "Fused elementwise - SiLU / swish"
ENTRYPOINT = "silu"
METRIC = "bandwidth"          # 1 read + 1 write, memory-bound
TOL = {"atol": 1e-3, "rtol": 1e-3}

N = 1 << 24


def make_inputs():
    torch.manual_seed(0)
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    return (x,)


def reference(x):
    # SiLU: x * sigmoid(x). The point: several ops fused, still ONE read + ONE write.
    return x * torch.sigmoid(x)


def bytes_moved(x):
    return 2 * x.numel() * x.element_size()
