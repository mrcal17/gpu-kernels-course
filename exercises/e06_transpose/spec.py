"""Harness contract for e06: transpose. Do not edit."""
import torch

TITLE = "Transpose - 2-D tiling & coalescing"
ENTRYPOINT = "transpose"
METRIC = "bandwidth"          # read M*N, write M*N
TOL = {"atol": 0.0, "rtol": 0.0}

M, N = 4096, 4096


def make_inputs():
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    return (x,)


def reference(x):
    return x.t().contiguous()


def bytes_moved(x):
    return 2 * x.numel() * x.element_size()
