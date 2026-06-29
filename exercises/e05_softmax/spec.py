"""Harness contract for e05: softmax. Do not edit."""
import torch

TITLE = "Softmax - numerically stable, row-wise"
ENTRYPOINT = "softmax"
METRIC = "bandwidth"          # read M*N, write M*N
TOL = {"atol": 1e-4, "rtol": 1e-4}

M, N = 4096, 2048


def make_inputs():
    torch.manual_seed(0)
    # spread the scale so the naive (no max-subtraction) version would overflow
    x = torch.randn(M, N, device="cuda", dtype=torch.float32) * 10.0
    return (x,)


def reference(x):
    return torch.softmax(x, dim=1)


def bytes_moved(x):
    return 2 * x.numel() * x.element_size()
