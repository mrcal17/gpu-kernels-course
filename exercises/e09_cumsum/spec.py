"""Harness contract for e09: cumulative sum (scan). Do not edit."""
import torch

TITLE = "Cumsum - the scan pattern (advanced)"
ENTRYPOINT = "cumsum"
METRIC = "bandwidth"          # read M*N, write M*N
TOL = {"atol": 1e-2, "rtol": 1e-3}

M, N = 4096, 2048            # a row fits one block -> intra-block scan


def make_inputs():
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    return (x,)


def reference(x):
    return torch.cumsum(x, dim=1)


def bytes_moved(x):
    return 2 * x.numel() * x.element_size()
