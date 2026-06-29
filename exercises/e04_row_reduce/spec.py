"""Harness contract for e04: row reduction. Do not edit."""
import torch

TITLE = "Row reduction - sum each row"
ENTRYPOINT = "row_sum"
METRIC = "bandwidth"          # read M*N, write M -> dominated by the read
TOL = {"atol": 1e-2, "rtol": 1e-3}   # fp32 reduction order differs from torch

M, N = 4096, 4096


def make_inputs():
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    return (x,)


def reference(x):
    return x.sum(dim=1)


def bytes_moved(x):
    # read every element once, write one result per row
    return (x.numel() + x.shape[0]) * x.element_size()
