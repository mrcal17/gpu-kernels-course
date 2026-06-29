"""Harness contract for e03: copy / bandwidth. Do not edit."""
import torch

TITLE = "Copy - find your bandwidth ceiling"
ENTRYPOINT = "copy"
METRIC = "bandwidth"          # 1 read + 1 write; the purest bandwidth test
TOL = {"atol": 0.0, "rtol": 0.0}

N = 1 << 25                   # bigger array -> steadier bandwidth measurement


def make_inputs():
    torch.manual_seed(0)
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    return (x,)


def reference(x):
    return x.clone()


def bytes_moved(x):
    return 2 * x.numel() * x.element_size()
