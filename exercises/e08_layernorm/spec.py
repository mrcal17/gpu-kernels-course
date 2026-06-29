"""Harness contract for e08: layernorm. Do not edit."""
import torch
import torch.nn.functional as F

TITLE = "LayerNorm - fused reduce + normalize + affine"
ENTRYPOINT = "layernorm"
METRIC = "bandwidth"
TOL = {"atol": 1e-2, "rtol": 1e-3}

M, N = 4096, 2048
EPS = 1e-5


def make_inputs():
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    weight = torch.randn(N, device="cuda", dtype=torch.float32)
    bias = torch.randn(N, device="cuda", dtype=torch.float32)
    return (x, weight, bias)


def reference(x, weight, bias):
    return F.layer_norm(x, (x.shape[1],), weight, bias, eps=EPS)


def bytes_moved(x, weight, bias):
    # dominated by reading x and writing out; weight/bias are tiny
    return 2 * x.numel() * x.element_size()
