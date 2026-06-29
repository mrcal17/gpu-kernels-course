"""Harness contract for e10: autotuned matmul over ragged shapes. Do not edit."""
import torch

# Make the reference TRUE fp32 (no TF32) so correctness is judged fairly against
# an fp32-accumulating kernel.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

TITLE = "Autotuned matmul - mask any shape, let the tuner pick the tile"
ENTRYPOINT = "matmul"
METRIC = "flops"
TOL = {"atol": 1e-1, "rtol": 1e-2}   # fp32 accumulation order differs from torch

# Deliberately non-power-of-2 and not multiples of 16/32/64/128, so the M, N,
# AND K dimensions are all ragged and every tile needs masking.
M, K, N = 1023, 1025, 769


def make_inputs():
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    return (a, b)


def reference(a, b):
    return a @ b


def flops(a, b):
    M_, K_ = a.shape
    _, N_ = b.shape
    return 2 * M_ * N_ * K_       # one multiply + one add per inner-product term
