"""Harness contract for e12: quantized matmul. Do not edit."""
import torch

TITLE = "Quantized matmul - dequantize int8 weights in the inner loop"
ENTRYPOINT = "quant_matmul"
METRIC = "flops"
# Reference and entry BOTH dequantize the SAME b_q with the SAME scale, so the
# int8 quantization error cancels -- what remains is fp16 rounding (of a and of
# the fp16 output) plus fp32-accumulation-order differences across K. Outputs are
# O(100) from the K=1024 sum, so rtol (which scales with magnitude) carries the
# slack; a kernel that forgets the scale misses by thousands.
TOL = {"atol": 1e-1, "rtol": 1e-2}

M, K, N = 512, 1024, 768


def make_inputs():
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)

    # Build B in float, then symmetric (zero-point 0) per-output-channel int8
    # quantize. One scale per COLUMN of B (per output channel): amax reduces over
    # the K rows (dim=0) and keeps the N columns.
    b_f = torch.randn(K, N, device="cuda", dtype=torch.float32)
    scale = b_f.abs().amax(dim=0, keepdim=True) / 127.0          # (1, N) fp32
    b_q = torch.clamp(torch.round(b_f / scale), -127, 127).to(torch.int8)  # (K, N) int8

    # The learner never sees b_f -- they must reconstruct b_hat = b_q * scale
    # exactly as reference() does.
    return (a, b_q, scale)


def reference(a, b_q, scale):
    # Dequantize back to float, accumulate in fp32, cast result to fp16.
    return (a.float() @ (b_q.float() * scale)).to(torch.float16)


def flops(a, b_q, scale):
    M_, K_ = a.shape
    _, N_ = b_q.shape
    # Still 2*M*N*K -- the dequant multiply is lower-order (lecture 2c: quant
    # changes bytes moved, not the flop count).
    return 2 * M_ * N_ * K_
