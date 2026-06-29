"""Harness contract for e11: flash attention forward. Do not edit."""
import math

import torch
import torch.nn.functional as F

TITLE = "Flash attention forward - tile the KV blocks, never write the scores"
ENTRYPOINT = "flash_attention"
METRIC = "flops"
# fp16 attention with an fp32-accumulated online softmax. A correct manual
# fp32-softmax reference matches SDPA to ~5e-4 max abs error -- comfortably
# inside atol=1e-2, while still catching a real bug (wrong scale, missing
# rescale). Looser than e05's fp32 softmax (1e-4) because the data is fp16.
TOL = {"atol": 1e-2, "rtol": 1e-2}

B, H, N, D = 2, 4, 512, 64


def make_inputs():
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return (q, k, v)


def reference(q, k, v):
    # SDPA's default scale is exactly 1/sqrt(D); softmax is done in fp32
    # internally and the result is returned in fp16. Non-causal for v1.
    return F.scaled_dot_product_attention(q, k, v, is_causal=False)


def flops(q, k, v):
    B_, H_, N_, D_ = q.shape
    # Two matmuls dominate: scores QK^T is 2*B*H*N*N*D, output PV is
    # 2*B*H*N*N*D. The softmax ex/sum is lower-order and ignored (matches the
    # lecture 2b count).
    return 4 * B_ * H_ * N_ * N_ * D_
