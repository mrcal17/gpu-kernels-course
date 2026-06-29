"""e12: quantized matmul -- YOU write this file.

C = A @ dequant(B_q). A is fp16 (M, K). B arrives PRE-QUANTIZED as int8 (K, N)
with one fp32 scale per output channel (column of B), shape (1, N). The weights
were symmetrically quantized (zero-point 0), so dequant is a single multiply:
b_hat = b_q * scale_of_its_column.

This is your tiled matmul from e07/e10 with ONE new step in the inner loop: the
B tile loads as int8 and must be turned back into real numbers before tl.dot.
Read README.md (and lecture 2c) carefully.

Run:  python -m harness.runner e12 --watch
"""
import torch
import triton
import triton.language as tl


@triton.jit
def quant_matmul_kernel(
    # TODO: a_ptr (fp16), b_q_ptr (int8), scale_ptr (fp32), c_ptr (fp16);
    #       M, N, K;
    #       strides for a, b_q, c, and the stride of scale along its N axis;
    #       BLOCK_M, BLOCK_N, BLOCK_K: tl.constexpr
):
    # TODO: 2-D program ids -> which BLOCK_M x BLOCK_N tile of C this program owns
    # TODO: row/col offset vectors for this tile
    # TODO: an accumulator of shape (BLOCK_M, BLOCK_N), start at zero (fp32)
    # TODO: load this tile's per-output-channel scale vector ONCE, before the K
    #       loop -- it is indexed by column only and is constant across K (mask
    #       the N tail).
    # TODO: loop k from 0 to K in steps of BLOCK_K:
    #          load the (BLOCK_M x BLOCK_K) tile of A (fp16) and the
    #          (BLOCK_K x BLOCK_N) tile of B as int8 (mask the K tail, other=0);
    #          convert the int8 B tile to a float compute type and dequantize --
    #          multiply each column by its channel scale BEFORE the dot;
    #          accumulate a_tile @ b_hat_tile with tl.dot.
    # TODO: cast the accumulator to fp16 and write to C (mask the M/N edges)
    pass


def quant_matmul(a: torch.Tensor, b_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b_q.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # TODO: BLOCK_M/N/K; 2-D grid = (cdiv(M,BLOCK_M), cdiv(N,BLOCK_N));
    #       pass all strides INCLUDING the scale's column stride; launch.
    raise NotImplementedError("write the quantized tiled matmul kernel + launch")
    return c
