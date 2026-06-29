"""e13: fused SiLU with a hand-written backward -- YOU write this file.

Forward:  out = SiLU(x) = x * sigmoid(x), elementwise over a flat vector.
Backward: given grad_output = dL/d(out), produce grad_input = dL/dx.

The lesson (lecture 2d): wrap two Triton kernels in a torch.autograd.Function so
PyTorch can call your hand-written backward inside .backward(). Save only x in
the forward and RECOMPUTE sigmoid(x) in the backward -- recompute, don't store.

Run:  python -m harness.runner e13 --watch
"""
import torch
import triton
import triton.language as tl


@triton.jit
def silu_fwd_kernel(
    # TODO: x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    # TODO: 1-D program id -> the chunk of the flat vector this program owns
    # TODO: build the offset vector for this chunk and a tail mask
    # TODO: load x (masked), compute SiLU = x * sigmoid(x)
    #       (sigmoid can come from tl.sigmoid, or build it from tl.exp)
    # TODO: store the result (masked)
    pass


@triton.jit
def silu_bwd_kernel(
    # TODO: x_ptr, grad_out_ptr, grad_in_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    # TODO: 1-D program id -> the chunk of the flat vector this program owns
    # TODO: offsets + tail mask
    # TODO: load x and grad_output (masked)
    # TODO: recompute sigmoid(x); form the SiLU derivative from it
    #       (derive it yourself from out = x*sigmoid(x); the formula symbols
    #        will NOT be your variable names -- map them)
    # TODO: grad_input = grad_output * (the derivative); store it (masked)
    pass


def _silu_forward(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    # TODO: n_elements; pick BLOCK_SIZE; grid = (cdiv(n_elements, BLOCK_SIZE),);
    #       launch silu_fwd_kernel.
    raise NotImplementedError("write the fused SiLU forward launch")
    return out


def _silu_backward(x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    grad_input = torch.empty_like(x)
    # TODO: n_elements; grid; launch silu_bwd_kernel with x, grad_output, grad_input.
    raise NotImplementedError("write the fused SiLU backward launch")
    return grad_input


class SiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # TODO: launch the forward kernel; ctx.save_for_backward(x); return out
        raise NotImplementedError("write SiLUFunction.forward")

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: (x,) = ctx.saved_tensors; launch the backward kernel;
        #       return ONE gradient (forward took one tensor input)
        raise NotImplementedError("write SiLUFunction.backward")


def silu_fwd_bwd(x: torch.Tensor):
    # TODO: attach requires_grad to a CLONE of x (never the runner's input).
    # TODO: y = SiLUFunction.apply(that clone); y.sum().backward().
    # TODO: return (forward output DETACHED, the clone's .grad) as a tuple
    #       so the runner can grade both the forward and the backward.
    # Do NOT mutate or attach grad to the original x.
    raise NotImplementedError("wire up the autograd Function and return (out, grad_input)")
