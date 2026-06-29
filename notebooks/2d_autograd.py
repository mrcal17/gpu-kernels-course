import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2D: Autograd Integration

    > *"A kernel that can't do its own backward pass is a science project.
    > A kernel PyTorch can differentiate through is a layer."*

    You've written kernels that beat the framework on the forward pass. To put them
    in a real model you need one more thing: PyTorch has to be able to **call your
    kernel inside `loss.backward()`**. That means wiring it into autograd — teaching
    the framework what your op's gradient is, so it slots into the backward graph
    like any built-in.

    This lecture is about that wiring. Two mechanisms: the classic
    `torch.autograd.Function` (define `forward` and `backward` yourself) and the
    modern `torch.library` custom-op registration (which also plays nicely with
    `torch.compile`). Then the payoff question: **why write a fused backward at all,
    instead of letting autograd differentiate your forward automatically?** The
    answer is the same memory argument that drove every kernel in Part 2. We'll show
    the structure; the working fused backward is yours to write in the harness.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. What autograd needs from you

    PyTorch builds a **graph** of operations during the forward pass. Each node knows
    how to turn gradients of its *outputs* into gradients of its *inputs* — its
    vector-Jacobian product (VJP). `backward()` walks this graph in reverse, calling
    each node's VJP, chaining by the chain rule.

    When you call a built-in like `torch.matmul`, PyTorch already knows its VJP. When
    you call **your** Triton kernel, PyTorch sees an opaque tensor operation with no
    registered gradient — the graph has a hole, and `backward()` will either error or
    silently produce nothing. You have to supply the missing piece:

    - **forward:** run your kernel, return the output, and **save** whatever tensors
      the backward will need (via `ctx.save_for_backward`).
    - **backward:** given $\frac{\partial L}{\partial \text{output}}$ (the "upstream"
      or "grad_output"), return $\frac{\partial L}{\partial \text{input}}$ for **each**
      input, in the same order.

    For an op $y = f(x)$ with scalar loss $L$, the contract your backward implements
    is the VJP:

    $$\frac{\partial L}{\partial x}
      = \left(\frac{\partial y}{\partial x}\right)^{\!\top}
        \frac{\partial L}{\partial y}.$$

    You never form the Jacobian $\partial y/\partial x$ explicitly — you implement the
    *product* of it with the upstream gradient, which for most ops is itself a cheap
    kernel.

    > [PyTorch: Extending autograd](https://pytorch.org/docs/stable/notes/extending.html)
    > is the authoritative guide to `Function`, `ctx`, and the gradient contract.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        # A numpy stand-in for the autograd contract: forward + its VJP.
        # Op: y = x^2 elementwise, loss L = sum(y * upstream_weights).
        # Backward must return dL/dx = 2x * grad_output, matching x's shape.
        _rng = np.random.default_rng(0)
        x = _rng.normal(size=5).astype(np.float64)
        grad_output = _rng.normal(size=5).astype(np.float64)   # dL/dy from upstream

        # forward: save x for the backward
        y = x * x
        saved_x = x

        # backward: VJP of y = x^2  is  dL/dx = (dy/dx)^T dL/dy = 2x * grad_output
        grad_input = 2.0 * saved_x * grad_output

        # finite-difference check that grad_input is correct
        eps = 1e-6
        fd = np.empty_like(x)
        for _i in range(len(x)):
            _xp = x.copy(); _xp[_i] += eps
            _xm = x.copy(); _xm[_i] -= eps
            Lp = ((_xp * _xp) * grad_output).sum()
            Lm = ((_xm * _xm) * grad_output).sum()
            fd[_i] = (Lp - Lm) / (2 * eps)

        print("=== The VJP contract: forward saves x, backward returns dL/dx ===")
        print(f"  analytic grad_input (2x * grad_output): "
              f"{np.array2string(grad_input, precision=5)}")
        print(f"  finite-difference   grad_input        : "
              f"{np.array2string(fd, precision=5)}")
        print(f"  max abs error: {np.abs(grad_input - fd).max():.2e}")
        print("\n  This is exactly what backward() must return -- same shape as the input.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. `torch.autograd.Function` — the class skeleton

    The classic mechanism is a subclass of `torch.autograd.Function` with two
    `@staticmethod`s. `ctx` is a scratchpad that carries tensors from forward to
    backward. Here is the **skeleton** — the signatures and where your kernel calls
    go. The actual kernel launches and the fused backward math are left for you to
    fill in (that's the exercise).

    ```python
    import torch

    class MyFusedOp(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, weight):
            # 1. launch your Triton FORWARD kernel here -> out
            out = my_forward_kernel(x, weight)        # <-- your kernel
            # 2. stash whatever backward needs (inputs, intermediates, shapes)
            ctx.save_for_backward(x, weight)
            # (non-tensor data goes on ctx directly, e.g. ctx.scale = ...)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            # grad_output is dL/d(out), same shape as out
            x, weight = ctx.saved_tensors
            # 3. launch your Triton BACKWARD kernel(s) here
            grad_x, grad_w = my_backward_kernel(grad_output, x, weight)  # <-- yours
            # 4. return one gradient PER forward input, in input order.
            #    Return None for inputs that don't need a gradient.
            return grad_x, grad_w

    # call it like any differentiable op:
    y = MyFusedOp.apply(x, weight)     # NOT MyFusedOp() -- use .apply
    loss = y.sum()
    loss.backward()                    # autograd calls MyFusedOp.backward for you
    ```

    Three rules that trip everyone up the first time:

    1. **`backward` returns one gradient per `forward` input, positionally.** Two
       inputs → two return values. Use `None` for inputs that don't need grad (e.g. a
       `tl.constexpr`-style flag passed as an argument).
    2. **Only `save_for_backward` tensors** you truly need — saved tensors are kept
       alive for the whole backward, so over-saving is the autograd version of the
       memory waste this whole part fights.
    3. **Call via `.apply`, never the constructor.** `.apply` is what registers the
       node in the graph.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. `torch.library` — the modern custom op

    `autograd.Function` works, but it's invisible to `torch.compile`, which sees it as
    an opaque blob it can't reason about or fuse around. The modern path is to
    register your kernel as a first-class **custom operator** with `torch.library`
    (the `@torch.library.custom_op` decorator). You then register a separate
    "backward" via `register_autograd`, and a `register_fake` (a meta function that
    returns correctly-shaped empty tensors so the compiler can trace shapes without
    running the kernel).

    ```python
    import torch

    @torch.library.custom_op("mylib::fused_op", mutates_args=())
    def fused_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return my_forward_kernel(x, weight)           # <-- your kernel

    @fused_op.register_fake                            # shape/dtype only, no compute
    def _(x, weight):
        return torch.empty_like(x)

    def _backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        return my_backward_kernel(grad_output, x, weight)   # <-- yours

    def _setup_context(ctx, inputs, output):
        x, weight = inputs
        ctx.save_for_backward(x, weight)

    fused_op.register_autograd(_backward, setup_context=_setup_context)
    ```

    What you buy by doing it this way:

    - **`torch.compile` can see through it** — it knows the op's schema, shapes
      (via the fake/meta fn), and gradient, so it can fuse, reorder, and CUDA-graph
      around your kernel instead of treating it as a barrier.
    - **It behaves like a native op** under `vmap`, `torch.export`, serialization, and
      multiple dispatch backends.

    For a kernel you only ever call eagerly, `autograd.Function` is fine. For one
    headed into a compiled production model, `torch.library` is the right home.

    > [PyTorch Custom Operators tutorial](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
    > walks through `custom_op`, `register_fake`, and `register_autograd`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. The dataflow: forward saves, backward consumes

    The diagram traces a single op through both passes. Forward runs your kernel and
    *parks* the tensors backward will need on `ctx`. The reverse pass receives the
    upstream gradient `grad_output`, pulls the saved tensors back, and runs your
    backward kernel to produce a gradient for **each** input. The dashed arrow is the
    `ctx` handoff — the only channel between the two passes.
    """)
    return


@app.cell
def _():
    def _run():
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        _fig, _ax = plt.subplots(figsize=(9.5, 4.4))
        _ax.set_xlim(0, 12)
        _ax.set_ylim(0, 6)
        _ax.axis("off")
        _ax.set_title("Autograd dataflow: forward saves to ctx, backward consumes it")

        def _box(x, y, w, h, label, fc, ec):
            _ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=True,
                          facecolor=fc, edgecolor=ec, linewidth=1.8))
            _ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                     fontsize=9, color=ec, weight="bold")

        # --- forward row (top, left -> right) ---
        _ax.text(0.2, 5.5, "FORWARD", fontsize=10, color="#3060c0", weight="bold")
        _box(0.4, 4.2, 1.5, 0.9, "x, weight", "#eef3ff", "#5b8def")
        _box(2.6, 4.2, 2.2, 0.9, "forward kernel", "#dff0e4", "#4c9f70")
        _box(5.5, 4.2, 1.3, 0.9, "out", "#eef3ff", "#5b8def")
        _ax.annotate("", xy=(2.6, 4.65), xytext=(1.9, 4.65),
                     arrowprops=dict(arrowstyle="->", color="#888"))
        _ax.annotate("", xy=(5.5, 4.65), xytext=(4.8, 4.65),
                     arrowprops=dict(arrowstyle="->", color="#888"))

        # ctx in the middle
        _box(4.4, 2.5, 1.8, 0.9, "ctx\n(saved tensors)", "#f5f0ff", "#8a63d2")
        _ax.annotate("save_for_backward", xy=(5.3, 3.4), xytext=(5.3, 4.2),
                     ha="center", fontsize=7, color="#8a63d2",
                     arrowprops=dict(arrowstyle="->", color="#8a63d2",
                                     linestyle="--"))
        _ax.annotate("saved_tensors", xy=(5.3, 1.9), xytext=(5.3, 2.5),
                     ha="center", fontsize=7, color="#8a63d2",
                     arrowprops=dict(arrowstyle="->", color="#8a63d2",
                                     linestyle="--"))

        # --- backward row (bottom, right -> left) ---
        _ax.text(0.2, 0.3, "BACKWARD", fontsize=10, color="#a33", weight="bold")
        _box(5.3, 0.9, 1.6, 0.9, "grad_output", "#fde0e0", "#d65f5f")
        _box(2.5, 0.9, 2.3, 0.9, "backward kernel", "#fde0e0", "#d65f5f")
        _box(0.4, 0.9, 1.6, 0.9, "grad_x,\ngrad_weight", "#fde0e0", "#d65f5f")
        _ax.annotate("", xy=(4.8, 1.35), xytext=(5.3, 1.35),
                     arrowprops=dict(arrowstyle="->", color="#888"))
        _ax.annotate("", xy=(2.0, 1.35), xytext=(2.5, 1.35),
                     arrowprops=dict(arrowstyle="->", color="#888"))

        # note on the right
        _ax.text(7.2, 3.0,
                 "one grad returned\nper forward input\n(positional; None if\n"
                 "no grad needed)",
                 fontsize=8, color="#555",
                 bbox=dict(boxstyle="round", fc="#fafafa", ec="#ddd"))

        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Why a *fused* backward matters

    You could skip all this: write only the forward as small PyTorch ops and let
    autograd build the backward automatically. It would be correct. It would also be
    slow — for the same reason naive attention was slow (`2b`).

    Autograd's auto-generated backward differentiates **each small op separately**,
    and every intermediate it needs gets **materialized in HBM** on the forward pass
    and **re-read** on the backward. A fused op collapses that chain: it computes the
    whole forward in one kernel, saves only the *minimal* tensors, and computes the
    whole gradient in one (or two) kernels — recomputing cheap intermediates on the
    fly rather than storing them. Fewer kernel launches (`0b`), and far less HBM
    traffic (`0c`).

    Concretely, an unfused forward of $N$ chained elementwise/reduction ops writes
    ~$N$ intermediate tensors and launches ~$N$ kernels each way; the fused version
    launches **one** kernel each way and saves **one or two** tensors. The
    interactive below counts exactly that — toggle fusion and watch the op count and
    saved-tensor count collapse. This is *why* FlashAttention ships a hand-written
    backward instead of leaning on autograd, and why your `e13` exercise does too.

    > [Triton LayerNorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
    > implements a fused forward *and* fused backward inside one
    > `autograd.Function` — the exact pattern `e13` asks for.
    """)
    return


@app.cell
def _(mo):
    fuse_toggle = mo.ui.switch(value=True, label="fuse the op chain")
    n_ops_slider = mo.ui.slider(start=2, stop=12, step=1, value=6,
                                label="length of the op chain")
    mo.vstack([n_ops_slider, fuse_toggle])
    return fuse_toggle, n_ops_slider


@app.cell
def _(fuse_toggle, n_ops_slider):
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        n = int(n_ops_slider.value)
        fused = bool(fuse_toggle.value)

        if fused:
            # one fused forward kernel + one fused backward kernel;
            # save only the input(s) -- recompute the rest.
            fwd_launches = 1
            bwd_launches = 1
            saved = 1
            hbm_intermediates = 0
        else:
            # each op launches a fwd kernel and a bwd kernel; each writes an
            # intermediate that must be read back during backward.
            fwd_launches = n
            bwd_launches = n
            saved = n              # every intermediate kept for its backward
            hbm_intermediates = n

        cats = ["fwd kernel\nlaunches", "bwd kernel\nlaunches",
                "saved tensors", "HBM intermediates"]
        vals = [fwd_launches, bwd_launches, saved, hbm_intermediates]
        colors = ["#5b8def", "#8a63d2", "#e0a458", "#d65f5f"]

        _fig, _ax = plt.subplots(figsize=(8, 4.0))
        _bars = _ax.bar(cats, vals, color=colors, edgecolor="none")
        for _b, _v in zip(_bars, vals):
            _ax.text(_b.get_x() + _b.get_width() / 2, _v + 0.15, str(_v),
                     ha="center", fontsize=10, weight="bold")
        _ax.set_ylim(0, max(n, 1) + 1.5)
        _ax.set_ylabel("count")
        _state = "FUSED" if fused else "UNFUSED"
        _ax.set_title(
            f"{_state}: chain of {n} ops   ->   "
            f"{fwd_launches + bwd_launches} total launches, {saved} tensors saved")
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    - **A kernel isn't a layer until it has a backward.** Wire it into autograd —
      `autograd.Function` for eager use, `torch.library` for `torch.compile`-friendly
      production — and PyTorch differentiates through it like any built-in.
    - **Backward implements the VJP, positionally.** Return
      $\big(\partial y/\partial x\big)^{\!\top} \text{grad\_output}$ — one gradient per
      input, in input order, `None` where no grad is needed. Never form the Jacobian;
      implement the product.
    - **Save the minimum; recompute the rest.** Saved tensors live for the whole
      backward. The fused-kernel discipline — save one or two tensors, recompute cheap
      intermediates — is the same HBM-traffic argument from `2b`/`2c`, now on the
      backward pass.
    - **Fuse the backward, don't let autograd auto-derive it.** One forward kernel +
      one backward kernel beats $N$ small ops each way, in both launch overhead and
      memory traffic. This is the last piece that turns your Part-2 kernels into real,
      trainable layers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Now go write it

    Bring it all together: a kernel that lives inside a real training loop. Open the
    harness:

    ```bash
    python -m harness.runner e13 --watch
    ```

    `e13` asks you to wrap a fused op in a `torch.autograd.Function` with **both** a
    forward and a hand-written **fused backward** — the skeleton in §2 is your
    starting template, and the VJP contract in §1 is what your backward must satisfy.
    The harness checks your gradients against `torch.autograd.gradcheck`, so they have
    to be exactly right, not just close. The metric is bandwidth — the whole point is
    that your fused backward moves far less of it than the autograd-derived one would.

    That closes Part 2. From here, Part 3 re-derives every one of these patterns in
    CUDA C++, to the metal.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [2C: Quantization](../2c_quantization/) &nbsp;|&nbsp; Next: [3A: The CUDA C++ Execution Model](../3a_cuda_model/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
