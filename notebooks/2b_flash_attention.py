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
    # 2B: Flash Attention

    > *"The scores matrix is a ghost. FlashAttention's whole trick is to compute
    > attention as if that matrix existed — without ever letting it touch memory."*

    This is the marquee lecture of Part 2. Attention is the workhorse of modern
    models, and the naive implementation has a fatal flaw: it materializes an
    $N \times N$ scores matrix in slow HBM, so its memory cost grows as $O(N^2)$.
    Double the sequence length and you quadruple the memory traffic — the wall every
    long-context model hits.

    **FlashAttention** removes that matrix entirely. It tiles the computation over
    blocks of keys/values and maintains a handful of **running statistics** — a
    running max, a running sum, and a running output — updated block by block with an
    **online softmax**. The full scores matrix is never written down; only $O(N)$
    state lives in fast SRAM. The crux of the whole method is the *rescaling rule*
    that lets you fold a new block into the running result correctly, and we will
    derive it carefully. The kernel itself you'll write in the harness.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. Attention, and the matrix that kills it

    Given queries $Q\in\mathbb{R}^{N\times d}$, keys $K\in\mathbb{R}^{N\times d}$,
    and values $V\in\mathbb{R}^{N\times d}$, attention is:

    $$S = \frac{QK^\top}{\sqrt d}\in\mathbb{R}^{N\times N},\qquad
      P = \mathrm{softmax}(S)\ \text{(row-wise)},\qquad
      O = P\,V\in\mathbb{R}^{N\times d}.$$

    The naive recipe computes these as three passes, and the problem is the middle
    object: $S$ (and $P$) are **$N \times N$**. For $N = 8192$ that's 67 million
    entries — **256 MB in fp32 per head**, written to HBM after the matmul and read
    back for the softmax, then written again, then read for $PV$. The math is cheap;
    the memory traffic for that ghost matrix is what dominates.

    $$\text{naive HBM traffic} \;=\; O(N^2)\ \text{(scores)} \;+\; O(Nd)\ \text{(IO)}.$$

    Worse, $O(N^2)$ memory means you simply **run out of HBM** at long context —
    before you run out of compute. Attention is not compute-bound; it is
    **memory-bound on a matrix that shouldn't exist**. That reframing is the entire
    opportunity.

    > [Vaswani et al., "Attention Is All You Need" (2017)](https://arxiv.org/abs/1706.03762)
    > defines the operation; the cost analysis here motivates FlashAttention.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        # Make the N^2 problem concrete in bytes.
        print("=== HBM footprint of the N x N scores matrix (fp16, 1 head) ===\n")
        print(f"  {'seq len N':>10s} {'scores entries':>16s} {'scores MB':>12s} "
              f"{'O(N) state MB':>14s}")
        print("  " + "-" * 56)
        d = 64
        for N in [512, 1024, 2048, 4096, 8192, 16384]:
            scores_mb = (N * N * 2) / 1e6          # N*N fp16
            # flash keeps only running stats + the O(N x d) output, not N x N
            state_mb = (N * d * 2 + 2 * N * 4) / 1e6
            print(f"  {N:>10d} {N * N:>16,d} {scores_mb:>12.1f} {state_mb:>14.2f}")
        print("\n  The scores column grows quadratically; the state column linearly.")
        print("  At N=16384 the ghost matrix alone is ~537 MB/head -- flash needs ~2 MB.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. The flash idea: tile, and never materialize

    FlashAttention keeps $Q$, $K$, $V$ in HBM but processes attention in **tiles
    that fit in SRAM**. The structure: a program owns one block of queries
    $Q_i$ (say `BLOCK_M` rows). It loops over blocks of keys/values $K_j, V_j$ (each
    `BLOCK_N` rows), and for each KV block it:

    1. computes the small score tile $S_{ij} = Q_i K_j^\top / \sqrt d$
       — only `BLOCK_M x BLOCK_N`, which lives entirely in SRAM;
    2. folds $S_{ij}$ into a **running output** for $Q_i$ using the online-softmax
       update (§3);
    3. throws $S_{ij}$ away.

    Because each query block carries forward only a few statistics — a per-row
    running max $m$, a per-row running denominator $\ell$, and the partial output
    accumulator $O$ — the full $N\times N$ matrix is **never assembled**. HBM traffic
    drops from $O(N^2)$ to $O(N^2/M_{\text{sram}})$ in the worst case and is
    bandwidth-friendly because $Q$, $K$, $V$ are each read in a streaming, coalesced
    pass.

    The catch — and the reason this isn't trivial — is the softmax. Softmax needs the
    max and the sum **over the entire row**, but we're only ever looking at one
    `BLOCK_N`-wide slice at a time. You cannot normalize until you've seen every key.
    The online-softmax rescaling is what resolves this.

    > [Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and Memory-Efficient Exact
    > Attention with IO-Awareness" (2022)](https://arxiv.org/abs/2205.14135), and
    > [Dao, "FlashAttention-2" (2023)](https://arxiv.org/abs/2307.08691) for the
    > improved work partitioning.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Online softmax — the crux

    Take one query row. Its (unscaled) scores against all keys are $s_1,\dots,s_N$.
    The numerically stable softmax-weighted output is

    $$o = \sum_{k} \frac{e^{\,s_k - m}}{\ell}\, v_k,\qquad
      m = \max_k s_k,\qquad \ell = \sum_k e^{\,s_k - m},$$

    where subtracting the row max $m$ keeps $e^{(\cdot)}$ from overflowing (the same
    stabilization you built in `1d`). The trouble: $m$ and $\ell$ depend on **all**
    keys, but we process keys one block at a time. We need to maintain $m$, $\ell$,
    and $o$ **incrementally**, fixing them up as new blocks arrive.

    **Setup.** After processing some keys we hold the running triple
    $(m, \ell, o)$ — the max, denominator, and (denominator-scaled) output computed
    *so far*. A new block arrives with its own local max $\tilde m$, and per-key
    exponentials $\tilde p_k = e^{\,s_k - \tilde m}$, local sum
    $\tilde\ell = \sum \tilde p_k$, and local weighted value
    $\tilde o = \sum \tilde p_k\, v_k$.

    **The fix-up.** The new global max is $m^{\text{new}} = \max(m, \tilde m)$.
    Everything previously exponentiated relative to the *old* max $m$ must be
    re-based to $m^{\text{new}}$. Multiplying $e^{s-m}$ by the correction factor
    $\alpha = e^{\,m - m^{\text{new}}}$ turns it into $e^{\,s - m^{\text{new}}}$.
    The new block, scored relative to $\tilde m$, gets the analogous factor
    $\beta = e^{\,\tilde m - m^{\text{new}}}$. So the running statistics update as:

    $$\boxed{\;
      m^{\text{new}} = \max(m, \tilde m),\quad
      \alpha = e^{\,m - m^{\text{new}}},\quad
      \beta = e^{\,\tilde m - m^{\text{new}}}
      \;}$$
    $$\boxed{\;
      \ell^{\text{new}} = \alpha\,\ell + \beta\,\tilde\ell,\qquad
      o^{\text{new}} = \alpha\,o + \beta\,\tilde o
      \;}$$

    Each step **rescales the entire accumulated state** by $\alpha$ (cheap: one
    multiply per row) before adding the new block's contribution. The division by the
    final $\ell$ is deferred to the very end — only once, after the last block:
    $o \leftarrow o / \ell$. No full row, no full matrix; just three running numbers
    per query row, corrected block by block.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why the rescaling is *exact*, in one line

    The accumulator stores $o = \sum_{k\le\text{seen}} e^{\,s_k - m}\,v_k$ relative to
    the current running max $m$. When $m$ jumps to $m^{\text{new}}$, every term should
    have been $e^{\,s_k - m^{\text{new}}}$; multiplying the whole sum by
    $\alpha = e^{\,m - m^{\text{new}}}$ converts every stored term at once because
    $e^{\,s_k - m}\cdot e^{\,m - m^{\text{new}}} = e^{\,s_k - m^{\text{new}}}$.
    Exponentials factor through the sum, so one scalar per row repairs the entire
    history. That is the whole trick — and it's why FlashAttention is **exact**, not
    an approximation. The simulation below checks it against a textbook softmax.
    """)
    return


@app.cell
def _():
    def _run():
        import numpy as np

        # One query row vs N keys, processed in KV blocks via the online update.
        # We verify the running (m, l, o) triple reproduces the exact softmax output.
        _rng = np.random.default_rng(7)
        N, d = 20, 4
        BLOCK_N = 6                       # process keys 6 at a time (ragged tail)
        s = _rng.normal(size=N) * 1.5     # per-key scores for this row
        V = _rng.normal(size=(N, d))      # values

        # ----- reference: stable softmax, all at once -----
        m_ref = s.max()
        p_ref = np.exp(s - m_ref)
        o_ref = (p_ref[:, None] * V).sum(0) / p_ref.sum()

        # ----- online: running triple updated per block -----
        m = -np.inf                       # running max
        l = 0.0                           # running denominator
        o = np.zeros(d)                   # running (unnormalized) output

        for j0 in range(0, N, BLOCK_N):
            sj = s[j0:j0 + BLOCK_N]
            Vj = V[j0:j0 + BLOCK_N]
            m_tilde = sj.max()                       # local max of this block
            p_tilde = np.exp(sj - m_tilde)           # local exps
            l_tilde = p_tilde.sum()                  # local denominator
            o_tilde = (p_tilde[:, None] * Vj).sum(0) # local weighted values

            m_new = max(m, m_tilde)
            alpha = np.exp(m - m_new) if np.isfinite(m) else 0.0  # rescale old state
            beta = np.exp(m_tilde - m_new)                        # scale new block

            l = alpha * l + beta * l_tilde
            o = alpha * o + beta * o_tilde
            m = m_new

        o_online = o / l                  # deferred normalization, exactly once

        print("=== Online softmax vs exact softmax (1 row, 20 keys, blocks of 6) ===")
        print(f"  exact   output: {np.array2string(o_ref, precision=5)}")
        print(f"  online  output: {np.array2string(o_online, precision=5)}")
        print(f"  max abs error : {np.abs(o_ref - o_online).max():.2e}")
        print("\n  ~1e-16: the online update is EXACT, not an approximation.")

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. The block loop, in pictures

    The diagram below is the FlashAttention forward pass for one query block $Q_i$.
    Read it left to right: $Q_i$ is loaded once into SRAM, then the kernel sweeps the
    KV blocks $K_j, V_j$, and at each step the running triple $(m, \ell, O)$ is
    rescaled and updated by the §3 rule. The $N\times N$ scores matrix (faint, in the
    background) is the thing that is *never* written to HBM — only the small green
    score tile $S_{ij}$ ever exists, and only inside SRAM.
    """)
    return


@app.cell
def _():
    def _run():
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        _fig, _ax = plt.subplots(figsize=(9.5, 4.6))
        _ax.set_xlim(0, 12)
        _ax.set_ylim(0, 6)
        _ax.axis("off")
        _ax.set_title("FlashAttention forward: one query block sweeps the KV blocks")

        # The ghost N x N scores matrix in the background (never materialized).
        for _r in range(4):
            for _c in range(4):
                _ax.add_patch(mpatches.Rectangle(
                    (0.4 + _c * 0.9, 4.2 - _r * 0.42), 0.85, 0.38,
                    fill=True, facecolor="#f2f2f2", edgecolor="#dcdcdc"))
        _ax.text(0.4, 4.75, "ghost  S = QK^T  (N x N, never in HBM)",
                 fontsize=8, color="#aaaaaa")

        # Q_i block (loaded once)
        _ax.add_patch(mpatches.Rectangle((0.4, 1.0), 0.9, 2.4,
                      fill=True, facecolor="#eef3ff", edgecolor="#5b8def", linewidth=2))
        _ax.text(0.85, 0.7, "Q_i", color="#3060c0", fontsize=10, weight="bold",
                 ha="center")

        # KV blocks streamed
        for _j in range(3):
            _x = 2.1 + _j * 1.25
            _ax.add_patch(mpatches.Rectangle((_x, 2.6), 0.95, 0.8,
                          fill=True, facecolor="#fff3e0", edgecolor="#e0a458"))
            _ax.text(_x + 0.47, 3.0, f"K_{_j}", color="#a96f1c", fontsize=8, ha="center")
            _ax.add_patch(mpatches.Rectangle((_x, 1.6), 0.95, 0.8,
                          fill=True, facecolor="#fde0e0", edgecolor="#d65f5f"))
            _ax.text(_x + 0.47, 2.0, f"V_{_j}", color="#a33", fontsize=8, ha="center")
            # score tile (transient, in SRAM only)
            _ax.add_patch(mpatches.Rectangle((_x, 0.5), 0.95, 0.7,
                          fill=True, facecolor="#dff0e4", edgecolor="#4c9f70"))
            _ax.text(_x + 0.47, 0.85, f"S_i{_j}", color="#2e6b48", fontsize=7,
                     ha="center")
            # arrow into the running-stats box
            _ax.annotate("", xy=(7.0, 2.5), xytext=(_x + 0.95, 2.5),
                         arrowprops=dict(arrowstyle="->", color="#bbb", lw=0.8))

        _ax.text(4.4, 4.0, "for each KV block:\n  S_ij = Q_i K_j^T / sqrt(d)\n"
                 "  rescale (m, l, O) by alpha, beta\n  O += beta * (P_ij V_j)",
                 fontsize=8, color="#555",
                 bbox=dict(boxstyle="round", fc="#fafafa", ec="#ddd"))

        # running statistics box
        _ax.add_patch(mpatches.Rectangle((7.4, 1.2), 4.2, 2.6,
                      fill=True, facecolor="#f5f0ff", edgecolor="#8a63d2", linewidth=2))
        _ax.text(9.5, 3.5, "running stats (SRAM)", color="#5b3aa0", fontsize=9,
                 weight="bold", ha="center")
        _ax.text(7.6, 2.95, "m  : per-row running max", fontsize=8, color="#444")
        _ax.text(7.6, 2.55, "l   : per-row running sum", fontsize=8, color="#444")
        _ax.text(7.6, 2.15, "O : running weighted output", fontsize=8, color="#444")
        _ax.text(7.6, 1.6, "at the end:  O <- O / l", fontsize=8, color="#2e6b48",
                 weight="bold")

        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The block-loop structure (pseudocode)

    Here is the *shape* of the kernel — enough to see the control flow, but the
    actual loads, masks, and the `tl.dot` are yours to write in the exercise. Note
    the single deferred division at the end.

    ```python
    # one program handles a BLOCK_M-row block of queries: Q_i in SRAM
    m_i = full([BLOCK_M], -inf)          # running max,  per query row
    l_i = zeros([BLOCK_M])               # running denominator
    acc = zeros([BLOCK_M, d])            # running (unnormalized) output

    for j in range(0, N, BLOCK_N):       # stream KV blocks
        k_j, v_j = load K[j], V[j]       # BLOCK_N rows each
        s_ij  = dot(q_i, k_j.T) * scale  # BLOCK_M x BLOCK_N score tile (SRAM)
        # --- online-softmax update (the crux you derived in section 3) ---
        m_new = max(m_i, rowmax(s_ij))
        p_ij  = exp(s_ij - m_new[:, None])
        alpha = exp(m_i - m_new)         # rescale the running state
        l_i   = alpha * l_i + rowsum(p_ij)
        acc   = alpha[:, None] * acc + dot(p_ij, v_j)
        m_i   = m_new

    acc = acc / l_i[:, None]             # normalize ONCE, at the end
    store O[i] = acc
    ```

    The shapes, the masks for ragged $N$, the causal mask (if any), and getting
    `tl.dot` / the running rescale right in Triton are the parts the harness leaves
    to you. The math above is the entire specification.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. The payoff: linear memory

    The interactive below contrasts HBM traffic for naive attention (which writes and
    re-reads the $N\times N$ scores) against FlashAttention (which streams $Q,K,V$ and
    keeps only $O(N)$ state) as you grow the sequence length. The naive curve is the
    quadratic that ends every long-context dream; the flash curve is near-linear.
    Slide $N$ up and watch the gap explode — at long context it's orders of magnitude.
    """)
    return


@app.cell
def _(mo):
    seqlen_slider = mo.ui.slider(start=512, stop=32768, step=512, value=4096,
                                 label="sequence length N")
    seqlen_slider
    return (seqlen_slider,)


@app.cell
def _(seqlen_slider):
    def _run():
        import numpy as np
        import matplotlib.pyplot as plt

        d = 64                # head dim
        bytes_ = 2            # fp16
        N_user = int(seqlen_slider.value)

        Ns = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768])

        # Naive: read Q,K,V (Nd each); write S (N^2); read S for softmax; write P;
        # read P and V for PV; write O. Dominated by the N^2 terms.
        def naive_bytes(N):
            qkv = 3 * N * d * bytes_
            scores = 2 * (N * N * bytes_)        # write + read of S/P (the killer)
            out = N * d * bytes_
            return qkv + scores + out

        # Flash: read Q,K,V once (K,V re-read per Q-block, but tiled and coalesced);
        # write O once; O(N) running state stays in SRAM, not HBM.
        def flash_bytes(N):
            qkv = 3 * N * d * bytes_
            out = N * d * bytes_
            return qkv + out                     # no N^2 term

        naive = np.array([naive_bytes(int(n)) for n in Ns]) / 1e6   # MB
        flash = np.array([flash_bytes(int(n)) for n in Ns]) / 1e6

        nb = naive_bytes(N_user) / 1e6
        fb = flash_bytes(N_user) / 1e6
        ratio = nb / fb

        _fig, _ax = plt.subplots(figsize=(8, 4.2))
        _ax.plot(Ns, naive, "-o", color="#d65f5f", linewidth=2,
                 label="naive  (O(N^2): materializes S)")
        _ax.plot(Ns, flash, "-o", color="#4c9f70", linewidth=2,
                 label="flash  (O(N): running stats)")
        _ax.axvline(N_user, color="#999", linestyle="--", linewidth=1)
        _ax.scatter([N_user], [nb], color="#d65f5f", s=80, zorder=5)
        _ax.scatter([N_user], [fb], color="#4c9f70", s=80, zorder=5)
        _ax.set_xscale("log")
        _ax.set_yscale("log")
        _ax.set_xlabel("sequence length N  (log)")
        _ax.set_ylabel("HBM traffic per head (MB, log)")
        _ax.set_title(f"N={N_user:,}:  naive {nb:,.0f} MB  vs  flash {fb:,.1f} MB"
                      f"   ->  {ratio:,.0f}x less traffic")
        _ax.legend(loc="upper left", fontsize=8)
        _ax.grid(True, which="both", alpha=0.15)
        _fig.tight_layout()
        return _fig

    _run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why this matters for the kernels you'll write

    - **Fuse the whole pipeline; never write the scores.** The win isn't a faster
      matmul — it's *deleting* the $N\times N$ HBM round-trip. Keep $S_{ij}$ in SRAM,
      use it, discard it. This is the §1 lesson of fusion at its most dramatic.
    - **Online softmax is the enabling primitive.** A running max + running sum +
      running output, rescaled by $\alpha,\beta$ per block, computes an *exact*
      softmax you never had the full row for. The same pattern generalizes to any
      "reduce over a dimension you're streaming" problem.
    - **Defer the normalization.** Divide by $\ell$ exactly once, at the end. Dividing
      per block would be wrong (the denominator isn't final) and wasteful.
    - **It's memory-bound-friendly by design.** $Q,K,V$ stream in coalesced passes and
      the working set fits SRAM, so you sit near the bandwidth roof (`0d`) instead of
      drowning in $O(N^2)$ traffic. This is *the* kernel that made long context
      practical.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Now go write it

    This is the big one. Open the harness and implement the FlashAttention forward
    pass:

    ```bash
    python -m harness.runner e11 --watch
    ```

    `e11` gives you $Q, K, V$ and asks for $O$ — your kernel must tile over KV blocks
    and carry the running $(m, \ell, O)$ triple with the online-softmax update you
    derived in §3, never materializing the scores matrix. The pseudocode in §4 is the
    skeleton; the loads, masks, `tl.dot`, and the per-block rescale are yours. The
    metric is FLOP/s, but the real prize is that the memory stays $O(N)$ — check it
    against a naive reference and watch it hold at long sequence lengths where naive
    runs out of room.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    [&#8593; Course home](../) &nbsp;|&nbsp; &#8592; Prev: [2A: Autotuning](../2a_autotuning/) &nbsp;|&nbsp; Next: [2C: Quantization](../2c_quantization/) &#8594;
    """)
    return


if __name__ == "__main__":
    app.run()
