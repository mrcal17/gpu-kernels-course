# Site Review — Accuracy & Brevity

> Cross-model-style swarm judgment over all 26 lecture notebooks. **Method:** one
> reviewer per notebook (accuracy + brevity), then a high-effort **adversarial
> verifier** per notebook instructed to *refute* every finding (default: "the
> lecture is fine") — so false positives (e.g. "correcting" the real `sm_120`
> specs, nitpicking order-of-magnitude latencies, or calling beginner scaffolding
> "bloat") are filtered before they reach this report.

**Tally:** accuracy 17/21 standing · brevity 9/42 standing · 1/26 notebooks fully clean.

`CONFIRMED` = verifier agreed it's a real defect. `KEPT (unmatched)` = verifier
didn't return a matching verdict, kept for a human look. Refuted findings are
listed at the bottom for transparency.

---

## Confirmed / standing accuracy issues (17)

### `0d_occupancy_and_roofline.py` — MAJOR · CONFIRMED
- **Where:** Worked occupancy math code cell — print comment, line 155: "64 regs/thread halves blocks (8 -> 4); 50 KB smem caps at 6 by SRAM."
- **Claim:** 50 KB smem caps at 6 by SRAM.
- **Issue:** The same code computes by_smem = floor(100 KB / 50 KB) = 2 blocks for the (32, 50) row, so 50 KB of shared memory caps that configuration at 2 blocks (16 warps, 33% occupancy), not 6. The printed table directly above shows b_smem = 2 and blocks = 2 for that row, so the comment contradicts the code's own output. The '6' is the threads-limiter value for a different row, not the smem cap here.
- **Fix:** 50 KB smem caps at 2 blocks by SRAM (16 warps, 33% occupancy) — smem is the binding limiter for that row.
- **Verifier:** Verified against the code. In occ(), by_smem = int(SMEM_KB // smem_kb) = int(100.0 // 50) = 2 for the (32, 50) row, so blocks = min(6, 8, 2) = 2. The printed table for that row shows b_smem=2 and blocks=2. The line-155 comment claims '50 KB smem caps at 6', which contradicts the code's own output: it caps at 2, not 6. The '6' is the threads limiter (1536//256 = 6), not the shared-memory cap. This is a real, fixable defect — a beginner reading this would learn the wrong binding value for the smem-limited row. Confirmed.

### `1e_tiling_matmul.py` — MAJOR · CONFIRMED
- **Where:** Section 4 intro, "### The climax: slide the tile along the roofline"
- **Claim:** The fully general rectangular form ... is the harmonic-mean expression $I \approx \tfrac{1}{2}\,\tfrac{2}{1/T_m + 1/T_n}$ in elements — for square tiles $T_m = T_n = T$ it collapses to $T/4$ in FLOP/byte for fp32.
- **Issue:** The leading 1/2 factor is spurious and makes the formula internally inconsistent with its own claimed result. The correct reuse intensity per element loaded is I_elem = 2*Tm*Tn/(Tm+Tn) = 2/(1/Tm + 1/Tn) FLOP/element (this IS the harmonic mean of Tm and Tn, no extra 1/2). For square tiles that is T FLOP/element, and dividing by 4 bytes/fp32 gives T/4 FLOP/byte — matching section 4. As literally written, (1/2)*(2/(1/Tm+1/Tn)) evaluates to T/2 in elements for square tiles, which divided by 4 bytes is T/8 FLOP/byte, NOT the T/4 the sentence claims it collapses to. So the formula as printed does not produce the stated T/4.
- **Fix:** Drop the 1/2: the rectangular element-form intensity is $I \approx \tfrac{2}{1/T_m + 1/T_n}$ FLOP/element (the harmonic mean of $T_m,T_n$); for square tiles that is $T$ FLOP/element, and dividing by 4 bytes (fp32) gives $T/4$ FLOP/byte.
- **Verifier:** Verified by direct derivation. For a Tm×Tn output tile with K-window Tk, each K-step loads A (Tm·Tk) + B (Tk·Tn) = Tk(Tm+Tn) elements and does 2·Tm·Tn·Tk FLOPs, so intensity = 2·Tm·Tn/(Tm+Tn) = 2/(1/Tm+1/Tn) FLOP/element — the harmonic mean of Tm,Tn with NO extra 1/2. At Tm=Tn=T that is T FLOP/element, /4 bytes = T/4 FLOP/byte, matching section 4. The printed (1/2)·(2/(1/Tm+1/Tn)) = 1/(1/Tm+1/Tn) evaluates to T/2 FLOP/element for square tiles → T/8 FLOP/byte, which contradicts the sentence's own stated collapse to T/4. The leading 1/2 is spurious and makes the parenthetical internally inconsistent. A real math error; the live interactive uses the correct I=T/4, so it doesn't corrupt the main result, but a careful author would drop the 1/2.

### `2b_flash_attention.py` — MAJOR · KEPT (unmatched)
- **Where:** ## 5. The payoff: linear memory  (flash_bytes in the interactive code cell)
- **Claim:** def flash_bytes(N): qkv = 3*N*d*bytes_; out = N*d*bytes_; return qkv + out   # no N^2 term  — and the chart labels flash as 'flash (O(N): running stats)' on an axis titled 'HBM traffic per head (MB)'.
- **Issue:** This conflates flash's memory FOOTPRINT (genuinely O(N) resident state) with its HBM TRAFFIC (total bytes moved). In real FlashAttention each program re-reads K and V once per query block, so K,V HBM reads scale as O(N^2 * d / BLOCK_M) — still super-linear in N, just divided by the SRAM/block factor. The model counts K and V exactly once, making traffic strictly linear. The code's own comment ('K,V re-read per Q-block, but tiled and coalesced') acknowledges the re-read but the formula omits it, and it directly contradicts the correct §2 statement that traffic drops to O(N^2/M_sram). At N=32768, BLOCK_M=128 the modeled flash traffic is ~17 MB vs a realistic ~2.2 GB.
- **Fix:** Model flash K,V traffic with the re-read factor, e.g. qkv = N*d*b + (N/BLOCK_M)*(2*N*d*b); out = N*d*b, and label the curve O(N^2/M_sram) (still far below naive) rather than O(N). Alternatively retitle the axis/chart to 'resident state' / 'memory footprint' if the O(N) claim is meant to be about footprint, not traffic — but as written the axis says HBM traffic, so the asymptote is wrong.

### `3e_occupancy_tuning.py` — MAJOR · CONFIRMED
- **Where:** §1 Registers are the binding limiter — "R \le 42 registers/thread for 100% occupancy" and the closing code-cell note "42 regs/thd is the last rung at 100%."
- **Claim:** $R_{\max} = 65536/1536 \approx 42.6 \Rightarrow R \le 42$ registers/thread for 100% occupancy; and the code cell labels r≤42 as "full occupancy" / "42 regs/thd is the last rung at 100%."
- **Issue:** The 42 figure comes from the CONTINUOUS bound (65536/1536≈42.6), which assumes per-thread allocation with no rounding. But §1 and every code cell adopt the 256-regs/warp granularity model, and under THAT model 42 regs/thread rounds to 42×32=1344→1536 regs/warp → 65536/1536 = 42 warps = 88%, not 100%. The true last rung at 100% is 40 regs/thread (40×32=1280→1280 regs/warp → 51→48 warps = 100%). The code cell's own output prints "42 → 88%" in the occupancy column on the very row it annotates "full occupancy," so the prose and note contradict the number the code computes. A beginner is told 42=100% while the table shows 42=88%.
- **Fix:** Under the granularity model the lecture teaches, the last register count that still allows 48 warps (100%) is 40 regs/thread; 41–42 already drop to 42 warps (88%). State that the continuous bound gives 42.6 but warp-granularity rounding pulls the real 100% ceiling down to 40, and fix the code-cell note threshold (r≤40, not r≤42) so the 'full occupancy' label matches the computed occ column.
- **Verifier:** Verified against the notebook's own granularity model (256 regs/warp), which §1 prose and every code cell adopt. Running that model: r=40 -> 40*32=1280 regs/warp -> 65536/1280=51 warps capped to 48 = 100%; r=41 and r=42 -> 1536 regs/warp -> 65536/1536=42 warps = 88%. So the last register count still at 100% is 40, not 42. The contradiction is concrete and self-inflicted: the code cell's loop (lines 118-127) computes and prints occ = _w/WARPS_CAP, which prints 88% on the r=42 row, while the same loop annotates that row 'full occupancy' (line 121, _r<=42) and line 130 prints '42 regs/thd is the last rung at 100%.' A beginner reads 'full occupancy' next to a printed 88%. The 42.6 continuous bound is a true upper bound, but the lecture commits to the warp-granularity model and must round to 40. This is a genuine technical error a careful author would fix (threshold should be r<=40).

### `0a_orientation.py` — MINOR · CONFIRMED
- **Where:** Section: 'The latency you are hiding' — "the memory hierarchy spans **three orders of magnitude** in latency"
- **Claim:** the memory hierarchy spans three orders of magnitude in latency
- **Issue:** The chart this sentence introduces uses cycles = [1, 30, 200, 500]. Register (1) to DRAM (500) is a 500x span = ~2.7 orders of magnitude, not three (three orders = 1000x). The claim overstates the span shown in its own figure.
- **Fix:** Either say 'nearly three orders of magnitude' / 'over two orders of magnitude,' or bump the DRAM value to ~500-1000 cycles to genuinely cover 3 decades. Register->DRAM as drawn is ~500x.
- **Verifier:** The sentence explicitly introduces 'the bar chart below,' and that chart's own data is cycles=[1,30,200,500]. Register(1)->DRAM(500) is a 500x span = log10(500) ~= 2.7 decades, i.e. nearly three but not three orders of magnitude (three = 1000x). This is not a nitpick of the order-of-magnitude latencies themselves (those values are fine and match the ground-truth ~1/~30/~200/~500 guidance); it is an internal inconsistency between a definite prose claim and the numbers in the figure it points at. 'Over two' or 'nearly three' orders, or bumping DRAM to ~1000 cycles, would fix it. Minor but a real defect a careful author would correct.

### `0a_orientation.py` — MINOR · CONFIRMED
- **Where:** Chart title (latency cell): "Memory hierarchy: each level ~10x slower than the last"
- **Claim:** each level ~10x slower than the last
- **Issue:** The plotted data is [1, 30, 200, 500], whose step ratios are 30x, 6.7x, and 2.5x — none is ~10x, and they do not compound at ~10x/level. The title's per-level rule of thumb contradicts the numbers in the same figure.
- **Fix:** Use a title that matches the data, e.g. 'each level is dramatically slower than the last (log scale)' or 'register -> DRAM spans ~500x'. If '~10x per level' is the desired teaching point, pick values like [1, 10, 100, 500] so the figure actually shows it.
- **Verifier:** The chart title states a per-level rule of thumb (~10x each step), but the plotted bars are [1,30,200,500], giving step ratios of 30x, 6.7x, and 2.5x. None is ~10x, and they conspicuously do not compound at ~10x/level (the first jump is 30x, the last only 2.5x). The title therefore contradicts the data in its own figure. The geometric mean step is ~7.9x, so an honest title ('register->DRAM spans ~500x', or data like [1,10,100,500] to actually show ~10x/level) would resolve it. Again this is an internal prose-vs-figure inconsistency, not a disagreement with the ground-truth latencies. Minor but genuine.

### `1a_triton_model.py` — MINOR · CONFIRMED
- **Where:** ## 6. tl.constexpr and the launch grid
- **Claim:** Marking it `constexpr` lets Triton's compiler bake the value into the generated code — it can unroll loops, size register tiles, and pick the number of warps per program based on it.
- **Issue:** In Triton, the number of warps per program is set by the separate `num_warps` launch/meta parameter (default 4), not derived by the compiler from BLOCK_SIZE. Autotuning varies num_warps independently of BLOCK_SIZE. Saying the compiler picks warp count 'based on' BLOCK_SIZE overstates the coupling and could leave a beginner thinking num_warps is automatic.
- **Fix:** Keep the 'unroll loops / size register tiles' part (correct), but clarify that the number of warps per program is a separate knob (`num_warps`, default 4) that the programmer or autotuner sets; BLOCK_SIZE only determines how many elements those warps must cover, not the warp count itself.
- **Verifier:** This is a genuine, if minor, Triton error. The 'unroll loops / size register tiles' part is correct, but the number of warps per program is set by the separate `num_warps` launch/meta parameter (default 4), chosen by the programmer or autotuner — it is NOT derived by the compiler from BLOCK_SIZE. Autotuning varies num_warps independently of BLOCK_SIZE. Saying the compiler picks warp count 'based on' BLOCK_SIZE attributes an automatic coupling that does not exist and could leave a beginner thinking num_warps is set for them by block size. A careful author would clarify that num_warps is a separate knob. (Note: §7's slider line says 'typically more warps per program' — that one is hedged and describes a real practical correlation, so it is fine; the §6 claim is the firmer, incorrect one.)

### `1b_memory_coalescing.py` — MINOR · CONFIRMED
- **Where:** §4 code cell output: "the 8 ms row is what a stride-8 pattern costs."
- **Claim:** "The fastest row (~0.6 ms) rides the coalesced ceiling; the 8 ms row is what a stride-8 pattern costs."
- **Issue:** Internally inconsistent with the notebook's own transaction model. Stride-8 = 8 distinct segments = 8 transactions = 12.5% efficiency. Relative to the ~0.6 ms coalesced row (95% of peak), a stride-8 copy would take ~8x as long = ~4.8 ms and achieve ~107 GB/s (~12% of peak). The 8 ms row in the table is 64 GB/s = 7.1% of peak, which corresponds to roughly stride-13–14, i.e. a worse pattern than stride-8. The timing values in the table are all arithmetically correct; only this labeling annotation is wrong.
- **Fix:** Either change the annotation to reference a ~4.8 ms / ~107 GB/s row for stride-8, or relabel the 8 ms row as a worse-than-stride-8 (or simply "badly strided/scattered") pattern. Easiest fix: add a 4.8 ms row to the timing list and point the stride-8 remark at it.
- **Verifier:** Confirmed. The annotation contradicts the notebook's own transaction model. bytes_moved = 512 MB; at 8.00 ms the table yields 64 GB/s = 7.1% of 896 (the cell prints this). But the notebook's own stride model (§2 numpy cell and §3 plot) computes stride-8 = arange(32)*32 bytes spanning 8 distinct 128B segments = 8 transactions = 12.5% efficiency = ~112 GB/s, which corresponds to ~4.8 ms for the same 512 MB. The 8 ms / 64 GB/s row is ~7.1%, i.e. worse than stride-8 (roughly stride-13-14). So pointing the stride-8 remark at the 8 ms row is internally inconsistent with the very accounting the lecture teaches two sections earlier. A careful author would either add a ~4.8 ms row or relabel the 8 ms row as a worse-than-stride-8/scattered pattern. The timing arithmetic itself is correct; only the label is wrong. Genuine, fixable defect.

### `1b_memory_coalescing.py` — MINOR · CONFIRMED
- **Where:** §2 "natural sizes are 32, 64, and 128 bytes, and a 128-byte transaction is the common unit (it matches the L2 cache line ...)"
- **Claim:** Transactions come in 32/64/128-byte natural sizes and 128 bytes "matches the L2 cache line."
- **Issue:** This is the classic textbook coalescing model and is fine as a teaching abstraction, but on modern NVIDIA GPUs (Maxwell onward, including Blackwell) the L2/DRAM access granularity is a 32-byte sector; 128 bytes is the L1/global cache-LINE size, serviced as four 32-byte sectors. Calling 128 bytes "the L2 cache line" is imprecise. Not a falsehood that breaks the mental model (the per-warp transaction accounting still works), but the cache-line attribution is slightly off.
- **Fix:** 128 bytes is the L1/global cache line (= a full warp of float32); L2/DRAM is accessed in 32-byte sectors. Phrase 128B as the cache-line / coalescing unit rather than specifically "the L2 cache line," or note the 32-byte sectoring as the finer granularity.
- **Verifier:** Confirmed as a minor but real hardware mis-attribution. The 32/64/128-byte transaction sizes and the per-warp segment-counting model are the standard, acceptable teaching abstraction and are not the defect. The specific phrase "it matches the L2 cache line" is factually off for this card's architecture: on Maxwell-through-Blackwell NVIDIA GPUs the L1/global cache line is 128 bytes (= a full warp of float32, which the notebook also correctly states), while L2/DRAM is accessed at 32-byte sector granularity. So 128B is the L1/coalescing/cache-line unit, not specifically "the L2 cache line." This is a checkable attribution error a knowledgeable author would correct (e.g. drop "L2" or note the 32B sectoring), not merely stylistic. Severity is minor: it does not break the per-warp transaction accounting, but it states a wrong fact about where the 128B granularity lives.

### `2b_flash_attention.py` — MINOR · CONFIRMED
- **Where:** ## 1. Attention, and the matrix that kills it
- **Claim:** For N = 8192 that's 67 million entries — 256 MB in fp32 per head
- **Issue:** 8192^2 * 4 bytes = 268.4 MB in SI units (the same /1e6 convention the adjacent code cell uses). '256 MB' is the value in MiB (256.0 MiB), so the unit label is inconsistent with the notebook's own SI byte accounting; a beginner comparing to the code cell (which prints SI MB) sees a mismatch.
- **Fix:** Either '268 MB (fp32)' to match the SI convention used in the code cells, or '256 MiB' if binary units are intended. The entry count (67M) is correct.
- **Verifier:** Confirmed as a minor but genuine inconsistency. 8192^2 * 4 bytes = 268,435,456 bytes = 268.4 MB under the SI (/1e6) convention the adjacent code cell uses throughout (scores_mb = (N*N*2)/1e6). The figure '256' is exactly the binary value 256.0 MiB, so the prose reports the MiB number while labeling it 'MB' and while every code cell in the same notebook prints SI MB. A beginner cross-checking the prose against the code's convention would compute 268 MB and see a mismatch. The entry count (67M) is correct. Fix is trivial: '268 MB (fp32)' to match SI, or '256 MiB' if binary units are intended. Real defect a careful author would fix, severity minor.

### `3c_warp_primitives.py` — MINOR · CONFIRMED
- **Where:** §4, code comment: `int count = __popcount(votes);` and prose `__popcount (population count) then turns the bitmask into a tally`, plus the takeaways bullet `__ballot_sync + __popcount`
- **Claim:** __popcount(votes) is used as the CUDA device intrinsic to count set bits in the ballot result.
- **Issue:** CUDA's device population-count intrinsic is named __popc (and __popcll for 64-bit), not __popcount. __popcount is a GCC/glibc builtin (__builtin_popcount) that does not exist as a CUDA device function; a learner copying this into a kernel will hit a compile error. The notebook elsewhere uses exact, copy-pasteable intrinsic names (__shfl_down_sync, __ballot_sync, __activemask), so this is an inconsistency that teaches the wrong symbol.
- **Fix:** Use __popc(votes) (32-bit) — e.g. `int count = __popc(votes);` — and refer to the intrinsic as __popc throughout.
- **Verifier:** Verified in the notebook: `__popcount` appears in the code comment (line 249: `int count = __popcount(votes);`), the prose (line 252: '`__popcount` (population count) then turns the bitmask into a tally'), and the takeaways bullet (line 406: '`__ballot_sync` + `__popcount`'). CUDA's actual device population-count intrinsic is `__popc` (32-bit) / `__popcll` (64-bit). `__popcount` is not a CUDA device function — it is a glibc/GCC-style spelling (`__builtin_popcount`). A learner copying `int count = __popcount(votes);` into a .cu kernel hits a compile error (undefined identifier). The notebook is otherwise precise with real intrinsic names (`__shfl_down_sync`, `__shfl_xor_sync`, `__ballot_sync`, `__activemask`, `__all_sync`, `__any_sync`), so this is a genuine wrong-symbol error that teaches a non-existent intrinsic, not a stylistic quibble. Could not refute. Fix: use `__popc(votes)` and refer to `__popc` throughout.

### `3d_memory_banks.py` — MINOR · CONFIRMED
- **Where:** Section 2 'Shared memory's 32 banks' — sentence after the bank(a)=a mod 32 setup
- **Claim:** The 32 banks can serve 32 different banks simultaneously in one cycle — that's what makes shared memory fast.
- **Issue:** Garbled wording: banks cannot 'serve banks.' The intended meaning is that the 32 banks can each service one request, i.e. 32 distinct addresses (one per bank) in a single cycle. As written it is a self-referential non-sentence. It is recoverable because the very next bullet ('the 32 lanes hit 32 distinct banks') states the rule correctly.
- **Fix:** The 32 banks can each serve one request simultaneously — 32 distinct addresses (one per bank) in a single cycle — that's what makes shared memory fast.
- **Verifier:** Verified verbatim at lines 182-183. The sentence is genuinely garbled: 'banks can serve ... banks' is self-referential and does not state a true proposition. Banks serve requests/addresses, not other banks. The intended (correct) idea is that the 32 banks can each service one request, i.e. 32 distinct addresses (one per bank) per cycle. This is not a stylistic quibble — as literally written it is a non-statement in a beginner-grounding lecture where bank vocabulary is being defined for the first time. It is recoverable only because the immediately following bullet (line 189: 'the 32 lanes hit 32 distinct banks') states the rule correctly, but the defining sentence itself is still a real wording error a careful author would fix. Confirmed as a minor accuracy/clarity defect.

### `3e_occupancy_tuning.py` — MINOR · CONFIRMED
- **Where:** §5 Worked reading — Kernel B: "theoretical occupancy 33% (limiter: registers, 96/thread)"
- **Claim:** Kernel B has theoretical occupancy 33% with the register limiter binding at 96 registers/thread.
- **Issue:** Inconsistent with the notebook's own granularity model. 96 regs/thread → 96×32=3072 regs/warp → 65536/3072 = 21 warps → 21/48 = 44% occupancy, not 33%. 33% occupancy (16 warps) corresponds to ~124–128 regs/thread, not 96. The example pairs a register count and an occupancy that the model the lecture just taught does not connect.
- **Fix:** Either change the occupancy to 44% (to match 96 regs/thread) or change the register count to ~128/thread (to match 33% occupancy). The figures are illustrative, but as written the reg-count/occupancy pair contradicts the granularity model used throughout.
- **Verifier:** Under the granularity model the lecture teaches and uses in its §4 calculator, 96 regs/thread -> 96*32=3072 regs/warp -> 65536/3072 = 21 warps -> 21/48 = 44%, not 33%. 33% occupancy is 16 warps, which corresponds to 128 regs/thread (4096 regs/warp -> 16 warps). The reg-count/occupancy pair is internally inconsistent with the model just taught one section earlier. While the numbers are labeled illustrative, an illustrative example should not contradict the very model the lecture uses; pairing 96 regs with 33% mis-teaches the relationship. Minor severity (a single example, not the core derivation), but a real defect.

### `3e_occupancy_tuning.py` — MINOR · CONFIRMED
- **Where:** §2 The two levers — `__launch_bounds__(256, 6)` example: "forces the compiler to fit each thread in 65536 / 1536 ~= 42 registers ... = full occupancy"
- **Claim:** 6 blocks × 256 threads = 1536 threads = 100% occupancy forces each thread into ~42 registers.
- **Issue:** The block arithmetic (6×256=1536=48 warps) is correct, but the "~42 registers" budget for 100% inherits the §1 error: under the warp-granularity model the lecture uses, 42 regs/thread yields only 42 warps (88%), so the compiler must hit ≤40 regs/thread to actually achieve the requested 1536 resident threads. Saying ~42 registers gives full occupancy is the same off-by-a-rung mistake.
- **Fix:** State the per-thread budget needed for 6 resident blocks as ≤40 registers/thread (granularity-rounded), noting the continuous 65536/1536≈42.6 is an upper bound that rounding tightens to 40.
- **Verifier:** The block arithmetic is correct (6*256=1536 threads = 48 warps = full occupancy). The defect is the '~42 registers' budget, which inherits the §1 off-by-a-rung error: under the warp-granularity model the lecture uses, 42 regs/thread yields only 42 warps (88%, 1536 regs/warp), so to actually fit 6 resident blocks (1536 resident threads) the compiler must hit <=40 regs/thread. The text should state <=40 regs/thread (granularity-rounded), noting the 65536/1536≈42.6 continuous figure is an upper bound rounding tightens to 40. Same root cause as Finding 1, so confirmed for consistency.

### `3g_tensor_cores.py` — MINOR · CONFIRMED
- **Where:** §5, code block: "mma.sync ... (and on Hopper/Blackwell, the warpgroup-wide `wgmma` / `tcgen05` family)"
- **Claim:** On Hopper/Blackwell the instruction is "the warpgroup-wide `wgmma` / `tcgen05` family".
- **Issue:** The adjective "warpgroup-wide" is accurate for Hopper's `wgmma` but not for Blackwell's `tcgen05.mma`, which is a different issue model: it is launched by a single thread (or thread pair) and reads/writes operands in dedicated Tensor Memory (TMEM) rather than across a warpgroup's registers. Lumping `tcgen05` under "warpgroup-wide" slightly mis-describes the 5th-gen Blackwell path the learner's own card uses.
- **Fix:** Phrase it as e.g. "the warpgroup-wide `wgmma` (Hopper) and the Tensor-Memory-based `tcgen05` family (Blackwell)", or just drop the single adjective so it does not apply to both. The detail is forward-referenced to 4a, so a light touch is fine.
- **Verifier:** The single adjective "warpgroup-wide" is applied to both wgmma and tcgen05, but it is only accurate for Hopper's 4th-gen wgmma (128-thread warpgroup cooperating, operands distributed across warpgroup registers). Blackwell's 5th-gen tcgen05.mma has a different issue model: it is launched by a single thread (or thread pair), and operands reside in dedicated Tensor Memory (TMEM) rather than across a warpgroup's registers. Describing tcgen05 as "warpgroup-wide" mischaracterizes the actual Blackwell tensor-core path — which is precisely the architecture (sm_120) the learner's own 5070 Ti uses and that the notebook elsewhere flags as "new on your card." This is a real, fixable technical inaccuracy, not stylistic. Minor severity (forward-referenced to 4a), but confirmed: a careful author would split the adjective (e.g. "the warpgroup-wide wgmma (Hopper) and the Tensor-Memory-based tcgen05 family (Blackwell)") or drop it.

### `4a_blackwell.py` — MINOR · KEPT (unmatched)
- **Where:** ## 2. 5th-generation tensor cores — lineage bullets vs. the chart in the following code cell
- **Claim:** Prose bullet: "Hopper (4th gen): adds FP8 (E4M3 / E5M2)..." while the chart in the next cell marks FP8 as available on Ada: `fp8 = [np.nan, 2.6, 4.0, 5.2]` with column 1 = Ada (sm_89), and the comment says "FP8 lands at Ada/Hopper."
- **Issue:** Minor internal tension in attribution: the prose lineage credits FP8 to Hopper (4th gen) only, but the chart (correctly) shows FP8 tensor-core support already on Ada (sm_89). Both Ada and Hopper have FP8 tensor cores, so the chart is right and the prose bullet is slightly under-credited. Not a falsehood, just an inconsistency a careful reader will notice.
- **Fix:** Make the prose match the chart: FP8 (E4M3/E5M2) tensor-core support appears on the Ada/Hopper generation, not Hopper alone. e.g. "Ada/Hopper (4th gen): add FP8 (E4M3/E5M2); Hopper additionally adds the warpgroup wgmma instruction."

### `4a_blackwell.py` — MINOR · KEPT (unmatched)
- **Where:** Two throughput charts use different normalizations
- **Claim:** Chart 1 (§2 code cell) normalizes to Ampere-FP16 = 1 and shows Blackwell FP16=2.6, FP8=5.2, FP4=10.4; the interactive chart (§3) normalizes to FP16=1.0, FP8=2.0, FP4=4.0.
- **Issue:** Not a technical error (each chart is internally self-consistent and both are explicitly labeled "illustrative"), but the two different baselines for "relative throughput" sit close together and could momentarily confuse a beginner into thinking the numbers disagree.
- **Fix:** Optionally note in one of the captions that the two charts use different baselines (cross-generation vs. within-Blackwell), or align them. No factual correction needed.

---

## Confirmed / standing brevity issues (9)

### `1c_reductions.py` — MINOR · redundancy · CONFIRMED
- **Where:** ## 2. Tree reduction, paragraph ending 'one $\oplus$ in parallel with the others.'
- **Issue:** The closing sentence 'Each step is a halving of live values, every surviving lane doing one $\oplus$ in parallel with the others.' restates the point already made earlier in the same cell ('Each step **halves** the number of live values' plus the $n \to n/2 \to \dots \to 1$ chain). The halving idea is stated three times in one cell.
- **Suggestion:** Drop the trailing 'Each step is a halving of live values, ...' sentence; the halving is already established by the diagram and the earlier sentence.

### `1e_tiling_matmul.py` — MINOR · redundancy · KEPT (unmatched)
- **Where:** Section 4, last paragraph: "The interactive below lets you feel that dot slide along the roofline as you turn the tile-size knob." followed by section 4's "### The climax" subsection ("Now make the payoff interactive ... Drag T from tiny to large and watch the dot climb")
- **Issue:** The promise of an interactive tile-size knob sliding the dot along the roofline is stated three separate times in close succession: end of section 4 prose, then the climax intro, then again inside that same paragraph. The 'slide the dot / turn the knob' image is repeated nearly verbatim.
- **Suggestion:** Keep the vivid statement once (the climax subsection intro) and trim the trailing sentence of section 4 to a plain handoff, or vice versa.

### `1e_tiling_matmul.py` — MINOR · redundancy · KEPT (unmatched)
- **Where:** Closing summary section "## Why this matters for the kernels you'll write", first bullet vs. section 1 and the §3/§4 body
- **Issue:** The first summary bullet ("2-D decomposition is the pattern ... pid_m, pid_n, the offs_m[:, None] + offs_n[None, :] broadcast, masking both dimensions") restates the section-1 transpose lesson almost word-for-word. A recap bullet is reasonable scaffolding, but this one re-explains rather than just pointing back.
- **Suggestion:** Compress to a one-line pointer (e.g. 'the 2-D broadcast + both-dims mask from section 1 recurs in every matrix kernel') instead of re-listing the mechanics.

### `1g_scan.py` — MINOR · redundancy · CONFIRMED
- **Where:** Section 4 'Totals and the trade' bullets, plus '### Work vs depth, made tangible' intro, plus 'Why this matters' bullet 2
- **Issue:** The work-vs-depth trade-off and the 'HS wins small N / Blelloch wins large N, no free lunch' framing is stated in §2 ('opposite corners of that work-vs-depth trade'), restated in the §4 trade bullets ('opposite corners of the work/depth plane — exactly the trade §2 set up'), restated again in the slider intro ('There is no free lunch — only which axis you'd rather spend'), and a fourth time in the takeaways ('three points on that plane'). For an advanced/optional module the point lands by the second pass; the third and fourth restatements are the same idea in fresh adjectives.
- **Suggestion:** Keep the §2 setup and the §4 bullets (which carry the concrete ~2N / ~2logN numbers). Trim the slider intro's editorializing ('Watch the contrast... There is no free lunch') to a one-line caption, since the plot already shows the contrast.

### `1g_scan.py` — MINOR · redundancy · CONFIRMED
- **Where:** Section 1: 'The two are one shift apart. Exclusive is inclusive shifted right by one...' followed by the table read-off 'Read off the relationship: the exclusive row is the inclusive row slid one cell to the right...'
- **Issue:** The inclusive/exclusive shift relationship is stated in prose, then boxed as a formula, then the worked table is added, then a sentence re-narrates the exact same shift-by-one-and-prepend-zero relationship that was just stated two sentences above the table. The table itself already demonstrates it.
- **Suggestion:** Drop or shorten the post-table 'Read off the relationship...' sentence — the table plus the earlier prose+box already establish both directions (shift, and +x_i recovers inclusive).

### `3e_occupancy_tuning.py` — MINOR · redundancy · KEPT (unmatched)
- **Where:** §3 — "Register-heavy, low-occupancy, and fast, all at once."
- **Issue:** This trailing sentence restates the immediately preceding sentence ("Those kernels want the extra registers — capping them to chase occupancy would slow them down") and the paragraph's thesis. The point about Volkov kernels being register-heavy yet fast has already been made twice in the same paragraph.
- **Suggestion:** Drop the standalone clause; the prior two sentences already deliver it.

### `4a_blackwell.py` — MINOR · redundancy · KEPT (unmatched)
- **Where:** The recurring "don't assume sm_120 has it — check the CC docs for CUDA 13.1" thesis
- **Issue:** This single point is restated in nearly identical wording in at least eight places: the intro ("check the arch docs for your toolkit"), §1 caution 2 ("query it, don't recall it"), §2 caution ("Treat the bars below as illustrative"), §3 ("version- and CC-dependent — confirm against the CUDA 13 docs"), §4 caution ("toolchain-version-dependent. Check the Triton release notes"), §5 (twice: "not something to assume" and "treat clusters/DSMEM as 'check the arch docs'"), all of §6 (the four-step query workflow), and "Why this matters" bullet 3 ("query, read the CC table for CUDA 13.1, confirm, then design"). One framing statement plus §6's concrete workflow would carry the whole load; the per-section repetitions are the same sentence reworded.
- **Suggestion:** Keep the intro framing and the §6 workflow as the canonical statements. Trim the per-section cautions to a one-clause pointer (e.g. "(sm_120 support: see §6)") instead of re-deriving "it's version-dependent, don't assume, check the docs" each time.

### `4a_blackwell.py` — MINOR · redundancy · KEPT (unmatched)
- **Where:** ## 5 — final paragraph "So the actionable rule..."
- **Issue:** This paragraph ("query the device, read the per-CC feature table, and confirm — don't port an H100 cluster kernel to your 5070 Ti on faith") restates the immediately preceding paragraph's point ("is not something to assume... gated by compute capability and CUDA version") and pre-states §6's entire workflow, which follows directly after.
- **Suggestion:** Drop the standalone "actionable rule" paragraph in §5 and let §6 (which is the actionable workflow) carry it; or collapse the two §5 closing paragraphs into one sentence.

### `4a_blackwell.py` — MINOR · redundancy · KEPT (unmatched)
- **Where:** Closing sections "Why this matters for the kernels you'll write" and "Where this is used"
- **Issue:** Both closing sections re-cover the same four themes (narrow floats as the throughput lever, TMA, clusters/DSMEM caveat, measure-don't-trust) and both point forward to the 4b capstone. "Why this matters" bullet 1 and "Where this is used" first bullet make the same FP8/FP4-menu-for-4b point.
- **Suggestion:** These two sections could be merged into one short "What carries into 4b" wrap-up; the per-theme recap in "Why this matters" overlaps the section bodies it summarizes.

---

## Refuted by the verifier (4 accuracy, 33 brevity)

Kept here only for the record — the adversarial pass judged these *not* real defects.

**Accuracy:**
- `0b_execution_model.py` — warp time ≈ time(then) + time(else) when the warp diverges — _The equation (lines 374-375) is presented in the context of a single data-dependent if/then/else — the prose immediately above (line 369-372) describes exactly a two-way branch, and the divergence cod_
- `0b_execution_model.py` — a first batch fills every SM to capacity, and as those blocks finish, the next batch moves in — _The 'waves' framing (lines 308-312) is standard CUDA pedagogy (NVIDIA's own occupancy material uses the 'wave' abstraction). The reviewer explicitly admits this is 'a pedagogical simplification rather_
- `1a_triton_model.py` — ...let the compiler narrate the 1024 lanes. — _This is an epigraph — deliberately flavorful framing — not a body definition, and it does not teach a falsehood. With BLOCK_SIZE=1024 there are 1024 element-slots, each handled by one SIMT lane-positi_
- `2a_autotuning.py` — gbytes = (a.numel() + b.numel() + M * N) * a.element_size() — _The byte count is the standard minimal-traffic estimate (A read once + B read once + C written once) and the bytes/s ÷ 1e9 → GB/s conversion is dimensionally correct. For a plain GEMM with no beta-acc_

**Brevity:**
- `0b_execution_model.py` (redundancy) — ### The words you'll need (keep this open) — final line 'Don't memorize the table — just k — _The two statements live in different cells separated by the entire glossary table. The intro cell (lines 28-29) frames the lecture's overall pedagogical approac_
- `0b_execution_model.py` (redundancy) — ## 1. Throughput, not latency — 'so pin them down first (they're in the glossary too)' — _Per the audience guidance, a single re-statement of a definition in running prose is acceptable beginner scaffolding and should NOT be flagged as bloat — and th_
- `0c_memory_hierarchy.py` (redundancy) — Section 1 staircase, Registers bullet ('They are a *budget*... caps how many warps can be  — _The finding's own framing is muddled. The section-1 Registers bullet is NOT a 'reuse on-chip' statement — it makes a different point (registers are a budget tha_
- `0c_memory_hierarchy.py` (redundancy) — Section 3 prose ('most simple kernels finish their math long before the data arrives and s — _The three passages serve three distinct roles, not one repeated point. The intro line ('arithmetic got cheap, moving data did not') states the course thesis as _
- `0d_occupancy_and_roofline.py` (redundancy) — Sections "Why this matters for the kernels you'll write" ("Diagnose before you optimize... — _The two sections serve distinct functions and do not deliver the same conclusion twice. 'Why this matters' is a conceptual recap (four principles: diagnose-firs_
- `1a_triton_model.py` (redundancy) — ## 4 ("the exact analogue of CUDA's `if (i < n)` guard") and "The same kernel, side by sid — _The finding claims the mask = CUDA `if (i < n)` equivalence is stated 'three or four' times, but the explicit CUDA-guard analogy actually appears only twice: in_
- `1a_triton_model.py` (redundancy) — ## 7 visualization prose: "This is the whole §3–§6 story in one image." followed by "### S — _The two visualizations illustrate distinct ideas, not the identical point. The static §7 figure shows the per-lane anatomy at one fixed n/BLOCK_SIZE (which spec_
- `1b_memory_coalescing.py` (redundancy) — §4 "never time a single launch (launch overhead and the clock's resolution dominate)" and  — _Rejected. The closing section is explicitly a recap ("## Why this matters for the kernels you'll write"), and restating the key operational rule in a summary is_
- `1b_memory_coalescing.py` (redundancy) — §3 misaligned bullet "~50–90% depending on BLOCK_SIZE" vs §2 misaligned "~50% on that warp — _Rejected. This is not a redundancy/padding/too-terse defect — there is no repeated sentence, restated heading, or filler. The two figures describe the same effe_
- `1c_reductions.py` (redundancy) — ## 4. Per-block partial reductions — 'For the **row reduce** of `e04` this is the easy cas — _The §4 paragraph (lines 338-340) is the substantive first statement with its justification ('each row is independent'). The summary bullet (lines 437-438) lives_
- `1c_reductions.py` (redundancy) — ## 5. Numerical considerations (c) and '## Why this matters' bullet 4 — _The three mentions occupy distinct pedagogical registers: §5(a) line 363 explains WHY (float non-associativity → tolerance), the code-cell print line 417 is the_
- `1d_softmax.py` (redundancy) — Bottom nav cell: "Next: [1E: Tiling & Matmul](../1e_tiling_matmul/)" — _The footer nav is strictly sequential (Prev: 1C -> this 1D -> Next: 1E), matching the course's linear ordering. The 2b references are a deliberately-labeled 'Te_
- `1d_softmax.py` (redundancy) — §3 demo cell (Case A) and the §2 demo cell both run on `_x = [1.0, 2.0, 3.0, 95.0]` — _Two parts of the finding don't hold. First, the claim that the static visualization reuses the same row is inaccurate: the viz cell (line 343) uses a different _
- `1f_fused_norms.py` (redundancy) — Section 'Why this matters for the kernels you'll write', bullet 'Reduce → map is the same  — _The reviewer's count of 'at least three times' is inflated. The intro (lines 28-33) states the reduce-then-map *shape* as setup ('The shape is the reduce-then-m_
- `1f_fused_norms.py` (redundancy) — §3 'Why fuse', cross-reference 'reduce → map is the same shape as softmax (`1c`/`1d`)' vs  — _The summary bullet reads 'the same shape as softmax (1c/1d)' where '(1c/1d)' is a parenthetical cluster pointing to the two prior lectures that built the reduce_
- `2a_autotuning.py` (redundancy) — §4 item 2 and the §4 visualization cell, shared-memory formula — _The three appearances are not redundant prose. One is the conceptual model stated once in §4 prose; the other two are live executable code inside two distinct i_
- `2b_flash_attention.py` (redundancy) — ### Why the rescaling is *exact*, in one line / online-softmax code cell output / ## Why t — _Refuted. The finding itself concedes the claim/proof/empirical-check/summary split is 'mostly justified for a crux idea' and only objects to the wording overlap_
- `2c_quantization.py` (redundancy) — ## Why this matters for the kernels you'll write — _This is a single, short, clearly-signposted end-of-lecture recap of a five-section lecture. The judging guidance explicitly treats one concise closing recap as _
- `2c_quantization.py` (redundancy) — Section 4, item 4: 'Re-quantize the output only if the next layer wants low precision.' — _The reviewer explicitly states 'Not padding within section 4 itself' and 'Acceptable as-is.' The only concern is mild conceptual overlap between a section-4 pip_
- `2d_autograd.py` (redundancy) — ## 5. Why a *fused* backward matters — "Autograd's auto-generated backward differentiates  — _The two prose paragraphs do not in fact overlap heavily: paragraph 1 gives the QUALITATIVE mechanism (intermediates materialized in HBM and re-read; fused colla_
- `2d_autograd.py` (redundancy) — Final summary section "## Why this matters for the kernels you'll write" — bullets "Save t — _This is an explicit, clearly-labeled closing recap of a multi-section lecture. Restating section takeaways is the defining function of a summary, not redundancy_
- `3a_cuda_model.py` (redundancy) — §6, 'CUDA error %s at %s:%d -> %s' uses cudaGetErrorName for the first %s and cudaGetError — _Not redundant. cudaGetErrorName returns the enum SYMBOL (e.g. "cudaErrorIllegalAddress") which the reader greps for in headers/docs, while cudaGetErrorString re_
- `3b_shared_tiling.py` (redundancy) — Section 4, 'Why tiling raises operational intensity' — sentences 'This is the whole reason — _The two flagged sentences serve distinct functions and are not literal restatements of each other. 'This is the whole reason shared memory exists: to convert re_
- `3b_shared_tiling.py` (redundancy) — Final 'Why this matters' bullet list, e.g. 'Tiling is a roofline move' and 'Tile size is a — _The reviewer itself concedes the closing recap is 'justified scaffolding' for a thorough beginner lecture. The claim that it 're-derives rather than points back_
- `3e_occupancy_tuning.py` (redundancy) — §3 closing + "Why this matters" bullet 3 + §6 exercise step 3 — _The three locations serve distinct structural roles: an in-context conclusion to the Volkov section, an end-of-lecture takeaway bullet (whose entire purpose is _
- `3f_async_pipelining.py` (redundancy) — ## Why this matters for the kernels you'll write (four bullets) — _This is the lecture's single closing recap, which the rubric explicitly protects for a beginner-grounding lecture ('a short recap is justified'). A summary by d_
- `3f_async_pipelining.py` (redundancy) — ## 6 closing paragraph vs. the 'You now understand Triton's num_stages' bullet — _The §6 sentence is body prose concluding the Triton walkthrough ('Now you know what the compiler is doing on your behalf, and why the optimum is shape-dependent_
- `3g_tensor_cores.py` (redundancy) — §3 body ("The rule of thumb: halving the input bit-width roughly doubles tensor-core throu — _Section 5 develops the "the MMA is one instruction, keeping it fed is the engineering" thesis in depth; the recap bullet compresses it into a single sentence at_
- `3g_tensor_cores.py` (redundancy) — §Why-this-matters bullet "Feeding them is the hard part" vs §5 ("feed the tensor cores fas — _The three instances are not interchangeable repetition. §3 establishes the rule conceptually where it is first earned. The §4-viz intro instance is captioning w_
- `4b_capstone.py` (redundancy) — ## Why this matters — "The two non-negotiables, restated because they're where capstones g — _The three 'correctness gates speed' touches serve distinct functions and none repeat verbatim: §2 is an operational milestone rule, M1's 'Correct is not optiona_
- `4b_capstone.py` (padding) — ### Now go write it — "When it passes, you've written a real GPU kernel, benchmarked it li — _The finding overcounts: the epigraph is about benchmarking against a strong baseline, not the full-loop thesis. The remaining instances serve different rhetoric_
- `7a_study_guide.py` (redundancy) — ## 3. Cheat-sheets — intro cell and §§3.1–3.4 (full) — _The §§3.1–3.4 'full' versions are NOT verbatim copies of the dropdown cheat-sheets — they are deliberately condensed. The dropdown renders full markdown tables _
- `7a_study_guide.py` (redundancy) — ## Why this matters and ### Where this is used (final two prose cells) — _The two cells serve genuinely distinct functions and follow the course's standard 'Why this matters / Where this is used' section pattern. 'Why this matters' is_

---

## Fully clean notebooks (1)

`home.py`
