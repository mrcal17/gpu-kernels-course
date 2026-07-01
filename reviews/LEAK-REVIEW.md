# Exercise Leak Review — does the scaffolding spoil the exercise?

> Agent-swarm audit of all 19 exercise dirs (13 Triton + 6 CUDA). **Lens:** do the
> stub TODOs + README hints hand over the answer (exact API calls, variable names,
> dims/block sizes, literal code) instead of conceptual nudges? **Method:** one
> reviewer per exercise + a high-effort adversarial verifier that refuses to flag
> the harness contract (entrypoint names, spec.py, given wrapper plumbing).

**Raw tally:** 99 raised · 78 confirmed · 21 refuted.

> **NOTE — e01 quarantined:** `e01_vector_add/kernel.py` currently holds the
> learner's completed, passing solution (it was solved in-session), so its
> findings are the learner's own code, not stub scaffolding. Excluded from the
> actionable list below; restore the blank stub to re-audit it fairly.

**Actionable confirmed leaks (excluding e01): 71 across 17 exercises.**

---

## `c01_vector_add` — 6 leaks (3 critical, 2 major, 1 minor)

### CRITICAL — kernel.cu:14-15 — TODO: "compute this thread's global index / i = blockIdx.x*blockDim.x + threadIdx.x"
- **Leaked:** i = blockIdx.x*blockDim.x + threadIdx.x
- **Why:** This is the single most important idiom the exercise exists to teach (the README itself calls it 'the global-index idiom' and hint #2 says 'Re-derive the idiom; don't look it up'). The TODO writes the exact expression, the exact variable name `i`, and the exact multiply/add ordering of all three built-ins, so the learner copies it instead of deriving it.
- **Conceptual rewrite:** Compute the flat global position this thread is responsible for by combining its block index, the block size, and its position within the block.

### CRITICAL — kernel.cu:16 — TODO: "guard the ragged tail with if (i < n) before touching memory"
- **Leaked:** guard the ragged tail with if (i < n) before touching memory
- **Why:** Hint #3 deliberately makes deciding the bound condition the work ('decide what condition keeps a thread in bounds'). The TODO writes the exact guard expression `if (i < n)`, so the learner no longer derives the comparison or which side it goes on.
- **Conceptual rewrite:** Before this thread reads or writes anything, guard against the ragged tail: only proceed when this thread's index actually falls inside the array.

### CRITICAL — kernel.cu:17 — TODO: "inside the guard, out[i] = a[i] + b[i]"
- **Leaked:** out[i] = a[i] + b[i]
- **Why:** Writes the entire kernel body line verbatim, including indexing all three pointers with `i`. Nothing of the kernel's compute is left to derive.
- **Conceptual rewrite:** Inside the guard, do the elementwise work: write the sum of the two inputs at this thread's position into the output at the same position.

### MAJOR — kernel.cu:21-22 — TODO: "choose blockDim (a multiple of 32) and / gridDim = ceil(n / blockDim)"
- **Leaked:** gridDim = ceil(n / blockDim)
- **Why:** States the exact launch-geometry computation (grid = ceil-division of n by block size) with both variable names filled in. The learner only has to transcribe the ceil-div rather than reason out how many blocks cover n elements. (The 'multiple of 32' nudge is acceptable conceptual guidance; the verbatim grid formula is the leak.)
- **Conceptual rewrite:** Pick a threads-per-block count, then size the grid so that every element is covered even when the block size does not evenly divide n (i.e. round the block count up).

### MAJOR — kernel.cu:23 — TODO: "launch the kernel with <<<grid, block>>>(d_a, d_b, d_out, n)"
- **Leaked:** <<<grid, block>>>(d_a, d_b, d_out, n)
- **Why:** Writes the exact launch-syntax call with the execution-configuration brackets, both config variables, and the full filled-in argument list. Configuring and firing the launch is half of what the README says the learner must write ('solve(), which configures the launch and fires the kernel'); this hands over the syntax and the arguments.
- **Conceptual rewrite:** Launch the kernel with your chosen grid and block configuration, passing the three device pointers and the length.

### MINOR — README.md:38-40 — Hint #2: "combine blockIdx.x, blockDim.x, and threadIdx.x into a single global index"
- **Leaked:** combine `blockIdx.x`, `blockDim.x`, and `threadIdx.x` into a single global index
- **Why:** Hint #2 explicitly tells the learner to 'Re-derive the idiom; don't look it up,' yet it names the complete set of the three exact built-ins to combine. With all three operands handed over, the only remaining work is the arithmetic ordering — which kernel.cu:15 then also gives away. Naming the existence of block/lane indexing would be a fair nudge; enumerating the exact three identifiers is a leak.
- **Conceptual rewrite:** Turn this thread's (block, lane) coordinates into one flat position by combining the index of its block, the number of threads per block, and its index within the block. Re-derive the arithmetic; don't look it up.

---

## `c02_tiled_matmul` — 3 leaks (1 critical, 2 major)

### CRITICAL — kernel.cu:31 (TODO in matmul_tiled K-loop) — "inner loop over TILE: acc += As[ty][k] * Bs[k][tx]."
- **Leaked:** inner loop over TILE: acc += As[ty][k] * Bs[k][tx].
- **Why:** This is the literal kernel body of the inner-product step, written out with the accumulator, the loop variable, and — critically — the exact index placement (As[ty][k] vs Bs[k][tx]). The whole point of the shared-tile exercise is deriving which on-chip index each operand reads, including the row/col asymmetry between the A-tile and B-tile. The learner copy-pastes instead of working out that As is indexed [thread-row][k] while Bs is indexed [k][thread-col].
- **Conceptual rewrite:** Inner loop: walk k across the staged tile and accumulate the product of the A-tile element on this thread's tile-row and the B-tile element on this thread's tile-column. Work out which of (tile-row, tile-col, k) indexes each shared array yourself.

### MAJOR — kernel.cu:40-41 (TODO in solve) — "launch matmul_tiled with a 2-D block dim3(TILE, TILE) and a 2-D grid that covers an N x N output (round the grid up ...)."
- **Leaked:** launch matmul_tiled with a 2-D block dim3(TILE, TILE) and a 2-D grid that covers an N x N output (round the grid up so the ragged tail is still covered)
- **Why:** States the exact block dimension (dim3(TILE, TILE)) the learner should derive from 'one thread per output element of a TILE x TILE tile'. Block shape is a config decision the exercise asks them to make; handing dim3(TILE, TILE) removes that step. (The round-up-grid instruction is fine conceptually; the explicit block dim is the leak.)
- **Conceptual rewrite:** Choose a 2-D block whose thread count matches one output element per thread of a single tile, and a 2-D grid sized so every output tile (including the ragged tail) is covered. Derive both extents yourself.

### MAJOR — kernel.cu:26-32 (TODO block describing the K loop)
- **Leaked:** loop over K in steps of TILE. Each step: - cooperatively load ONE element of the A-tile and ONE of the B-tile ... - __syncthreads() AFTER the loads, BEFORE the inner-product loop. - inner loop over TILE: ... - __syncthreads() AFTER compute, BEFORE the next iteration overwrites the tiles.
- **Why:** Taken as a whole this is a filled-in step-by-step recipe (load one element each, sync, inner loop, sync) that fixes the exact ordering and barrier placement. README hints 2-3 already cover stage/sync/reuse and the two-barrier reasoning conceptually and ask the learner to 'say out loud which is which'; restating the precise placement here as an imperative checklist removes the derivation those hints set up.
- **Conceptual rewrite:** Loop over K one tile-width at a time. Each step you'll cooperatively stage the next A and B tiles, place the barriers you reasoned about in the hints, and accumulate from on-chip data. Decide the order and where each barrier goes yourself.

---

## `c03_warp_reduce` — 1 leaks (1 major)

### MAJOR — kernel.cu:27 (TODO: reduce within the warp using __shfl_down_sync with delta = 16,8,4,2,1)
- **Leaked:** TODO: reduce within the warp using __shfl_down_sync with delta = 16,8,4,2,1
//       and the FULL_MASK = 0xffffffff participation mask.
- **Why:** Hands over (a) the exact warp-shuffle intrinsic to call, (b) the complete 5-step delta sequence 16,8,4,2,1, and (c) the participation mask. The warp fold IS the core lesson of this exercise (it is c03's named 'first warp primitive'). After reading this the learner writes the fold by transcription, deriving nothing. This directly contradicts README hint 2, which deliberately says 'Work out the delta sequence yourself' — the stub leaks precisely what the README withholds.
- **Conceptual rewrite:** TODO: fold the 32 lanes of a warp into lane 0 using the down-shuffle intrinsic from lecture 3c. It is a log2(32)-step tree; at each step a lane adds in the value from a fixed offset above it. Derive the offset for each step (hint 2 in the README) and keep the full participation mask so every lane stays converged.

---

## `c04_transpose` — 6 leaks (4 major, 2 minor)

### MAJOR — kernel.cu:19-20 (TODO: declare a __shared__ tile sized TILE x (TILE+1) — the +1 is the bank-conflict fix)
- **Leaked:** declare a __shared__ tile sized TILE x (TILE+1) — the +1 is the bank-conflict fix (column neighbors land in consecutive banks).
- **Why:** States the exact padded tile shape (TILE x (TILE+1)) directly in the stub. The entire point of the exercise (per README hint 3, 'work out on paper *why* a width of TILE+1 shears the bank mapping') is deriving the +1 pad. The TODO writes the dimension and the rationale, so the learner copies the shape instead of discovering that padding the inner dim by 1 breaks the column-onto-one-bank collision.
- **Conceptual rewrite:** TODO: declare a __shared__ tile big enough to stage one block's worth of data. A column read of an unpadded tile whose width is a multiple of 32 collides on one bank — adjust the inner dimension so column neighbors fall in different banks.

### MAJOR — kernel.cu:26 (TODO: store the loaded value into the shared tile at [threadIdx.y][threadIdx.x])
- **Leaked:** store the loaded value into the shared tile at [threadIdx.y][threadIdx.x], guarded by the edge check
- **Why:** Writes the exact shared-memory write index expression. Choosing which thread coordinate maps to the row vs column of the staged tile is part of the work; here it is filled in verbatim, so paired with the swapped-read TODO the learner no longer has to reason about the [y][x] write / [x][y] read asymmetry that makes the transpose work.
- **Conceptual rewrite:** TODO: store the loaded value into the staged tile, indexing it by this thread's position within the block, guarded by the edge check (rows/cols are not always multiples of the tile size).

### MAJOR — kernel.cu:37 (TODO: read the shared tile with the indices SWAPPED ([threadIdx.x][threadIdx.y]))
- **Leaked:** read the shared tile with the indices SWAPPED ([threadIdx.x][threadIdx.y]) and write to `out`, guarded by the edge check
- **Why:** Writes the exact transposed shared-memory read index expression verbatim. The swap of the two thread coordinates when reading the tile back IS the transpose trick at the heart of the exercise. Handing over [threadIdx.x][threadIdx.y] means the learner copies the core line instead of deriving that the column read of shared memory is what produces the transpose.
- **Conceptual rewrite:** TODO: read the staged tile back transposed — i.e. with the two in-block coordinates exchanged relative to how you stored it — and write to `out`, guarded by the edge check.

### MAJOR — kernel.cu:42-43 (solve TODO: choose a square block dim3(TILE, TILE) and a 2-D grid ... ceil-div on each axis)
- **Leaked:** choose a square block dim3(TILE, TILE) and a 2-D grid that covers the rows x cols input (ceil-div on each axis).
- **Why:** Names the exact block construction (dim3(TILE, TILE)) and the exact grid-sizing rule (2-D, ceil-div on each axis). Picking the block shape and computing the covering grid is launch-config work the learner is supposed to derive; this states both the API form and the dimensions.
- **Conceptual rewrite:** TODO: pick a 2-D block whose threads map onto one tile, and size a 2-D grid so the blocks cover the whole input even when a dimension isn't an exact multiple of the tile.

### MINOR — kernel.cu:22-24 (TODO: compute the global (row, col) this thread READS ... consecutive threadIdx.x walk consecutive `cols`)
- **Leaked:** compute the global (row, col) this thread READS from `in`. Lay the block out so consecutive threadIdx.x walk consecutive `cols` — that makes the global READ coalesced.
- **Why:** Specifies the exact thread-axis-to-data-axis mapping (threadIdx.x -> consecutive cols). The learner is supposed to derive which thread axis must stride contiguous global addresses for coalescing; naming threadIdx.x and the cols axis removes that derivation, though the actual offset arithmetic (pid*tile + threadIdx) is still left to write.
- **Conceptual rewrite:** TODO: compute the global (row, col) this thread reads from `in`. Arrange the mapping so that, within a warp, consecutive threads read consecutive global addresses — that is what makes the read coalesced.

### MINOR — README.md:35-36 (Hint 3: pad the inner dimension by 1 (`TILE × (TILE+1)`))
- **Leaked:** The fix is **one character** of the declaration: pad the inner dimension by 1 (`TILE × (TILE+1)`) so column neighbors land in consecutive banks.
- **Why:** States the exact padded shape TILE × (TILE+1) and that the fix is padding the inner dimension by 1. Although it then asks the learner to work out *why*, the *what* (the literal +1 on the inner dim) is the deliverable of hints 2-3; giving the exact declaration leaves only the rationale to derive, not the fix itself.
- **Conceptual rewrite:** Hint 3: the fix is a single change to the tile's declaration — adjust one of its dimensions so that elements in the same column no longer map to the same bank. Work out on paper what change to the width achieves that and why.

---

## `c05_pipelined_matmul` — 2 leaks (1 critical, 1 minor)

### CRITICAL — kernel.cu:35 (TODO comment: 'declare DOUBLE-buffered shared tiles: __shared__ float As[2][TILE][TILE], Bs[2][TILE][TILE];')
- **Leaked:** __shared__ float As[2][TILE][TILE], Bs[2][TILE][TILE];
- **Why:** This is the actual solution line written out verbatim: the learner copy-pastes the shared-memory declaration instead of deriving it. It hands over (a) that there are TWO shared arrays (A-tile and B-tile), (b) the exact leading dim [2] that encodes the double buffer, (c) the [TILE][TILE] shape, and (d) the float type and variable names As/Bs. Deriving 'double-buffering needs two of each tile, each TILE x TILE' is exactly the structural step the exercise is about.
- **Conceptual rewrite:** Replace with a conceptual nudge: 'declare your shared tiles, but with an extra dimension so you hold two copies of each (one being read while the other is filled) -- derive that dimension and the tile shape from your c02 tile.' Do not write the declaration.

### MINOR — kernel.cu:47 (TODO comment: 'ping-pong the buffer index each iteration (buf ^= 1)')
- **Leaked:** ping-pong the buffer index each iteration (buf ^= 1)
- **Why:** Hands over both the variable name (buf) and the exact swap expression (buf ^= 1). The learner no longer has to figure out how to alternate between the two buffers each K-step; the idiom is written for them.
- **Conceptual rewrite:** 'each iteration, flip which of the two buffers is current vs. next -- decide how you track and toggle that index.' Name neither the variable nor the XOR trick.

---

## `c06_wmma_matmul` — 11 leaks (2 critical, 7 major, 2 minor)

### CRITICAL — kernel.cu:28-33 (TODO) and README hint 4 line 44
- **Leaked:** //         fragment<matrix_a, 16,16,16, half, row_major>  a_frag;
//         fragment<matrix_b, 16,16,16, half, col_major>  b_frag;
//         fragment<accumulator, 16,16,16, float>         acc_frag;
- **Why:** Writes the three fragment declarations verbatim — the template parameters (matrix_a/matrix_b/accumulator), the exact 16,16,16 tile dims, the element types (half vs float), the layouts (row_major / col_major), and even the variable names. The whole 'figure out the fragment templates, dims, and layouts' lesson (explicitly called 'the subtle part' in hint 4) is handed over as copy-pasteable code.
- **Conceptual rewrite:** Declare three fragments for one tile: two input operand fragments (A and B) and one accumulator. Work out from the WMMA docs each fragment's template arguments — operand role, the tile's three dimensions, the element precision (inputs vs accumulator), and the memory layout each operand expects.

### CRITICAL — kernel.cu:41-43 (TODO)
- **Leaked:** //         - load_matrix_sync(a_frag, A_tile_ptr, lda);
//         - load_matrix_sync(b_frag, B_tile_ptr, ldb);
//         - mma_sync(acc_frag, a_frag, b_frag, acc_frag);
- **Why:** Writes the inner-loop body as filled-in calls: the exact load function (load_matrix_sync), the exact MMA call (mma_sync) and crucially its full argument order with the accumulator passed as both source and destination (acc_frag, a_frag, b_frag, acc_frag). Identifying these calls and the accumulate-in-place argument pattern IS the core exercise; it is handed over verbatim.
- **Conceptual rewrite:** Each K-iteration: load this warp's A-tile and B-tile into their fragments (there is a load-fragment-from-memory call that takes a pointer and a leading dimension), then issue one warp-collective tile-MMA that multiplies the two operand fragments and accumulates into the accumulator. Work out the argument order for the MMA call, including how the running accumulator is threaded through it.

### MAJOR — kernel.cu:17-19 (TODO comment)
- **Leaked:** // TODO: #include <mma.h> as well, and bring the WMMA names into scope with
//       `using namespace nvcuda::wmma;` (the fragment/load/store/mma symbols
//       all live in nvcuda::wmma).
- **Why:** Learner no longer has to discover which header the WMMA API lives in (<mma.h>) nor the exact namespace incantation `using namespace nvcuda::wmma;` — both are written verbatim, copy-pasteable.
- **Conceptual rewrite:** Note that the WMMA fragment/load/store/mma types live in a dedicated CUDA header and a sub-namespace — find which header to include and how to bring those symbols into scope.

### MAJOR — kernel.cu:35 (TODO)
- **Leaked:** // TODO: fill_fragment(acc_frag, 0.0f) to zero the accumulator.
- **Why:** Names the exact API call and its arguments verbatim (fill_fragment(acc_frag, 0.0f)); the learner doesn't have to find the zeroing function or its signature.
- **Conceptual rewrite:** Before the K-loop, zero the accumulator fragment (there is a WMMA helper that fills a fragment with a constant).

### MAJOR — kernel.cu:37 (TODO)
- **Leaked:** // TODO: loop k from 0 to N in steps of 16. Each iteration:
- **Why:** States the exact loop bounds (0 to N) and the step (16) instead of letting the learner derive that the K dimension is walked one tile-K (16) at a time.
- **Conceptual rewrite:** Loop over the K dimension one tile at a time, accumulating each tile-MMA into the accumulator. Derive the step from the tile's K extent.

### MAJOR — kernel.cu:47-48 (TODO)
- **Leaked:** // TODO: store_matrix_sync the accumulator to C's tile (leading dim N,
//       layout mem_row_major).
- **Why:** Names the exact store API (store_matrix_sync), states the leading dimension to pass (N), and names the exact layout enum (mem_row_major) — leaving nothing to derive about how the result tile is written back.
- **Conceptual rewrite:** After the K-loop, write the accumulator fragment out to this warp's tile of C (there is a store-fragment call; you supply the destination pointer, the buffer's leading dimension, and the memory layout C is stored in).

### MAJOR — kernel.cu:51-56 (TODO in solve) and README spec line 11 / harness N=512
- **Leaked:** // TODO: choose a launch config that gives ONE WARP per 16x16 output tile.
//       There are (N/16) x (N/16) output tiles. A block holds several warps
//       (blockDim.x must be a multiple of 32); size grid/block so every tile
//       is covered exactly once.
- **Why:** Gives the grid arithmetic directly: there are (N/16) x (N/16) tiles, one warp per tile, blockDim.x a multiple of 32. Deriving the tile count and the warp-per-tile launch geometry is part of the launch-config exercise; here it is spelled out, leaving only the literal block size to pick.
- **Conceptual rewrite:** Pick a launch geometry so that exactly one warp lands on each output tile and every tile is covered once. Work out how many tiles there are from N and the tile size, and remember a block's thread count must be a whole number of warps.

### MAJOR — README.md hint 3 lines 40-43
- **Leaked:** Allocate an fp32 `accumulator` fragment, zero it with `fill_fragment`, then loop `k` over `K` in steps of 16: load an A-tile fragment and a B-tile fragment, do one `mma_sync` into the accumulator, advance K.
- **Why:** Restates the full step-by-step recipe in prose with the exact API names (fill_fragment, mma_sync) and the exact step (16), duplicating the kernel.cu leak in the README so even a learner who deleted the stub comments still gets the filled-in sequence.
- **Conceptual rewrite:** Mirror the c02 structure: allocate an fp32 accumulator fragment, zero it, then walk K one tile at a time — load an A-tile and B-tile fragment and accumulate one tile-MMA per step. Keep the accumulator in fp32 throughout.

### MAJOR — README.md hint 4 lines 44-50
- **Leaked:** Look at the fragment templates: `matrix_a` is `row_major`, `matrix_b` is `col_major`. ... for a row-major `N×N` matrix, that's `N`.
- **Why:** Hint 4 itself says layouts/leading dimensions are 'the subtle part' (i.e. the thing to derive), then gives the answer: A is row_major, B is col_major, and the leading dimension is N. The learner no longer derives the layout choice or the stride.
- **Conceptual rewrite:** Layouts and leading dimensions are the subtle part. Decide which layout each operand fragment should declare and what stride load_matrix_sync needs — relate the leading dimension to how a row-major NxN buffer is laid out in memory, and reason about what declaring B col_major over a row-major buffer implies.

### MINOR — kernel.cu:24-26 (TODO in wmma_matmul) and README hint 1 line 34
- **Leaked:** threadIdx.x / 32 tells you which warp you are within the block / (threadIdx.x / 32 picks the warp within the block)
- **Why:** Deriving that a warp is 32 lanes and that warp-within-block = threadIdx.x / 32 is part of the warp-mapping work; the exact expression is given so the learner copies it instead of reasoning about the warp size.
- **Conceptual rewrite:** A warp is a fixed number of lanes; from threadIdx work out which warp within the block you belong to, then map that warp to a (tileRow, tileCol) in C.

### MINOR — kernel.cu:56 (TODO)
- **Leaked:** // TODO: launch wmma_matmul<<<grid, block>>>(d_A, d_B, d_C, N);
- **Why:** Writes the launch statement verbatim, including the <<<grid, block>>> syntax and the exact argument list, so the learner copies rather than writing the launch.
- **Conceptual rewrite:** Launch your kernel with the grid/block you chose, forwarding the device pointers and N.

---

## `e03_copy_bandwidth` — 3 leaks (3 minor)

### MINOR — kernel.py:16 (TODO inside copy_kernel signature)
- **Leaked:** # TODO: in ptr, out ptr, n_elements, BLOCK_SIZE: tl.constexpr
- **Why:** Hands the learner the full kernel parameter list AND the exact Triton annotation tl.constexpr, so they no longer have to derive which arguments the kernel needs or that the block size must be a compile-time constant marked with tl.constexpr.
- **Conceptual rewrite:** # TODO: declare the kernel's parameters -- the data it reads/writes, how many elements there are, and the per-program tile size (think about which of these must be known at compile time).

### MINOR — kernel.py:18 (TODO inside copy_kernel body)
- **Leaked:** # TODO: program id -> offsets -> mask -> load -> store
- **Why:** Gives the complete ordered sequence of the kernel body as a recipe. The learner no longer has to figure out the structure of a memory-bound elementwise kernel (compute this program's index, build its offsets, guard the tail, read, write) -- only the API calls per step remain to derive.
- **Conceptual rewrite:** # TODO: each program handles one contiguous tile of the array. Work out which elements this program owns, guard against running past the end, then move the data from input to output.

### MINOR — README.md:22 (Hint 1)
- **Leaked:** Identical structure to `e01`: program-id → offsets → mask → load → store.
- **Why:** Repeats the exact ordered operation pipeline in the README, reinforcing the kernel-body recipe so the learner does not derive the structure themselves. Pointing back to e01 alone would be fine; spelling out the step sequence is the leaky part.
- **Conceptual rewrite:** Same structure as `e01`, just with no arithmetic between the load and the store -- revisit how that kernel mapped programs to elements and guarded the tail.

---

## `e04_row_reduce` — 4 leaks (4 minor)

### MINOR — README.md:19 (Hint 2)
- **Leaked:** Pass them in (`x.stride(0)`, `x.stride(1)`).
- **Why:** The learner no longer has to discover the exact API for querying a tensor's strides. The conceptual half ('you need the row stride and column stride') already teaches the idea; the parenthetical hands the precise method calls, which is the part of the work that was supposed to be derived.
- **Conceptual rewrite:** Pass them in (PyTorch tensors expose their per-dimension strides through a method -- find it).

### MINOR — README.md:24 (Hint 4)
- **Leaked:** Triton reduces a vector to a scalar with a reduction op along an axis (look for `tl.sum`).
- **Why:** Identifying the Triton reduction function is part of the exercise; naming tl.sum directly removes that lookup. This is the bar's own canonical minor-leak example (a hint that names the exact function).
- **Conceptual rewrite:** Triton reduces a vector to a scalar with a reduction op along an axis -- look in tl for the function that sums a block.

### MINOR — README.md:16-17 (Hint 1)
- **Leaked:** launch `M` programs, program `i` owns row `i`. The grid is 1-D with `M` entries.
- **Why:** States the grid configuration (rank and size) outright. The learner should derive that 'one program per row' implies a 1-D grid of size M; here the exact grid shape is given, so there is no dim to work out.
- **Conceptual rewrite:** launch one program per row, with program i owning row i -- decide the grid's shape and size from that.

### MINOR — kernel.py:15 (row_sum_kernel TODO)
- **Leaked:** TODO: x ptr, out ptr, strides you need, N (cols), BLOCK_SIZE: tl.constexpr
- **Why:** Enumerates the full kernel parameter list (down to the BLOCK_SIZE: tl.constexpr annotation), so the learner no longer derives which arguments the kernel needs or how BLOCK_SIZE must be declared. 'strides you need' is left abstract (good), but the rest of the signature is handed over.
- **Conceptual rewrite:** TODO: declare the parameters this kernel needs -- the input/output pointers, whatever stride and dimension info you must index with, and a compile-time tile size.

---

## `e05_softmax` — 4 leaks (4 minor)

### MINOR — README.md:26 (Hint 4)
- **Leaked:** `tl.max` and `tl.sum` reduce along the block axis; `tl.exp` is your map.
- **Why:** The learner no longer has to discover which Triton language functions perform a block-axis reduction (max, sum) or the elementwise exp. Identifying these API calls is part of the work; here all three are named verbatim.
- **Conceptual rewrite:** There are block-level reduction primitives that collapse your loaded row to a single max and a single sum, and an elementwise exponential that maps the whole vector at once -- find them in the Triton language reference.

### MINOR — README.md:24-25 (Hint 3)
- **Leaked:** mask them to `-inf` for the max, `0` for the sum
- **Why:** States the exact identity values to substitute for masked-out tail lanes, so the learner doesn't have to reason that a max needs a -inf identity and a sum needs a 0 identity.
- **Conceptual rewrite:** Masked-out tail lanes must not pollute the reduction -- give them whatever neutral value leaves a max unchanged, and a different neutral value that leaves a sum unchanged. Work out what each identity is.

### MINOR — README.md:18-19 (Hint 1)
- **Leaked:** compute `m = max(row)`, compute `e = exp(row - m)`, compute `s = sum(e)`, write `e / s`.
- **Why:** Gives the full ordered operation sequence with the solution expressions filled in (exp(row - m), e / s) and even the variable names. This is the numerically-stable softmax algorithm itself written out as near-pseudocode rather than described conceptually. Borderline: the algorithm IS the named concept of the exercise, but writing the exact expressions/variables crosses from naming the concept into handing the recipe.
- **Conceptual rewrite:** Per row, the stable recipe is: find the row's maximum, subtract it before exponentiating, sum the exponentials, then normalize each entry by that sum. Translate each of those four steps into code yourself.

### MINOR — kernel.py:15
- **Leaked:** # TODO: x ptr, out ptr, strides, N, BLOCK_SIZE: tl.constexpr
- **Why:** Names the kernel's exact parameter list including which argument must be a tl.constexpr, so the learner doesn't derive the signature. Mild: deciding what the kernel needs (pointers, strides, N, a compile-time block size) is part of the design, but the README already commits the learner to a BLOCK_SIZE-covers-a-row design so this mostly restates the contract.
- **Conceptual rewrite:** # TODO: decide what this kernel needs as arguments -- the data pointers, whatever stride info you need to walk rows, the row length, and a compile-time tile width.

---

## `e06_transpose` — 5 leaks (4 major, 1 minor)

### MAJOR — README.md:18-19 (Hint 1, "Two program ids")
- **Leaked:** **Two program ids:** `tl.program_id(0)` and `tl.program_id(1)` pick the tile's row and column blocks.
- **Why:** Hands the exact API call AND both axis arguments. The learner no longer has to discover that the program index comes from tl.program_id, nor derive that a 2-D grid means two calls with axes 0 and 1. Finding/naming this function is part of the work for a first 2-D grid.
- **Conceptual rewrite:** A 2-D launch grid gives each program two index coordinates (one per grid axis). Find the Triton call that returns the current program's index along a given axis, and use it twice to identify which row-block and which col-block this program owns.

### MAJOR — README.md:20-22 (Hint 2, "Tile offsets")
- **Leaked:** build `rows = pid_m*BLOCK_M + arange(0, BLOCK_M)` and similarly for cols; combine into 2-D offset arrays with broadcasting (`rows[:, None]`, `cols[None, :]`).
- **Why:** Writes the offset-vector expression essentially verbatim (including variable names pid_m, rows, the BLOCK_M scaling and arange call) and hands the exact broadcasting syntax rows[:, None] / cols[None, :]. Deriving how to turn a block index into the range of offsets it owns, and how to outer-broadcast two 1-D ranges into a 2-D index grid, is the central skill this exercise teaches.
- **Conceptual rewrite:** Turn each block index into the range of element indices that block covers (offset the block's start by a 0..BLOCK range). Then form a 2-D index grid from the two 1-D ranges by broadcasting one along rows and the other along columns.

### MAJOR — README.md:25 (Hint 4, "Edge mask")
- **Leaked:** **Edge mask:** `(rows < M) & (cols < N)` for ragged tiles.
- **Why:** Writes the boolean mask expression verbatim, including the variable names and the elementwise-and. The learner no longer has to derive how to guard both dimensions against overrun for ragged tiles.
- **Conceptual rewrite:** Tiles at the matrix edge can run past the real bounds in either dimension. Build a 2-D boolean mask that is true only where both the row and the col index are still in range, and pass it to the load/store.

### MAJOR — kernel.py:27 (transpose wrapper TODO)
- **Leaked:** 2-D grid = (cdiv(M,BLOCK_M), cdiv(N,BLOCK_N));
- **Why:** Writes the grid tuple verbatim: both the ceil-div helper by name and the exact two-element tuple shape with arguments filled in. Deriving the grid shape (how many tiles cover the matrix, and that it is 2-D) is part of launching a 2-D kernel.
- **Conceptual rewrite:** Choose a 2-D grid with enough programs to cover the whole matrix in both dimensions (round up each dimension by your tile size -- look for a Triton ceil-div helper).

### MINOR — kernel.py:16 (transpose_kernel signature TODO)
- **Leaked:** # TODO: x ptr, out ptr, M, N, strides, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
- **Why:** Spells out the full parameter list and explicitly types the two tile-size params as tl.constexpr. Deciding which arguments the kernel needs (and that block sizes must be compile-time constexpr) is minor design work the learner is told to do; naming them as constexpr removes that small step.
- **Conceptual rewrite:** # TODO: declare the kernel parameters you need -- the two pointers, the matrix dims, the strides for each tensor, and the tile sizes (think about which params must be compile-time constants).

---

## `e07_matmul` — 6 leaks (1 critical, 3 major, 2 minor)

### CRITICAL — kernel.py:35  (TODO in matmul() wrapper)
- **Leaked:** 2-D grid = (cdiv(M,BLOCK_M), cdiv(N,BLOCK_N))
- **Why:** The learner no longer derives the launch grid at all — the exact grid tuple, both ceil-div calls, and the M-rows/N-cols axis assignment are written verbatim, ready to paste. Deriving 'one program per output tile -> ceil-div each output dim' is core matmul-launch work.
- **Conceptual rewrite:** # TODO: choose tile sizes and build a 2-D launch grid so that, together, the
#        programs cover every output tile (each program owns one tile).
#        Pass all strides; launch.

### MAJOR — kernel.py:22  (TODO in kernel stub)
- **Leaked:** an accumulator of shape (BLOCK_M, BLOCK_N), start at zero (fp32)
- **Why:** States the exact accumulator shape and dtype. Deriving that the accumulator matches the output tile dimensions and must be fp32 for accuracy is part of the tiling lesson; here it is handed over.
- **Conceptual rewrite:** # TODO: allocate a register accumulator sized to the output tile this program
#        owns, initialized to zero; pick a dtype that preserves accumulation accuracy.

### MAJOR — kernel.py:24-25  (TODO in kernel stub)
- **Leaked:** load an (BLOCK_M x BLOCK_K) tile of A and a (BLOCK_K x BLOCK_N) tile of B (mask the K tail), accumulate the tile product (look for tl.dot)
- **Why:** Gives the exact A-tile and B-tile shapes (which dimension pairs with BLOCK_K on each operand) and names the exact primitive tl.dot. Working out the operand tile shapes and discovering that Triton has a block-matmul primitive is the heart of this exercise; both are spelled out.
- **Conceptual rewrite:** # TODO: load the slice of A and the slice of B needed for this K-step (mask the
#        K tail), then multiply-accumulate the two blocks. Triton has a primitive
#        that does a block-by-block matmul into an accumulator — find it.

### MAJOR — README.md:20-24  (Hints 1-3)
- **Leaked:** 2-D program ids -> a `BLOCK_M x BLOCK_N` block of C ... allocate a `(BLOCK_M, BLOCK_N)` fp32 accumulator initialized to 0 ... load an `(BLOCK_M, BLOCK_K)` tile of A and a `(BLOCK_K, BLOCK_N)` tile of B
- **Why:** Restates every tile/accumulator shape explicitly: output tile (BLOCK_M,BLOCK_N), accumulator (BLOCK_M,BLOCK_N) fp32, A-tile (BLOCK_M,BLOCK_K), B-tile (BLOCK_K,BLOCK_N). Working out which block dim pairs with K on each operand and that the accumulator mirrors the output tile is the derivation the exercise is meant to exercise.
- **Conceptual rewrite:** Each program owns one output tile. Allocate an accumulator matching that tile (accuracy-preserving dtype, zero-initialized). In the K loop, load the slice of A and the slice of B that line up along K for this step, and accumulate their block product — let the matmul definition tell you which tile dimension pairs with K on each operand.

### MINOR — README.md:25  (Hint 3)
- **Leaked:** Triton has a primitive that multiplies two 2-D blocks and accumulates — look for `tl.dot`.
- **Why:** Identifying the block-matmul primitive is part of the exercise; the conceptual description ('a primitive that multiplies two 2-D blocks and accumulates') is the right level, but appending the literal API name tl.dot removes the discovery step. The conceptual half alone would suffice.
- **Conceptual rewrite:** Triton has a primitive that multiplies two 2-D blocks and accumulates into your accumulator — look for it in the language reference.

### MINOR — README.md:38  (Going for performance)
- **Leaked:** try `BLOCK_M=BLOCK_N=64 or 128`, `BLOCK_K=32 or 64`.
- **Why:** Hands concrete working block-size values. Choosing/tuning tile sizes is explicitly framed as learner work ('Tile sizes are everything'); giving the exact numbers removes the experimentation. Borderline since they are framed as things to 'try' for perf rather than required, but they are specific block sizes.
- **Conceptual rewrite:** Tile sizes are everything — experiment with power-of-two BLOCK_M/BLOCK_N in the tens-to-low-hundreds range and a smaller BLOCK_K, and watch how FLOP/s responds. e10 automates this search.

---

## `e08_layernorm` — 4 leaks (3 major, 1 minor)

### MAJOR — kernel.py:21 (TODO inside layernorm_kernel stub)
- **Leaked:** # TODO: mean = sum(row)/N ; var = sum((row-mean)^2)/N
- **Why:** The exact reduction expressions for both mean and variance are written verbatim as comments in the body the learner is told to fill. The learner only has to mechanically translate sum->tl.sum; they no longer derive the two-moment computation that is the substance of the per-row pass.
- **Conceptual rewrite:** # TODO: reduce the row to its mean, then to its variance (deviation-squared, averaged). Decide what to divide by.

### MAJOR — kernel.py:22 (TODO inside layernorm_kernel stub)
- **Leaked:** # TODO: norm = (row - mean) * rsqrt(var + eps)
- **Why:** The complete normalization expression, including the eps-inside-rsqrt placement and the multiply-by-reciprocal-sqrt form, is given verbatim. The learner doesn't have to reason about where eps goes or that rsqrt/1-over-sqrt is the move; it is copy-translate.
- **Conceptual rewrite:** # TODO: center the row and rescale it to unit variance (remember eps for numerical safety -- think about where it belongs).

### MAJOR — kernel.py:23 (TODO inside layernorm_kernel stub)
- **Leaked:** # TODO: out = norm * w + b   (w, b indexed by column)
- **Why:** The full affine expression is written out and the indexing-by-column decision is stated, so the learner derives neither the scale-then-shift form nor the realization that weight/bias are per-column and reuse the row offsets.
- **Conceptual rewrite:** # TODO: apply the affine transform; figure out which axis weight/bias are indexed along and which offsets that reuses.

### MINOR — README.md:21-22 (Hints, item 2)
- **Leaked:** Normalize: `(row - mean) * rsqrt(var + eps)`. Triton has `tl.rsqrt` (or use `1/tl.sqrt`).
- **Why:** Names the exact API (tl.rsqrt) and writes the normalization expression including eps placement. Identifying the rsqrt helper and where eps sits is part of the work; the formula here is allowed-ish but naming tl.rsqrt and giving the verbatim expression hands both. Stating eps goes inside the sqrt is the derived insight removed.
- **Conceptual rewrite:** Normalize by centering and dividing by the (eps-stabilized) standard deviation -- Triton has a reciprocal-sqrt helper if you look for it. Think about whether eps belongs inside or outside the root.

---

## `e09_cumsum` — 3 leaks (3 minor)

### MINOR — README.md:21-22 (Hint 2 "The primitive")
- **Leaked:** within a block, Triton can do an inclusive scan for you -- look for `tl.cumsum` (or the more general `tl.associative_scan`). Using it is fair
- **Why:** The learner no longer has to discover the exact Triton scan primitive; the function names tl.cumsum and tl.associative_scan are handed over verbatim, so identifying the API is no longer part of the work. Note the author explicitly justifies this (the stated lesson is *why* scan needs masking, not finding the op), which softens it but does not change that the exact call is named.
- **Conceptual rewrite:** Within a block, Triton can perform an inclusive scan for you, so you don't need to hand-roll Hillis-Steele here. Search the Triton language module for a cumulative-sum / associative-scan primitive -- the point of this exercise is understanding *why* scan needs special handling, not reimplementing the scan itself.

### MINOR — kernel.py:21-22 (TODO in cumsum_kernel)
- **Leaked:** inclusive scan along the block axis (Triton has a cumulative-sum op; look for tl.cumsum / tl.associative_scan).
- **Why:** Duplicates the README leak inside the stub: names tl.cumsum / tl.associative_scan directly at the point of implementation, so the learner copies the op name rather than locating it. Only deriving which axis to scan along is left.
- **Conceptual rewrite:** inclusive scan along the block axis (Triton provides a cumulative-sum / associative-scan primitive for in-block scans -- find it in the Triton language module).

### MINOR — kernel.py:17 (TODO in cumsum_kernel signature)
- **Leaked:** # TODO: x ptr, out ptr, strides, N, BLOCK_SIZE: tl.constexpr
- **Why:** Enumerates the full parameter set the learner would otherwise have to reason about (that strides and N must be passed, and that BLOCK_SIZE is a tl.constexpr), reducing signature design to transcription. Borderline plumbing, but it does spell out work (which args to plumb through, and the constexpr mechanism) the learner is told to do.
- **Conceptual rewrite:** # TODO: declare the kernel parameters. Think about what each program needs: the input/output base pointers, enough stride info to walk to its row, the row length, and a compile-time tile width.

---

## `e10_autotuned_matmul` — 5 leaks (1 critical, 4 major)

### CRITICAL — kernel.py:47 (TODO inside matmul() wrapper)
- **Leaked:** grid = lambda META: (cdiv(M, META['BLOCK_M']), cdiv(N, META['BLOCK_N']))
- **Why:** Writes the entire launch-grid expression verbatim: the learner no longer has to derive that the grid is 2-D, that each axis is a ceil-div of a problem dim by the corresponding block size, that block sizes are read out of the autotune META dict by the keys 'BLOCK_M'/'BLOCK_N', or even the cdiv call. The README's own hint 4 deliberately stops at 'the grid is a function of META' -- this TODO undoes that restraint and hands over the answer.
- **Conceptual rewrite:** # TODO: the tile sizes now come from the chosen autotune config, not constants.
#       That means the launch grid can't be a fixed tuple anymore -- it has to
#       ask the config how big the tiles are. Triton lets the grid be a callable
#       that receives the chosen meta-parameters. Build a 2-D grid that covers
#       all row-tiles and all column-tiles (round UP so ragged edges still get a
#       tile). Then launch the kernel, passing every pointer, the three sizes,
#       and all six strides.

### MAJOR — kernel.py:36 (final-store TODO in matmul_kernel)
- **Leaked:** (row < M) & (col < N) so the ragged last tiles do not write out of bounds
- **Why:** Writes the store-mask boolean expression verbatim, including the chosen variable names (row, col) and the bitwise-AND combination of the two edge conditions. The learner no longer has to derive that the 2-D store mask is the conjunction of a row-in-bounds and a column-in-bounds test; they copy it.
- **Conceptual rewrite:** # TODO: write the accumulator out to C, but only the lanes that actually fall
#       inside C -- the last row-tile and last column-tile hang off the edge, so
#       guard the store with a 2-D mask that is true only where this tile's rows
#       AND columns are still within bounds.

### MAJOR — kernel.py:24-26 (signature TODO in matmul_kernel)
- **Leaked:** # TODO: a, b, c pointers; M, N, K; strides for a, b, c (stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn); BLOCK_M, BLOCK_N, BLOCK_K: tl.constexpr
- **Why:** Hands over the complete kernel parameter list: the exact set and names of all six strides, and that the three block sizes are constexpr named BLOCK_M/BLOCK_N/BLOCK_K. Designing the kernel signature -- realizing you need per-axis strides for a 2-D matmul and which dims they index -- is part of the work for a from-scratch matmul. This is the kernel's internal interface, not the harness contract (the harness only fixes the matmul(a,b) entrypoint). Naming the strides also implicitly reveals the index layout of each operand.
- **Conceptual rewrite:** # TODO: declare the kernel parameters. You'll need the input/output pointers,
#       the three problem dimensions, the strides needed to address each 2-D
#       operand (think about how many strides a 2-D tensor needs), and the tile
#       sizes -- which must be compile-time constants so they can come from the
#       autotune config.

### MAJOR — kernel.py:18-21 (autotune-decorator TODO above matmul_kernel)
- **Leaked:** # TODO: add an @triton.autotune decorator here. configs=[ triton.Config({...block sizes...}, num_warps=..., num_stages=...), ... a small, diverse menu YOU design ... ], key=[...]
- **Why:** Spells out the decorator's call shape almost completely: that it is @triton.autotune with a configs=[...] list of triton.Config(...) objects, each taking a meta dict plus num_warps and num_stages keyword args, and a key=[...]. Identifying the autotune API surface (that Config carries num_warps/num_stages and that the decorator takes configs/key) is part of the exercise. The block-size values are still left blank, which keeps it from being fully critical, but the API skeleton is handed over rather than derived.
- **Conceptual rewrite:** # TODO: this kernel should be autotuned. Find Triton's autotune decorator and
#       give it a menu of candidate configurations to search over. Each
#       candidate fixes a set of tile sizes plus the scheduling knobs you met in
#       lecture 2a. You'll also tell it which arguments, when they change, force
#       a fresh search.

### MAJOR — kernel.py:30 (accumulator TODO in matmul_kernel)
- **Leaked:** an accumulator of shape (BLOCK_M, BLOCK_N), start at zero (fp32)
- **Why:** States the exact accumulator shape (BLOCK_M x BLOCK_N) and dtype (fp32). Per the teaching bar, tensor shapes should be derived, not given. The learner no longer has to reason that the accumulator must match the output tile dimensions and accumulate in fp32 for accuracy.
- **Conceptual rewrite:** # TODO: create an accumulator that holds this program's output tile and
#       initialize it to zero. Think about what shape it must be (it has to hold
#       the tile you'll eventually store) and what precision you want to
#       accumulate in for accuracy.

---

## `e11_flash_attention` — 3 leaks (1 critical, 1 major, 1 minor)

### CRITICAL — kernel.py:32-38 (flash_kernel KV-loop TODO)
- **Leaked:** score tile  s_ij = tl.dot(q_i, k_j^T) * scale   (lives only in SRAM)
          online-softmax update:
             new_max = max(old_max, rowmax(s_ij))
             p       = exp(s_ij - new_max)
             alpha   = exp(old_max - new_max)            (rescale the old state)
             denom   = alpha * denom + rowsum(p)
             acc     = alpha * acc + tl.dot(p, v_j)
             old_max = new_max
- **Why:** This is the complete kernel body. Every load-bearing expression of the flash-attention inner loop is written out verbatim: the scaled score dot, the running-max update, the exponentiated probabilities, the rescale factor alpha, the denominator recurrence, and the accumulator recurrence. The learner does not derive the online-softmax recurrence (the entire point of the exercise, unlocked from lecture 2b) -- they transcribe these lines and rename s_ij/p/alpha/denom/acc to their own variables. It also hands over where the *scale multiply goes (on the score tile) and that rowmax/rowsum reductions are needed. Directly contradicts README hint 2, which tells the learner the formula's variable names won't match and they must map them and derive shapes themselves -- here the formula is pre-mapped onto concrete per-step assignments.
- **Conceptual rewrite:** loop over the KV blocks; for each, score your query block against the key block (remember the scale), then apply the online-softmax update you derived in 2b: refresh the running max, exponentiate the scores against it, compute the rescale factor for the OLD state, and fold both the denominator and the output accumulator forward. Keep the score/probability math in fp32.

### MAJOR — kernel.py:34 and 38 (s_ij = tl.dot(q_i, k_j^T); acc += tl.dot(p, v_j))
- **Leaked:** s_ij = tl.dot(q_i, k_j^T) * scale  ...  acc = alpha * acc + tl.dot(p, v_j)
- **Why:** Names the exact API call (tl.dot) for BOTH matmuls and shows the operand for each (q_i with k_j transposed; p with v_j). README hint 6 deliberately makes 'which operand needs transposing' part of the work ('Think carefully about which operand needs transposing'), but the stub already writes k_j^T, resolving the transpose the hint asked the learner to reason out. Identifying that both score and output steps are tl.dot matmuls and their operand orientation is exercise work, not contract.
- **Conceptual rewrite:** the score step and the output-accumulation step are each a matmul that maps to tensor cores; figure out which operand of each must be transposed so the dimensions line up (see hint 6).

### MINOR — kernel.py:28-29 (init-state TODO)
- **Leaked:** init the running triple per query row -- running max at -inf, running denominator at 0, output accumulator at 0 (all fp32, sized to the query block / head dim).
- **Why:** Gives the exact initial values for all three running statistics (max = -inf, denom = 0, acc = 0). Deriving that the running max starts at -inf (so the first real max wins) and the sums start at 0 is a small but genuine reasoning step the learner is otherwise meant to do. Borderline with README hint 2, which already states the three stats but says 'running max at -inf' there too, so the concept is duplicated; the kernel stub restates the exact init values rather than the concept.
- **Conceptual rewrite:** initialize the three running statistics so the first KV block's contribution is taken as-is (think about what starting value for the running max makes the first real max win, and what neutral values the two sums need). Keep them in fp32.

---

## `e12_quantized_matmul` — 3 leaks (1 critical, 1 major, 1 minor)

### CRITICAL — kernel.py:47 (quant_matmul launch TODO)
- **Leaked:** 2-D grid = (cdiv(M,BLOCK_M), cdiv(N,BLOCK_N));
- **Why:** The learner no longer derives the grid construction at all -- the full 2-D grid tuple is written out with the ceil-div helper named and the variables (M, BLOCK_M, N, BLOCK_N) already filled in. Computing how many BLOCK_M x BLOCK_N tiles cover an MxN output is the core launch-logic step this TODO claims to ask of them; it is handed over copy-pasteable.
- **Conceptual rewrite:** 2-D grid: enough programs to cover every BLOCK_M x BLOCK_N tile of the MxN output (recall the ceil-div helper from e07/e10).

### MAJOR — README.md:29 (hint 2) + kernel.py:6 (docstring) + kernel.py:37 (inner-loop TODO)
- **Leaked:** Dequantizing a weight is one multiply: `b_hat = b_q * scale_of_its_column`.  /  b_hat = b_q * scale_of_its_column  /  multiply each column by its channel scale BEFORE the dot
- **Why:** This single dequant multiply is THE one new operation the whole exercise is built around ('only the B load changes'). Writing b_hat = b_q * scale as a formula hands over the exact operation the learner is meant to derive from 'symmetric, per-channel quant'; nothing is left to work out but the variable names.
- **Conceptual rewrite:** Symmetric (zero-point 0) quantization means dequant is a single scale multiply -- recover each weight by scaling the stored int8 value by its channel's scale. Derive from the spec what 'one multiply' looks like and which factor it uses.

### MINOR — kernel.py:28 (accumulator TODO)
- **Leaked:** an accumulator of shape (BLOCK_M, BLOCK_N), start at zero (fp32)
- **Why:** States the accumulator's exact shape and its fp32 dtype directly, which the bar says should be derived, not given. The learner no longer has to reason that the accumulator spans the output tile this program owns or that the running sum must be wide (fp32).
- **Conceptual rewrite:** a zeroed accumulator covering the output tile this program owns, kept in a precision wide enough to sum across all of K (see hint on accumulating wide).

---

## `e13_fused_silu_backward` — 2 leaks (1 major, 1 minor)

### MAJOR — kernel.py:45 (TODO inside _silu_forward)
- **Leaked:** # TODO: n_elements; pick BLOCK_SIZE; grid = (cdiv(n_elements, BLOCK_SIZE),);
- **Why:** Writes the launch-grid as a literal solution expression: it names the ceil-div helper (cdiv), spells out the exact 1-D grid tuple including the trailing-comma single-element shape, and fills in the variable names. The learner no longer has to derive that the grid is ceil(n_elements / BLOCK_SIZE) as a 1-element tuple, nor figure out that a ceil-div helper is what produces the program count -- they can transcribe the line. The same pattern is then trivially reused for the backward launch at line 53.
- **Conceptual rewrite:** # TODO: how many elements are there, and how many programs do you need to cover them all with your chosen chunk size? Build the launch grid from that, then launch the kernel.

### MINOR — README.md:60-61 (Going for performance)
- **Leaked:** `BLOCK_SIZE` of 1024-4096 is the usual sweet spot for a flat 1-D map; bigger blocks mean fewer programs but more registers per program.
- **Why:** Hands the concrete block-size value the learner is supposed to choose/tune. The teaching bar explicitly disallows giving block sizes directly; the learner no longer has to reason from element count to a sensible chunk size.
- **Conceptual rewrite:** Block size is a tuning knob for a flat 1-D map: too small and you launch more programs than needed; too large and each program burns more registers. Try a few powers of two and watch the bandwidth number to find the sweet spot.

---

## Quarantined — e01 (file holds user's solution)

- CRITICAL — kernel.py:29 (and the TODO above it, line 27-28) — _#   1. which block am I?         -> tl.program_id
    pid = tl.program_id(axis=0_
- CRITICAL — kernel.py:31 (and TODO line 30) — _#   2. which indices do I own?   -> that block id, BLOCK_SIZE, tl.arange
    off_
- CRITICAL — kernel.py:33 (and TODO line 32) — _#   3. don't run off the end     -> a boolean mask
    mask = n_elements>offs_
- CRITICAL — kernel.py:35-37 (and TODO line 34) — _#   4. load, add, store          -> tl.load / tl.store (pass the mask!)
    a = _
- CRITICAL — kernel.py:50 (and TODO line 48-49) — _# TODO: compute the 1-D launch grid ... (Look for a Triton ceil-div helper.)
   _
- CRITICAL — kernel.py:52 (and TODO line 51) — _# TODO: launch -> vector_add_kernel[grid](... , BLOCK_SIZE=...)
    vector_add_k_
- MAJOR — kernel.py:47 (and TODO line 46) — _# TODO: choose a BLOCK_SIZE (a power of two is conventional -- why?).
    BLOCK__

## Fully clean exercises (1)

`e02_fused_elementwise`
