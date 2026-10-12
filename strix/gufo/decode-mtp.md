# gufo (Qwen3.8-Flash-Next, gfx1151): decode, verification and MTP study

Read-only study of `~/src-ref/gufo` (MIT). No builds or runs. All paths are
relative to the gufo root. Abbreviations used below:

- `M/` = `src/models/qwen38_flash_next/`
- `K/` = `src/models/qwen38_flash_next/kernels/rocm/`
- `Q/` = `src/models/qwen38_flash_next/kernels/rocm/mmq/`
- `D/` = `docs/models/qwen3.8-flash-next/`

Note: the `kernels/rocm/...` paths in the brief live under the model directory
(`M/kernels/rocm`), not at the repo root.

Model geometry (from `M/config.hpp:24-68`): hidden 2560, vocab 248320, HC 4
streams with low rank 320 (so HcDim = 10240), 24 q heads / 2 kv heads / head
dim 256, full attention every 4th layer, indexer 4 heads x 128, top_k 2048
tokens, compress ratio 4, GDN k-heads 16, v-heads 48, head dim 128, conv 4,
512 experts, 10 used, expert_ff 640, shared ff 640, PLE n-gram (bigram and
trigram, 8 heads each, dim 160, conv 4).

---------------------------------------------------------------------------

## 0. Headline numbers and how they compare with ours

Published (D/BENCHMARKS.md:197-224, Oct 10 2026, UD-Q4_K_XL + shared Q8_0 MTP):

- AR tg128: 27.98 tok/s at d0, 28.02 at 32K, 27.13 at 128K.
- MTP mixed: 33.80 (d0) .. 38.01 (16K); repetitive: 65.25 (d0), 45-54 deeper.
- Acceptance in those runs (D/artifacts/single-mtp-gufo.json): mixed 50-63%,
  0.66-1.03 accepted drafts per cycle; repetitive 88.6%, 3.74 accepted per
  cycle at d0.

Their calibrated cycle cost table (`M/mtp_costs.hpp:18-43`), C1, context 0,
index = number of drafts (index 0 = "headless MTP catch-up plus ordinary
decode", comment `M/mtp_costs.hpp:13`):

| drafts | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ms (d0) | 38.5 | 47.3 | 56.1 | 65.7 | 74.1 | 82.2 | 89.7 | 96.8 |
| ms (d4K) | 39.7 | 49.3 | 59.8 | 69.7 | 79.1 | 89.1 | 98.6 | 108.8 |
| ms (d32K) | 42.5 | 51.9 | 61.5 | 71.1 | 82.5 | 92.5 | 103.6 | 112.6 |

So gufo is roughly **38.5 ms + ~8.3-8.8 ms per drafted token** (d0), falling
to ~7 ms for the 6th/7th draft. Compare:

- ours: 29.2 ms + 9.4 ms per draft (verify row 6.8, of which routed 4.8;
  draft 2.6, of which full-vocab head 1.5)
- Halogen: 25.6 ms + 6.0 ms per draft.

Interpretation:

- gufo's base cost is much worse than ours (their plain AR is 35.7 ms/token,
  and the MTP "width 0" cycle adds ~2.8 ms of predictor catch-up on top).
- gufo's marginal per-draft cost (~8.5 ms) is ~10% below ours even though
  their draft head is the **full Q8_0 vocabulary head** (248320 x 2560 Q8_0
  = ~644 MiB, `D/EXPERIMENTS.md:72` "retaining another 644 MiB"; ~3.1 ms at
  215 GB/s). If the head is ~3.1 ms and the MTP body ~1 ms, the target
  verify row is only ~4.5 ms marginal in their setup, versus our 6.8 ms.
  That decomposition is my inference; gufo publishes no per-kernel split.
- Their routed experts (UD Q4_K gate/up + Q5_x/Q8_0 down) are larger per
  token than our requantized file, which makes the low verify-row figure
  more notable. The likely reasons are in sections 2 and 5.

---------------------------------------------------------------------------

## 1. Decode matvec design (T = 1..8 rows)

### 1.1 Routing between vector and matrix paths

- `kVecBatch = 8` (`K/executor.cpp:81`). `MatrixRows(n)` is true only in
  prefill or for n > 8 (`K/executor.cpp:92-94`); `ExpertMatrixRows(n)` only
  for n > 32 (`K/executor.cpp:95-97`).
- So **every C1 decode and verify forward (T <= 8) takes the vector (mmvq)
  kernels**. Int8 WMMA is used only for 9..48 rows, i.e. batched multi-session
  verification (section 2.4).
- Dense Q8_0 entry: `Executor::Dense(w, x, ...)` quantizes x (or reuses
  producer rows) then calls `qfn_mmq_q8_0_dense_vec_preq` (`K/executor.cpp:1452-1468`,
  `1545-1552`). Non-Q8 small tensors (F16 router stack, F16 alpha/beta stack,
  indexer) go to `SmallGemm` (`K/executor.cpp:1590-1593`).

### 1.2 Q8_1 activation rows written by producer kernels

The activation quantize pass is mostly eliminated at decode. The kernel that
produces an activation also writes its Q8_1 row (block of 32: int8 codes,
half2 {d, sum}), and the executor tracks "ready" rows:

- Ready-row cache: `ReadyQ8` / `DropReadyQ8` / `Quantize` (`K/executor.cpp:1411-1450`).
  Two alternating Q8 slots so an input stays valid across one other
  quantization, also under graph replay (`K/executor.cpp:1439-1442`).
- Lane-level producer quantizer `QuantizeQ8_1Lane` (`K/kernels.hip.cpp:82-102`):
  32-lane amax and sum via `XorLane`, writes the code and, on lane 0 of the
  block, `ds = {amax/127, sum}`. Same values as the mmq quantizer.
- Producers that emit Q8_1 at decode:
  - HC combine + RMSNorm of next mixer -> `xn` and `xn_q8_1`
    (`K/kernels.hip.cpp:647-740`, call `K/executor.cpp:1846-1854`).
  - HC mix epilogue -> `mixed` and `mixed_q8_1` (`K/kernels.hip.cpp:319`,
    `K/executor.cpp:1965-1975`); consumed by ssm_in, qkv, router? (router is
    F16 so no), shared expert and routed gate/up (`K/executor.cpp:1767`).
  - SiLU-scale of the 320-wide HC low-rank vector -> Q8_1 for the HC up
    projection (`SiluScaleQ8_1`, `K/executor.cpp:1899-1907`).
  - GDN epilogue (per-head RMSNorm x sigmoid(z)) -> Q8_1 for ssm_out
    (`K/kernels.hip.cpp:1936-1937`, `K/executor.cpp:2185-2225`).
  - Attention output gate (`SigmoidMulQ8_1`) -> Q8_1 for attn_out
    (`K/executor.cpp:2421-2427`).
- Still quantized separately: shared-expert down input, routed-down slot rows
  (with expert reordering, 1.4 below), PLE embeddings.
- EXPERIMENTS records this bundle (producer Q8_1, up-front loads,
  DPP/permlanex16 reductions, branch-free Q4_K/HC reads) as **+5.8-6.0% tg
  at d0, +7.1-7.2% at d32K**, bit-identical (`D/EXPERIMENTS.md:105`).

### 1.3 Dense Q8_0 GEMV kernel (`mul_mat_vec_q8`)

`Q/mmvq.hip.cpp:10-75`, launched by `launch_q8<T>` (`Q/mmvq.hip.cpp:634-654`):

- **One wave32 = one block = one weight row**; grid = rows, block = 32 threads
  (`<<<rows, 32>>>`, `Q/mmvq.hip.cpp:638-652`). `__launch_bounds__(32, 1)`.
- Each lane handles vdr = 2 ints of a Q8_0 block, 4 lanes per 32-element
  block, 8 blocks (256 elements) per wave iteration
  (`Q/mmvq.hip.cpp:19-31`; `VDR_Q8_0_Q8_1_MMVQ 2`, `Q/vecdotq.hpp:195`).
- **All T tokens (1..8) are computed in the same wave**: the weight ints are
  read once per k-iteration and dotted with each token's Q8_1 block
  (`Q/mmvq.hip.cpp:32-45`), integer `dp4a` (`ggml_hip_dp4a`) per 4 bytes.
  So a verify of width T reads each dense weight once.
- Optional fused gate: `has_gate` runs gate and up rows together and applies
  SiLU before storing (`Q/mmvq.hip.cpp:41-44`, `57-59`, `69-72`) - used for
  the shared expert (`K/executor.cpp:1474-1491`).
- Reduction for T > 1 uses a **reduce-scatter butterfly** (`scatter_reduce_sum`,
  `Q/common.hpp:225-273`): with 8 sums, the first three levels each halve the
  number of sums a lane carries, so 8 independent sums cost ~1 butterfly + a
  few extra DPP ops instead of 8 full butterflies; lane `l` stores sum
  `l >> shift` (`Q/mmvq.hip.cpp:47-61`). Same pairing order, so bitwise
  identical to 8 separate butterflies.
- Shuffles are **DPP `row_xmask` for offsets < 16 and `v_permlanex16` for 16**
  instead of `ds_bpermute` (`Q/common.hpp:182-201`).
- Few-row/long-K shapes get **up-front loads**:
  - `mul_mat_vec_q8_ahead<20>` for T=1, rows <= 64, K % 5120 == 0 (the MTP
    block's 4-row HC inject at K=10240): each lane issues 20 blocks of loads
    before the first dp4a (`Q/mmvq.hip.cpp:82-135`, selection
    `Q/mmvq.hip.cpp:643-646`).
  - HC down projection 320 x 10240 (only 320 waves): `qfn_q8_hc_down_kernel`
    prefetches two iterations then `sched_barrier` (`Q/qfn_mmq.hip.cpp:596-670`,
    used for N <= 8 at `Q/qfn_mmq.hip.cpp:689-692`). Recorded as "HC Q8 weight
    prefetch ... retained for 1-8 rows" (`D/EXPERIMENTS.md:90`).
  - HC combine kernel loads the whole row into registers before any store
    (`K/kernels.hip.cpp:661-681`) because loads behind stores to `res` waited
    one by one.
- Rejected variants (useful negative results): wider dense rows 56-64 per
  block, phased/packed token tiles (`D/EXPERIMENTS.md:104`), Q8 activation
  reuse across output rows (`D/EXPERIMENTS.md:103`), more vocab rows per block
  (`D/EXPERIMENTS.md:85`, `137`), vocab head prefetch (`D/EXPERIMENTS.md:70`),
  planar head layout (+0.8% only, `D/EXPERIMENTS.md:72`).

### 1.4 Quantized expert GEMV (`mul_mat_vec_q_moe`)

`Q/mmvq.hip.cpp:249-349`, dispatch `Q/mmvq.hip.cpp:759-828`:

- Block = (32, block_tokens): one wave per token (slot), `rows_per_wave` rows
  per wave (2 by default; 4 for the K=640 down with T>1 for Q8_0/Q5_1, 8 for
  Q5_1 above 20 slot rows) (`Q/mmvq.hip.cpp:788-816`).
- Grid = (rows/rows_per_wave, experts_used, token blocks). Expert id loaded
  per wave, invalid ids write zeros (`Q/mmvq.hip.cpp:285-301`).
- Per-format max tokens per block: 4 for Q4_K/Q5_K, 8 otherwise
  (`Q/mmvq.hpp:16-19`).
- Gated variant (`gated=true`): wave 0 = gate rows, wave 1 = up rows of the
  same expert, SiLU(g)*u via LDS in the same kernel (`Q/mmvq.hip.cpp:323-341`).
  This is the T=1 routed gate/up launch: grid ((640+1)/2, 10), block (32,2)
  (`Q/mmvq.hip.cpp:867-883`).
- Q4_K dot: branch-free scale unpacking (`q4_K_scale_pair`, three word loads
  then a select instead of a branch whose sunk load stalled all in-flight
  loads, `Q/vecdotq.hpp:819-837`), used by `vec_dot_q4_K_q8_1`
  (`Q/vecdotq.hpp:839-875`).
- Down projection slot rows are re-quantized **in expert order**
  (`quantize_q8_1_by_expert`, `Q/quantize.hip.cpp:53-58`, selection
  `Q/qfn_mmq.hip.cpp:408-446`): slots sharing an expert become adjacent in the
  same block so their weight loads coincide; outputs go back to the source
  slot via `dst_slots` (`Q/mmvq.hip.cpp:289-292`).

### 1.5 Which tensor uses which kernel at decode (T <= 8)

| Tensor (per layer) | Type (UD-Q4_K_XL as loaded) | Kernel | Input source |
| --- | --- | --- | --- |
| HC down 320 x 10240 (attn and ffn mixers) | Q8_0 | `qfn_q8_hc_down_kernel<T>` | `xn_q8_1` from combine |
| HC up 10240 x 320 | Q8_0 | `mul_mat_vec_q8<T>` | SiluScaleQ8_1 output |
| HC inject | F32 fused into epilogue (or Q8 for MTP block -> `_ahead<20>`) | epilogue | - |
| ssm_in = stacked ssm_qkv + ssm_gate (`K/device_model.cpp:251-256`) | Q8_0 | `mul_mat_vec_q8<T>` | `mixed_q8_1` |
| ssm_alpha + ssm_beta stack (F32 -> F16, `K/device_model.cpp:158-235`) | F16 | `SmallGemm` | F32 `mixed` |
| ssm_out 2560 x 6144 | Q8_0 | `mul_mat_vec_q8<T>` | GDN epilogue Q8_1 |
| attn qkv stacked (`K/device_model.cpp:264-266`) | Q8_0 | `mul_mat_vec_q8<T>` | `mixed_q8_1` |
| attn_out | Q8_0 | `mul_mat_vec_q8<T>` | SigmoidMulQ8_1 |
| indexer k/q | small | `Dense` -> SmallGemm/Q8 | F32 |
| router + shared-expert gate input stack (`K/device_model.cpp:291`) | F16 | `SmallGemm` | F32 |
| shared expert gate+up | Q8_0 | `mul_mat_vec_q8<T, has_gate>` (fused SiLU) | `mixed_q8_1` |
| shared expert down | Q8_0 | quantize + `mul_mat_vec_q8<T>` | F32 |
| routed gate+up | Q4_K (or Q5_K/Q8_0) | T=1: `mul_mat_vec_q_moe<.,2,gated>`; T=2..8: `group_moe_slots` + `mul_mat_vec_moe_grouped<.,T>` | `mixed_q8_1` |
| routed down | Q5_1 / Q8_0 (Q5_K/Q4_K supported) | `quantize_q8_1_by_expert` + `mul_mat_vec_q_moe<.,4 or 8>` | F32 SwiGLU rows |
| output head 248320 x 2560 | Q8_0 | `mul_mat_vec_q8<T>` | HC head mix |

GB/s: gufo reports **no decode GEMV bandwidth**. The only bandwidth figures
are prefill: routed gate/up/down stream 0.9-1.1 GB/layer at 160-190 GB/s,
combines ~200 GB/s, dense prefill 30-34 TFLOPS (`D/EXPERIMENTS.md:155-161`),
and "routed kernels already stream expert weights at roughly two thirds of
DRAM bandwidth" (`D/EXPERIMENTS.md:139`).

---------------------------------------------------------------------------

## 2. Routed experts at T=1 and at verify widths T=2..8

### 2.1 Flow (`Executor::Moe` / `MoeExperts`, `K/executor.cpp:2432-2611`)

1. Router + shared gate logits in one F16 GEMV (`K/executor.cpp:2437-2440`).
2. `RouterTopK` (`K/executor.cpp:2441-2442`; kernel `K/kernels.hip.cpp:2919`).
3. For > 32 rows only: expert counts downloaded for tile maps
   (`K/executor.cpp:2443-2454`); not at decode.
4. Shared expert queued before the routed experts (`K/executor.cpp:2455-2473`).
5. Routed gate/up via `GatedExperts` (`K/executor.cpp:1757-1812`): at decode
   `qfn_mmq_moe_gated_vec_preq` with the ready `mixed` Q8_1 rows
   (`K/executor.cpp:1763-1783`).
6. Routed down via `Experts(..., slots = T*10, n_used = 1)`
   (`K/executor.cpp:2586-2590`) -> `qfn_mmq_moe_vec` (`K/executor.cpp:1723-1731`).
   The comment at `K/executor.cpp:1719-1722` states verification keeps the
   vector path "even when top-k expansion produces more than 32 slot rows".
7. `MoeEpilogue` weighted sum + shared expert (`K/executor.cpp:2602-2609`).

### 2.2 Gate/up at T = 2..8: union of experts, each read once

`mul_mat_vec_moe_gated` (`Q/mmvq.hip.cpp:830-884`):

- `group_moe_slots` (one block, <= 256 threads, `Q/mmvq.hip.cpp:357-427`):
  for the T*10 (token, slot) anchors, the first anchor with a given expert
  becomes the group **leader**; it records a per-token slot bitmask (which of
  the token's 10 slots selected this expert) (`Q/mmvq.hip.cpp:370-396`).
  Non-leaders are empty groups. Inactive (negative) ids get -2 so the kernel
  writes zeros. For Q4_K with T >= 3 the anchors are split into two ordered
  lists: groups with >= 2 active tokens and groups with exactly 1
  (`Q/mmvq.hip.cpp:394`, `398-426`).
- `mul_mat_vec_moe_grouped<type, T>` (`Q/mmvq.hip.cpp:430-548`): block = 64
  threads (gate wave + up wave), 2 output rows per block; grid
  ((640+1)/2, anchors) or (.., 2) when walking the lists
  (`Q/mmvq.hip.cpp:600-632`). For each k-chunk it loads the expert's weights
  once and accumulates every active token column (`Q/mmvq.hip.cpp:481-512`);
  SwiGLU applied in-kernel via LDS and written to every slot in the mask
  (`Q/mmvq.hip.cpp:514-545`).
- Q4_K multi-token groups use `Q4MoeFragment`: the packed Q4_K fragment
  (2 ints, scale pair, dm) is decoded once and dotted with each token
  (`Q/vecdotq.hpp:1286-1320`, used `Q/mmvq.hip.cpp:483-496`). Single-token
  groups go to a second launch compiled with `single_request=true`, a
  one-column body with no per-token mask checks (`Q/mmvq.hip.cpp:620-623`,
  `459-473`). EXPERIMENTS: "Q4 shared-expert weight reuse ... separate
  compact kernel for single-request experts; faster repetitive C1/C4/C8"
  (`D/EXPERIMENTS.md:91`).
- Launch count for routed gate/up: T=1: 1; T=2: 2 (group + grouped);
  T >= 3 (Q4_K): 3 (group + multi + single).

### 2.3 Down at T = 2..8

- Not grouped. All T*10 slot rows are separate inputs (K=640). Re-quantized
  by expert (`Q/qfn_mmq.hip.cpp:408-446`) so equal-expert slots sit in the same
  block and hit the same weight lines; `launch_moe<Q5_1, 4>` up to 20 slots,
  `<Q5_1, 8>` above, `<Q8_0, 4>` (`Q/mmvq.hip.cpp:788-816`).
- Explicit Q5 down grouping/reordering across tokens was tried and rejected
  ("slower on mixed routing", `D/EXPERIMENTS.md:96`), as were 16/32-row Q5
  expert tiles and Q5 wave64 (`D/EXPERIMENTS.md:104`).
- "Q5 high-bit expansion: exact integer multiply/mask replaces repeated
  shifts" (`D/EXPERIMENTS.md:92`).

### 2.4 Int8 WMMA "verification up to 48 rows"

- `mul_mat_q8_decode_batch<token_tiles>` (`Q/mmvq.hip.cpp:137-225`): 256
  threads, 16 weight rows per block, 8 waves split K, 16-token tiles x up to 3
  (48 rows); `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` with four separate
  K8 sums per wave to reproduce the scalar kernel's rounding, LDS transpose
  then one-thread tree reduction.
- Only for dense Q8 shapes `(K=2560, rows >= 1024)` or `(K=320, rows=10240)`
  and **9 <= tokens <= 48** (`Q/mmvq.hip.cpp:685-711`); >48 rows multiples of
  32 use the grouped grid version (`Q/mmvq.hip.cpp:691-699`).
- So it is for batched multi-session verify (e.g. C4 x width 8 = 32 rows),
  never C1. `D/EXPERIMENTS.md:102`: "Retained through 48 input rows for wide
  target projections/heads ... Small batches retain vector kernels."
- Batched Q4 gate/up above 8 rows: `group_moe_batch` + wave64
  `mul_mat_vec_moe_batch` with groups of <= 8 tokens per expert
  (`Q/mmvq.hip.cpp:550-597`, `Q/moe_wave64.hip.cpp:1-60`, dispatch
  `Q/mmvq.hip.cpp:836-843`); `D/EXPERIMENTS.md:98`.

### 2.5 How verify rows are batched (C1)

- The verify forward is a single target forward of width
  `chain = [anchor, d1..dk]` (`M/engine.cpp:997-1006`) in
  `ForwardMode::kVerify`; every layer processes all k+1 rows in one launch per
  op. Logit rows for all k+1 positions stay on device
  (`VerificationLogits()`), not downloaded for greedy (`M/engine.cpp:997-999`).
- Graph key includes n, logit rows, speculative, sparse, download flag
  (`K/executor.cpp:2832-2836`), so each width replays its own captured graph.

---------------------------------------------------------------------------

## 3. MTP policy

### 3.1 Cycle structure (`Session::PrepareDecode` / `FinishDecode`, `M/engine.cpp:761-975`)

1. Choose width = 1 + drafts from the length controller
   (`M/engine.cpp:780-789`).
2. Sample the **anchor** from the previous cycle's frontier logits on the CPU
   (`M/engine.cpp:794`).
3. `DraftCatchUp(anchor, propose=true)`: one MTP forward over
   `[tokens accepted since the MTP position ..., anchor]` using the target's
   kept hidden rows, which also emits the first draft
   (`M/engine.cpp:813-816`, replay list `M/engine.cpp:536-553`,
   `M/engine.cpp:555-573`). Catch-up and the first proposal share one MTP pass.
4. Remaining drafts: one single-row MTP forward each, carrying the MTP block's
   own residual (`hidden_row = -1`) (`M/engine.cpp:835-846`).
5. Target verify forward of width k+1 (`M/engine.cpp:1000-1006`).
6. Greedy: `GreedyMtpPredictions` = batched GPU argmax over the k verify rows
   (penalty-aware variant too), download k small candidates, one sync
   (`K/executor.cpp:3824-3894`). Accept the longest matching prefix
   (`M/engine.cpp:884-918`). A grammar-forbidden argmax falls back to
   downloading the remaining rows once (`M/engine.cpp:894-907`).
7. `Rollback(keep)` restores recurrent state and downloads only the frontier
   logit row (`K/executor.cpp:3056-3127`).
8. `draft_length_.Observe(...)`, then `MtpRewind(base)` so the next cycle's
   catch-up recomputes accepted rows with true target hidden states
   (`M/engine.cpp:964-968`, `K/executor.hpp:374-382`).
9. Sampled mode: a rejected draft's residual sample becomes the next cycle's
   anchor via `DeferSample`, avoiding an extra target pass
   (`M/engine.cpp:969-973`).

### 3.2 Adaptive depth: `MtpLengthController` (`M/mtp_policy.hpp:17-161`)

- Cap 7 drafts (`kMaxMtpDraftTokens = 7`, `M/mtp_policy.hpp:15`).
- Per-depth discounted counts: `successes[d]`, `failures[d]`, prior 1.5 / 0.5
  (`M/mtp_policy.hpp:97-99`), decay 0.75 per observation
  (`M/mtp_policy.hpp:67-70`). Only the first rejected depth is a failure; the
  unverified tail and a budget-limited fully accepted chain are censored
  (`M/mtp_policy.hpp:27-29`, `59-71`).
- P(accept at depth d) = s/(s+f), except depths already explored with no
  rejection ever, which are treated as 1.0 (`M/mtp_policy.hpp:126-132`).
- `BestLength`: for L = 1..cap, survival product, expected tokens
  `1 + sum survival`, score = expected / (cost[L]/cost[0]); pick the max; 0
  if plain decode wins (`M/mtp_policy.hpp:133-156`). Costs from the table,
  interpolated over context (`M/mtp_costs.hpp:49-75`).
- Probing: after a fully accepted chain shorter than the cap, probe
  `2 * drafted` (bounded, every 16 cycles once explored)
  (`M/mtp_policy.hpp:74-82`).
- AR fallback: if best length is 0, run 16 AR tokens then retry with at least
  1 draft (`M/mtp_policy.hpp:39-47`, `87-89`, `91-94`).
- Length depends on acceptance history only, never timing, to keep seeded
  replay deterministic (`M/mtp_policy.hpp:27`).

### 3.3 Batch controller (greedy, C > 1 only) (`M/mtp_policy.hpp:167-278`)

- Picks one width for the whole physical batch, maximizing
  `sum ExpectedTokens(width) / cost[width]` with a 1% hysteresis
  (`M/mtp_policy.hpp:191-205`).
- Live cost learning per (concurrency, context bin): EMA 0.25 after two warmed
  samples; transitions excluded because catch-up belongs to the previous chain
  (`M/mtp_policy.hpp:222-251`).
- Forced headless (width 0) cycles 2 of every 32 to measure the AR alternative,
  exploration of best+1 every 16 cycles (`M/mtp_policy.hpp:186-189`,
  `215-217`).

### 3.4 Cost model numbers (`M/mtp_costs.hpp`)

Full table in section 0. Concurrency cohorts C1/C2/C4/C6/C8 and contexts
0/4096/32768; above 32K extrapolated along the measured slope with a
monotonic marginal guard (`M/mtp_costs.hpp:49-75`). Measured 2026-09-20 with
`qwen38_flash_next_gpu_probe --cost-audit` (`M/mtp_costs.hpp:11-17`).
C1 marginal per draft at d0: 8.8, 8.8, 9.6, 8.4, 8.1, 7.5, 7.1 ms.

### 3.5 Draft head

- The MTP block reuses the **target's output tensor** (`model_->output()`) as
  its vocabulary head (`K/executor.cpp:2630-2636`), full 248320 rows, Q8_0.
- `D/EXPERIMENTS.md:87`: "Full Q8 vocabulary head: Retained; private Q4
  shortlist removed. Sampled top-64 proposals use exact target verification."
  A **planar head layout** improved isolated kernels but only +0.8% in full
  MTP decoding and cost another 644 MiB, rejected (`D/EXPERIMENTS.md:72`).
- Greedy: on-GPU argmax and a 4-byte download of the draft token
  (`K/executor.cpp:2640-2647`). Sampled: top-64 candidate selection on GPU
  (`MtpTopCandidates`, `K/kernels.hip.cpp:6910-6918`) and a download of ids
  and logits (`K/executor.cpp:2649-2660`).
- Each draft step is a separate graph replay **followed by a host sync**
  (`Run(..., synchronize=true)` default, `K/executor.cpp:2710-2711`; MTP key
  at `K/executor.cpp:3963-3969`) because the host needs the token to feed the
  next MTP step.
- MTP body details (`K/executor.cpp:3992-4197`): embedding of the shifted
  token, full-width RMSNorm of the HC*H hidden row, split `nextn_eh_proj`
  into embedding and hidden halves (`K/device_model.cpp:110-150`), one
  attention layer with its own KV/indexer, HC mixers, MoE (10 routed +
  shared), hidden carry copied to `session.mtp_.h`.

### 3.6 Exact sampling (`M/mtp_sampling.hpp`)

- Greedy: deterministic drafts compared to the target argmax.
- `VerifyDraft` for non-greedy deterministic drafts: sample from the target
  row; on mismatch restore the RNG so a stop at this prefix consumes the same
  draws as AR (`M/mtp_sampling.hpp:109-126`).
- Stochastic: proposal q over the top-64 candidates, masses quantized to
  2^24 units so q sums to exactly 1 (`M/mtp_sampling.hpp:32-69`); target p
  computed in FP64 over the full vocab; accept if u * q(x) < p(x), else sample
  the residual (`M/mtp_sampling.hpp:76-107`). QUALITY notes draft sampling
  "uses the full Q8 head's top 64 logits" (`D/QUALITY.md:51-54`).

### 3.7 Repetitive-text tricks

- **No prompt lookup / n-gram drafting.** Grep for prompt lookup / lookahead /
  suffix structures finds nothing in the model or core. `ngram.cpp` is the
  model's **PLE n-gram embedding table** read from disk with 128 `pread`
  workers and a 128 MiB row cache (`M/ngram.cpp:17-24`, `M/ngram.hpp:21-47`),
  not a drafter.
- The repetitive-text speedups come from MTP acceptance (88.6%) plus the
  controller's probe doubling, which quickly reaches width 7
  (`M/mtp_policy.hpp:74-82`).
- PLE detail relevant to verify: draft tokens' n-gram rows are hashed and
  fetched from disk per verify, with per-prefix n-gram history snapshots for
  rollback (`K/executor.cpp:1994-2033`). The decode graph is **split at the
  PLE layer** so layers before it replay while the host waits for the rows
  (`K/executor.cpp:2837-2862`).

### 3.8 Rollback of GDN recurrent state

- Snapshot slots allocated lazily up to the drafted depth
  (`EnsureRollback`, `K/executor.cpp:1291-1335`). Size per extra row is
  `GdnRollbackRowFloats` (row 0 full state, later rows compact).
  `D/EXPERIMENTS.md:130`: "one full state plus exact FP32 update operands ...
  seven-draft cap about 147 MiB/session".
- Inside the verify GDN kernel (`GdnKernel`, `K/kernels.hip.cpp:1736-1854`):
  - grid (v_heads=48, 128/32), block 128 threads, 4 lanes per state row, each
    lane holds 32 floats of a 128-wide row **in registers for all T tokens**;
    the state is read once and written once per verify regardless of T
    (`K/kernels.hip.cpp:1755-1771`, `1850-1853`).
  - After token 0 it writes the **full state** to snapshot row 0
    (`K/kernels.hip.cpp:1829-1833`), only when T >= 2.
  - For tokens t >= 1 (except the last) it writes only the **operands**: key
    per k-head, decay and beta per v-head, and the scalar error per state row
    (`K/kernels.hip.cpp:1834-1847`). That is 16*128 + 2*48 + 48*128 = 8288
    floats (~33 KB) per token per layer, versus 3 MiB for a full state.
- Restore on rejection (`RestoreGdnStateKernel`, `K/kernels.hip.cpp:1856-1879`,
  launcher `K/kernels.hip.cpp:6432-6438`): one thread per state element starts
  from row 0 and replays `keep-1` updates with the exact original FMA order.
  Cost: read 3 MiB + write 3 MiB per GDN layer (x36 = ~226 MB, ~1 ms) plus
  operand reads. No restore when all drafts are accepted
  (`K/executor.cpp:3080-3082`).
- Conv state: rolling snapshots written by the fused `SsmConv4Kernel<true>`
  for T <= tokens-per-thread (`K/kernels.hip.cpp:2022-2088`, `6464-6474`),
  restored by a memcpy of slot `keep-1` (`K/executor.cpp:3094-3098`).
- PLE conv history and n-gram window also restored per slot
  (`K/executor.cpp:3101-3111`). Attention KV: just move `position_` back and
  clamp pooled blocks (`K/executor.cpp:3112-3118`).
- Cost of this scheme per cycle: every verify with T >= 2 writes 36 x 3 MiB =
  ~113 MB of row-0 snapshots (~0.5 ms at 215 GB/s), accepted or not. My
  estimate; gufo does not publish it.
- C4 profile: "rollback replay 14.5 ms" over a 32-token-per-request trace
  (`D/EXPERIMENTS.md:163-168`).

---------------------------------------------------------------------------

## 4. Fused kernels, launch counts, hipGraph

### 4.1 hipGraph use

- `Executor::Run` captures on first reuse of a shape key (an eager warm-up
  first, because the first pass grows the GEMM arena), stores per session,
  replays with `hipGraphLaunch` (`K/executor.cpp:2664-2712`). Capture mode
  `hipStreamCaptureModeThreadLocal` so snapshot workers do not break it
  (`K/executor.cpp:2676-2680`; `D/EXPERIMENTS.md:132`).
- Target forward graphs exist for n <= 8 outside prefill
  (`K/executor.cpp:2829-2831`), keyed on (n, logit rows, speculative, sparse,
  download) (`K/executor.cpp:2832-2836`), and are **split in two at the PLE
  layer** (prefix graph, wait for disk rows, suffix graph)
  (`K/executor.cpp:2840-2865`).
- All host inputs (tokens, positions, block counts, hidden row index) live in
  pinned buffers that graph nodes read, and kernels read the position from a
  device control block (`K/executor.cpp:2810-2818`, `2897-2905`), so a graph
  replays at any position. Attention scratch is sized for max context so
  replay works at later positions (`K/executor.cpp:2326-2334`).
- MTP forwards are graphs too (n <= 8), separate keys for catch-up vs.
  recursive steps, token vs. candidates output (`K/executor.cpp:3960-3976`).
- Captured shared stages for batched target decoding were tried and rejected
  (`D/EXPERIMENTS.md:129`).

### 4.2 Fusions present at decode

- Stacked projections that share an input: ssm_qkv+ssm_gate, attn q/k/v,
  router + shared-expert gate input, alpha+beta (`K/device_model.cpp:155-295`).
- HC combine = residual update + next mixer's RMSNorm + Q8_1 (and MoE epilogue
  on the wide route) in one kernel (`K/executor.cpp:1814-1857`,
  `K/kernels.hip.cpp:647-740`, `882`).
- HC mix epilogue = mix + inject + Q8_1 write (`K/kernels.hip.cpp:273-375`).
- SiLU-scale + Q8_1 (`K/kernels.hip.cpp:1030`).
- Shared expert gate+up+SiLU in one GEMV (`Q/mmvq.hip.cpp:41-44`).
- Routed gate+up+SwiGLU in one kernel at all decode widths (2.2).
- Short conv + rolling history + rollback snapshots fused for 1-8 tokens
  (`D/EXPERIMENTS.md:93`; `K/kernels.hip.cpp:6464-6474`).
- GDN epilogue: per-head RMSNorm x sigmoid(z) + Q8_1 (`K/kernels.hip.cpp:1887-1966`).
- Attention: fused norm/rope/KV store (`PrepareAttention`,
  `K/executor.cpp:2266-2270`), split-K decode attention with 8 splits per row
  plus merge (`K/executor.cpp:107`, `2412-2418`), sigmoid gate + Q8_1
  (`K/executor.cpp:2421-2427`).

### 4.3 Launch count per decode token (my count from the code; gufo does not publish it)

GDN layer, T=1:

- HC attn mix: hc_down GEMV, SiLU-scale-Q8, hc_up GEMV, epilogue = 4
- ssm_in GEMV, alpha/beta SmallGemm = 2
- SsmConv4, GdnPrep, Gdn, GdnEpilogue = 4 (`K/kernels.hip.cpp:6469-6537`)
- ssm_out GEMV = 1
- combine (+norm, Q8_1) = 1
- HC ffn mix = 4
- router SmallGemm, RouterTopK = 2
- shared gate+up GEMV, quantize, shared down GEMV = 3
- routed gate+up = 1 (T=2: 2, T>=3 Q4_K: 3)
- routed down quantize-by-expert + GEMV = 2
- MoE epilogue = 1
- combine = 1

That is **~26 launches per GDN layer** (T=1), ~28-29 for T >= 3. Full
attention layers are similar plus about 4-10 attention launches (more past
2048 tokens for the indexer/selector). Total roughly **1,200-1,350 kernels per
token** across 48 layers, plus the head. These run as two graph replays
(prefix/suffix of the PLE layer). Per verify cycle add: GPU argmax kernels +
1 sync (`K/executor.cpp:3878-3893`) and the rollback (1 kernel + 1 memcpy per
GDN layer when rejecting, then 1 sync, `K/executor.cpp:3083-3122`). Per draft:
one MTP graph replay + 1 host sync.

---------------------------------------------------------------------------

## 5. What makes their verify row cheap relative to a T=1 token

Ordered by likely impact (my assessment):

1. **Routed experts: union grouping with one weight read per distinct expert**
   across the T rows, Q4_K fragment decoded once and dotted with each token,
   SwiGLU in-kernel, plus a separate one-column launch for experts that only
   one token selected (2.2). The down projection is ordered by expert so
   shared experts coincide in a block (2.3).
2. **Every dense GEMV handles all T rows in the same wave** with the weight
   loaded once and a reduce-scatter butterfly over DPP/permlanex16 (1.3).
   Dense weights therefore cost T=1 bandwidth for any T <= 8; only activation
   loads (L2-resident Q8_1) and dp4a work scale with T.
3. **GDN recurrence keeps the state in registers across all T tokens**; the
   3 MiB/layer state is read and written once per verify, and rollback
   operands are compact (3.8). The only T-independent extra cost is the
   ~113 MB row-0 snapshot write.
4. **Producer-written Q8_1 activations** remove most quantize launches, and
   the same rows feed several consumers (ssm_in, shared expert, routed
   gate/up) (1.2).
5. **One graph per width**, positions read on device, a single sync for greedy
   argmax over all verify rows, and only the frontier logit row downloaded
   (3.1, 4.1).
6. **Attention at decode** is split-K over 8 blocks per row, so extra rows add
   query work but not KV passes beyond each row's own sweep (`K/executor.cpp:2412-2418`).
7. **Catch-up folded into the first draft**: the MTP replay over accepted rows
   also proposes draft 1, so the cycle needs one fewer MTP pass
   (`M/engine.cpp:813-816`).

What gufo does **not** do (so it is not their advantage): no draft
vocabulary shortlist (full Q8 head per draft, ~3 ms), no prompt lookup, no
device-side draft chaining (one host sync per draft), no int8 WMMA for C1
verify.

---------------------------------------------------------------------------

## 6. Candidate ideas for our engine

Per drafted token (ours 9.4 ms = verify row 6.8 + draft 2.6):

- A. **Draft head shortlist** (Halogen does this; gufo dropped its Q4
  shortlist for exactness of *sampled* proposals). For greedy, a shortlist
  only changes acceptance, never output. Cuts most of our 1.5 ms per draft.
  Keep the full head for sampled requests, or a shortlist plus an
  out-of-list fallback.
- B. **Measure verify-row routed cost against the union bound.** Our routed
  share is 4.8 ms per verify row. Gufo reads each distinct expert once per
  verify (2.2). Instrument union size per verify and the achieved GB/s of
  the verify expert kernels. If we re-read an expert per token, or our
  per-row kernels hit lower GB/s than T=1, grouping by expert with a slot
  mask is the gufo pattern to copy (`Q/mmvq.hip.cpp:357-548`). Also copy the
  split launch for single-token experts, and decoding a Q4 fragment once
  then dotting it with every token.
- C. **Down projection in expert order** (`quantize_q8_1_by_expert`): cheap to
  add, no kernel rewrite.
- D. **Non-expert verify cost (~2.0 ms/row)**: check that every dense GEMV
  processes all T rows in one pass over the weights. Use the reduce-scatter
  butterfly for multi-row reductions and DPP/permlanex16 instead of
  ds_bpermute. Gufo's bundle, which also includes up-front loads for few-row
  long-K matrices, is worth +6-7% (`D/EXPERIMENTS.md:105`).
- E. **GDN rollback**: compact operands instead of per-token full states
  (3.8). Alternative that also avoids gufo's ~0.5 ms/cycle row-0 snapshot:
  double-buffer the state, so verify reads A and writes B, and on rejection
  replay `keep` steps from A into B with stored operands. No full-state
  write on full acceptance.
- F. **Remove the per-draft host sync**: gufo syncs after each draft to read
  the token. Device-side argmax that feeds the next MTP step's embedding
  lookup in the same graph saves k syncs per cycle (we may already do this).
- G. **Fold MTP catch-up into the first proposal** (one MTP pass over
  accepted rows + anchor), if not already.
- H. **Controller**: gufo's survival model (per-depth discounted Beta counts,
  censored tails, probe x2 after a full accept, AR fallback with a 16-token
  retry) maximizes expected tokens / cost[L]. With our linear cost
  (29.2 + 9.4 d), the same argmax applies directly.

Plain decode (ours 34-36 tok/s, already ahead of gufo's 28):

- I. Producer-written Q8_1 rows and stacked same-input projections to cut
  quantize launches and passes (1.2, 4.2).
- J. Up-front loads (`_ahead<20>`, 2-step prefetch + `sched_barrier`) for
  shapes with too few rows to hide latency: HC down 320 x 10240, MTP inject
  4 x 10240 (1.3).
- K. Branch-free Q4_K scale reads, so a sunk load does not wait on every
  in-flight load (`Q/vecdotq.hpp:819-837`).
- L. Fuse the residual combine with the next RMSNorm and the Q8_1 write
  (`K/kernels.hip.cpp:647-740`).
- M. PLE disk gather overlap: split the graph at the PLE layer
  (`K/executor.cpp:2837-2865`), if our PLE rows are not already resident.
