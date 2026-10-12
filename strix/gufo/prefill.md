# gufo: how Qwen3.8-Flash-Next prefill reaches ~1,700 tok/s on gfx1151

This is a read-only study of `~/src-ref/gufo` (MIT). Nothing was built or run.
Path abbreviations used below:

- `X` = `src/models/qwen38_flash_next/kernels/rocm/executor.cpp`
- `XH` = `.../executor.hpp`
- `K` = `.../kernels.hip.cpp`
- `KH` = `.../kernels.hpp`
- `BL` = `.../blaslt.cpp`
- `DM` = `.../device_model.cpp`
- `E` = `src/models/qwen38_flash_next/engine.cpp`
- `D` = `docs/models/qwen3.8-flash-next/`
- `A` = `docs/models/qwen3.8-flash-next/artifacts/`

Model geometry, from `config.hpp:22-72`:

- 48 trunk layers, of which 36 are GDN. A layer is full attention when `(layer+1) % interval == 0`.
- hidden 2560, with 4 HC streams, so the residual is 10240 wide. HC low rank is 320.
- 24 query heads × 256, 2 KV heads × 256, rotary 64.
- Indexer: 4 heads × 128, top-k budget 2048 tokens, compress ratio 4 (a 4-token block), so 512 selected blocks.
- GDN: 16 K heads and 48 V heads, head dim 128, conv kernel 4, value dim 6144. The qkv conv channels are 2·16·128 + 48·128 = 10240.
- MoE: 512 experts, top-10, expert_ff 640, shared_expert_ff 640.
- Vocabulary 248320.
- PLE (n-gram embeddings) is injected at one layer only (`ple_layer`, config.cpp:186). It is not per layer.

---

## 0. Headline numbers and the measured breakdown (question 6)

Benchmarks (`D/BENCHMARKS.md:80-87`). These are AR pp at about 2048 tokens:

| Depth | pp tok/s |
| ---: | ---: |
| d0 | 1702.93 |
| d4K | 1624.56 |
| d8K | 1619.43 |
| d32K | 1556.91 |
| d128K | 1430.01 |

llama.cpp b11069 on the same machine gets about 490.

d0 pp2048 profile from 2026-09-21 (`D/EXPERIMENTS.md:100-106`):

- The run is GPU-bound: 1362.8 ms of kernel time in a 1380 ms span.
- Shares of kernel time:
  - routed expert GEMMs 35%
  - dense F16 projections 29%
  - HC combines 10.5%
  - GDN recurrence 9.5%
  - HC down/inject/shared projections 7%
  - attention 5.5%
- "The dense projections run at 30–34 TFLOPS against the measured 59 TFLOPS matrix ceiling and the combines at about 200 GB/s."
- "The routed gate/up and down kernels stream 0.9–1.1 GB of expert weights per layer at 160–190 GB/s."

A later d0 pp2048 profile (`A/prefill-normalization.json` "profile") measured gpu_kernel_ms 1195 and span 1198.9. Shares:

| Kernel group | Share |
| --- | ---: |
| routed_moe_f16 | 34% |
| dense_f16 | 29% |
| hc_combine | 10.7% |
| gdn | 8.7% |
| int8_dense | 7.4% |

Dispatch counts for one pp2048 pass (`A/prefill-long-context.json` → `unused_activation_and_routing_work/profile`):

| Variant | Dispatches | gpu_ms | wall_ms |
| --- | ---: | ---: | ---: |
| Baseline | 1629 | 1199.9 | 1204.1 |
| Candidate | 1533 | 1184.6 | 1188.6 |

That is about 32 kernels per layer and more than 99.6% GPU-busy. Host and launch overhead is negligible.

There is a d128K pp2048 profile (`A/prefill-final-rows.json` "baseline_profile_128k_ms"). Total kernels 1382.5 ms, span 1388.4.

| Kernel group | ms |
| --- | ---: |
| routed_moe | 390.1 |
| dense_f16 | 374.3 |
| sparse_attention | 126.5 |
| hyperconnection_residual | 143.8 |
| gated_delta_net | 110.6 |
| dense_int8 | 85.8 |
| selector_scoring | 78.0 |
| exact_selection | 17.8 |

There is also a d32K split: MoE 29.1%, dense projections 34.9%, attention/indexing 12.6% (`D/EXPERIMENTS.md:96-98`).

Per-layer arithmetic at d0 pp2048, from the 1195 ms profile:

- routed MoE ≈ 406 ms / 48 ≈ 8.5 ms per layer for 2048 tokens
- dense F16 ≈ 347 ms
- HC combine ≈ 128 ms
- GDN ≈ 104 ms (≈ 2.9 ms per GDN layer)
- int8 dense ≈ 88 ms

Microbenchmarks:

- GDN recurrence at 4096 rows: 3.50 ms per layer (old 4.61). At 512 rows: 0.42 ms (`A/prefill-long-context.json` microbenchmarks.gdn_rows_ms).
- SSM out-projection at 4096 rows: 3.20 ms (ssm_padded_projection_ms).
- Dense shapes at 2048 tokens, W8A8 → F16 WMMA ms (`X:108-115`):

| Shape | W8A8 ms | F16 ms | Route kept |
| --- | ---: | ---: | --- |
| 16384×2560 | 5.7 | 4.6 | F16 |
| 13312×2560 | 4.3 | 3.9 | F16 |
| 10240×320 | 1.02 | 0.79 | F16 |
| 2560×2560 | 0.81 | 0.74 | F16 |
| 2560×640 | 0.24 | 0.21 | F16 |
| 2560×6144 | 1.8 | 2.1+ | int8 |
| 640×2560 | 0.19 | 0.23 | int8 |
| 320×10240 | 0.40 | 1.3 | int8 |

- Fused attention QKV projection at 4096 tokens: 8.8 → 7.9 ms from row-tile grouping, then 7.3 ms from BK=1 for 3 blocks per WGP (`K:6159-6163`).
- SSM out-projection launch ordering: 5.7 → 3.4 ms at 4096 tokens (`K:6260-6264`).
- hipBLASLt F16 on the same dense shapes reaches only 19–26 TFLOPS. gufo's Q8→F16 WMMA kernels reach 30–34 (`D/EXPERIMENTS.md:87`, `tools/qwen-flash/dense_blaslt_sweep.hip`).

Rough FLOP accounting (mine). A GDN layer is about 126 M MAC per token:

- ssm_in 42M, ssm_out 15.7M, HC 2×(3.3+3.3)M
- routed 10×3×640×2560 = 49M
- shared expert 4.9M, router 1.3M

48 layers come to about 12 GFLOP per token. At 1700 tok/s that is about 20 TFLOPS effective. At 780 tok/s it would be about 9.4 TFLOPS.

---

## 1. Prefill chunking, overlap, streams/graphs, host overhead

**Chunk size.** `kPrefillChunkTokens = 4096` and `kPrefillTailTokens = 128` (`E:23-32`).

- `PrefillCapacity` is 4096 (`E:204-206`).
- `max_batch = PrefillThroughCapacity()` = 4224 (`E:145`, `E:208-211`).
- So any prompt of 4224 tokens or fewer runs as one forward. pp2048 is a single 2048-row forward.
- Rationale (`E:23-28`): 4096-token chunks prefill 24K–102K prompts 6–11% faster than 2048. pp4096 is +5% at d0 and +6% at d32K.
- With an 8192-token chunk, results stayed within run-to-run drift (no gain) and it costs about 3 GiB more scratch (`D/EXPERIMENTS.md:21`).

**Split loop** (`Session::Feed`, `E:597-653`).

- Chunk n is `min(capacity, remaining)`.
- It computes `next`, the following chunk, and passes it to `Executor::Forward(... ForwardMode::kPrefill, checkpoint, next)` (`E:624-632`).
- One logit row is requested per chunk.

**Prefill arithmetic is chunk-width independent.**

- `thread_local prefill_phase` makes `MatrixRows()` and `ExpertMatrixRows()` true for any row count (`X:79-97`).
- So a short prefill tail still uses the wide GEMM, WMMA expert and attention kernels.
- In the "dense split at the indexer budget", queries below position 2048 always take dense tiles. This keeps logits independent of batch size (`X:2370-2396`; `D/EXPERIMENTS.md:79`).

**N-gram (PLE) prefetch** (`X:1994-2125`, `XH:529-537`, `XH:668-686`).

- `PleFetch` hashes the batch's n-gram rows on the CPU and starts asynchronous `pread`s into `host_emb_`. That buffer is coherent pinned memory that the GPU reads directly, with no upload ("Direct mapped n-gram input", `D/EXPERIMENTS.md:16`).
- The forward enqueues layers 0..ple_layer-1. Only `Ple()` waits for the rows (`X:2092`, `WaitPle`).
- Once this chunk's rows are in, `Ple()` immediately starts `PrefetchPle(next)` (`X:2095-2098`). That read overlaps the rest of this chunk and the next chunk's early layers. Prefetched rows are claimed only when tokens and n-gram history match exactly (`X:1999-2007`).
- 128 reader threads with queue depth 128 serve the reads (`D/EXPERIMENTS.md:93`). Their measured gain is 6–7% at 5K–8K prompts, mostly time-to-first-token.

**Streams and graphs.**

- There is a single HIP stream `stream_` (`X:957-959`). Every prefill kernel is launched eagerly on it.
- `Run()` captures or replays a hipGraph only when `graph = !prefill_phase && n <= kVecBatch(8) && ...` (`X:2826-2831`), so prefill is never graphed.
- Second-stream experiments were all rejected:
  - Shared-expert and inject on a side stream ran 0.5–0.9% slower (`D/EXPERIMENTS.md:88`).
  - SSM gate rows on a second stream beside the GDN recurrence slowed the recurrence from 3.75 to 6.76 ms per layer (`D/EXPERIMENTS.md:21`).

**Host syncs inside a forward.** There is exactly one per MoE layer.

1. The router writes ids, `ExpertCounts` runs, and a D2H `hipMemcpyAsync` of 512 counts is followed by an event record (`X:2443-2453`).
2. The shared-expert GEMMs are enqueued after that copy (`X:2455-2473`). The GPU therefore keeps working while the host waits.
3. `RouteHints` calls `hipEventSynchronize(counts_ready_)`, builds an exact tile list on the CPU, and uploads it with `hipMemcpyAsync` (`X:1640-1707`).

The profile shows this costs essentially nothing (span ≈ kernel time).

**Launch count** is about 1533 per pp2048, roughly 32 per layer; see §2 for the per-layer list. Experiments that removed 96 launches gave about +0.4% (`D/EXPERIMENTS.md:8`).

**Final-layer pruning** (`X:2933-2992`). In AR (no MTP) with `n_logits == 1`, the last layer's FFN and MoE run only on the final aligned tile, `(n-needed)/128*128` rows skipped with needed ≥ 96. Its attention computes only the last query tile (`last_only`). This gained +1.2% at d0 (`A/prefill-final-rows.json`).

**MTP mode.** After each prefill chunk the predictor runs `MtpForward(kv_only)` over the chunk (`X:2875-2883`). This is not part of the AR numbers.

---

## 2. Per-op kernels in one prefill layer (n ≥ 1024, e.g. pp2048)

### Tensor types in UD-Q4_K_XL, as the code requires them

- **Q8_0** (`X:1538-1543`, `DM:159-232`): all dense matrices.
  - GDN ssm_qkv and ssm_gate, stacked at load into `ssm_in` 16384×2560.
  - ssm_out 2560×6144.
  - attn_q/k/v, stacked into `attn_qkv` 13312×2560: q and gate interleaved per head as 24×2×256 = 12288 rows, plus k 512 and v 512.
  - attn_out 2560×6144.
  - HC mixer down 320×10240 and up 10240×320, for both hc_attn and hc_ffn.
  - shared expert gate/up 640×2560 and down 2560×640.
  - Token embedding, output head (the "Q8 vocabulary head"), and probably PLE key/value.
- **F32 in the GGUF, narrowed to F16 at load:** router + shexp_gate_inp, stacked into 513×2560, and ssm_alpha + ssm_beta, stacked into 96×2560 (`DM:155-232`, `Stack()`).
- **BF16:** indexer_q (512×2560) and indexer_k (128×2560). This is inferred from the BF16 handling in `BL:118-129` and `X:1611-1622`.
- **Routed experts:** gate/up are Q4_K or Q5_K (UD mixes per layer). Down is Q5_1 or Q8_0 (`X:2509-2515`, `D/EXPERIMENTS.md:42,84`).
- Q6_K is rejected outright (`DM:301-310`).

### Order of ops per layer

From `ForwardBody`, `X:2913-2994`:

```
[PLE at ple_layer only]
HcMix(hc_attn, normed=true)   -> mixed (+ F16 copy + tiled-Q8 copy)
LinearAttention | Attention   -> block_out
Combine(res, hc_ffn.norm)     -> res += ..., xn F16 + xn tiled Q8
HcMix(hc_ffn, normed=true)
Moe                            -> (epilogue deferred)
Combine(res, next layer's hc_attn.norm)  [MoE epilogue fused here]
```

### 2a. HC mix (hyper-connections)

Implemented by `HcMix`, `X:1859-1992`. The previous `Combine` has already produced the grouped RMSNorm of `res` (4 groups of 2560) as F16 rows `xn_half` and as tiled Q8 `xn_q8t`. That is the `normed=true` / `xn_half_` route, enabled by `wide_mixer_`, `X:1161-1163`.

**HC down: `HcDownF16Gemm`** (`K:5308-5316`).

- `W8A8BlockedWmmaGEMMKernel<BM=64,BN=128,BK=4,WM=2,WN=4,kHcDown=true>`, int8 WMMA. Grid is (tokens/128, 5) over m=320, K=10240.
- The epilogue fuses `SiLU(x/4)` and writes F16 (`K:4304-4313`, `HcScaledSilu` `K:4066-4072`).
- This shape (320×10240) is the one where int8 wins: 0.40 vs 1.3 ms (`X:113-115`).

**HC up + mix: `HcMixF16Gemm`** (`K:6172-6194`).

- `DenseF16GEMMKernel<256,128,BK=1,WM=4,WN=2,kRowGroup=8,kHcMix=true>` over Q8_0 10240×320. The A operand is F16 low-rank, and K=320 is tiny.
- The weight rows are interleaved in-kernel so 4 consecutive tile rows are the 4 streams of one hidden element (`K:5646-5651`).
- The epilogue (`K:5942-6033`) computes `mixed = 0.25 * Σ_s xn[s] * sigmoid(gate[s])`. It writes F32 mixed (optional), F16 mixed into `x_half`, and tiled-Q8 mixed into `x_q8t`. These are the next projections' staged inputs.
- A separate `HcMixEpilogueF16Kernel<false>` computes the 4 inject logits, as F32 dot products of inject_w with xn, in partial sums (`K:6187-6192`).
- `half_only` skips the F32 and Q8 copies when only the F16 copy is consumed. That is the GDN layer, whose ssm_in and alpha/beta both read F16 (`X:2921-2926`).
- **Launches: 3.**

### 2b. GDN (linear-attention) layer

Implemented by `LinearAttention`, `X:2127-2227`.

**In-projection qkv + z, with the conv fused.**

- `DenseF16SsmGemm` (`K:6290-6313`): `DenseF16GEMMKernel<256,128,BK=2,WM=8,WN=1,kRowGroup=2,kSsmConv=true>`.
- Q8_0 weights are dequantized to F16 while committing to LDS. Activations are the F16 `x_half` rows. The kernel uses F16 WMMA (`v_wmma_f32_16x16x16_f16`) with F32 accumulate over the whole K.
- The epilogue transposes results through LDS in 32-token × 32-row tiles. For the 10240 conv channels it computes the 4-tap causal conv + SiLU in-LDS from the 3 previous token rows, writing `conv_out` directly (`K:6079-6120`).
- Raw qkv rows are written only at tile boundaries: the first and last 3 tokens of each 32-token tile, the chunk tail, and checkpoints. z rows are always written (`K:6091-6095`).
- `SsmConvBoundaryKernel` then fills the first 3 tokens of each tile, using the previous chunk's history for t<3 (`K:5512-5535`).
- Condition: `n >= 1024 && !speculative` (`X:2143-2151`).
- **Launches: 2.**

**alpha/beta.**

- `Dense(ssm_alpha_beta)` uses F16 weights with m=96, which routes to `BlasLt::Gemm` → `UnquantizedF16Gemm` (`BL:137-145`).
- That is `DenseF16GEMMKernel<64,64,2,2,4,...,kHalfWeights=true>` (`K:6196-6206`), with two separate K16 accumulator chains for precision (`K:5760-5763`, `K:5825-5827`).
- It reads the cached `x_half` (`X:1613-1622`, `half_src_` cache).
- **Launches: 1.**

**Conv history.** `HistoryShiftKernel` + `CopyKernel` update the rolling 3-row state (`K:6497-6504`). **Launches: 2.**

**Gates.**

- `GdnPrepKqKernel` computes per (token, k-head) `rsqrt(|k|²+eps)`, `rsqrt(|q|²+eps)/sqrt(d)`, and `k·q` (`K:1380-1410`).
- `GdnPrepAbKernel` computes `decay = exp(a·softplus(alpha+dt))` and `beta = sigmoid(beta)` (`K:1412-1426`).
- q and k are not normalized in memory. The scan applies the scalar factors.
- **Launches: 2.**

**Scan: `GdnRowSplitKernel`** (`K:1442-1706`, launched at `K:6516-6519`).

- **Sequential over tokens.** It is NOT chunked (no WY/UT transform) and does NOT use WMMA. State, arithmetic and state storage are all FP32.
- Grid is `(128/64 = 2, v_heads = 48)` = 96 blocks of 128 threads (4 waves).
- Each block owns 64 state rows of one head. Each lane owns 2 rows × 32 keys in registers (`float4 s[2][8]`). 4 lanes cover a 128-wide row, and reductions use DPP xor-adds (`K:1428-1440`).
- q/k/v/scales/decays for 4 tokens are staged through LDS per "window", with 3 windows of global loads in flight in named registers (`K:1549-1595`).
- Per token: decay the state, compute `u = S·k` and `p = S·q`, then `delta = (v - u·inv_k)·beta`, `out = (kq·update + p)·q_scale`, and `S += update·k`.
- Measured 3.5 ms per layer at 4096 rows, which is about 0.85 µs per token per layer.
- The "two rows per lane / 4 waves" layout gave a 24% speed-up over one row per lane (`A/prefill-long-context.json`), adapted from the mikealanni fork.
- **Launches: 1.**

**Norm/gate epilogue.**

- `GdnEpilogueKernel` applies the RMSNorm × norm_w × SiLU(z) gate.
- It writes F16 rows directly, padded to stride 6144+32 to avoid a 12 KiB pitch (`X:2177-2185`, `K:6531-6537`).
- **Launches: 1.**

**Out-projection.**

- `DenseF16Gemm(..., input_stride)` with `DenseF16GEMMKernel<256,128,2,8,1,kRowGroup=5,...,kStrided=true>` (`K:6227-6237`).
- Q8_0 2560×6144 runs as F16 WMMA. "5 row tiles of a token tile first" shares each activation tile in L2: 5.7 → 3.4 ms.
- **Launches: 1.**

**GDN layer total:** about 10 launches for the mixer body.

### 2c. Full-attention layer

Implemented by `Attention`, `X:2229-2430`.

**Q/K/V/gate projection, fully fused.**

- `AttentionF16Gemm` (`K:6146-6170`): `DenseF16GEMMKernel<256,128,BK=1,WM=8,WN=1,kRowGroup=4,kAttention=true>`. The grid is (tokens/128, 52 head tiles).
- Each 256-row tile is exactly one q head, gate head, k head or v head.
- The epilogue (`K:5856-5940`) does per-head RMSNorm (q/k), NEOX RoPE on 64 dims, writes q and gate as F32, and **writes K and V straight into the F16 KV cache** at `position + tok`.
- So there are no separate norm, RoPE or StoreKv launches.
- Condition: `n >= 1024` and the exact geometry (`X:2244-2260`). Otherwise it uses Dense + `PrepareAttention`.

**Indexer K.**

- `Dense(indexer_k)` with BF16 m=128: `NarrowActivations(x → bf16)`, then `BlasLt::Gemm`.
- If hipBLASLt picks algo 4438 (MT32x32x64), gufo substitutes its own `DenseBf16Gemm` instead. That kernel is about 4× faster and reproduces the library's K order bitwise (`BL:118-129`, `K:6208-6219`).
- `StoreRows` writes the raw F32 keys to a power-of-two ring (`X:2303-2311`).

**Sparse selection.** This only happens when `start_pos + n > 2048` (`X:2821`), so at d0 pp2048 there is none.

- `indexer_q`: BF16 512×2560 GEMM, then RmsNorm, then RoPE.
- `PoolIndexerBlocks`: mean over 4 raw keys, RMSNorm, RoPE at block start, F16 block keys (`KH:419-431`).
- `SelectBlocks` runs in query chunks of at most 512 (`X:2337-2356`):
  - `SelectScoreKernel` (`K:2341-2391`) is scalar FP32 FMA, with one thread per (query, block). It scores against all 4 heads with `relu(dot)` summed. It is not WMMA, because they keep FP32 queries so top-k boundaries don't flip.
  - `SelectMarkKernel` (`K:2441-2596`) is an exact top-512 per query: a sampled-window 4096-bin LDS histogram, then 8-bit radix refinement, then a ballot tie-break. The output is a visibility bitmap.
- At 128K: scoring 78 ms and selection 18 ms per pp2048 (§0).

**Attention core: `WmmaCausalAttentionKernel`** (`K:3360-3960`, launcher `K:6693-6738`).

- F16 WMMA flash-style: 16-key tiles, online softmax, PV, and the **sigmoid output gate fused**.
- `__launch_bounds__(256, 2)`.
- **Dense window** (queries below 2048): 16 queries × 2 heads per block, grid (n/16, 12).
- **Sparse window:** 4 queries × 12 heads (one KV group) packed into 3 row tiles. Key tiles are gathered from the union of those 4 queries' selected blocks, compacted in LDS (capacity 4·512+4) with per-query membership bits. At 16K depth the union is 846 blocks vs 2478 for 32 queries.
- The next tile's K/V is prefetched to registers behind the S/softmax/PV phases.
- Q is converted to F16 pre-scaled by 1/16. K and V come from the F16 cache.
- "Tiles cut across selection windows / 64-block carry" gave +6.7% AR at 32K and +16.5% at 128K (`D/EXPERIMENTS.md:89`).

**Out-projection.**

- `Dense(attn_out)`: Q8_0 2560×6144 has K > 2560, so it is not on the F16 route.
- `QuantizeQ8TiledVec4Kernel(ctx)`, then `W8A8Gemm` → `W8A8GemmWave64` (`K:5325-5328`). That is the shared Qwen27B wave64 int8 kernel `WKQuantA8BlockedWmmaGEMMKernel<128,128,2,4,1,Q8_0,64>` (`w8a8_wave64.hip`).
- So for 2560×6144 they chose int8 W8A8 (1.8 ms vs 2.1+ ms F16 at 2048 tokens).
- Note that the GDN ssm_out of the same shape takes F16 with padded F16 input instead (`X:2177-2211`), because its producer can emit F16 directly.

### 2d. Combine (between mixer and FFN)

`HcCombineF16` (`X:1840-1844`; `KH:108-116`; `HcCombineVec4Kernel`, `K:748`) does all of the following in one pass per token:

- `res[t][s][:] += block_out[t][:] * 2·sigmoid(inject[t][s]/4)` over the 4 streams
- the next mixer's 4-group RMSNorm × gamma
- writes the norm as F16 (`xn_half`) and as tiled Q8 (`xn_q8t`) for the next HC down W8A8

**Launches: 1.** HC combine is 10.7% of time and runs at about 200 GB/s.

### 2e. MoE

Implemented by `Moe` (`X:2432-2495`) and `MoeExperts` (`X:2497-2611`).

**Router + shared-expert gate.**

- One stacked F16 GEMM of 513×2560 runs on the own `UnquantizedF16Gemm` (`BL:137-145`). It reads the cached `x_half` written by the HC mix epilogue.
- `RouterTopKKernel<512>` does softmax over 512 and an iterative top-10 with wave argmax, then renormalizes (`K:2918-2998`). **Launches: 2.**

**Counts.** `ExpertCounts` writes counts, followed by a D2H copy and an event (`X:2443-2453`).

**Shared expert** (`GatedDense`, `X:1470-1522`).

- gate and up are Q8_0 640×2560, with rows < 2048, so they take the int8 route. `W8A8Gemm<128,128,2,4,2>` reads `x_q8t`, which the mix epilogue already wrote, so there is no quantize pass.
- `SwigluHalf` writes F16 into a private `shexp_half` (`X:1500-1509`).
- down (2560×640) uses `DenseF16Gemm<256,128,1,4,2,kRowGroup=2>` (`K:6272-6275`).
- **Launches: 4.**

**Host tile map.** After `hipEventSynchronize`, the CPU builds `(expert | tile<<16)` lists (`X:1640-1707`):

- the 48-row tile list for the down projection
- a 64-row list
- if `tiles_128·4 ≤ tiles_64·3`, a 128-row list for gate/up

`routed_tile_rows_` = 48 when slots ≥ 16·512 (n ≥ 820), else 16 (`X:119-129`). The list is uploaded with `hipMemcpyAsync`. Only non-empty tiles launch.

**Compaction** (`RoutedCompact`, `K:5362-5374`).

- `RoutedPadBoundsKernel` (1 block) does an exclusive scan of 16-padded counts and clears padding rows.
- `RoutedScatterKernel` uses atomics to fill `rows_token[c]` and `rows_slot[c]`.
- This only builds index maps. **No activation gather or copy happens.** **Launches: 2.**

**Routed gate+up with SwiGLU fused.**

- `RoutedGatedF16Gemm` → `RoutedF16GEMMKernel<Q4_K|Q5_K, BM=128, BN=64|128, BK=2, kPair=true>` (`K:5454-5498`, kernel `K:4370-4990`).
- The grid is (640/64 = 10, n_tiles). Each block computes 64 gate rows (waves 0-3) and the matching 64 up rows (waves 4-7) for one expert × token tile.
- Activations are read straight from the F16 `x_half` token rows via `rows_in`, so the gather is fused into the activation fetch (`K:4547-4567`).
- Weights stay packed in LDS: 4-bit codes plus a per-(row, K32) half2 (scale, bias) word. They are decoded to F16 after the LDS read, per wave and only for its own 16 rows, using the 1024-magic byte-permute trick (`K:4396-4436`, `K:4728-4802`).
- So LDS per stage is 11–12 KB and 5 blocks are resident per WGP (`K:4370-4377`).
- The paired variant keeps the whole 144-byte Q4_K superblock's codes in registers across 4 stages, selected with explicit `v_cndmask` (`K:4612-4637`). That was worth +0.7% (`D/EXPERIMENTS.md:94`).
- The epilogue pairs gate and up in LDS and computes `silu(g)·u`, written as F16 at `rows_out` (`rows_slot`) (`K:4848-4900`).
- Launch condition: `n >= 1024 && tile_rows == 48` (`X:2539-2545`). Otherwise there are two `RoutedF16Gemm` launches with SwiGLU in the second epilogue.
- **Launches: 1.**

**Routed down.**

- `RoutedF16Gemm<Q5_1|Q8_0, BM=128, BN=48|64, BK=2>`. The grid is (2560/128 = 20, n_tiles).
- It reads the F16 SwiGLU rows by `rows_slot` and writes F16 `down_e[token·10+slot][2560]` (`X:2560-2580`, `K:4903-4951`).
- The 64-row tiles are used when they don't add more than 1/3 padding.
- **Launches: 1.**

**MoE reduce.** This is deferred (`moe_pending_`, `X:2592-2601`) into the next `Combine`. `HcCombineMoeF16Kernel` (`K:876-1017`) does all of the following without ever materializing block_out:

- forms `Σ_k w_k·expert_out[t,k] + sigmoid(gate)·shared`
- applies the HC residual update for the 4 streams
- computes the next layer's 4-group RMSNorm
- writes xn as F16 and tiled Q8

**Launches: 1.**

Measured weight streaming: 0.9–1.1 GB per layer at 160–190 GB/s, about two-thirds of DRAM bandwidth (`D/EXPERIMENTS.md:84,104-106`).

**Rejected MoE variants** (`D/EXPERIMENTS.md:42,84`; `A/prefill-normalization.json` "rejected"):

- 96-row and mixed expert tiles (−1.1%)
- 256-row down blocks
- streaming loads (−20%)
- expert-ordered intermediates
- the four-phase loop
- the int8 `moe_gemm_i8_bench.hip` prototype (see §5)

### 2f. Final norm and head

- Only the requested logit rows (1 per chunk) run: `PrefillPhase head_phase(false)`, so the decode GEMV path, then `HcMix(hc_head)` and `Dense(output)` as a Q8 GEMV (`X:3034-3043`).
- The "skip unused final HC normalization" change was neutral (`D/EXPERIMENTS.md:25`).

### Per-layer launch tally

| Layer type | Launches |
| --- | ---: |
| GDN layer | about 3 (mix) + 10 (GDN) + 1 (combine) + 3 (mix) + about 12 (MoE) + 1 (combine/MoE) ≈ 30 |
| Attention layer (dense window) | about 3 + 5–7 + 1 + 3 + 12 + 1 ≈ 27 |

This matches 1533 dispatches per 48 layers.

---

## 3. Activation quantization and narrowing

gufo prefill never uses ggml q8_1 / MMQ. `qfn_mmq_*` (the vendored llama.cpp MMQ in `mmq/`) is only used for:

- decode/verify (≤ 8 rows)
- fallback paths, such as a non-Q4_K/Q5_K expert layout

See `X:1426-1450`, `X:1710-1755`, and `mmq/VENDOR.md`.

For n > 8 or in prefill phase, `Quantize()` returns early: "the tiled path quantizes per call" (`X:1432-1434`).

There are two staged activation forms for wide batches:

1. **F16 rows** (`s_.x_half`), for every F16 WMMA GEMM. These are the dense Q8_0 projections with K ≤ 2560 and rows ≥ 2048, the routed experts, the router and alpha/beta, and BF16 for the indexer.
2. **"Tiled Q8"** (`s_.x_q8t`, `xn_q8t`), for int8 WMMA (`K:3960-3970`, `K:4030-4062`).
   - Per 32-wide block, the absmax scale is in FP32 (not q8_1's half d+s), with no sum term.
   - It is stored directly in WMMA B-fragment order: tile(16 tokens, kb) = [16×16 B lo | 16×16 B hi | 16 FP32 scales] = 576 B (`K:447-460`).
   - The buffer is padded to 128-token macro tiles (`K:5283-5291`).

**Producers write these forms directly (fused quantization/narrowing):**

- `HcCombineF16` and `HcCombineMoeF16` write xn as F16 plus tiled Q8 (`K:876-1017`).
- The HC mix epilogue (`HcMixF16Gemm`, `HcMixEpilogueVec4F16`) writes mixed as F16 into `x_half` plus tiled Q8 into `x_q8t` (`K:5991-6027`, `X:1932-1960`).
- The GDN epilogue writes F16 padded rows (for ssm_out), or tiled Q8 / Q8_1 (`K:6531-6537`).
- `SwigluHalf` writes F16; `SwigluQ8Tiled` writes tiled Q8 (`X:1500-1517`).
- The routed gate/up epilogue writes F16 SwiGLU rows, the down GEMM's input. The routed down writes F16 rows, read by the fused combine.
- The HC down epilogue writes F16 `SiLU(x/4)`, the HC up's input.

**The cache** (`q8t_src_/rows/cols`, `half_src_/rows/cols/bf16`, `XH:705-719`) lets a later projection over the same source skip its conversion. Examples:

- router and alpha/beta both reuse the `x_half` the mix wrote
- shared-expert gate/up reuse `x_q8t`

**Only remaining standalone conversion passes per layer:**

- `QuantizeQ8TiledVec4Kernel` for attention ctx → attn_out
- `NarrowActivations` → BF16 for the indexer projections, attention layers only

---

## 4. Load-time precomputation

From `DM` as a whole. There are no repacks, transposes or dequantized F16/BF16 copies of quantized weights. Weights are DMA'd in their GGUF byte layout. Each tensor gets a 4096-byte zeroed tail margin for over-reads (`DM:17-23`, `DM:72-110`).

The only load-time transforms are:

- **Row-stacking of tensors that share an input** (`DM:155-232`), with Q8_0 kept as is:
  - `ssm_in` = ssm_qkv ‖ ssm_gate (16384×2560)
  - `attn_qkv` = q ‖ k ‖ v (13312×2560)
- **F32 stacks narrowed to F16 on the GPU** after upload, with the F32 source freed (`DM:204-221`, `DM:405-421`):
  - `router` = router ‖ shexp_gate_inp (513×2560)
  - `ssm_alpha_beta` = alpha ‖ beta (96×2560)
- **The MTP eh_proj split** into two halves (`DM:111-153`).

Rejected alternatives:

- "Persistent packed Q8 SSM weights" and "Transient F16 SSM weights" (`D/EXPERIMENTS.md:58-59`).
- "Pinned weights and aligned Q8 head layouts", which would have added 644 MiB (`D/EXPERIMENTS.md:17`).

In-kernel tricks replace repacking:

- The HC-mix GEMM interleaves the 4 stream gate rows via an index remap (`K:5646-5651`).
- The F16 attention GEMM maps each 256-row tile to a head type (`K:5862-5867`).

Scratch is sized once for max_batch = 4224 (`X:942-1100`). The 4096 chunk adds about 1.3 GiB at 262K context (`D/EXPERIMENTS.md:80`).

---

## 5. GEMM tile shapes, WMMA variants, LDS and occupancy

All of these kernels are wave32 with 256 threads (8 waves) unless noted.

**F16 WMMA** uses `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` (`K:2311-2313`). BF16 uses `..._bf16_w32` (`K:2318-2321`).

**Int8 WMMA** uses `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` (`K:3980-3983`). The wave64 variant is used only for attn_out (`w8a8_wave64.hip`).

**`DenseF16GEMMKernel<BM,BN,BK,WM,WN,kRowGroup,...>`** (`K:5552-6144`).

- Q8_0 codes and the scale are fetched one stage ahead into registers. At LDS commit they are dequantized to F16, so the LDS holds F16 weights: `code ^ 0x80` → `byte_perm` into the mantissa of 1152 → `hadd2(-1152)` → `hfma2(scale)`, exact (`K:5707-5743`).
- Activations are F16 rows copied as is.
- LDS layout is `[BK][rows][4 × 16 B chunks]`, with a chunk XOR swizzle `c ^ ((row>>1)&3)` that avoids bank conflicts (`K:5586-5597`).
- The inner loop does 2 WMMAs (lo/hi K16) per (row tile, token tile) per K32 block. A `sched_barrier` keeps the next-stage loads above the WMMAs (`K:5776-5837`).
- The epilogue transposes through LDS so every global store is a full 128 B line of 32 rows for one token (`K:6035-6143`).
- LDS is `BK·(BM+BN)·64 B`:
  - 256×128, BK=2: 48 KB, which gives 2 blocks per WGP
  - BK=1 variants: 24 KB, which gives 3 blocks per WGP (`K:6159-6163`)
- About 221 VGPRs (`D/EXPERIMENTS.md:86`).
- `kRowGroup` changes launch order only: N row tiles of the same token tile are adjacent in dispatch so the activation tile is shared in L2. This is worth up to 40% on ssm_out.

Instances:

| Use | BM×BN, BK | WM×WN | kRowGroup | Notes |
| --- | --- | --- | ---: | --- |
| ssm_in (fused conv) | 256×128, BK2 | 8×1 | 2 | |
| attn_qkv (fused norm/RoPE/KV) | 256×128, BK1 | 8×1 | 4 | |
| ssm_out (strided F16 input) | 256×128, BK2 | 8×1 | 5 | |
| other 16384/13312×2560 | 256×128, BK2 | 8×1 | – | |
| hc up 10240×320 | 256×128, BK1 | 4×2 | 8 | |
| shexp down 2560×640 | 256×128, BK1 | 4×2 | 2 | |
| default | 256×128, BK1 | 4×2 | – | |
| m ≤ 512 | 64×128, BK2 | 2×4 | – | |
| < 96 tokens | 128×64, BK4 | 4×2 | – | |
| router / alpha-beta (F16 weights) | 64×64, BK2 | 2×4 | – | dual K16 chains |
| indexer (BF16) | 64×128, BK2 | 2×4 | – | K-rotation to match hipBLASLt bitwise |

So a wave owns 2×4 or 4×2 16×16 tiles (64×64 per wave on the wide route). The comment at `K:6250-6251` says 64×64 wave tiles were chosen because F16 fragments are twice the bytes of int8.

**`W8A8BlockedWmmaGEMMKernel<BM,BN,BK,WM,WN>`** (`K:4074-4339`). Ported from the Qwen27B route.

- Weights are raw 34-byte block_q8_0, staged as int8 into LDS, with dw scales per (row, kb).
- Activations are tiled Q8.
- Per K32 block: 2 iu8 WMMAs into a fresh int32 accumulator, then `acc += (dw·dx)·float(c)` across 8 values. That is the per-block float epilogue (`K:4241-4258`).
- LDS for 128×128 BK=2 ≈ 18 KB.
- Instances:
  - 128×128 BK2 4×2 (wide)
  - 64×128 BK4 2×4 (m ≤ 512, HC down)
  - 128×64 BK4 4×2 (< 96 tokens)
- clang 23 needs a `sched_barrier` to avoid spills (`K:4260-4265`).

**`RoutedF16GEMMKernel<type,BM=128,BN∈{16,48,64,128},BK=2,kPair>`** (`K:4438-4990`).

- Codes are packed in LDS (Q4_K: 2×16 B per row and stage; Q5_1: + high-bit word; Q8_0: 4×16 B), plus a half2 (scale, bias) per row and kb.
- Activations sit in LDS as `[kb][quarter][token (+1 pad)][16 B]`.
- Each wave decodes its own 16 rows to F16 in registers, then runs F16 WMMA over all token tiles. Token tiles beyond the bucket are skipped (`live_tok_tiles`).
- Stage LDS is 11–12 KB, giving 5 blocks per WGP (`K:4376-4377`).
- The wider BN amortizes decode over more tokens. That is why 48/64/128-row tiles are used when buckets are full.

**Why F16 WMMA instead of int8 for dense and experts.** The authors' rationale is at `K:4370-4377` and `X:108-115`.

- On gfx11, iu8 WMMA has the same 16×16×16 per-instruction rate as f16, so int8 buys no matrix throughput.
- The int8 path must apply 2 scales (and, for K-quants, a min/sum correction) per K32 block in VALU, which competes with the WMMA issue.
- F16 with dequant-on-stage accumulates the whole K in F32 with zero per-block epilogue.
- int8 is kept only where K is long and M small (HC down 320×10240, shexp gate/up 640×2560) or K=6144 (attn_out). There, F16 activation rows would not stay cache-resident.

Measured efficiency is 30–34 TFLOPS out of 59 for dense F16. An int8 Q4_K prototype (`tools/qwen-flash/moe_gemm_i8_bench.hip`) "was meant to replace" the vendored MMQ tier but did not survive. Production experts are F16.

**Rejected tile/occupancy experiments:**

- LDS double-buffer prefetch: 221 → 253 VGPRs, with spills, and 3–14% slower (`D/EXPERIMENTS.md:86`).
- 64-token HC down tiles: 23% slower (`D/EXPERIMENTS.md:85`).
- Four-wave Q8 reduction lost to eight (`D/EXPERIMENTS.md:63`).
- SSM wave/tile variants (`D/EXPERIMENTS.md:72-73`).
- hipBLASLt F16 for dense: 19–26 TFLOPS (`D/EXPERIMENTS.md:87`).

---

## 6. Additional observations relevant to the comparison

- **Exactness discipline.** Many kernels insert `asm volatile("" : "+v"(x))` to pin FP order (`K:5890-5906`, `K:4877-4884`). This costs little and is not a speed lever.
- **Router precision.** The router and alpha/beta GEMMs use own kernels with two K16 chains, not hipBLASLt, because library edge tiles change the order and errors amplify (`BL:134-145`). This was done for correctness, not speed.
- **Depth scaling.** d0 → d128K drops only 1703 → 1430 tok/s, because:
  - the sparse selector is bounded (`select_score_floats_`, chunks of 512 queries)
  - attention cost scales with the union of 512-block selections, not the context
  - pre-budget queries use dense tiles
- **MALL / L2 awareness.** Strix Halo has a 32 MiB MALL. Launch-order grouping (`kRowGroup`) and "keep the two KV heads of nearby queries together" (`K:3435-3439`) exploit it.
- **No CPU work on the critical path.** The CPU only hashes n-grams, builds the expert tile map and enqueues work. GPU span ≈ kernel time.
