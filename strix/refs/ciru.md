# CIRU "Qwen3.8-Flash-CIRU-STRIX-IU4" — read-only study

Repo: `~/src-ref/Qwen3.8-Flash-CIRU-STRIX-IU4` (MIT, llama.cpp derivative, single squashed commit `bac7ad8`).
All paths below are relative to that root. Nothing was built or run.

## 0. What this repo actually is

- **CIRU is mostly an integrator, not a kernel author.** Since v4.0 the engine is pwilkin's Strix Halo
  llama.cpp reference (commit `f5daaa3c…`), which in turn reproduced Halogen's fast-prefill result.
  CIRU's own contribution is "the compatibility integration and validation" (README.md:55,
  docs/PROVENANCE.md:5-7, CIRU_RELEASE.json:137-147). So most kernels (`mmb.cu`, `hc-*.cu`, `qsa.cu`,
  GDN tiled, indexer) are pwilkin's, several explicitly "Halogen-style" (hc-mix.cu:83, hc-cn.cu:103,
  qwen4exp.cpp:746).
- CIRU-specific additions: Q4_1 routed-expert admission for the MMB kernels (mmb.cu:72-98, 729-733,
  848-850, 899-901), the CIRUPLE1 disk pager for the n-gram embedding (src/models/qwen4exp-ple-pager.*),
  the MTP LM-head shortlist (qwen4exp.cpp:20-118, 918-941), the incremental QSA (indexer) cache
  (src/llama-memory-hybrid-idx.cpp:86-110, qwen4exp.cpp:1305-1330), the Q8_0 HC gate-mix admission
  (`CIRU_HC_Q8_FUSE`, mmb.cu:796), an FA occupancy knob for reproducibility (fattn-common.cuh:1130-1142),
  and the "Kairic Boost" launcher preset (scripts/ciru/run-server.sh:246-247).
- Base: ggml-org/llama.cpp `f5e85d43` + Qwen-experimental/MTP port (Ryan Monsurate) (docs/PROVENANCE.md:52-54,
  THIRD_PARTY_NOTICES.md:7-8). No local upstream checkout exists to diff, so "upstream" judgements below are
  from code shape and comments (e.g. mma.cuh WMMA iu8, mmvq tables, mmq configs are upstream-style).

## 1. What "IU4" is

**IU4 is a name for a tensor-type mix, not a new quant format.** Direct statement:
docs/PROVENANCE.md:95 — "No tensor is stored on disk as a custom `IU4_A640` type; the standard launcher uses
ordinary Q4_1 storage, expands packed values to byte lanes for the IU8 WMMA matrix path, and does not
activate the separate native IU4/E3 bank path." `grep -ri iu4 ggml/ src/` finds nothing in kernel code.
(The `e3-qr05-locked/` AMD helper dir named in THIRD_PARTY_NOTICES.md:9 is not present in this tree.)

Composition of the target GGUF (docs/PROVENANCE.md:82-95), 1,223 tensors, 79,397,818,720 B:

| Type | Count | Notes |
|---|---:|---|
| Q4_1 | 144 | = 48 layers x {gate, up, down} routed experts; 75,497,472,000 B |
| Q5_K | 328 | dense trunk ("protected core") |
| Q8_0 | 290 | incl. HC matrices (CIRU_HC_Q8_FUSE admits Q8_0 HC gate, mmb.cu:796-800) |
| Q5_1 | 48 | one per layer (role not stated) |
| BF16 | 25 | |
| F32 | 388 | norms, router (MMB "F32 split" GEMM for the F32 MoE router, mmb.cu:482-483) |

- Q4_1 layout (standard ggml): 32-weight block = fp16 d + fp16 m + 16 B nibbles = 20 B -> **5.0 bpw**,
  asymmetric `w = d*q + m`, no rotation, no codebook. Kernel decode of exactly this: mmb.cu:71-98.
  Check: 48*512*3*640*2560 weights * 5/8 B = 75.5 GB, matches PROVENANCE:95.
- Rest of core = 3.90 GB (PROVENANCE:95). Total package incl. PLE: 131.8 GB
  (benchmarks/fidelity/data/baselines.json "size_bytes").
- MTP head: separate Q8_0 GGUF, 4,135,893,248 B (PROVENANCE:72); its LM head is Q8_0 2560x248320
  (asserted at qwen4exp.cpp:921).
- PLE (n-gram hash embedding) sidecar "CIRUPLE1": 320,001,536 rows x 160 B FP8-E4M3 (one global BF16
  scale), 4 KiB pages of 25 rows + 96 B trailer, 52,429,053,952 B (qwen4exp-ple-pager.h:9-23, 115-117).
- **Production tool / calibration**: not in this repo. No conversion script, no imatrix, no requant recipe;
  all docs say weights were never requantized since v2 (PROVENANCE:67, V2_RELEASE.md:70,
  V3_1_RELEASE.md:5). `conversion/qwen4exp.py` is the stock HF->GGUF converter. "Orca" is a second
  variant sharing the runtime (docs/RUNNING.md:135-143), format not described either.
- No rotation/Hadamard/trellis anywhere — compared with our HT4 trunk (4.0 bpw trellis), CIRU spends
  more bits (Q5_K core, 5.0 bpw experts) for worse fidelity (see 2.4).

## 2. Reported numbers

All on Ryzen AI Max+ 395 / 8060S gfx1151, 128 GB, NixOS, ROCm "10" (TheRock), single slot.

### 2.1 Prefill
- v4.0 own weights, served, MTP on: 992-1,001 tok/s on five cold 30.8K prompts, 948 at 65K,
  631 at 4K, 1,003 at 16K, 748 at 261,888 (docs/qualification/v4.0.0/QUALIFICATION.md:11-24).
- Target-only native pilot PP16384: 959.5 PP / 24.27 TG, batch 16384 (same file :28).
- **pwilkin's reference package on its own (IQ4_NL) weights: 1,361 +- 11 PP; same engine with the 36 kernel
  env vars unset: 641 PP** (same file :30). That 2.12x is the value of the MMB/HC/QSA kernel bundle.
  CIRU's Q4_1/Q5_K weights don't get all of it ("Some reference optimizations remain unavailable for
  incompatible quantization types", README.md:63; profiles/strix-halo-production.env:3).
- v4.4: 12,960-token prompt 824 -> 889 tok/s; 245,760-token prefill 751 -> 761; 512K target-only 604.7
  (docs/qualification/v4.4.0/QUALIFICATION.md:5-10, 45).
- Pre-pwilkin (v2.0) engine was ~370 tok/s (docs/V2_RELEASE.md:43-50).

### 2.2 Decode
- Plain (target-only): 24.27 tok/s at 16K (v4.0 QUALIFICATION.md:28); v2 sweep 22.6 @512 -> 6.75 @131K
  (V2_RELEASE.md:43-50).
- MTP (depth 3 in v4.3+): 44.53 tok/s on the 12,960-token replay, 353/469 drafts accepted = 75%
  (docs/qualification/v4.4.0/IU4-RESULTS.json "incident-r0"); short HumanEval 0-9 panel 60.35 tok/s, 64.07 with
  Boost (v4.4.0/QUALIFICATION.md:24-29); HumanEval full run MTP6 66.46 tok/s at 4K ctx (README.md:69).
  245K cold decode 7.62 -> 21.14 tok/s from the v4.4 changes (v4.4.0/QUALIFICATION.md:9).
  Hermes agent pooled generation 34.7 tok/s (README.md:13).
- Multi-user: **none**. MTP only qualified with one slot; launcher rejects multi-slot MTP
  (docs/RUNNING.md:121-123). No AR multi-user numbers.

### 2.3 Memory
- Idle system RAM used ~104 GB with the server loaded (IU4-RESULTS.json baseline_memory.idle);
  min available 6.82 GiB on the 15-request panel, 7.70 GiB at 261,888 ctx (v4.0 QUALIFICATION.md:35, 38).
- Fits only because the 52 GB PLE stays on disk behind a 4 GiB page cache (profile :49-53,
  CIRU_RELEASE.json:94). F16 target+draft KV, ctx 262144, ubatch 8192.

### 2.4 Quality
- Community fidelity benchmark (benchmarks/fidelity/README.md "Frozen protocol"): 16 windows x 2,048 tokens,
  score positions 1,024-2,047 = 16,384 full-vocab distributions, reference = **original Transformers BF16 logits
  of Qwen/Qwen3.8-Flash-Next**, forward KL(ref||cand), bootstrap CIs over 8 repo clusters, F16 KV, ubatch 512.
  Results (benchmarks/fidelity/data/baselines.json):
  - IU4: KL **0.0600** [0.052, 0.069], top-1 90.6%, PPL 2.234
  - AP-Q4_K_XL (101 GB): KL 0.0369, top-1 91.7%, PPL 2.196
  - AP-Q5_K_XL (121 GB): KL 0.0311, top-1 91.9%, PPL 2.190
  So IU4 is the least faithful of the three despite being the largest file.
- Older tiny panel (64 distributions): KL 0.030 vs BF16 (V2_RELEASE.md:20-27), an earlier one 0.223
  (BENCHMARKS.md:68-78). Not comparable to our KL 0.047 (our reference is the Q4 GGUF, not BF16).
- Task scores: HumanEval 95.7% / HE+ 93.9% (README.md:69), Hermes Agent 20 98.5/100 (README.md:13).

## 3. Kernel design

### 3.1 Prefill GEMM ("MMB", pwilkin; ggml/src/ggml-cuda/mmb.cu)
- Path: **dequantize weights to BF16 in LDS, BF16 WMMA (`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`)**
  — not int8. mmb.cuh:3; WMMA at mmb.cu:216, 318, 439-440. Activations converted F32->BF16 once per tensor and
  cached by tensor identity for the whole graph (`mmb_bf16_activation`, mmb.cu:583-592; cache API mmb.cuh:9-18).
  Gated by `LLAMA_MMB=1`, only when T >= `LLAMA_MMB_MIN_T` (512) (mmb.cu:655-656, 738).
- Tile: 256 threads = 8 wave32, `__launch_bounds__(256, 2)`, K-step 64, LDS row stride 64+8 BF16
  (mmb.cu:15). Pipeline = global->register prefetch of step k+1 while WMMA runs on step k, then dequant+store
  to LDS after a barrier (mmb.cu:200-226). Epilogue via a 1 KB/wave LDS stage, writes F32 and/or BF16
  (mmb.cu:105-137).
- Dequant tricks: IQ4_NL LUT applied with two `v_perm_b32` per 4 nibbles instead of indexed loads, and
  `kv*d` computed as `fma(kv+128, d, -128d)` so the BF16 result is bit-identical (mmb.cu:34-69). Q4_1 is
  plain `d*q+m` (mmb.cu:71-98). Q8_0 (mmb.cu:100-117).
- Dense tiles: 128x128 (warp 32x64) or 128x256 "big" when M>=6144 && K>=2560 (mmb.cu:771-786);
  "tall" 384x32/384x64 for M<=384, K>=4096 (the HC down|inject [10240->324] GEMM, activations read once,
  mmb.cu:755-760). Optional persistent **BF16 shadow copies** of IQ4_NL/Q6_K dense weights
  (cap 6 GB, `LLAMA_MMB_SHADOW`) so dense GEMMs skip dequant entirely (mmb.cu:594-660, 909-945).
- **MoE (routed experts)**:
  - Routing sort: for 512 experts / top-10 / 4096 < T <= 32768 a chunked counting sort (1024-route
    tiles, per-chunk expert offsets, prefix, scatter) (mmid.cu:183-299); else upstream `mm_ids_helper`.
  - **Two tile classes**: experts with >=128 rows get BN=128-token tiles, the rest BN=32 tiles; descriptors
    (expert | tile<<16) built by one 1024-thread prefix-scan kernel; empty slots early-exit
    (mmb.cu:546-569, 836-855, 355-357). Two launches per GEMM (big grid, small grid).
  - **Fused gate+up+SwiGLU**: one kernel holds gate and up weight tiles in two LDS buffers and two
    accumulator sets, epilogue `silu(g)*u` written as BF16 (mmb.cu:365-461, 871-905). Tile BM=64 weight rows x
    BN=128 tokens (warp 32x32), small class 64x32. Admitted only for Q4_1 K=2560,M=640 (mmb.cu:724-733, 862).
  - Down: BM=128 x BN=128 (warp 32x64), small 128x32; with `LLAMA_MMB_DOWN16` writes BF16 in place, consumed by
    a BF16 weighted-reduction kernel (moe-weighted-reduction.cu:58-87) whose only consumer is the fused HC
    combine reading BF16 block_out.
  - So the expert chain is BF16 end-to-end between kernels: x(bf16) -> GLU(bf16) -> down(bf16) ->
    weighted sum(bf16) -> hc_combine_norm.
- Below T=512 (MTP verify batches, short prompts) dispatch falls to upstream MMQ with **int8 WMMA**
  (`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`, mma.cuh:1324-1331), Q4_1 nibbles expanded to bytes in LDS
  (mmq-load-tiles.cuh ~284-285) — this is the "IU8 path" PROVENANCE:95 talks about. Upstream llama.cpp.
  RDNA3.5 MMQ tile table: mmq-config-rdna3-5.cuh:40-51 (Q4_1), :129-140 (Q5_K) — upstream-style.

### 3.2 Decode matvec
- Upstream MMVQ (dp4a on Q8_1-quantized activations), RDNA3.5 uses the RDNA2 parameter table
  (mmvq.cu:120-150), MoE through upstream `mul_mat_vec_q_moe` (mmvq.cu:870, 1069-1110). Small fork tweaks:
  DPP/permlanex16 warp reduce for Q8_0 (mmvq.cu:96-109), unrolled Q8_0 loop for small dense rows
  (mmvq.cu:713-728, 1135-1137), 4 rows/block for IQ4_NL MoE (mmvq.cu:1102-1108). Nothing Q4_1-specific.
  Decode is the weak spot (24 tok/s plain at 16K).

### 3.3 gfx1151 specifics
- Everything new is gated by `GGML_CUDA_CC_IS_RDNA3_5` and often by exact shapes (512 experts, top-10,
  2560/640, H=48, S_v=128, head dim 256, GQA 12): mmid.cu:285-288, mmb.cu:729-733,
  gated_delta_net.cu:399, qsa.cu:414-421, lightning-indexer.cu:668-680.
- `HSA_OVERRIDE_GFX_VERSION=11.5.1`, unified memory on, HIP-graph PM4 debug flag (profile :5-9);
  pinned "pwilkin HIP/ROCr" runtime build `7dda3ac6` (README.md:27).
- HIP 7.16 reports 128 KiB LDS/WGP vs 64 KiB before, changing FA split counts; knob clamps to the old
  bound for reproducibility (fattn-common.cuh:1130-1142).

## 4. Model-specific (Flash-Next / qwen4exp) work

### 4.1 Hyper-connections (4 streams)
- Graph form: grouped RMSNorm over each stream with [hc*n_embd] gamma; `lo = silu(down(xn)/hc)`;
  `gate = sigmoid(up(lo))`; mixed = mean over streams of `xn*gate`; inject = W_inject xn; combine:
  `res += block_out * 2*sigmoid(inject/hc)` (src/models/qwen4exp.cpp:481-575).
- Fusions (all pwilkin/Halogen-derived):
  - down|inject packed into one GEMM [10240 -> lr+hc] (`LLAMA_HC_PACK_DI`, qwen4exp.cpp:500-516) -> the
    "tall" MMB tile (mmb.cu:755).
  - **gate GEMM [320 -> 10240] + sigmoid + stream mix in one kernel** (`hc_gate_mix_kernel`, mmb.cu:258-345;
    matcher ggml-cuda.cu:3837-3845); CIRU extends it to Q8_0 HC weights (mmb.cu:796-813).
  - **hc_combine + next RMSNorm fused** (hc-cn.cu:37, 104; dispatch ggml-cuda.cu:3876-3910), optional BF16
    residual stream in place (`LLAMA_HC_RES16`) and BF16 copy of xn for the next MMB GEMM (`LLAMA_MMB_HC16`;
    BF16-only marking pass in graph_optimize, ggml-cuda.cu:4971-4995).
  - hc mix reduce kernels incl. BF16-input variant (hc-mix.cu:34-110).
- Bitwise-equivalence is a design rule: fused kernels reproduce op order/rounding with inline
  `v_mul_f32`/`v_add_f32` asm to block FMA contraction (hc-cn.cu:4-24, mmb.cu:250-255).

### 4.2 Gated DeltaNet
- Prefill GDN is **not chunked** ("TODO: Add chunked kernel", gated_delta_net.cu:388). RDNA3.5 path is a
  tiled sequential scan: grid H(48) x 1 x 2, 8 waves, each warp owns 8 state columns in registers, 16-token
  q/k/v tiles staged in LDS, two warp reductions per column per token (gated_delta_net.cu:198-375, 399-430).
  v4.4 adds float4 staging (`FLASH_K21_VECTOR_STAGE`) and precomputed exp(g) (`FLASH_K22_STAGED_DECAY`)
  (gated_delta_net.cu:405-416; profile :62-63).
- `keep_rs_t`: writes the last K per-token states into snapshot slots so MTP rejection rolls back without
  recompute (gated_delta_net.cu:349-361; delta-net-base.cpp:566-590; llama-memory-recurrent.cpp:193).
  Likely from the upstream/Monsurate MTP work.
- Short conv: fused direct 4-tap conv + SiLU with tail-only concat (gdn-conv.cu:7-40, `LLAMA_GDN_CONV`).
- Wave-per-row RMSNorm (+ optional sigmoid gate) for 256-wide per-head norms: upstream launch was
  "block-scheduling bound at ~75 GB/s" on ~786k rows (norm-gated.cu:6-10).

### 4.3 Sparse attention + indexer ("QSA")
- Indexer: 4 heads x 128, WMMA f16 score kernel keeping 16x64 key tiles in registers, relu x weight summed
  and mask added in-kernel (lightning-indexer.cu:596-665, 668-700); relu+head-sum fusion
  (idx-relu-sum.cu:4-21); keys pooled 4:1 by a fused `get_rows_mean4` (ggml-cuda.cu:3912-3923).
  Top-k selects whole blocks: `indexer_top_k + ratio - 1` cells (qwen4exp.cpp:1380-1500 comments).
- Attention ("qsa3", qsa.cu): queries in **groups of 4 share the union of their selected key blocks**
  (sorted-row check, 128-lane merge with prefix sum, qsa.cu:16-170), keys/values prepacked F16 in 4-key blocks,
  f16 WMMA for QK and PV (qsa.cu:177-400, WMMA at :275, :376). Only for >=128 queries (prefill), head dim 256,
  GQA 12 (qsa.cu:411-430). Decode uses the fallback dense/split path; dense shortcut when n_kv <= width
  (`LLAMA_QSA_DENSE_SHORTCUT`, qwen4exp.cpp:1679).
- **Incremental QSA cache (CIRU, v4.4)**: persistent F32 tensor of pooled block-level indexer keys per
  ratio-4 layer, [128, kv_size/4] (llama-memory-hybrid-idx.cpp:86-110); when the prefix matches, only the
  new `(n_tokens+3)/4+1` blocks are prepared instead of all n_blocks (qwen4exp.cpp:1305-1330). This is the
  likely source of the 245K-token decode 7.62 -> 21.14 tok/s (v4.4.0/QUALIFICATION.md:9).

### 4.4 MTP / speculation
- MTP head: separate Q8_0 GGUF, single nextn block; eh_proj input flattened over (hc, tokens) so it runs as
  one wide GEMM instead of 4-column matvecs (qwen4exp.cpp:780-799). Draft attention can use the same
  sparse QSA path as the trunk ("this is what Halogen does"), now from 1 token (`LLAMA_MTP_QSA_MIN_T=1`)
  (qwen4exp.cpp:745-760, 850-873; README.md:9).
- **LM-head shortlist (CIRU)**: at draft time the head computes only 32,768 of 248,320 rows: ids
  0..16383 always, plus the top-16,383 of the target's last logits (CPU 3-pass radix select over the
  11-bit buckets), plus the last token. Implemented as `mul_mat_id` with each vocab row as an "expert",
  then `set_rows` into a -inf vector (qwen4exp.cpp:57-118, 914-941; `CIRU_MTP_SHORTLIST=32768`, profile :57).
  Cuts the draft LM-head read from ~675 MB to ~89 MB per draft step.
- Depth: 3 is the tested default (shorter drafts waste less), was 6 (README.md:9, 18). Sampler
  `p-split 0.10`, `p-min 0` (run-server.sh:241-244).
- "Kairic Boost" = **upstream llama.cpp ngram-mod speculation chained before draft-mtp**:
  `--spec-type ngram-mod,draft-mtp --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 64 --spec-ngram-mod-n-max 64`
  (run-server.sh:246-247; common/ngram-mod.cpp). +6.2% on one short code panel (v4.4.0/QUALIFICATION.md:24-29).

### 4.5 N-gram embedding (PLE)
- One PLE layer, 3-gram, 8 heads per n-gram order -> 16 hashed lookups per token, row = 160 FP8 values;
  hash = XOR of token*multiplier, `% head_vocab + head_offset` (qwen4exp.cpp:222-300, 1985-2003).
- Module: key/value projections of the gathered rows, grouped norms, per-stream dot -> signed sqrt ->
  sigmoid gate, then a depthwise causal conv dilated by the n-gram size (as K shifted copies), SiLU, added
  to all hc streams (qwen4exp.cpp:2176-2267); fused PLE conv kernel (ple-conv.cu, `LLAMA_PLE_CONV`).
- **CIRUPLE1 pager**: the 52 GB table stays on disk; 4 KiB pages, page cache (4 GiB in production),
  16 decode workers / 32 bulk I/O workers; a prompt >= 4096 tokens triggers a detached CPU thread that hashes
  all n-grams and prefetches (qwen4exp.cpp:1972-2007); decode issues the 16 lookups early and joins with a
  500 µs soft deadline, falling back to synchronous completion (qwen4exp-ple-pager.h:51-111).
  v4.1: 32 I/O workers gave +4.4% 16K prefill (README.md:47).

## 5. Ideas for ds4, ranked by expected gain

Context: CIRU itself is **behind ds4** on prefill (889-1,000 vs our ~1,050), plain decode (24 vs 33-35)
and fidelity (KL 0.060 vs BF16, worse than a Q4_K_XL). What is worth taking is mostly pwilkin's kernels
and three CIRU runtime tricks.

1. **Incremental block-key cache for the indexer (CIRU, new)** — big win only at long context.
   If ds4 re-pools / re-norms / re-ropes all indexer block keys every decode step, keep a persistent
   [128, n_ctx/4] block-key tensor per sparse layer and update only the tail blocks
   (llama-memory-hybrid-idx.cpp:86-110, qwen4exp.cpp:1305-1330). CIRU saw 2.8x decode at 245K.
   Measure: plain + MTP decode at 32K/128K/240K, indexer per-step time per layer, exact top-k match vs current.
2. **Fuse expert gate+up+SwiGLU, keep expert intermediates BF16 end-to-end, two tile classes (pwilkin)** —
   targets our 47 ms/layer at 8K (gate/up ~20, down ~13 TFLOP/s). Details: BF16 dequant-to-LDS + BF16 WMMA,
   register-prefetched next K-step, 64x128 GLU tiles with two accumulators, 128x128 down tiles, 32-token
   tiles for experts with <128 rows, BF16 GLU out -> down -> BF16 weighted reduce -> fused HC combine
   (mmb.cu:365-461, 546-569, 817-905; moe-weighted-reduction.cu:58-87). Note that at 8K the mean is
   160 rows/expert, so the 128/32 split leaves a partial 128 tile per expert; a 128+32 remainder split may beat it.
   For our Q4_K/MXFP4 weights the dequant goes into the same slot as `mmb_dq_row40`. Upper bound if experts
   reach ~35 TFLOP/s: about -1.1 s of 7.8 s at 8K -> ~1,250 tok/s. Measure: per-layer expert ms at 2K/8K/32K,
   TFLOP/s per kernel, KL unchanged.
3. **MTP draft LM-head shortlist (CIRU, new)** — if ds4's draft step reads the full 248K-row head.
   32K rows = 16K lowest ids + top-16K by last target logits + last token (qwen4exp.cpp:68-118, 918-941).
   Saves ~0.5-0.6 GB of reads per draft token at Q8 (~2.5 ms at ~230 GB/s); at depth 3 maybe 5-10% MTP tok/s.
   Do the top-k on the GPU; CIRU does it on the CPU. Measure: draft step ms, acceptance rate (must not drop),
   MTP tok/s on code + chat.
4. **HC fusions and BF16 HC intermediates (pwilkin/Halogen)** — prefill, a few %.
   gate GEMM+sigmoid+mix (mmb.cu:258-345), combine+next-norm (hc-cn.cu), down|inject packed tall GEMM
   (qwen4exp.cpp:500-516, mmb.cu:755), BF16 residual/xn (ggml-cuda.cu:3876-3910, 4971-4995). Each 10240-wide
   F32 HC tensor at 8K is ~335 MB per pass. Measure: HC kernel share in an 8K prefill profile before/after.
5. **ngram-mod in front of MTP (upstream llama.cpp)** — cheap: n-gram match 24, 64-token proposals
   (run-server.sh:247). +6% on short code; probably more on edit/refactor prompts, ~0 on fresh prose.
   Measure: tok/s and accepted/drafted split by source, on code-edit vs chat.
6. **MTP depth 3 instead of deeper, sparse draft attention from 1 token** — config only (README.md:9, 18).
   Measure acceptance-weighted tok/s for depth 2/3/4 at 8K and 64K.
7. **Smaller kernels to check against ds4's profile**: wave-per-row RMSNorm for 256-wide head norms
   (norm-gated.cu:6-10, CIRU says the generic launch hit only ~75 GB/s); relu+head-sum indexer fusion
   (idx-relu-sum.cu); register-resident-key WMMA indexer (lightning-indexer.cu:596-665); 4-query union
   sparse attention (qsa.cu) if ds4's 32K prefill spends a lot in sparse attention.
8. **PLE paging (CIRU)** — memory, not speed. Only worth it if ds4 holds the full PLE table in RAM and
   needs room (e.g. for multi-user KV). Async prefetch on long prompts + early-issue lookups at decode
   (qwen4exp.cpp:1972-2007, qwen4exp-ple-pager.h:98-111). Measure: RSS, prefill/decode delta, page-miss p99.

Not worth copying: the GDN prefill scan (sequential, unchunked — gated_delta_net.cu:388), the Q4_1/Q5_K
weight mix (worse KL than Q4_K_XL at more bytes), and decode matvec (upstream MMVQ, 24 tok/s plain).
