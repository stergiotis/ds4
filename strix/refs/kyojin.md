# kyojin (Yamz-Labs exllamav3 fork for Strix Halo gfx1151): study notes

Repo: `~/src-ref/kyojin` (MIT, one commit). Read-only study; nothing built or run.
Paths below are relative to the repo root. `README.strix-halo.md` is cited as RSH.

## 0. TL;DR

- It runs **our model**: Qwen3.8-Flash-Next (`qwen4_exp`), with the GDN, QSA sparse attention + indexer, MoE 512/top-10,
  4-stream gated-residual hyper-connections, the PLE hashed n-gram table, and the MTP head. Weights are EXL3 trellis
  (mul1 codebook, 128-pt Hadamard, i.e. the same family as our HT4). The published pack is 95 GB at about 3.47 bpw,
  with **4-bit routed experts**.
- Published prefill **1481 tok/s at 8K, 1477 at 32K, 1347 at 256K**. Speculative decode is 45.7 / 48.3 / 66.8 tok/s
  (prose / chat / code). Plain decode is about 27 tok/s, slower than ds4's 33-35.
  Fidelity is KL 0.0216 and top-1 93.81 % against the official FP8 release.
- **Q2 answer (the key one):**
  - **Routed experts:** kyojin decodes the trellis *inside* the grouped GEMM. Each wave decodes into its own LDS
    stage, then reads WMMA fragments back.
  - **Dense layers:** it does **not** decode inside the GEMM. It reconstructs to an fp16 scratch, then runs rocBLAS,
    exactly like ds4. The one difference: at 1024 or more rows, both Hadamards and the sign vectors are folded into
    the reconstruct, so the GEMM runs on raw activations.
  - **Below 1024 rows, and for MoE:** Hadamards are applied **to activations** (the rotated basis), not to weights.
  - **Why the in-GEMM decode stops at MoE:** on gfx1151 the WMMA unit and the decode VALU share the SIMD, so time is
    at least WMMA time plus decode time. The decode is paid once per 128-row m-tile. For a 4096-row dense GEMM that
    would be 32 decodes of the same weights.

## 1. Models, formats, numbers

### Models
- Three models: Qwen3.8-Flash-Next (95 GB pack), GLM-5.3-Flash (99.7 GB), MiMo-V2.6-Flash (105.8 GB)
  (README.md:15-20, doc/models.md).
- Qwen arch file: `exllamav3/architecture/qwen4_exp.py:29-34`. Its header lists:
  - "QSA sparse full-attention layers (indexer-selected 4-token blocks over standard GQA)";
  - "low-rank gated-residual hyper-connections ... (4 fp32 streams, elementwise mixing)";
  - "sigmoid-gated GDN output norm";
  - "PLE layer injecting hashed n-gram embeddings (51.2B-row table, streamed from disk by default)".
- Config reads for the arch: hc_count=4, indexer heads, budget, compress ratio 4, PLE ids, ngram size
  (`qwen4_exp.py:64-82`).
- MTP drafter: `qwen4_exp_mtp.py`. Its draft experts are K6 (`tools/qwen/serve.py:553-556`).
- Pack composition is "bal pack: 4-bit routed experts, gate/up N 640, down N 2560" (`tools/qwen/serve.py:553-555`).
  The earlier turboderp 3.05 bpw pack is also referenced (RSH:71).
- Hard-coded Qwen shapes in the HIP kernels:
  - QSA: HD 256, 24 q heads, 2 kv heads, GRP 12, page 256, KPAD 2080 (= 4*block_topk+3 rounded)
    (`exllamav3/modules/attention_fn/qsa_prefill.hip:19-27`);
  - mixer: K1 = H*D = 10240, DD 2560, HS 4, RANK 320, NTOT 324 (`exllamav3/modules/gr_fn/gr_mix.hip:17-21`).

### Throughput (one Ryzen AI Max+ 395, 128 GB)

Method (doc/benchmarks.md:7-14): public packs, default env, greedy, thinking off, 256 tokens, median of 3. Prefill
is the server-reported prompt speed on cold prompts of 8K-256K. Qwen decode is client-timed from first to last
streamed token. Spec decode returns the same tokens as plain decode.

Qwen3.8-Flash-Next (doc/benchmarks.md:20-26, 39-43):

| | 8K | 32K | 64K | 128K | 256K |
|---|---|---|---|---|---|
| prefill tok/s | 1481 | 1477 | 1436 | 1396 | 1347 |
| spec decode tok/s | 46.5 | 46.4 | 46.4 | 45.1 | 44.3 |

- Card prompts: prose / chat / code = 45.7 / 48.3 / 66.8.
- A second box: 1486 / 1478 / 1454 / 1397 / 1349 (doc/benchmarks.md:26).

Their measurements of the rivals (doc/benchmarks.md:77-94):
- Gufo v0.8.0: 1250 / 1308 / 1289 / 1230 prefill, 35.3-31.4 decode.
- Strata: 39.0-35.0 decode.
- These are their numbers, on their box. Our competitor figures for gufo (1,620-1,700) are higher, so their method
  or settings differ.

Plain vs speculative decode (`tools/qwen/SERVE.md:166-171`, 3.47 bpw pack): **plain 27.3 tok/s** for every class.
Speculative:

| chat | prose | code | multi | copy |
|---|---|---|---|---|
| 42.2 | 38.7 | 52.2 | 42.6 | 54.7 |

Earlier optimisation ladder on the 3.05 bpw pack, single stream (RSH:71-88). Each row adds one change:

| step | tok/s |
|---|---|
| stock reconstruct+hgemm | ~17 |
| WMMA GEMV | 27.9 |
| `EXL3_MOE_CFG=2` | 30.2 |
| `EXL3_HIP_PREFILL_MIN_ROWS=2` | 32.7 |
| LDS bank-conflict pad | 34.9 |
| skinny GEMM for GDN b/a | 36.1 |
| **int8 mixer weights** | **40.0** (mean, 46.3 peak) |

Multi-user, measured two ways:
- Batch throughput (RSH:132-176). Synthetic aggregate: 47.0 (b=1), 70.3 (b=3), 79.4 (b=4), 86.1 (b=5).
  Realistic aggregate: 34.2 (b=1), 66.1 (b=4), **74-75 (b=5)**, then a dip to 56.1 at b=6. The dip is the "16-row
  rule": the grouped MoE runs 16-row chunks, and B*(ndt+1) rows should land just under a multiple of 16.
- `--sessions` server (`tools/qwen/SERVE.md:143-151`): 2 sessions give 49.6 total, 4 sessions give 51.4-54.4.

### Quality
- **Engine comparison** (doc/benchmarks.md:96-104):
  - Method: top-1 agreement and KL against the official FP8 release; 200 public neutral texts, 3084 scored
    positions, fixed seed, one scorer, row-cluster bootstrap.
  - Kyojin: **93.81 % top-1, KL 0.0216** (run-to-run 0.0008).
  - Strata: 93.64 %, 0.0316. Gufo (111 GB Q4_K_XL): 95.36 %, 0.0156.
- **Perplexity:** PPL 4.2259 on the 3.05 bpw pack, unchanged across every kernel change (RSH:85-95). It is wikitext-2
  test, 2048-token windows (`eval/ppl.py:18-36`).
  - RSH:229-232 warns that ppl.py reported the correct PPL on a build that gave NaN logits in decode. **PPL alone is
    not a kernel gate.**
- **Mixer int8 group scales + GPTQ sidecar:** decoded-token KLD 0.0504 -> 0.0455 (`tools/qwen/serve.py:545-548`).
- **No cross-engine comparison with our metric.** Ours is the PPL ratio vs a Q4 GGUF. To compare with them we would
  need a KL vs FP8 on their 3084 points.

## 2. EXL3/trellis in prefill GEMMs (most important)

### 2a. Routed experts: in-GEMM trellis decode (`exllamav3/exllamav3_ext/quant/exl3_moe_prefill.cu`)

The pipeline is 8 launches per MoE layer, independent of the expert count (`exl3_moe_prefill.cu:1-12`):
1. Metadata on device: expert offsets (exclusive scan), inverse order, and a compacted list of (expert, m-tile). No
   host sync (`:51-114`).
2. Gather plus the **input Hadamard on activations** (suh, 128-pt warp-shuffle FWHT). Each token row is read **once**
   and scattered to its top_k sorted slots (`:146-150`). Glue v2 fetches the top-k index chain once per wave and
   broadcasts it with shuffles (`:292-297`).
3. Grouped GEMM gate+up, with trellis tiles decoded into LDS **inside the GEMM** and `v_wmma_f32_16x16x16_f16`.
4. Output Hadamard (svh) of gate and up, SiLU(g)*u, and the input Hadamard of down, all in place.
5. Grouped GEMM down, **fp32 output**.
6. Output Hadamard of down, fused into a weighted top-k reduce. The reduce is in fp32, in fixed slot order, so it is
   deterministic (`:247-250`).

So the Hadamards are handled as online rotation of activations: x*su -> H128 -> GEMM with the raw decoded codebook
matrix M -> H128 -> *sv. That is exactly our HT4 W = diag(sv) H M H diag(su) structure.

**mpw2 kernel** (the default, `MPW_KERN=2`; `exl3_moe_prefill.cu:702-1021`). Shape and data flow:
- Tiles: MT = 128 rows (one expert per tile), 8 waves, NT = 128*NW columns, NW = 2 adjacent 16-column n-tiles per
  wave, KS = 2 k-slices (32 k) per iteration (`:714-724`). Uses 241-256 VGPRs and 34-37 KB LDS, with no spills
  (`:714`).
- **B (weights):** each wave owns its NW n-tiles for all MT rows.
  - It loads its own trellis words, stores them to a wave-private `sh_w`, and decodes them to fp16 in a wave-private
    `sh_b` in the CUDA m16n8k16 B-fragment layout. It then reads the 16x16 WMMA B fragments back from LDS.
  - No block barrier guards B, because LDS ops of one wave execute in order (`:871-907`).
  - NW = 2 halves the A-fragment LDS reads per WMMA.
- **A (activations)** is the only block-shared operand:
  - double-buffered, unpadded, with 16-byte chunks **XOR-swizzled**: `chunk c at slot c ^ ((r >> 1) & 3)`, which is
    conflict-free (`:726-731`);
  - global prefetch of A and W runs PF iterations ahead in a **register ring**, with clamped addresses so the load
    stream stays uniform (`:827-860`, `:953-958`).
- **One barrier per iteration, LDS-only:** `s_waitcnt lgkmcnt(0); s_barrier` instead of `__syncthreads()`. The
  latter would drain vmcnt, and with it the global prefetch in flight (`:654-656`, `:960-965`).
- Main step: store W(it), issue load W(it+PF), store A(it+1), issue load A(it+PF+1), decode, WMMA (`:961-982`).
- **Measured cost model** (`:703-707`):
  - "On gfx1151 WMMA, trellis-decode VALU and permlanes share the SIMD, so time >= WMMA + decode".
  - The decode alone is "~3 ms per 4096x2048 call (no-decode ablation), paid once per m-tile".
  - The older mpw kernel had "a fixed per-tile cost of ~7.7 ms per 4096x2048 launch at 8 rows/expert".
- **Variant trade-off:** MT 64 wins with uniform routing of 64 rows or fewer per expert. It loses on skewed routing,
  because experts with 65-128 rows decode twice (`:714-715`).
- **Measured win:** mpw2 is 2.6x faster per layer than the legacy K3 kernel (`exllamav3/modules/block_sparse_mlp.py:604-610`),
  and +3-5 % e2e prefill over the old grouped GEMM (`tools/qwen/serve.py:553`).

Instruction tricks, all bit-exact:
- **Fragment k-permutation (`MPW_LDM=1`, default):**
  - Each lane reads only its 8-k half of a fragment row. `v_permlanex16` fetches the other half from lane ^ 16
    (`:411-430`).
  - This works because "v_wmma_f32_16x16x16_f16 on gfx1151 computes the odd output rows from the upper half-wave's
    own A/B, and a k permutation applied to both operands leaves the dot product unchanged".
  - Measured +3.4 % e2e prefill at 4K (`:1386-1387`).
  - `MPW_OPT=2` replaces the permlanes with two direct LDS reads in the permuted k order: +2.3 % e2e at 4K/16K
    (`:763-765`, `:1485-1486`).
- **Quarter-rate multiply avoided:**
  - `x*0x83DCD12D` for a 16-bit x becomes two full-rate `v_mul_u32_u24` plus a shift-add, written in inline asm,
    because LLVM folds the C form back into a quarter-rate `v_mul_lo_u32` (`:733-745`, `codebook.cuh:170-182`).
- **Byte-sum in one instruction:** `v_sad_u8(p, 0, 0x6400)` gives byte_sum(p)+0x6400. Two sums are packed into a
  half2 and fed through one `hfma2` with kinv/kbias (`codebook.cuh:45-75`).
  - The original dp4a byte-sum is emulated with about 6 VALU ops; SAD is 1 op.
  - `v_dot4_u32_u8` is the alternative (`EXL3_MUL1_DECODE_DOT4`), measured null (RSH:30, 209-213).
- **mpw2x (`EXL3_MPW2X`; on in the MiMo lane, off for Qwen):**
  - The trellis pointer is read through an **addrspace(1) global pointer**. Through a generic pointer the W prefetch
    compiles to *flat* loads, which also count on lgkmcnt, so "every LDS wait ... drains the whole prefetch and
    exposes DRAM latency per step".
  - It also adds an A-fragment software pipeline and a K loop specialised on the row-group count (`:1023-1033`).
  - This is a cheap check for any ds4 kernel that does global loads plus LDS waits.
- **Tail-wave skip (OPT bit 8):** for N = 640 (gate/up), waves past N skip decode and WMMA (`:968-975`, `:1465-1470`).
- **Row thresholds:** the grouped path engages from `EXL3_MOE_WMMA_MIN_ROWS=32` (`block_sparse_mlp.py:69`). Small
  verify batches (2-31 rows) go to a deduplicating grouped GEMV (`EXL3_HIP_PREFILL_MIN_ROWS=2`, doc/env_vars.md:921).
- **No TFLOP/s figure for Qwen experts in the repo.** We cannot compare directly with our Q4_K gate/up at 20 TFLOP/s
  and MXFP4 down at 13. The kernel needs `size_n % 16`, K of 2, 3 or 4 bits, the mul1 or mcg codebook, and K
  divisible by 32.

### 2b. Dense linears: reconstruct-then-rocBLAS (`exllamav3/modules/quant/exl3.py`)

- **Path choice:**
  - Above 144 rows (`AUTO_RECONSTRUCT_THRESHOLD`, `exl3.py:21`), `reconstruct_hgemm`: decode the whole matrix to an
    fp16 scratch, then `hgemm_recon` (rocBLAS) (`exl3.py:575-650`).
  - At 1024 or more rows, the **"fused reconstruct"** emits *original-basis* weights: both 128-pt Hadamards and the
    su/sv signs are folded into the reconstruct kernel's shared-memory epilogue. The GEMM then runs on raw x, and
    the standalone input/output Hadamard launches disappear (`exl3.py:592-605`).
    - Docs: "previously ~14 % of long-chunk prefill GPU time; ~+6-7 % prefill throughput" (doc/env_vars.md:156-167).
    - It costs about 4x a plain reconstruct (k*n-proportional) against savings that scale with rows*(k+n), so the
      breakeven is about 400-900 rows.
  - Below 1024 rows: `had_r_128(x)` -> GEMM on the rotated-basis weights -> `had_r_128(y)`.
  - **Takeaway: they also pay one unpack pass per dense matrix per prefill chunk.** They amortise it with
    4096-token chunks (`EXL3_PREFILL_CHUNK=4096`), and a tail of up to 1024 rows is merged into the last chunk
    instead of a second pass (`EXL3_PF_NO_TAIL`, `EXL3_PF_TAIL_MERGE`; `tools/qwen/serve.py:518-526`).
- **hadP cache (off by default):** keeps reconstructed fp16 weights across chunks within an LRU byte budget
  (`exl3.py:98-140`). A shape is admitted by measured reconstruct cost in us per MB written: 6.2-9.8 us/MB, i.e.
  about 100-160 GB/s write (`exl3.py:132-138`).
- **Row-pitch padding**, the big dense-GEMM finding (`exl3.py:40-46`, `exllamav3_ext/pad_pitch.h:1-20`):
  - "An fp16 row of K = 6144 is 12288 B; at that pitch the rocBLAS GEMM reads A at about half rate (out_proj 50 % of
    peak in place). One extra 128 B per row runs the same product, bit-exact, at 87-90 %."
  - Rule: pad when `(K*2) % 1024 == 0` to pitch K+64. Producers write straight into the padded pitch
    (`row_pad_pitch`), or `pad_copy_in` copies once.
  - **Our K=2560 (5120 B), 4096 and 10240 all satisfy (K*2) % 1024 == 0, so they are hazard pitches.**
- **fp32-output trap** (`exl3.py:200-211`):
  - hipBLASLt fp16-in/fp32-out picks `MT64x32x8` at 6.2 TFLOP/s, against 34 TFLOP/s for the same GEMM with fp16 out.
    The GDN projections declared fp32 out and were 28 % of device time.
  - Fix: compute fp16, widen afterwards, 4.7x faster.
- **Dense tuning** (`exllamav3_ext/hgemm.cu:110-122`):
  - The default rocBLAS heuristic `MT128x128x16` gives about 24-27 TFLOP/s. Screening every solution from
    `rocblas_gemm_ex_get_solutions` per (n, k, m-class) runs 20-35 % faster.
  - Results are cached per rocBLAS version and arch, and a seed file ships (`exllamav3/model/dense_gemm_tune_seed.txt`).
  - Warm-up at load avoids per-shape screening (`tools/qwen/serve.py:728-735`).
  - Practical peak quoted: 34 TFLOP/s for fp16 out (`exl3.py:203`). GLM floors use 47 TFLOP/s WMMA peak and 256 GB/s
    (`tools/glm/prof_families.py:4`).
- **`EXL3_PF_MSPLIT=2048`:** runs big GEMMs (N or K of 16384 or more) as 2048-row blocks so the A block stays in the
  MALL. Bit-exact (`exl3.py:619-627`).
- **Compiler:** prefill is about 20 % slower if the HIP kernels are compiled by the system ROCm instead of the ROCm SDK
  (TheRock) compiler (doc/troubleshooting.md:29-31).

### 2c. Other prefill kernels (hand-written gfx11.5 WMMA)
- **Gated-residual mixer** (`gr_mix.hip:1-21`):
  - Two kernels: gr_g1 is dm = normed @ proj^T (324 x 10240) followed by silu/sigmoid; gr_g2 is t @ up^T with the
    mean of sigmoid(g)*normed fused in, so g is never written.
  - BM 32 or 64 by R; 64-row tiles at R of 3072 or more: 1.18 vs 1.35 ms at R=4096 (`gr_mix_hip.py:70`).
  - Tensile-style K stagger.
- **`EXL3_PF_GR_FUSE`:** residual apply folded into the next site's norm, +1.6-2.4 % (`serve.py:540-541`).
- **HIP prefill set:** the PLE, q/g deinterleave, GR and GDN HIP kernels together gave **+21-24 % prefill at 4K/16K**
  (`tools/qwen/serve.py:505-509`).

## 3. Decode matvec for trellis (`exllamav3/exllamav3_ext/quant/exl3_dec.cu`)

The design does **not** use WMMA for batch-1:
- Each decoded pair is two consecutive rows of one column, which is one `v_dot2_f32_f16` against a pair of input
  values (`exl3_dec.cu:1-13`, `:236-262`).
- Shape: 8 waves, 512 output columns per block (32 lanes x 16).
- Ragged k-split across waves and blocks; the last-arriving block reduces through an atomic counter
  (`:590-618`, `:1897-1907`).
- Input Hadamard into LDS and output Hadamard in the epilogue, so there is **one launch per GEMV**
  (`exl3.py:213-216`). Up to 4 matrices share x in one launch (`:624`).

Tricks:
- **Constants in SGPRs** via opaque `s_mov`: "the fully unrolled tile decode is instruction-cache bound on gfx1151,
  bytes matter" (`:118-160`).
- **Ring depth 1:** the unrolled tile decode is about 10 KB of code, and ring 2 thrashes the I-cache. Ring 1 is 10 %
  faster for K2 and K4; ring 3-4 is 25-40 % slower (`:336-345`).
- **One asm block per weight pair:** `v_pk_mul_lo_u16 + v_mad_u32_u16` (the product), then `v_sad_u8` and
  `v_sad_hi_u8`, which leave both codebook values packed in one register (`:162-196`). The asm keeps the scheduler
  from hoisting the products and spilling.
- **`sched_barrier` every 4 pairs** (`:256-258`).
- **Affine fold:** the codebook map v = h*kinv + kbias is folded into the block sums (sum h*x, then
  kinv*S + kbias*sum x).
  - Exact mode is the default for dense. The folded mode is used for routed MoE: GLM 309.7 -> 293.1 us per call.
  - Folded is not bitwise: rel-L2 6.2e-4, NLL at the noise floor (`:236-245`).
- **LDS staging pad:** a `[warps][32][4]` half2 buffer had 4-way and 8-way bank conflicts. Padding the inner extent
  to 5 (coprime with 32) gave **+10 % mean decode** (RSH:209-227). Their rule: "Any `__shared__` array of one-dword
  elements here needs an innermost extent coprime with 32."
- **R-row verify (`EXL3_GEMV_R_DEC1`):** each weight pair is decoded once and applied to all R rows, bit-identical
  (`:286-292`). Re-decoding per row made each extra verify row cost about 7 % of dense GEMV time.

Achieved bandwidth (RSH:85-93, 125-130):
- "practical single-kernel streaming rate ~135 GB/s (`read_bw_vs_size.py`)".
- Grouped-MoE GEMV 131-135 GB/s on unique-expert bytes, lm_head about 200 GB/s.
- Dense GEMVs at 55-65 % of the 236 GB/s DRAM peak, i.e. 130-153 GB/s.
- Whole verify forward 78 GB/s effective (4.9 GiB in 67 ms).
- **Our ds4 is at about 4.35 GB/token x 34 tok/s, roughly 150 GB/s, already at or above their level.** Their plain
  decode (27.3) is below ours.

Rejected or neutral levers, all measured:
- int8-activation GEMV: noise.
- CPU expert offload: a loss.
- Graph replay: neutral.
- Build variants (prefetch-after-stage, DOT4 decode, A-ring): all within ±2 %.
- A 2.05 bpw pack: **slower**, 25.7 vs 44.1 tok/s, "K=2 trellis tiles this part badly" (RSH:24-32, 79-80, 203-213).

Byte budget per trunk forward after the int8 mixers (RSH:87-89): routed experts 47 %, dense EXL3 GEMVs 40 %,
mixers 13 %.
- The mixer at fp16 is per site 324x10240 + 10240x320 halves = 2 x 6.3 MiB, with 96 sites (2 per layer).
  That is about **1.2 GiB/token in fp16**; int8 halves it.
- "int8 GatedResidual mixer weights (−22 % of decode bytes)" took decode from 36.1 to 40.0 tok/s (RSH:83).
- Group scales (fn 128 along H*D, up 64 along rank) plus a GPTQ-rounded sidecar recover the fidelity
  (`tools/qwen/serve.py:545-548`, `hyperconnections.py:510-519`).

## 4. MoE, GDN, attention kernels

MoE decode (`exl3_dec.cu:751-760`, `:1030-1048`, `:1317-1322`):
- Batch-1 uses 2 launches:
  - A: gate+up of all selected experts, with an epilogue fusing the output rotations, swiglu and the down input
    rotation;
  - B: down, with an epilogue fusing the output rotation, the routing weight and the sum over experts.
- The shared expert can be folded in as slot topk+1 (`EXL3_DEC_SHARED_FOLD`, `:2628-2631`).
- R-row "union" MoE for verify:
  - each *unique* expert across the R rows is decoded once (grid.y = unique experts);
  - the unique-expert table is built on device with no host sync;
  - a 3rd trivial combine kernel keeps the reduction order fixed.
- k-split width `EXL3_MOE_CFG=2` (4 waves) is the decode setting (doc/env_vars.md:927-928).

GDN prefill (`exllamav3/vendor/fla/hip/gdn_fused_h.hip:1-15`):
- One hand-written WMMA kernel fusing recompute_w_u + chunk_fwd_h + chunk_fwd_o of fla.
- One workgroup of 8 waves per (batch, value head); K = V = 128, BT = 64.
- Wave i owns state columns h[:,16i:16i+16] in registers as 8 WMMA C tiles.
- The A matrix (kkt solve) still comes from Triton.
- Bit-exact with the Triton chain; it can side-write mid-chunk checkpoints (CKPT).
- The GDN decode kernels reduce partials in a fixed order, so decode is reproducible (doc/env_vars.md:384-387).
- Our GDN scan is 0.85 us/token/layer, about 31 us/token over 36 layers, about 3 % at 1050 tok/s. Low priority.

QSA sparse prefill attention (`exllamav3/modules/attention_fn/qsa_prefill.hip:1-8`):
- One wave per (query row, kv head), 32-key chunks in index-list order.
- QK is one fp32 WMMA chain with A = K rows and B = q; then online softmax; then PV via WMMA with A = V^T and B = p.
- Budget 2048 tokens (KPAD 2080).
- The prefill curve stays flat with context (1481 at 8K, 1347 at 256K), so the indexer plus sparse path scale well.
  Our 32K figure is about 1,038, also flat.

Hyper-connections: one-pass hc_mix_norm for prefill (`EXL3_HC_1PASS_PF`); the fused `hc_pf_apply_mix_norm` writes and
reads the (R,4,D) stream stack once instead of twice, +1.5 % (RSH:59-65).

## 5. Speculative decoding

- **MTP:** chained depth up to ndt (default `--ndt 7`). The rule usually drafts up to 3 and goes deeper only on
  confident chains (`tools/qwen/SERVE.md:40`).
- **Shipped "mix" rule** (`tools/qwen/spec_policy.py:1-6`, `:211-243`, `serve.py:718-724`):
  - **Dynamic depth from the drafter's own top-1 probability:** draft depth i+1 only if the running product
    reach >= thf = 0.6; keep token i only if reach >= thv = 0.3.
  - **TierRule (the default):** past depth 3 it uses stricter thf 0.95 and thv 0.85, so 5+ row verifies happen only on
    very confident chains.
  - **n-gram lookup**, `ngram=(2,3,5)`: when the context suffix has an earlier match of length >= 5, the draft is its
    continuation (up to 3 tokens).
    - One MTP forward still runs, so the drafter cache has no hole.
    - An incremental hash index replaces the O(context) scan, which cost 7.7 ms/round at 128K (`spec_policy.py:28-45`).
  - Copy-heavy prompts reach 54.7 tok/s against 42 for chat (SERVE.md:170).
- **Verify is exact:**
  - `EXL3_VERIFY_ATTN_LOOP` runs the R-row attention as R single-row loops;
  - `EXL3_QSA_ROWINV` makes the split plan row-invariant;
  - result: greedy speculative equals plain bitwise (doc/env_vars.md:893-918).
- **Cost model** (RSH:97-130): `ms/forward = 32.4 + 9.04 x rows`.
  - The marginal row is expensive because more rows touch more unique experts.
  - Tree or wide speculation is a dead end on this part.
  - The static optimum is 2-3 drafts. They found the static vs dynamic difference unresolvable within 1.3 tok/s of
    run-to-run drift (RSH:183-200).

## 6. Ideas for ds4, ranked by expected gain

1. **Check the hyper-connection mixer weight format; quantise to int8 with group scales.**
   - Kyojin: -22 % decode bytes, +11 % decode (36.1 -> 40.0).
   - If ds4 keeps `input_mix_weight_down/up` + `block_inject` in F16 (about 1.2 GiB/token) or F32 (2.4 GiB), int8
     saves about 0.6 GiB of our 4.35 GB/token, **about +15 % plain and MTP decode**. If they are already HT4 or Q8,
     skip.
   - Measure: bytes per module per token (our `bytes_by_module` equivalent), then tok/s and KLD on our PPL/KL set.
   - Their fidelity recipe: group 128 along H*D for fn/down and 64 along rank for up, plus GPTQ rounding.
2. **Pad the activation row pitch for the dense prefill GEMMs.**
   - Our K = 2560 / 4096 / 10240 rows are 5120 / 8192 / 20480 B, all multiples of 1024 B, which is their measured
     half-rate hazard. They got 50 % -> 87-90 % of peak on K=6144.
   - Measure: dense GEMM TFLOP/s (we see about 29) at lda = K vs lda = K+64, then e2e prefill at 8K.
   - Applies equally to our own WMMA GEMMs if they read A rows from global at a 2^n pitch.
   - Also check every fp32-output GEMM (GDN projections) for the hipBLASLt 6 TFLOP/s kernel; compute fp16 and widen.
3. **Grouped MoE prefill rebuild along mpw2 lines** (targets our 47 ms/layer at 8K, about 29 % of prefill):
   - device-built (expert, 128-row tile) list, so the launch count is independent of E;
   - gather once, scatter to top-k; activation-side rotation if HT4 experts are ever used;
   - wave-private B decode stage with no block barrier on B;
   - XOR-swizzled double-buffered A;
   - register-ring prefetch PF = 2;
   - **LDS-only barriers** (`s_waitcnt lgkmcnt(0); s_barrier`) and **addrspace(1) weight pointers**, so LDS waits do
     not drain global prefetch;
   - the half-wave k-permutation fragment trick;
   - fp32 down output with a deterministic weighted reduce.
   - Measure gate/up and down TFLOP/s separately; MXFP4 down at 13 TFLOP/s is the weak one, with K = 640 only 20 k-iterations.
   - Also try MT 64 vs 128 on our real routing histogram: about 80 rows/expert at 4096 tokens x top-10 / 512 experts.
4. **Dense HT4 prefill: keep the unpack, but fold H/su/sv into it at 1024 or more rows, and amortise it.**
   - Their in-GEMM decode analysis (decode VALU serialises with WMMA, paid per 128-row m-tile) argues **against**
     in-GEMM decode for dense 4096-row GEMMs.
   - Rough cost: dense trunk about 2.9 B params gives about 5.8 GB of fp16 scratch per chunk, about 30-60 ms per
     4096-token chunk. That is only 1-2 % of a chunk at our speed **if the chunk is 4096**. If ds4 unpacks per
     512-token ubatch, it is 8x that.
   - Measure first: unpack kernel share of prefill time and the ubatch size. Then try 4096-row chunks with tail merge,
     and 2048-row M-splits for the GEMM so A stays in the 32 MB MALL.
   - If an in-GEMM variant is tried anyway, use MT >= 256 and compare against a no-decode ablation, as they did.
5. **Verify-cost reductions for MTP:**
   - decode each HT4 weight pair once for all R verify rows;
   - union-dedup routed experts across rows;
   - keep the rows of one verify, or B*(K+1), under the MoE kernel's row granularity.
   - Measure ms per forward vs rows and fit a + b*rows; theirs is 32.4 + 9.04.
   - Then a **dynamic depth rule plus n-gram drafts**: their TierRule 0.6/0.3, stricter 0.95/0.85 past depth 3, and
     n-gram (min 2, K 3, Lmin 5). Expect the largest gains on code and copy prompts; they report 66.8 code vs 45.7 prose.
6. **Trellis matvec micro-tricks** (ds4 decode is already about 150 GB/s, so small gains):
   - SGPR constants plus ring 1 for the I-cache;
   - the packed `v_sad_u8`/`v_sad_hi_u8` pair decode;
   - u24 multiply split, kept in asm;
   - affine fold into the block sums;
   - LDS extents coprime with 32 (their +10 % came from a bank conflict found by arithmetic).
   - Measure with a no-decode ablation build to see whether our matvec is decode-bound or bandwidth-bound.
7. **Toolchain:** check which hipcc builds our kernels. Their HIP kernels lose about 20 % prefill with the system ROCm
   compiler vs the TheRock SDK.

Not worth porting: CPU expert offload (a loss on unified memory), int8-activation GEMV (noise), graph replay
(neutral), lower-bit packs for speed (2 bpw is slower).
