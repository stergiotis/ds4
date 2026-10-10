# Qwen3.8-Flash-Next on Strix Halo: speed-up plan

Goal: the open ds4 path at halogen's speed (44-49 tok/s decode, ~1,700 tok/s
prefill), so the closed binary is no longer needed. Model:
`Qwen3.8-Flash-Next-Q4.gguf` (165 GiB, 69.7 GiB resident, n-grams on disk),
ROCm 7.1, gfx1151.

## Where the time goes now (2026-10-09)

`ds4-bench --rocm`, `speed-bench/promessi_sposi.txt`, 128 greedy tokens;
logs in `~/.local/share/ds4/qwen-prof/`:

| Context | Prefill | Decode |
|---|---:|---:|
| 512 | 308-324 tok/s | 20.4-20.5 tok/s |
| 8K | 422-436 | 20.0-20.5 |
| 32K | 392 | 19.9 |

These match the ROCm 10 numbers in docs/QWEN38_FLASH_NEXT.md (Q4: 466 / 440 /
405 prefill, ~19.9 decode), so ROCm 7.1 is not the handicap. Decode is
GPU-bound: wall 49.6 ms per token at 8K, GPU 49.5 ms, host enqueue 3.7 ms
(`DS4_GPU_TRACE=1`). Host staging adds 1.2-1.4 ms before the GPU starts
(`DS4_QWEN4_TIMING=1`).

Decode per token at 8K, from `DS4_GPU_TRACE=1`:

| Call | GPU ms | GB/s | Weights |
|---|---:|---:|---|
| experts gate/up (10 routed + shared, 48 layers) | 9.96 | 106 | Q4_K, shared Q8_0 |
| gdn qkv+z projections (36 layers) | 7.71 | 208 | Q8_0 |
| experts down | 5.35 | 94 | MXFP4, shared Q8_0 |
| hc combine+mix before the FFN (48) | 5.21 | 121 | F16 low-rank |
| hc mix before attention/GDN (48) | 4.79 | 131 | F16 low-rank |
| output mixer + head | 3.10 | 218 | Q8_0 |
| gdn norm + out projection | 3.08 | 195 | Q8_0 |
| router + top-k (48) | 2.50 | 101 | **F32** |
| attention q/k/v/indexer projections (12) | 2.46 | 185 | Q8_0 |
| attention prep + indexer + core | 1.46 | | |
| gdn conv + gates (36, 38 us each) | 1.38 | 26 | F32 |
| attention out projection | 0.99 | 202 | Q8_0 |
| gdn scan | 0.78 | | |
| moe reduce + combine | 0.55 | | |
| ple n-gram mix (layer 1) | 0.21 | 170 | |

The decode reads ~6.6 GB of weights per token:

| Group | Per token |
|---|---:|
| GDN projections, Q8_0 | 2.2 GB |
| Routed + shared experts | 1.55 GB |
| Hyper-connection low-rank matrices, F16 (4 x 6.55 MB per layer) | 1.26 GB |
| Head, Q8_0 | 0.68 GB |
| Attention projections, Q8_0 | 0.66 GB |
| Router, F32 | 0.25 GB |

At the ~215 GB/s the best kernels here reach, that is a floor of ~31 ms, or
~32 tok/s. **Perfect kernels on these formats cannot reach halogen.** Halogen
reads at most ~4.5-5 GB per token (4.2 bits per weight, 62 GiB resident
against ds4's 69.7). The gap is half kernel efficiency (49.6 ms against
the 31 ms floor) and half bytes (the dense Q8_0/F16/F32 tensors).

Prefill at 8K (one 8,192-token chunk, 18.4 s): tiled expert gate/up 33%, tiled
expert down 13%, the two hyper-connection mixes 24%, attention core 6%,
GDN projections 5%, GDN scan 5%.

## Ideas, ranked

Phase 1 is lossless (bit-identical, or float noise checked with
`teacher_forced`/KL as for Kolibri). Phase 2 changes weights, so it needs the
quality gates below.

### Phase 1: kernels (target ~33 ms per token, ~30 tok/s)

Progress on 2026-10-09: decode went from 20.75 / 20.43 to 24.44 / 23.81 tok/s
at 512 / 8K (+18% / +17%), and the GPU time per token at 8K from 49.6 to
44.0 ms. Every step was A/B'd on alternating runs (`qwen-prof/ab.sh`, or
`ab2.sh` for two builds). Every step kept a 128-token greedy reply
byte-identical, and teacher-forced NLL at T = 1 stayed within +-0.003 of
2.3096. Prefill is unchanged.

| Commit | Change | Decode at 512 / 8K |
|---|---|---:|
| 934a7a8 | Q4_K gate/up on Kolibri's block kernel, which also takes a Q8_0 shared slot | +5.4% / +4.7% |
| cb6ecc0 | MXFP4 down with one block per lane | +0.6% / +0.6% |
| 3a255ab | hc down 10240 -> 320 on 160 blocks instead of 40 (119 -> 186 GB/s) | +3.7% / +3.7% |
| 951ec38 | vectorized Q8_0 dot for those shared slots (`dot<8>` was scalar) | +7.0% / +6.7% |

Tried and reverted: staging the layer-1 n-gram rows after layer 0 was
queued. Decode did not change (22.83 -> 22.87, 22.33 -> 22.33), so the
host staging was not on the GPU's critical path.

2026-10-10: df9a5f2 merges the GDN alpha/beta projections into one 96-block
launch. Decode is +1.8% (24.40 -> 24.85 at 512, 23.83 -> 24.20 at 8K).

Two more were tried and reverted. Both were exact, but the kernels stayed
at ~20 us per call:
- hc_norm with one barrier pair for the injection sums: 20.5 -> 19.3 us.
- a wave-shuffle router top-k with 2 barriers per round instead of ~10:
  19.7 -> 19.6 us, with bit-identical NLL and output.

Small dependent kernels here cost ~20 us mostly in dispatch latency, so
for them only fusion, meaning fewer launches, pays.

`__restrict__` on the six hot decode kernels (moe_q4k, moe_down_mxfp4,
matvec_f16_long, gdn_gates_f32, hc_mix, moe_reduce) changed nothing. The
per-kernel trace times matched to within 1%, and NLL was bit-identical, so
the generated code was equivalent. Reverted. The host C compiler (gcc,
clang or icc) does not matter either: decode is GPU-bound, with 3.7 ms of
host enqueue overlapped with 43 ms of GPU time.

**halogen measured (2026-10-10, 0.15.3, its own `sweep -p 512,8192 -n 128
-d serial,mtp -r 2`, 10 prompt shapes, end to end over HTTP):**

| Decoding | tok/s, mean (min-max) | per token |
|---|---:|---:|
| serial, no speculation | 35.9 (33.5-38.1) | ~27.9 ms |
| MTP, 2.1-2.7 tokens committed per round | 55.0 (46.9-63.3) | ~18.2 ms |

Cold prefill is 1,694 tok/s at 512 and 1,844 at 8K; the repeat runs hit
halogen's prompt cache and are not usable.

**Bytes per token (2026-10-10, `strix/dram_bw.sh` during steady decode,
GPU 94-96% busy; logs in `qwen-prof/bytes-*`):**

| | Decode tok/s | DRAM read | Bytes per token |
|---|---:|---:|---:|
| ds4 (df9a5f2), 512 context, 2,048 greedy tokens | 24.4 | 172.6 GB/s | 7.08 GB |
| halogen serial, sweep tg1024 | 35.7 | 152.3 GB/s | 4.27 GB |
| halogen MTP, sweep tg1024 | 52.3 | 181.0 GB/s | 3.46 GB per committed token |

halogen's plain-decode lead is all bytes: it reads 40% fewer per token at a
lower bandwidth than ds4 already reaches. At 4.27 GB and 173 GB/s, ds4 would
decode at ~40 tok/s. **Phase 2 (weight formats) is the main lever and MTP
the second**; Phase 1 kernel work now has diminishing returns.

The plain-decode gap to ds4 (40.2 ms at 512 context) is 1.44x. Half of
halogen's headline speed is its MTP, which commits far more per round than
ds4's measured +10-12% suggests. Its EULA permits benchmarking and does
not prohibit reverse engineering, but asks that derived kernels not be
redistributed as one's own work. So kernel-level profiles of halogen may
inform this plan, but its code must not be ported here.

**ds4 MTP measured (2026-10-10, df9a5f2, `ds4 --mtp-timing`, greedy,
512 tokens, 8K context; logs in `qwen-prof/mtp-1010-1446`):**

| Prompt | plain | 1 draft (`DS4_QWEN4_MTP_DEPTH=2`) | auto (default) | 2 drafts (`=3`) | first-draft acceptance |
|---|---:|---:|---:|---:|---:|
| Go semver function | 24.0 | 31.8 | 34.9 | 36.5 (1.52x) | 92-93% |
| essay | 23.7 | 27.9 | 27.8 | 27.0 (1.14x) | 68% |
| train arithmetic | 24.0 | 32.4 | 36.3 | 37.2 (1.55x) | 94-97% |
| Python refactor | 23.9 | 31.3 | 33.0 | 34.7 (1.45x) | 87-89% |
| arithmetic, `--think`, 1,536 tokens | 24.2 | | 36.3 (1.50x) | | 93% |

Every non-thinking MTP run was byte-identical to its plain run. The
thinking run diverged at a near tie ("9:40 + 6" against "9:40+6"): the
2-3 row verify sums in a different order than T = 1 decode.

So ds4's MTP already matches halogen's ratio (1.53x over its serial
decode) on code and reasoning. Upstream's +10-12% was the old single-draft
build. The depth-3 policy is right for code. On prose the second draft
rarely pays, and auto falls back as designed.

What a verify costs (`DS4_GPU_TRACE=1`, 105 verify passes of 2.94 rows
each): 67.2 ms against 41.9 ms for a T = 1 forward (1.60x), while each
cycle commits ~2.46 tokens:

| Call | T = 1 | verify (T ~ 3) | |
|---|---:|---:|---|
| experts gate/up | 6.23 | 15.73 | ~2.7x the distinct experts, so mostly inherent |
| experts down | 3.93 | 9.59 | same |
| hc down | 3.42 | 6.50 | dense, same bytes: should stay near 3.4 |
| hc up+gate | 3.63 | 5.97 | dense, same bytes |
| gdn conv+gates | 0.76 | 1.65 | |
| router matvec | 1.69 | 2.01 | |

The hc kernels at T = 2-3 still use the old per-row path (Phase 1
item 8), and fixing them should save ~5 ms per verify, ~2 ms per
committed token (~7%).

**halogen parity estimate:** Phase 2's ~4.3 GB per token gives ~40 tok/s
plain. With 1.45-1.55x MTP on code, that is ~58-62 tok/s against halogen's
55. On prose it is ~46-47.

| # | Idea | Expected gain | Notes |
|---:|---|---|---|
| 1 | ~~Q4_K expert gate/up decode~~ done | 9.96 -> 6.29 ms (167 GB/s) | Kolibri reaches 194 GB/s; the 10-slot grid may still be tunable |
| 2 | ~~MXFP4 expert down decode~~ done | 5.35 -> 4.01 ms (125 GB/s) | latency-bound at ~10 MB per call. Next: coalesced 340-byte row loads through LDS, more rows per wave |
| 3 | Hyper-connection mix: ~~down projection~~ (done); norm (20 us x 97 = 1.95 ms) and up+gate (3.7 ms at 172 GB/s) left | ~2 ms | fold the norm into the down projection's block reduction |
| 4 | Router: matvec 36 us at 146 GB/s, softmax+top-k 20 us per call | ~1.2 ms | top-k: wave-level argmax instead of 80 block barriers |
| 5 | Fuse the small GDN kernels (conv+gates 38 us, scan 22 us, reduce 13 us) | ~1.5 ms | latency-bound, not bandwidth-bound |
| 6 | ~~Read the layer-1 n-gram rows asynchronously~~ | none | measured and reverted, see above |
| 7 | Q4_K / MXFP4 prefill tiles that read the packed formats directly | prefill +50-100% | Kolibri's Q4_K tile (8876191): 611/927/789 -> 1125/1494/1161 tok/s |
| 8 | MTP verify rows (T = 2-3) still use the `moe_mv_reuse` kernels; check whether their Q8_0 shared slot is scalar too | MTP speed | apply the same fixes to T = 2-3 |

### Phase 2: bytes per token (target ~4.5-5 GB, 40-45 tok/s)

| # | Change | Saves per token | Risk |
|---:|---|---:|---|
| 8 | Hyper-connection matrices F16 -> Q8_0 | 0.63 GB | low |
| 9 | Router F32 -> BF16 (as Kolibri) | 0.13 GB | top-k can flip on near ties |
| 10 | GDN and attention projections Q8_0 -> Q6_K (or Q5_K) | 0.66 (0.89) GB | the main quality question |
| 11 | Head Q8_0 -> Q6_K; shared expert Q8_0 -> Q4_K | 0.28 GB | low-medium |
| 12 | MTP on top (`--mtp`) | 1.45-1.55x on code, 1.15x on prose | measured, see above; a ds4 Qwen setup in hackathon_2026's `scripts/llm-serve.sh` (none yet) should pass `--mtp` |

**Disk:** a repacked GGUF leaves out the 95 GiB n-gram table and points at
the Q4 GGUF with `qwen4exp.ple.ngram_file` (or `DS4_QWEN_NGRAM_FILE`). The
GLM Q2 GGUF was deleted on 2026-10-10 to make room (145 GiB free after).

### Phase 2 results (2026-10-10)

`gguf-tools/qwen4_requant.py` rewrites chosen tensor groups and omits the
n-gram table; `--simulate` stores the rounded values in a format the runtime
already runs, so quality can be measured before kernels exist; `--imatrix`
takes a dense imatrix collected with `DS4_QWEN4_DENSE_IMATRIX=FILE` (59k
calibration tokens from `gguf-tools/imatrix/dataset`, 640 tensors, 2.5
minutes of prefill). Requantizing from Q8_0 is a double quantization; the
gates below decide.

**Quality** (teacher-forced, `qwen-prof/quant/eval.txt`: 8,791 tokens of
Italian prose, an English ADR, Go and Python; `strix/pplcompare.py` against
the Q4 GGUF, NLL 1.5706; logs in `qwen-prof/quant/`). KL is over the
reference's top-20 plus a rest bucket, a lower bound of the full KL (an
earlier estimator mixed the two top-20 lists and read about half as high):

| Variant (cumulative unless noted) | Saves per token | ppl | KL mean | top-1 |
|---|---:|---:|---:|---:|
| router F32 -> BF16 | 0.12 GiB | bit-identical (the weights are BF16 values) | 0 | 100% |
| hc low-rank F16 -> Q8_0 | 0.60 GiB | +0.13% | 0.020 | 94.6% |
| + GDN qkv/gate/out Q4_K (simulated) | 0.96 GiB | +1.5% | 0.060 | 89.4% |
| same, real Q4_K file (`qwen-v2.gguf`) | | **+1.14%** | 0.059 | 89.7% |
| same, imatrix | | +1.9% | 0.053 | 90.3% |
| + attention q/k/v/out, indexer, shared gate/up Q4_K | 0.39 GiB | +4.8% | 0.106 | 86.3% |
| + head Q4_K | 0.30 GiB | +5.6% | 0.114 | 85.3% |
| same, imatrix | | +3.8% | 0.103 | 85.2% |
| + hc down Q4_K | 0.15 GiB | +6.2% | 0.117 | 85.1% |
| imatrix: GDN + attention q/k/v, indexer, shared gate/up; **attn_output Q8_0** | | **+1.0%** | 0.060 | 89.5% |
| same + head Q4_K | | +1.6% | 0.065 | 87.9% |
| imatrix: GDN + attn_output Q4_K only | | +3.7% | 0.090 | 86.8% |

Nearly all of the attention loss sits in the 13 `attn_output` tensors (12
full-attention layers and the MTP layer, 0.1 GiB at Q8_0): with them at
Q8_0 the rest of attention costs almost nothing, and alone at Q4_K they
cost +3.7%. This matches what halogen's authors report (their 12 o_proj
promoted to 8 bits). The loss also sits where they say: by quartile of
reference entropy, the hardest quartile carries most of the NLL increase
(V2: +0.049 there, -0.004 to +0.008 elsewhere), and the 4-bit runs are
over-confident there (mean top-1 probability 0.280 in the reference,
0.289-0.300 in the 4-bit variants, highest with attn_output at Q4_K).

**Speed of the real V2 file** (`qwen-v2.gguf`, 68.1 GiB: hc Q8_0, router
BF16, GDN Q4_K), after the kernels below:

| | Q8 (df9a5f2) | V2 |
|---|---:|---:|
| GPU per token at 8K | 43 ms | 35.2 ms |
| decode 512 / 8K | 24.85 / 24.2 | 28.9-29.2 / 28.0-28.2 |
| prefill 512 / 8K / 32K | ~320 / 435 / 392 | 323 / 437 / 416 |
| MTP code / essay (512 tokens) | 34.9 / 27.8 | 38.2 / 31.7, identical to plain |

**V3 file** (`qwen-v3.gguf`, 67.6 GiB): V2 plus attention q/k/v, indexer
and head at Q4_K, imatrix for every Q4_K tensor, attn_output and the shared
expert kept at Q8_0. The five MTP-layer attention tensors have no imatrix
entry (that layer does not run in prefill) and use plain Q4_K.

| | Q8 (df9a5f2) | V2 | V3 | halogen 0.15.3 |
|---|---:|---:|---:|---:|
| ppl vs Q4 GGUF / KL | 0 | +1.14% / 0.059 | +1.59% / 0.064 | ? |
| decode 512 / 8K / 32K | 24.85 / 24.2 / 19.9 | 29.2 / 28.2 / | 31.1 / 30.4 / 30.0 | 35.9 (serial, sweep) |
| GPU per token at 8K | 43 ms | 35.2 ms | 33.1 ms | ~27.9 ms |
| DRAM read, decode at 512 | 172.6 GB/s | 151.4 GB/s | 153.6 GB/s | 152.3 GB/s |
| **bytes per token** | **7.08 GB** | **5.23 GB** | **4.78 GB** | **4.27 GB** |
| MTP code / essay (512 tokens) | 34.9 / 27.8 | 38.2 / 31.7 | 44.4 / 34.6 | 59.2 / 45.7 (API) |
| prefill 8K / 32K | 435 / 392 | 437 / 416 | 420-454 / 414 | 1,175-1,844 |

**How much of the gap the bytes explain.** Plain decode at 512 context
was 41.0 ms per token against halogen's 27.9 (13.1 ms). V3 takes 31.1 ms,
so the formats closed 9.9 ms (75%). ds4 and halogen now read DRAM at the
same rate (~153 GB/s); the remaining 3.2 ms is what the extra 0.51 GB per
token costs at that rate (0.51 / 153 = 3.3 ms). So plain decode is fully
explained by bytes per token; the candidates for the last 0.5 GB are the
hc matrices (0.60 GiB at Q8_0), the shared expert (0.24 GiB at Q8_0) and
the routed experts' format. The effective bandwidth also fell from 173 to
153 GB/s when the bytes shrank, because ~6.6 ms per token are small
latency-bound kernels (hc norm 2.0, attention core 1.5, top-k 0.95, GDN
scan and conv 1.6, reduce 0.56) that do not shrink with bytes.

The bytes do not explain MTP or prefill. MTP: halogen gains 1.65x (code)
and 1.27x (essay) over its serial decode, ds4 1.40x and 1.11x; the verify
pass at T = 2-3 (experts at ~60 GB/s, hc up) is the next lever. Prefill:
halogen is 2.7-4x faster at every length; the tiled expert GEMMs take 8.4
of the 17.6 s per 8K chunk.

**MTP verify, first pass (2026-10-10, V3):** routed experts and hc up
weights are now read once per verify instead of once per token
(`moe_q4k_rows`, `moe_down_mxfp4_multi`, `hc_mix_multi`). MTP 44.4 ->
49.6 tok/s on code, 34.6 -> 39.0 on the essay, output identical to plain
decode. The verify pass (~3 rows) takes 53.4 ms against 31.2 ms for one
token: experts gate/up 11.8 ms (6.0 at T = 1), down 9.8 ms (4.0; the
multi kernel did not help down), GDN qkv 6.3 ms (4.4), hc up 3.8 ms (2.5).

**Rejected: hc down at Q4_K (V4 file, deleted).** On top of V3: ppl +2.52%
(V3 +1.59%), KL 0.069, warm decode 32.5 / 31.6 tok/s (V3 33.0 / 32.1), MTP
43.8 / 37.2 with code acceptance down from 96% to 89%. The K = 10240,
M = 320 projection launches only 20 blocks in `matvec_q4k_rows` (2.71 ms
against 2.19 for `matvec_q8_long`). hc down stays Q8_0.

**Prefill:** Qwen's Q4_K expert gate/up now runs on Kolibri's tile kernel:
456 -> 619 tok/s at 8K, 417-441 -> 591 at 32K (warm, alternating). Per 8K
chunk (12.9 s): experts down (MXFP4, Qwen tile) 2.39 s, hc up+gate 1.56,
expert gate/up 1.31, attention core 1.05, hc down 1.03, GDN qkv 1.0, hc
norm 0.93. halogen: 1,175-1,844 tok/s.

Measurement note: the first ds4 run after halogen stops is 5-7% slower
(V3 decode at 8K: 30.2 then 32.2 tok/s with the same binary and file).
A/B queues need a discarded warm-up run; the V3 numbers above are first
runs, warm V3 decodes at ~33 / ~32 tok/s at 512 / 8K.

Kernels this needed (each A/B'd; greedy identical where the math is):

- `matvec_q4k_rows`: R = 4 rows per wave, so each x load serves four rows.
  One row per wave was x-bound: x traffic was ~7x the Q4_K weight bytes,
  and the K = 6144 out projection ran at 47 GB/s (6.65 ms); now 2.0 ms.
  Bit-identical to the one-row kernel and independent of T, so MTP verify
  rows match plain decode exactly.
- `qwen_q8_direct<12>`: the direct WMMA prefill with a Q4_K group decode.
  The pack-to-F16 + BLAS path took 6.2 s of GDN projections per 8K chunk,
  now 1.0 s.
- `matvec_q8_long<2,T>` for the Q8_0 hc down (3.63 -> 2.24 ms) and
  `matvec_bf16<T>` for the router (1.81 -> 0.87 ms), both T-invariant.
- The hc up GEMM at prefill now also takes Q8_0 (it excluded Q8_0, which
  cost 6.3 s per 8K chunk).

## How changes are checked

- **Speed:** `ds4-bench --rocm` at 512 / 8K / 32K, three runs, before and
  after, in the commit message; `DS4_GPU_TRACE=1` per call.
- **Phase 1 output:** greedy continuations byte-identical where the math is
  unchanged; otherwise teacher-forced KL and top-1 against the previous
  build.
- **Phase 2 quality:** `score_official` against Alibaba's continuations
  (docs/QWEN38_FLASH_NEXT.md), teacher-forced KL against the current Q4 GGUF,
  then the `scripts/llm-eval` suite (hard-smoke, Go tasks with the 24,576
  thinking budget) against halogen's 11/12 and 3/3. Halogen is itself
  4.2 bpw, so the bar is halogen's results, not the Q8 reference.
- **GPU use:** stop halogen (`systemctl --user stop halogen`) and restart it
  afterwards; no runs at night (scripts/llm-eval/README.md).
