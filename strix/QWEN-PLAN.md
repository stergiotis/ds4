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

**halogen measured (2026-10-10, 0.15.3, its own `sweep -p 512,8192 -n 128
-d serial,mtp -r 2`, 10 prompt shapes, end to end over HTTP):**

| Decoding | tok/s, mean (min-max) | per token |
|---|---:|---:|
| serial, no speculation | 35.9 (33.5-38.1) | ~27.9 ms |
| MTP, 2.1-2.7 tokens committed per round | 55.0 (46.9-63.3) | ~18.2 ms |

Cold prefill is 1,694 tok/s at 512 and 1,844 at 8K; the repeat runs hit
halogen's prompt cache and are not usable.

The plain-decode gap to ds4 (40.2 ms at 512 context) is 1.44x. Half of
halogen's headline speed is its MTP, which commits far more per round than
ds4's measured +10-12% suggests. Its EULA permits benchmarking and does
not prohibit reverse engineering, but asks that derived kernels not be
redistributed as one's own work. So kernel-level profiles of halogen may
inform this plan, but its code must not be ported here.

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
| 12 | MTP on top (`--mtp`, measured upstream at +10-12%) | up to ~1.5x | halogen's MTP gives 1.53x over its serial decode with 2.1-2.7 tokens per round; measure ds4's acceptance and verify cost here first |

Requantizing from the Q8_0 tensors is a double quantization; Q8_0 is close
enough to lossless that it should be acceptable, but the gates decide.
**Disk:** a repacked GGUF carries the 95 GiB n-gram table again unless the
loader can take the n-grams from the current file. 55 GiB are free after the
download, so either teach the loader a separate n-gram file or free space
(the GLM Q2 GGUF, 90 GiB, serves only the paused Q2 tier).

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
