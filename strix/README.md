# Strix Halo notes for this fork

Target: AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB unified memory,
Ubuntu 26.04, ROCm 7.1, GLM-5.3-Flash **Q4_K** (178 GiB) with SSD streaming.

## Tooling

- `strix/bench.sh MODEL LABEL [ds4-bench args]` runs ds4-bench with
  teacher-forced decode on `speed-bench/promessi_sposi.txt` (frontiers
  2048..8192, 64 decode tokens each). It records DS4_* settings, disk
  throughput, the GTT peak and full frontier logits.
- `strix/compare.sh BASELINE CANDIDATE` checks frontier logits with
  `gguf-tools/quality-testing/compare_frontier_logits.py` (exact float32 bits).

ds4 is deterministic run to run: two baseline runs gave identical logits. A
change in expert-cache size is not bit-identical, because cached and streamed
experts take different kernel paths. Measured drift: top-1 unchanged at every
frontier, RMSE 0.12-0.18, max-abs <= 0.91.

- `strix/telemetry.py` samples the machine without root. It decodes amdgpu
  `gpu_metrics` v3.0 (GFX busy, socket/GFX/core power, temperatures, clocks,
  throttle residency) and adds CPU load, NVMe reads and GTT use. Its DRAM
  read/write fields report implausible values on this APU; use `dram_bw.sh`.
- `strix/dram_bw.sh` measures DRAM read bandwidth on all 16 LPDDR5X channels
  from the data-fabric PMU (event `0x1f + 0x40*N`, 64 B per beat). It needs
  `modprobe amd_uncore` and `perf_event_paranoid=-1`; the UMC PMUs report 0
  counters on this APU.

## How the machine is used (measured 2026-09-29)

Qwen3.8-27B Q8 in llama.cpp (dense, fully in RAM) decodes at 17.3 tok/s:

- DRAM: 203 GB/s read, evenly over all 16 channels, which is 79% of the
  256 GB/s peak. Decode is memory-bound, so faster kernels will not help much.
- Power: socket pinned at the 120 W limit. The GPU draws ~41 W; about 60 W
  goes to memory, fabric and SoC, so moving data costs about half the budget.
- CPU: two host threads burn cores without doing work.
  - The GPU-sync wait spins in the HSA runtime. `HSA_ENABLE_MWAITX=1` turns it
    into MWAITX: core power 8 -> 1 W and socket 123 -> 113 W, at identical
    speed and output.
  - The HSA async-event thread polls completion signals in a `wait_any` loop
    with no pause, ~16 W; no environment switch covers it in ROCm 7.1.
- Heat: with both threads spinning, Tctl sits at 100 °C with continuous CPU
  thermal throttling. With MWAITX it is 93.6 °C with almost none. The GPU
  stays near 63-67 °C at its 2,900 MHz maximum.

GLM-5.3-Flash Q4_K in ds4 (sparse MoE, experts streamed from NVMe) uses
the same machine very differently:

| | Qwen3.8-27B Q8 | GLM-5.3-Flash Q4 |
|---|---:|---:|
| GPU busy / clock | 99.8% / 2,899 MHz | 53.6% / 2,194 MHz |
| Package power | 120 W (at the limit) | 62.8 W |
| DRAM read | 203 GB/s | 73 GB/s |
| NVMe read | 0 | 2.4 GB/s avg, 3.0 peak |
| Tctl / NVMe temp | 94-100 °C / 30 °C | 77 °C / 53 °C |

Qwen is throughput-bound: memory bandwidth and power are saturated. GLM is
latency-bound: it waits for expert reads, while power, DRAM bandwidth and the
GPU stay about half idle. Shortening those waits (see PLAN.md) is the lever
for GLM; for Qwen it would gain nothing.

## Settings (environment)

| Variable | Value | Why |
|---|---|---|
| `DS4_GLM_MEMORY_GUARD_RESERVE_GB` | 8 | The default reserve is 18 GiB only if GTT >= 108 GiB and 32 GiB otherwise. GTT = 107 GiB here, so the default shrinks the expert cache to 49.9 GiB. 2026-10-01: 8 raises ds4-server's planned cache at 64K from 69.1 to 73.1 GiB (the server does not grow past its plan); 8K prompt + 1,500 tokens: misses -10%, decode 2.93 -> 3.01 tok/s, output identical, GTT peak 93 GiB. |
| `DS4_SSD_AUTO_CACHE_PCT` | 90 | So that the 80% plan is not the binding limit. |
| `DS4_ROCM_STREAM_MODEL_CACHE_GB` | 24 | The layer-span cache defaults to GTT/3 (36 GiB) but the memory plan counts only 7.6 GiB of it. Prefill and the ROCm per-layer decode path both use it: at 12 GiB decode drops to 0.27 tok/s. |
| `DS4_ROCM_ENABLE_STREAMING_STATIC_DECODE_MAP` | 1 | Maps every layer's decode spans once (14.5 GiB) instead of remapping per layer. Decode +3-12% on top of the pointer tables, same GTT peak, logits bit-identical. |
| `HSA_ENABLE_MWAITX` | 1 | Waits on GPU signals with MWAITX instead of spinning. For ds4 (A/B, ABAB, 2026-09-30): decode unchanged (2.84/3.61/3.12/3.24 tok/s at 2K-8K both ways), logits bit-identical, core power 17.4 -> 15.3 W, socket ~79 W either way. ds4's ~115% CPU is streaming work, not spinning, so the gain is much smaller than Qwen's 8 -> 1 W. |
| `DS4_ROCM_STREAM_FREE_RESERVE_GB` | 4 | Free GTT the expert cache keeps while it grows (default 16). At 16 the cache stalls at ~4,790 of its 5,869-expert budget. 1,500-token coding run: 16 -> 8 -> 4 GiB gives 2.36 -> 2.60 -> 2.75 tok/s, misses 18.9 -> 15.5 -> 13.9%, GTT peak 91 -> 99 -> 103 GiB of 107, MemAvailable min 21 -> 13 -> 9 GB, output byte-identical. |
| `DS4_ROCM_STREAM_CACHE_POLICY` | s3fifo | S3-FIFO eviction instead of LRU (fork commit, opt-in). Misses -6% (coding) and -13% (olympiad), decode +3% and +7% at headroom 4, output byte-identical. |

GLM-5.3-Flash Q4_K, 2026-09-29 (steady decode tok/s, teacher-forced):

| Config | Expert cache | 4K | 6K | 8K | Prefill 8K | GTT peak |
|---|---:|---:|---:|---:|---:|---:|
| defaults | 49.9 GiB | 2.55 | 2.14 | 2.36 | 28.3 | 89.7 GiB |
| reserve 18 | 63.9 | 2.86 | 2.41 | 2.67 | 26.7 | 105.8 |
| reserve 12, span 12 | 63.9 (plan) | 0.27 | 0.27 | 0.27 | 23.8 | 82.0 |
| reserve 18, span 24 | 63.9 | 2.66 | 2.57 | 2.60 | 27.4 | 91.5 |
| **reserve 12, span 24** | **69.9** | **2.76** | **2.67** | **2.74** | 25.2 | 96.9 |
| reserve 12, span auto | 69.9 | fails: GTT out of memory at the 6K prefill | | | | |
| **reserve 12, span 24 + in-place decode** | 69.9 | 3.04 | 2.92 | - | - | - |
| **+ static decode map** | 69.9 | **3.42** | **3.12** | **3.21** | 25.2 | 96.5 |

Real task (2026-09-29): an 8,085-token code-review prompt, 4,096 generated
tokens, seed 42, 64K context, same build:

| | ds4 defaults | tuned environment |
|---|---:|---:|
| Expert cache | 49.1 GiB | 69.1 GiB |
| Prefill | 324 s | 335 s |
| Decode | 1.87 tok/s | 2.38 tok/s (+27%) |
| Total | 42.0 min | 34.3 min |

Expert-cache size and policy, 2026-09-30/10-01 (1,500 greedy tokens,
16K context, coding prompt unless noted):

| Config | Cache reached | Misses | Decode |
|---|---:|---:|---:|
| headroom 16 (default), LRU | 4,790 | 18.9% | 2.36 tok/s |
| headroom 8, LRU | 5,396 | 15.5% | 2.60 |
| headroom 4, LRU | 5,699 | 13.9% | 2.75 / 2.82 (repeat) |
| headroom 4, S3-FIFO | - | 13.1% | **2.90** |
| olympiad prompt, headroom 4, LRU / S3-FIFO | - | 14.9% / 13.0% | 2.74 / **2.93** |

Real tasks through `ds4-server` (64K context, sampled, so token counts
differ) with headroom 4 and S3-FIFO, against the same tasks on 2026-09-30
with headroom 16 and LRU: expr 2.30 -> 2.59, glob 2.42 -> 2.58, semver
2.23 -> 2.38 tok/s. The server stops at its planned budget (5,244 experts at
64K); `DS4_GLM_MEMORY_GUARD_RESERVE_GB=8` raises the plan to 73.1 GiB
(~5,548 experts) at the same 103 GiB GTT peak (CLI, 64K); through the server (8K-token review prompt, 1,500 greedy
tokens, no KV reuse) guard 8 vs 12 gives misses -10%, decode 2.93 -> 3.01
tok/s, identical output, GTT peak 93 GiB. ~60 W socket, Tctl 76 °C.

The bench (teacher-forced, prefill to 8K) is 4-6% *slower* at 4K-8K with
headroom 4 or 8 (e.g. 3.63 -> 3.42 tok/s at 4K), and its 6K/8K frontier
logits differ (max |d| ~1.0, same argmax). Cause: at headroom 16 the memory
is so tight that prefill falls back from the q8->fp16 weight cache to the q8
kernels ("q8 fp16 cache budget exhausted"); every lower headroom produces the
same, other logits. ds4 picks that path from free memory at run time, so
prefill numerics depend on memory pressure, not only on the input.

### Offline cache replay

`DS4_GLM_ROUTE_TRACE=FILE` writes every decode routing decision and its gate
weights; `strix/cachesim/` replays traces on the CPU. Replaying LRU at the
size the real cache reached reproduces ds4's own miss count exactly (19.72%).
Three 3,000-token traces (coding, olympiad, code review) at 5,869 slots:

| Policy | Misses vs LRU |
|---|---:|
| S3-FIFO (10% small queue, promote on first reuse) | -5 to -12% (-9% combined) |
| ARC | -0.3 to -1.2% |
| W-TinyLFU (1% window) | +3 to +34%; break-even needs a ~60% window |
| LRU, past layers first | +18 to +20% (A/B: decode -2 to -3.5%) |
| pinned hotlist trained on another trace | +4 to +23% |
| Belady OPT (bound) | -63% |

Miss rate against cache size (LRU, coding): 4,000 -> 24.9%, 4,790 -> 19.4%,
5,869 -> 13.3%, 7,000 -> 8.5%, 8,000 -> 5.5%. Size dominates policy.

Cross-layer prediction (layer L+1's router on layer L's FFN input) covers
52/64/71% of the *missing* experts at top-8/12/16, at 1.3/2.7/4.3 wasted
reads per layer step: below the 70%-at-top-12 bar the plan set.

### Learned eviction (offline, 2026-10-09)

`strix/cachesim/evictml.py` evicts, on each miss with a full cache, the one
of 64 sampled resident experts whose predicted next use is furthest. This is
LRB-style (Song et al., NSDI'20). The predictor is a LightGBM model over the
routing history:
- age and the last three reuse gaps
- access counts decayed over 32 / 256 / 2048 tokens
- the count and rate since the request started
- the layer, and how many layer steps until that layer is routed again

It is trained leave-one-domain-out: two traces for training, the third held
out for testing. Miss rates at 5,869 slots (venv and logs in
`~/.local/share/ds4/evals/evict-ml/`):

| Test domain | LRU | S3-FIFO | best single decayed count | learned (rank + 1 DAgger round) | oracle among the 64 samples | Belady |
|---|---:|---:|---:|---:|---:|---:|
| coding | 13.32% | 12.65% | 12.83% | 11.94% (-5.6%) | 5.39% | 5.05% |
| olympiad | 13.99% | 12.33% | 11.87% | 11.15% (-9.6%) | 5.43% | 5.07% |
| review | 12.28% | 10.89% | 10.74% | 9.93% (-8.8%) | 4.83% | 4.50% |

- Sampling 64 candidates loses little: the sampled oracle is within 0.4
  points of Belady, and sampled LRU equals LRU.
- The learned policy beats S3-FIFO on every held-out domain, by 6-10%.
- No single decayed-count rule matches it, and the best half-life differs
  by domain.
- The model leans on short-window decayed counts and the request's own usage
  rate; recency matters little.
- It still closes only about a seventh of the gap to the oracle.

At the measured elasticity (guard 12 -> 8: misses -10%, decode +3%), the
gain is about +2-3% decode, roughly what ~300 more cache slots would give.
The evidence is three ~3,000-token traces from three prompts. A runtime port
needs per-expert feature state in the cache and the trees in C (~20-50 us per
eviction). Parked: more context would need token ids and the think/answer
phase in `DS4_GLM_ROUTE_TRACE`, plus 10+ diverse traces (~4 GPU hours).

## Code changes

- **Q4_K one-token decode reads the selected experts in place.** Before each
  routed MoE step, ds4 copied all 8 selected experts from their streaming-cache
  slots into a compact buffer (~113 MiB device-to-device per layer at Q4_K,
  ~4.9 GB per token). Two pointer-table kernels
  (`moe_gate_up_mid_decode_q4K_qwarp32_ptrs_kernel`,
  `moe_down_q4K_sum6_qwarp32_ptrs_kernel`) now read the slots directly, using
  the pointer tables that `cuda_stream_selected_load` already uploads.
  - Speed: decode +10-14% at 2K/4K/6K on top of the tuned environment.
  - Correctness: frontier logits are bit-identical to the compact path, and
    256 greedy tokens (`--temp 0`) are byte-identical.
  - `DS4_ROCM_Q4K_DECODE_COMPACT=1` restores the copy for A/B tests.
  - Reading in place is safe for decode: the next layer's reads start only
    after its router ran on the default stream, which is after this layer's
    kernels.

- **Q4_K routed path.** The sorted/tiled Q4_K prefill kernels fault with
  `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`. With serialized launches the
  fault lands in `routed_moe down`. Q4_K now stays on the per-pair path;
  `DS4_ROCM_Q4K_TILED=1` restores the tiled path for debugging. Ruled out as
  causes: expert-id mismatch (tiles and per-pair kernels bucket the same
  `selected_exec` ids) and NULL activation pointers (the dot helpers skip
  them). Prefill speed on the per-pair path matches Q2 (~30 tok/s), because
  expert streaming dominates.

## Tried and dropped

- **Splitting streamed reads into 1 MiB slices** across the read workers made
  decode 0-9% slower and prefill waits 17% longer. During decode, the
  per-tensor jobs already keep the workers busy; the idle workers in an
  off-CPU profile were idle while the GPU computed.
- **Overlapping uploads with reads.** The read profiler measured upload at 5%
  of read time (17.6 s vs 336 s), which is not enough to justify it.
- **Counting only uncached spans against the span-cache limit.**
  `ds4_gpu_set_model_map_spans` counts spans that are already cached when it
  decides whether to flush the whole range cache. Fixing that changed nothing:
  38 vs 36 span reloads at 24 GiB, and still 0.27 tok/s at 12 GiB (131
  reloads). The 12 GiB collapse is genuine capacity thrash: decode needs more
  than 12 GiB of layer spans.

## How one decode layer overlaps reads (code read 2026-10-09)

The asynchronous selected-expert load is **on by default on ROCm**, and has
been since upstream added ROCm GLM support (ef8d923). On Metal the GLM path
is opt-in (`DS4_METAL_ENABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD`), which is
where the name comes from (`glm_graph_use_streaming_selected_async_load`,
`ds4.c:50982`). One streamed decode layer on ROCm:

1. The main thread queues attention and the router on the default HIP
   stream and records an event.
2. A worker thread (`metal_graph_selected_async_load_run`, `ds4.c:23961`)
   waits on that event, reads back the 8 ids, decides the Q2 tier, looks the
   experts up in the cache and hands the misses to the 16 read workers
   (`cuda_stream_selected_load`). It returns with the reads still in flight.
3. Meanwhile the main thread queues the shared expert (`shared_first`,
   `ds4.c:51266`), so it runs on the GPU while the ids come back.
4. The routed Q4_K step (`cuda_stream_selected_apply_ptrs`,
   `rocm/ds4_rocm_runtime.cuh:5544`) waits for **all** pending reads and
   their upload (`cuda_stream_selected_finish_pending_missing(0)`), then
   queues the in-place kernels for all 8 experts.

The only overlap left on the table is running the cached experts while the
misses arrive ("hit-first"). ds4 has it for the IQ2 gate path
(`rocm/ds4_rocm_moe_launch.cuh:1388`), but upstream disabled it
(`split_selected = 0`, line 709) because the split "can expose partially
updated selected-expert state to the default stream". That is a slot-reuse
race. Its ceiling for Q4_K is small: 8 experts are ~113 MiB per layer, about
0.6 ms at ~200 GB/s, so ~25 ms per token, of which the ~86% that hit could
overlap, against ~137 ms of read wait. At most ~7%.

Metal has two extras ROCm lacks: early command-buffer commits, which HIP
does not need because launches are already asynchronous, and read advice
(`F_RDADVISE`, `madvise(WILLNEED)`, `ds4.c:20085`). Read advice is opt-in on
Metal too, goes through the page cache that the O_DIRECT reads bypass, and
turbo-fieldfare measured it as unstable (see below). There is nothing left
to port.

So the read wait is a serial dependency, not missing overlap. A layer's ids
exist only after its router, and the next layer's attention needs this
layer's output. Only prediction, or verifying more than one token per
forward pass, gets reads off that chain.

## Compared with turbo-fieldfare (2026-10-09)

[turbo-fieldfare](https://github.com/drumih/turbo-fieldfare) streams Gemma 4
26B-A4B on 8 GB Apple Silicon. It keeps 1.35 GB of shared weights resident
and the 12 GB of routed experts in per-layer files, with 16 LFU slots per
layer (12.5% of experts). It decodes at 5-6 tok/s on an M2 and 31-35 on an
M5 Pro (MLX, resident: 76-82). It is the same design as GLM streaming here,
at a tenth of the scale, and it publishes all 103 experiments, failures
included (`docs/experiments/` there).

Where its results match ours, which supports what we dropped:

| Finding | turbo-fieldfare | here |
|---|---|---|
| Explicit reads beat demand-paged mmap | 0.50 vs 3.97 tok/s | pread + O_DIRECT |
| Replacement policy matters less than capacity | LRU -> LFU: I/O -11% | S3-FIFO: misses -9%; size dominates |
| Distance to Belady (the offline optimum) stays large | 8-11 hit-rate points | -63% misses |
| Cross-layer prediction is too weak | copy previous layer: 7% | next router on this layer's input: 52% at top-8 |
| Speculative / Markov prefetch | slower, contends with demand reads | not built (prediction demoted) |
| Trace-trained policy fails on holdouts | layout -16%, slot split lost | hotlist +4-23% misses |
| Read advice | `F_RDADVISE` won and lost by state | not used (O_DIRECT) |
| Shared expert overlapping reads | +7.5% | already done; ~1 ms per token here |
| Hit-first routed compute | +14.4% | disabled upstream (race); <=7% ceiling |
| Quantized KV cache | K4/V4 failed quality | Kolibri FP8 KV: KL 0.104 |
| Compressing quantized experts | zstd 10% | not tried |
| `mlock` of resident weights | no gain (contention, not eviction) | not tried |

Their hit-first and shared-expert gains are large because their per-layer
compute after the router is a big share of a token. Here the read wait
(~137 ms) dwarfs it (~25 ms).

Things they found that may still apply here:

- Grouped full attention (one KV head read once for all query heads that
  share it) gave 2.2x decode at 110K context, once the reduction order was
  made to match the old kernel. GLM has 64 query heads on one KV latent; the
  ds4 ROCm DSA kernel has not been checked for per-head K/V reloads.
- Chunked prefill so that one expert read serves many rows (32 -> 128 rows:
  1.8x). This is what the faulting Q4_K tiled path would give us.
- Their method rules agree with ours and add one: an unclear A/B is called
  inconclusive and the default stays. They also forced identical token ids
  in A/B pairs (teacher forcing) so that changed routing could not
  confound a speed result.

## GLM-5.3 Q2 resident decode (2026-10-25)

The whole Q2 file (89.9 GiB) in GTT, no `--ssd-streaming`, 5a9924d, clang 21.
Results under `~/.local/share/llm-eval/glm53-q2-ds4-20261025-1142/` and
`~/.local/share/ds4/bench/20261025-1610-q2-decode-baseline/`.

| Profile, socket | Prefill 2K/4K/6K/8K | Decode 2K/4K/6K/8K | GTT peak |
|---|---|---|---:|
| balanced (morning) | 61.9 / 52.3 / 50.3 / 48.3 | 10.61 / 10.52 / 10.40 / 10.37 | 95.8 GiB |
| performance, 45 W cap | 52.9 / 44.3 / 43.1 / 41.4 | 9.67 / 9.64 / 9.58 / 9.48 | 95.0 GiB |

In the afternoon the socket sat at a hard 45 W under GPU load (slow-PPT
throttle engaged, GPU 2,077 MHz of 2,900, Tctl 63 °C); in September it
reached 120 W. Long generations through ds4-server fall from 11.6 tok/s at
1K context to 9.2 at 32K. The three llm-eval Go tasks: 1/3 (semver; expr
and glob spent the 32K budget thinking). Perplexity on `strix/ppl/`
(1,000 tokens): Go 1.771, Markdown 9.259; the dumps are the reference for
later lossy changes.

Where a token goes (45 W run):
- **GPU-bound, not launch-bound.** `DS4_GLM_DECODE_HOST_TIMING=1`: the host
  queues a token in 5.0-5.4 ms and then waits 95-101 ms. HIP graphs would
  hide host time that is already hidden.
- **1,745 kernel launches per token** (HIP launch log, symbols mapped from
  the binary): 373 `quantize_q8_0_f32` next to 373 Q8_0 matvecs, because each
  matvec quantizes its input again even when the input is shared; 112 BF16
  matvecs, 90 each of the mHC split/expand kernels, 2 routed-MoE kernels per
  layer.
- **The 11 flushes per token** (every 4 layers; `cudaDeviceSynchronize` on
  ROCm): `DS4_GLM_DECODE_FLUSH_INTERVAL=0` is +0.5-2% with bit-identical
  frontier logits.
- **Per stage** (`DS4_ROCM_DECODE_STAGE_PROFILE=1`, 2K, 106 ms/token with its
  syncs):

| ms/token | Stage | Note |
|---:|---|---|
| 29.2 | attn_output, 45 layers | KDA projections; matches the matvec bench |
| 27.1 | routed_moe, 42 layers | 2.4 GB of IQ2_XXS/Q2_K experts at ~90 GB/s |
| 18.7 | attention, 11 MLA layers | 1.7 ms/layer at 2K; per-head re-reads of the shared KV latent would explain it |
| 9.1 | attn_norm + ffn_norm, 90x | mHC mixing in several small kernels, ~100 us each |
| 8.3 | shared expert | ~135 GB/s |
| 3.1 | router | F32 4096x288 and top-k |

- **Dense matvecs** (`tests/bench_glm_matvec_rocm`): Q8_0 projections of
  8 MiB and up reach 188-226 GB/s, near the ~240 GB/s the chip delivers; the
  Q4_K kda_q/k shape 130 GB/s; matvecs under 2 MiB cost 7-19 us, mostly
  fixed overhead.

Exact (bit-identical) work in order of expected gain: routed-MoE decode
kernels toward bandwidth (up to ~13 ms), grouped MLA attention with the
reduction order kept (up to ~15 ms), Q4_K KDA matvec (~4 ms), one quantize
per shared input, mHC kernel fusion.

Done the same day, each bit-identical by `DS4_PPL_DUMP` on
`strix/ppl/go.txt` (byte `cmp` against the baseline dump; `strix/compare.sh`
only sees prefill) and, for the MoE kernels, the hash in
`tests/bench_glm_moe_rocm`. All under the 45 W cap, so the absolute numbers
are low:

| Change | Stage before -> after | Decode end to end |
|---|---|---|
| 0bd300a MoE loads in flight | one MoE step 1,023 -> 909 us | ~2-3% expected, not resolvable under throttling |
| 3c0c510 MLA attention per head pair | 19.7 / 17.6 -> 5.2 ms/token | 2K 9.07 -> 9.99, 4K 8.88 -> 9.91 tok/s (ABBA) |
| 09d478b BF16 hc matvec chunks | norm stages 11.6 / 12.7 -> 9.3 / 9.6 ms/token | ~2% |

Tried and dropped: the activation row of the Q4_K matvec in LDS (122 ->
120 us, noise). The Q4_K matvec reaches ~155 GB/s and is likely
ALU-bound at the capped clock. Quantizing each shared Q8_0 input once
would save ~200 launches (~1 ms) but needs the call sites in ds4.c to
share it. Under the cap, effects below ~3% are not measurable here:
end-to-end throughput drifted 9-20% within one session as the box heated.

## Known test failures (clang 21, 2026-10-25)

Two model-free ROCm tests fail at 5a9924d when built with the system hipcc
(ROCm 7.1, Ubuntu clang 21.1.8); neither comes from this fork's changes.
Everything else in `make test-rocm` and the other model-free ROCm and CPU
targets passes.

- `make test-deepseek41-rocm`: the bulk V4.1 F16 projection
  (`ds4_gpu_dsv41_projection_rows`) is not bit-identical to the per-row
  `ds4_gpu_matmul_f16_tensor` reference at shape 12 (width 20480, 24
  outputs, 33 rows); `tests/test_deepseek41_rocm.c:837` compares with
  `memcmp`. Shapes up to 7 rows at width 20480, and 513 rows at width 512,
  are exact. The test stops at the first failure, so later shapes are
  unchecked.
- `make test-qwen4-rocm`: `moe mxfp4 E=2560 F=640 slots=10 T=2:
  reduce+combine` reaches max|d| 2.66e-3 (rel 1.17e-3, got -0.282767 vs ref
  -0.285426), ~59x over its 2e-5 relative tolerance; the q8_0 case of the same
  check is at 3.9e-7. Suspected cause: clang 21 vs the clang 23 of
  `strix/hipcc23.sh`, not yet confirmed with a clang 23 build.

## Open

- The root cause of the Q4_K tiled-kernel fault.
- Decode is still I/O-stall bound: GPU ~48% busy, disk ~2.4 GB/s against
  ~6 GB/s available. The disk runs at ~4.45 GB/s while a layer waits and
  sits idle during attention; the asynchronous load already exists (see
  above).
