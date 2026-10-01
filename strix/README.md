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

## Open

- The root cause of the Q4_K tiled-kernel fault.
- Decode is still I/O-stall bound: GPU ~48% busy, disk ~2.4 GB/s against
  ~6 GB/s available. ROCm lacks Metal's asynchronous selected-expert load and
  readahead (`DS4_METAL_ENABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD`).
