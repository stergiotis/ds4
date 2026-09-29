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

## Settings (environment)

| Variable | Value | Why |
|---|---|---|
| `DS4_GLM_MEMORY_GUARD_RESERVE_GB` | 12 | The default reserve is 18 GiB only if GTT >= 108 GiB and 32 GiB otherwise. GTT = 107 GiB here, so the default shrinks the expert cache to 49.9 GiB. |
| `DS4_SSD_AUTO_CACHE_PCT` | 90 | So that the 80% plan is not the binding limit. |
| `DS4_ROCM_STREAM_MODEL_CACHE_GB` | 24 | The layer-span cache defaults to GTT/3 (36 GiB) but the memory plan counts only 7.6 GiB of it. Prefill and the ROCm per-layer decode path both use it: at 12 GiB decode drops to 0.27 tok/s. |
| `DS4_ROCM_ENABLE_STREAMING_STATIC_DECODE_MAP` | 1 | Maps every layer's decode spans once (14.5 GiB) instead of remapping per layer. Decode +3-12% on top of the pointer tables, same GTT peak, logits bit-identical. |

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
