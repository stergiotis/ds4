# GLM-5.3-Flash on Strix Halo: optimization plan

Scope: ds4 ROCm backend, gfx1151 (Ryzen AI Max+ 395, 128 GB unified memory,
GTT 107 GiB), GLM-5.3-Flash **Q4_K** (178 GiB) with SSD expert streaming.
Measurements live in [README.md](README.md). This file ranks what is left.

## Where decode time goes now

Steady decode is 3.2-3.6 tok/s (about 300 ms per token over 42 MoE layers).
From the ds4 async profiler (`DS4_ROCM_GLM_STREAMING_ASYNC_PROFILE=1`, 1024
decode tokens):

| Phase | Per token | Share |
|---|---:|---:|
| Attention and router on the GPU, then routing readback to the host | ~162 ms | 54% |
| Waiting for missing experts to arrive from the NVMe | ~137 ms | 46% |
| Shared expert, flushes | ~1 ms | <1% |

The disk peaks at 4.45 GB/s during decode, against ~3.8-6.3 GB/s available.
The expert cache holds ~70-77 GiB of ~145 GiB of Q4 experts; ds4 grows it
after prefill (5,301 to 5,877 experts).

## Done

| Change | Decode gain | Where |
|---|---:|---|
| Expert-cache budget (guard reserve 12, cache 90%, span cache 24 GiB) | +16% (real task +27%) | `scripts/ds4/env.sh` in hackathon_2026 |
| In-place Q4_K decode through pointer tables (no 4.9 GB/token copy) | +10-14% | `c0c7350` |
| Static decode map (existing opt-in) | +3-12% | `env.sh` |
| Q4_K prefill fault avoided (per-pair path) | stability | `ebb4f50` |

Measured and dropped: chunked reads, overlapping uploads with reads, and
counting only uncached spans against the span-cache limit (see README).

## Remaining potential, ranked

### 1. Cross-layer expert prefetch (expected 1.15-1.3x decode, high effort)

Idea: while layer L computes, read the experts layer L+1 will probably need,
so they are in the cache when L+1 routes.

Measured feasibility (`DS4_GLM_PREDICT_PROBE=1`, branch `prefetch-probe`):
running layer L+1's router on layer L's FFN input catches **60.7%** of L+1's
real experts. By layer that is 0.41-0.77, and it is highest in layers 30-40.
Most steps catch 4-6 of 8.

Expected effect: it could hide roughly half of the ~137 ms of read wait per
token, but the ~40% wrong predictions add disk traffic on a disk already near
its peak.

Steps:
1. Extend the probe to report recall of the top 12 and top 16 candidates, and
   the share of *missed* experts covered, i.e. predicted experts that were not
   already cached. That share is what prefetch actually saves.
2. If misses covered at top-12 exceed ~70%, build it:
   - a second, lower-priority job set in the read pool (today it runs one set
     at a time);
   - prefetch slots that eviction skips until they are used or superseded;
   - running L+1's router right after L's router, with an asynchronous
     readback, on the same worker pattern as the existing selected async load;
   - a per-layer switch, so prefetch runs only where recall is high.

   The template is CUDA's `ds4_gpu_stream_expert_cache_prefetch`.
3. Acceptance: decode speed via `strix/bench.sh`, and byte-identical greedy
   output. Prefetch must only change when data arrives, never results.

### 2. Faster routing handoff (unknown gain, medium effort)

The 54% "attention + router + readback" phase includes a GPU-to-host sync per
layer before reads can start. Measure GPU kernel time against sync latency
(for example with `rocprofv3`) before changing anything. If the sync
dominates, a spin-wait on the event, or pinned-memory polling of the routing
result, could shorten each of the 42 handoffs.

### 3. Warm start with a GLM-5.3 hotlist (first-request speed, low effort)

A cold cache decodes at ~2.2-2.9 tok/s, a warm one at 3.2-3.6. ds4 can seed
the cache from a popularity list (`DS4_ROCM_STREAMING_EXPERT_HOTLIST`,
`--ssd-streaming-preload-experts`), but it ships none for GLM-5.3. Its
profiler that writes such lists is Metal-only (`ds4.c` ~70717); un-gate it for
ROCm, record a list from typical coding sessions, and preload it at startup.

### 4. Root cause of the Q4_K tiled prefill fault (prefill speed, medium effort)

The sorted/tiled Q4_K kernels fault (`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`
in `routed_moe down`). Ruled out: an expert-id mismatch and NULL activation
pointers. Prefill already runs at ~25-30 tok/s on the per-pair path, and
expert streaming dominates it, so the expected gain is modest. Next step:
bounds-check instrumentation inside the tiled kernel.

### 5. Upstream reports (no local gain, reduces fork drift)

- The Q4_K tiled fault on gfx1151, with the serialized-launch reproduction.
- ds4's memory plan omits the layer-span cache (up to GTT/3), so budgets that
  look safe run out of GTT mid-prefill.
- The 108 GiB GTT threshold for the GLM-5.3 guard reserve penalizes 128 GB
  hosts whose GTT is set just below it.
- The in-place Q4_K decode (`c0c7350`), offered as an upstream PR.

## Not worth pursuing

- Resident/missing split compute for Q4_K. The expert compute that could
  overlap with reads is small next to the reads themselves.
- Pushing the GPU cache further by shrinking the span cache: below ~24 GiB,
  decode thrashes (0.27 tok/s at 12 GiB).
