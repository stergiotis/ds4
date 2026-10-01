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
| `HSA_ENABLE_MWAITX=1` | none (cores -2 W) | `env.sh` |
| Expert-cache free headroom 16 -> 4 GiB (`DS4_ROCM_STREAM_FREE_RESERVE_GB`) | +17% (1,500-token run), real tasks +7-13% with S3-FIFO | `env.sh` |
| S3-FIFO eviction (`DS4_ROCM_STREAM_CACHE_POLICY=s3fifo`) | +3-7% on top | this branch |

Measured and dropped: chunked reads, overlapping uploads with reads, counting
only uncached spans against the span-cache limit, evicting past layers first
(-2 to -3.5%), W-TinyLFU, ARC, a pinned popularity hotlist (see README).

## Remaining potential, ranked

Decode is bound by expert reads, and the replay shows cache size dominates:
LRU misses fall from 19.4% at 4,790 experts to 13.3% at 5,869 and 5.5% at
8,000. Everything that adds effective capacity or removes bytes per miss
ranks first.

### 1. Server budget at 64K (small, measured half)

`ds4-server` does not grow its cache after prefill; at 64K context it stops at
the planned 5,244 experts. `DS4_GLM_MEMORY_GUARD_RESERVE_GB=8` raises the plan
to ~5,548 at the same 103 GiB GTT peak in the CLI. Next: the three Go tasks
through the server with guard 8, and a long-prefill (8K+) server request, as
the safety check.

### 2. Q2 copies for low-weight misses (expected -40% expert reads, large)

The Q2 file has the same tensors, shapes and expert order, at exactly half the
bytes per expert (IQ2_XXS gate/up, Q2_K down). Replaying traces with gate
weights: loading misses below the median gate weight as Q2 cuts reads by
41-42% with ~9% of the gate-weight mass computed at Q2 (upgrade-on-reuse
saves 42-49% but puts 14-15% of the mass at Q2). Output changes, so it needs
the hard-smoke and Go-task evals before and after.

Code survey: partial sums work (the router weight is applied in gate/up, the
down kernels only sum), so a layer can run a Q4 subset and a Q2 subset and
add them. The work is in the streaming plumbing: the single global pending
load and override, a second slab size class with a byte-based budget and
eviction, a per-map read fd (pread is tied to one fd today), reading gate
weights back with the ids, and the Q2 file must never become the current
model map. There is no in-place IQ2 decode kernel, so Q2 experts pay the
compact copy.

### 3. Faster routing handoff (unknown gain, medium effort)

The 54% "attention + router + readback" phase includes a GPU-to-host sync per
layer before reads can start. Measure GPU kernel time against sync latency
(for example with `rocprofv3`) before changing anything. If the sync
dominates, a spin-wait on the event, or pinned-memory polling of the routing
result, could shorten each of the 42 handoffs.

### 4. Cross-layer expert prefetch (demoted: ~1.1x at best, high effort)

Measured with the extended probe: top-12 prediction covers 64% of missing
experts, below the 70% bar; top-8 covers 52% at 1.3 wasted reads per layer
step. With the disk already at ~2.5 GB/s average, wasted reads compete with
demand reads. Revisit only after items 1-2 cut the miss traffic.

### 5. Root cause of the Q4_K tiled prefill fault (prefill speed, medium effort)

The sorted/tiled Q4_K kernels fault (`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`
in `routed_moe down`). Ruled out: an expert-id mismatch and NULL activation
pointers. Prefill already runs at ~25-30 tok/s on the per-pair path, and
expert streaming dominates it, so the expected gain is modest. Next step:
bounds-check instrumentation inside the tiled kernel.

### 6. Upstream reports (no local gain, reduces fork drift)

- The Q4_K tiled fault on gfx1151, with the serialized-launch reproduction.
- ds4's memory plan omits the layer-span cache (up to GTT/3), so budgets that
  look safe run out of GTT mid-prefill.
- The 108 GiB GTT threshold for the GLM-5.3 guard reserve penalizes 128 GB
  hosts whose GTT is set just below it.
- The 16 GiB streaming free reserve stalls the expert cache ~1,080 experts
  short of its budget on 128 GB hosts.
- Prefill picks the q8->fp16 weight cache or the q8 kernels from free memory
  at run time, so logits depend on memory pressure.
- The in-place Q4_K decode (`c0c7350`) and S3-FIFO, offered as upstream PRs.

## Not worth pursuing

- Resident/missing split compute for Q4_K. The expert compute that could
  overlap with reads is small next to the reads themselves.
- Pushing the GPU cache further by shrinking the span cache: below ~24 GiB,
  decode thrashes (0.27 tok/s at 12 GiB).
- A warm-start popularity hotlist (was item 3): routing popularity differs by
  domain; a hotlist trained on one trace raised misses on another by 4-23%.
