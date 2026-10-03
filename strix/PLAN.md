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
| Memory guard 12 -> 8 (server plan at 64K: 69.1 -> 73.1 GiB) | +3% (misses -10%) | `env.sh` |
| Q2 tier, opt-in (`DS4_GLM_Q2_FILE`, `DS4_GLM_Q2_WEIGHT_MAX=0.30`, `DS4_GLM_Q2_POOL_EXPERTS=1000`) | wall -13/-14% (1,500 scored tokens), ppl within 0.3% | `f5386ee`, `fbb45ef` |

Measured and dropped: chunked reads, overlapping uploads with reads, counting
only uncached spans against the span-cache limit, evicting past layers first
(-2 to -3.5%), W-TinyLFU, ARC, a pinned popularity hotlist (see README).

## Remaining potential, ranked

Decode is bound by expert reads, and the replay shows cache size dominates:
LRU misses fall from 19.4% at 4,790 experts to 13.3% at 5,869 and 5.5% at
8,000. Everything that adds effective capacity or removes bytes per miss
ranks first.

### 1. Q2 tier: validate, measure end to end, then decide the default

Status (2026-10-01, paused by request): implemented and committed, opt-in.
A selected expert that is not in the Q4 cache, has gate weight < 0.30 and is
not the heaviest of its step is computed from the Q2 GGUF, kept in a
1,000-slot LRU pool (6.6 GiB). With nothing below the threshold the output
is byte-identical. Measured on 1,500 scored tokens against the Q4 reference
(`DS4_PPL_DUMP` + `strix/pplcompare.py`): 19-22% of expert uses (12-13% of
the gate-weight mass) at Q2, perplexity x0.998-1.001, top-1 agreement 94%
(Go) / 89% (Markdown), wall time -13/-14%. The full Q2 model: x1.135, 81%.
Task check, cut short: hard-smoke cases 1-6, 10, 11 all passed with the
tier (as with plain Q4).

Remaining, in order (the scripts are in `~/.local/share/ds4/evals/`):
1. Finish the task validation: `tier/eval-queue.sh` (hard-smoke case 12,
   then the three Go tasks through ds4-server at 64K with the tier). Check
   the server path's pool allocation against GTT (the plan leaves ~10 GiB).
2. End-to-end speed on identical workloads: `overall/overall-queue.sh`
   (drop its wait-for-eval gate). Configs: V = near-vanilla (`ebb4f50`,
   upstream + Q4 prefill fix, ds4 defaults; worktree `~/repo/ds4-vanilla`),
   E = same code + `env.sh`, T = this branch + `env.sh`, Q = T + tier.
   Workloads: perplexity scoring of 1,500 Go tokens (identical tokens), and
   the 8K review prompt + 1,000 greedy tokens (prefill and decode t/s).
   Expected from piecewise numbers: ~1.9 -> ~3.0-3.1 tok/s decode.
3. If both hold: enable the tier in `scripts/ds4/env.sh` (needs the fork
   pushed and `DS4_REF` moved).
4. Refinements: express the threshold as a quantile of the gate weights
   (portable across models); S3-FIFO for the pool; a per-layer threshold;
   upgrading a Q2 expert to Q4 when it keeps coming back.

### 2. Faster routing handoff (unknown gain, medium effort)

The 54% "attention + router + readback" phase includes a GPU-to-host sync per
layer before reads can start. Measure GPU kernel time against sync latency
(for example with `rocprofv3`) before changing anything. If the sync
dominates, a spin-wait on the event, or pinned-memory polling of the routing
result, could shorten each of the 42 handoffs.

### 3. Cross-layer expert prefetch (demoted: ~1.1x at best, high effort)

Measured with the extended probe: top-12 prediction covers 64% of missing
experts, below the 70% bar; top-8 covers 52% at 1.3 wasted reads per layer
step. With the disk already at ~2.5 GB/s average, wasted reads compete with
demand reads. Revisit only after item 1 cuts the miss traffic.

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
