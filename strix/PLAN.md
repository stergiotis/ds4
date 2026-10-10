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

Blocked (2026-10-10): the GLM Q2 GGUF was deleted to make room for the Qwen
Phase 2 files. Resuming needs a ~90 GiB re-download, and the disk is nearly
full (8.2 GB free of 935 GB after `qwen-v3.gguf`). Item 2 does not need the
Q2 file, apart from its step 3.

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

### 2. Defer missing experts by one layer (largest upside, small quality cost)

From KTransformers' "Expert Deferral" (SOSP'25, §4): the lowest-scoring
experts of layer k add their output to the residual one layer late, so their
work overlaps layer k+1 instead of blocking it. On DeepSeek-V3, deferring 3
of 8 cut layer time 26% and raised decode 33%; LiveBench dropped 0.5% with 6
deferred, against 13.3% for skipping the same experts. They keep at least
2 immediate experts and never defer on the last layer.

Here the deferred set would be the misses, not the lowest scores. That
attacks the serial dependency item 4 describes, which this plan had assumed
only prediction or multi-token verification could break. Per layer, the
misses cost ~3.3 ms of reads (137 ms / 42) against ~3.9 ms of attention and
routing (162 ms / 42), and the disk work per token (~45 misses x 14.1 MiB
~0.63 GB at 4.45 GB/s) fits under the compute. Ceiling: ~300 -> ~170-200 ms
per token (~5-6 tok/s), after which the disk is the floor. A layer with many
misses still stalls at the end of layer k+1.

1. Quality first, without speed work: an emulation that waits for the reads
   as today but adds the missing experts' output after layer k+1. Rules as
   in the Q2 tier: the heaviest expert of a step is always immediate.
   Judge with `DS4_PPL_DUMP` + `strix/pplcompare.py` on the 1,500 scored
   tokens; only ~1 of 8 experts is deferred here, milder than their 3-6.
2. Speed: `cuda_stream_selected_apply_ptrs` waits for all pending reads
   before the routed step. Deferral moves the wait for the misses to the
   next layer's routed step. The deferred experts' slots must stay pinned
   until then; a slot-reuse race is why upstream disabled hit-first
   (`rocm/ds4_rocm_moe_launch.cuh:709`).
3. Interaction with the Q2 tier: both serve low-weight misses, Q2 with fewer
   bits on time, deferral with Q4 bits late. Measure each and the
   combination.

### 3. Cache-aware routing (cheap, lossy, weak evidence)

From Colibrì (`CACHE_ROUTE`, docs/tuning.md): when a cached expert sits just
outside the top 8 and its score is close to a selected expert that would
miss, use the cached one. It removes misses instead of hiding them and
generalises the Q2 tier. The only number is one benchmark row, 3.33 vs 2.4
tok/s on a GB10, with no quality data. Needs the routing readback to carry
~12 ranks and their scores. Judge like the Q2 tier: misses per token,
perplexity and top-1 agreement, as a function of the score-ratio threshold.

### 4. GLM MTP on the streamed Q4 path (unmeasured, cheap to measure)

The read wait is serial: a layer's ids exist only after its router, and the
next layer needs this layer's output (README, "How one decode layer overlaps
reads"). The asynchronous load and the shared-expert overlap are
already in place. Verifying two tokens per pass
pays for one attention pass and one routing handoff per layer instead of
two. Those make up the memory-bound 54% phase, which costs about the same
for two rows as for one. Reads become the union of both tokens' misses. ds4
already has GLM's built-in MTP (`--mtp`, commits up to two tokens;
docs/SPECULATIVE_DECODING.md), but there is no recorded run on streamed Q4
with the tuned `env.sh`.

1. Run the 8K review prompt with 1,000 greedy tokens, with and without
   `--mtp --mtp-timing`. Record acceptance, decode tok/s, misses per token
   and the GTT peak (the MTP block and its state take memory from the expert
   cache).
2. Check which routed path the 2-3-row verify pass takes. The in-place
   Q4_K kernels and the selected-load table require `n_tokens == 1`
   (`rocm/ds4_rocm_moe_launch.cuh:719`), so a verify pass falls through to
   the batch path. If MTP loses, a two-row in-place variant over the union
   of experts is the fix to try.
3. Batched verify reorders floating point, so greedy output may differ.
   Judge it like the Q2 tier: perplexity and top-1 agreement on scored
   tokens, then the task evals.

Rough bound: at ~70% acceptance, a pass costs ~162 ms plus ~1.7-1.9x one
token's read wait for 1.7 tokens, which works out to about +20-30% decode.
That holds only if the verify pass is about as cheap as a decode step, and
it ignores the draft block's own cost.

Colibrì measured MTP while streaming as a net loss on GLM-5.2 (Strix Halo):
expert loads rose from ~331 to ~914 per token at 52% acceptance, 2.8x rather
than the 1.7-1.9x assumed above. ds4's larger cache absorbs more of the
union, but record misses per token first; they decide this item.

### 5. Shorten the 54% phase: routing handoff and attention (unknown gain, medium effort)

The 54% "attention + router + readback" phase includes a GPU-to-host sync per
layer before reads can start. Measure GPU kernel time against sync latency
(for example with `rocprofv3`) before changing anything. If the sync
dominates, a spin-wait on the event, or pinned-memory polling of the routing
result, could shorten each of the 42 handoffs. A third option is
KTransformers': start the expert reads from a host function queued on the
stream (`hipLaunchHostFunc`) instead of a host sync. With whole-token graphs
it gave them 1.23x decode, where launch overhead had been 21-73% of GPU time
(from the paper's sections 2.3 and 6.4; a second reading reached only the
abstract, so treat the figure as unconfirmed).

Also check whether the DSA decode attention reads the shared KV latent once
per query head (64 heads, 1 KV head). Reading it once per head group was
turbo-fieldfare's 2.2x at 110K context (README, "Compared with
turbo-fieldfare"); at the 8-64K contexts used here the gain would be smaller.

### 6. Cross-layer expert prefetch (demoted: ~1.1x at best, high effort)

Measured with the extended probe: top-12 prediction covers 64% of missing
experts, below the 70% bar; top-8 covers 52% at 1.3 wasted reads per layer
step. With the disk already at ~2.5 GB/s average, wasted reads compete with
demand reads. Revisit only after item 1 cuts the miss traffic. Colibrì's
numbers on Strix Halo agree: real next-layer loads at 71.6% recall raised the
hit rate from 75-77% to 86-87% but decode only ~2%, and hint-only prefetch
under O_DIRECT lost 30% (issue #200).

### 7. Root cause of the Q4_K tiled prefill fault (prefill speed, medium effort)

The sorted/tiled Q4_K kernels fault (`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`
in `routed_moe down`). Ruled out: an expert-id mismatch and NULL activation
pointers. Prefill already runs at ~25-30 tok/s on the per-pair path, and
expert streaming dominates it, so the expected gain is modest. Next step:
bounds-check instrumentation inside the tiled kernel.

### 8. Upstream reports (no local gain, reduces fork drift)

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

- Resident/missing split compute ("hit-first") for Q4_K. The cached
  experts' compute is ~25 ms per token against ~137 ms of read wait, a
  ceiling of ~7%. Upstream disabled the IQ2 version for a slot-state race
  (`split_selected = 0`, `rocm/ds4_rocm_moe_launch.cuh:709`).
- Porting Metal's asynchronous selected-expert load: ROCm has had it on by
  default since ef8d923 (README, "How one decode layer overlaps reads").
  Metal's early commits are unnecessary under HIP. Its read advice
  (`F_RDADVISE`, `madvise(WILLNEED)`) acts on the page cache, which the
  O_DIRECT reads bypass, and turbo-fieldfare found it unstable.
- Compressing experts, `mlock` of the resident weights, packed K/V caches:
  turbo-fieldfare measured all three and dropped them (zstd saves 10%;
  mlock recovered 0 ms; K4/V4 failed quality). The Kolibri FP8 KV result
  here agrees.
- Pushing the GPU cache further by shrinking the span cache: below ~24 GiB,
  decode thrashes (0.27 tok/s at 12 GiB).
- A warm-start popularity hotlist (was item 3): routing popularity differs by
  domain; a hotlist trained on one trace raised misses on another by 4-23%.
- Dropping low-weight experts outright (top-k 8 -> 6, adaptive top-p,
  zeroing low-gate misses). Cheaper reads, but KTransformers measured
  skipping at 13.3% LiveBench loss against 0.5% for deferral, and Colibrì's
  zeroing cost +2.9% perplexity at gate < 0.03 and +41% at < 0.05 (OLMoE).
  The Q2 tier and item 2 serve the same experts without dropping them.
- Computing experts on the CPU as the main path (KTransformers' design). It
  relies on CPU DRAM bandwidth far above PCIe; here CPU and iGPU share the
  same 256 GB/s. The idle cores may still consume deferred experts (item 2).
- A second NVMe as a mirror with striped reads: Colibrì measured +37.5%
  (0.80 -> 1.10 tok/s), but this host has one drive. Hardware, not code.

## Sources from other engines (clean-room, 2026-10-10)

Read from papers, READMEs, docs and issue threads only, no source code.
Their numbers are the authors' own, often single runs on other models and
hardware.

- KTransformers: SOSP'25 paper
  (https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf),
  https://lmsys.org/blog/2025-10-22-KTransformers. Items 2 and 5.
- Colibrì (C, CPU/CUDA/Vulkan/Metal, GLM-5.2 on Strix Halo at 1.83 tok/s):
  https://github.com/JustVugg/colibri, docs/tuning.md, docs/benchmarks.md,
  docs/multidisk.md, issue #200,
  https://sleepingrobots.com/dreams/colibri-744b-moe-strix-halo/. Items 3,
  4 and 6. Its O_DIRECT vs buffered reads on Strix Halo: +65%.
- deepseek-v4-flash-mlx (https://github.com/ssd-moe/deepseek-v4-flash-mlx):
  LRU expert cache, parallel pread, top-k 6 -> 5 for ~+10%. Nothing new.
- ornith-flight (https://github.com/instax-dutta/ornith-flight): never run
  on the real model; nothing to take.
