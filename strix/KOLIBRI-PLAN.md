# Kolibri-1 on Strix Halo: speed-up plan

Scope: ds4 ROCm backend on gfx1151 (Ryzen AI Max+ 395, 128 GB unified memory,
LPDDR5X with about 256 GB/s theoretical bandwidth), Kolibri-1 as the lossless
F8 GGUF (75.7 GiB) and the Q4_K-experts GGUF (42.8 GiB). How the model is
implemented, the correctness harness and the benchmark method are in
[docs/KOLIBRI.md](../docs/KOLIBRI.md). This file ranks what could make it faster.
The decode gains below are estimated from per-call times
(`DS4_KOLIBRI_TRACE=1`) and byte counts; each item states how it would be
checked.

## Where the time goes now

Measured on 2026-10-04 with `tests/kolibri/bench.sh` (median of 3 runs):

| GGUF | prefill 512 / 8K / 32K, tok/s | decode at 512 / 8K / 32K, tok/s |
|---|---|---|
| F8 | 674 / 922 / 776 | 36.5 / 36.1 / 31.8 |
| Q4_K experts | 611 / 918 / 773 | 36.8 / 35.6 / 31.4 |
| F8, after item 1 | 706 / 922 / 776 | 41.4 / 39.3 / 36.1 |
| Q4_K experts, after item 1 | 609 / 916 / 772 | 40.7 / 38.6 / 35.5 |

The per-call breakdown below predates item 1.

### Decode

Per-call times from `DS4_KOLIBRI_TRACE=1` (timing events between calls, no
added syncs; the trace itself costs about 4.5%, 36.5 -> 34.9 tok/s). F8 at
512 context, per decode token: GPU 28.4 ms, host enqueue 1.4 ms.

| Call, per token (F8, 512 context) | Calls | Time | Per call | Bandwidth |
|---|---:|---:|---:|---:|
| experts gate/up (6 routed + shared) | 50 | 4.48 ms | 90 us | 211 GB/s |
| LM head (BF16, 128000 × 2560) | 1 | 3.92 ms | 3.9 ms | 167 GB/s |
| o projection | 50 | 3.85 ms | 77 us | 211 GB/s |
| q projection | 50 | 3.78 ms | 76 us | 215 GB/s |
| attention, sliding layers | 40 | 2.84 ms | 71 us | 15 GB/s |
| experts down | 50 | 2.36 ms | 47 us | 200 GB/s |
| router matvec (F32) | 50 | 2.35 ms | 47 us | 84 GB/s |
| FFN reduce + norm | 50 | 1.24 ms | 25 us | |
| attention, full layers | 10 | 0.78 ms | 78 us | 15 GB/s |
| attention residual + norm | 50 | 0.72 ms | 14 us | |
| router top-k | 50 | 0.59 ms | 12 us | |
| k and v projections | 100 | 1.11 ms | 11 us | 121 GB/s |
| q/k norm, RoPE, KV store | 50 | 0.35 ms | 7 us | |

- **Decode is not launch-bound.** The host enqueues a token in 1.4 ms while
  the GPU works for 28.4 ms, so each interval above is GPU time. That rules
  out GPU graph capture as a fix (it is also measured neutral to slower on
  gfx1151 in another fork) and kernel fusion as an end in itself.
- **The large matvecs are at bandwidth.** q, o and the F8 experts read at
  200-215 GB/s, the practical ceiling here. Together they are 14.5 ms of
  the 28.4 ms, and only fewer bytes make them faster.
- **The time lost is in small calls.** Sliding attention reads about 1 MB
  of K/V per call (512 keys) and takes 71 us at any context; full attention
  takes 78 us at 512 keys and 162 us at 8K (16 MB, 105 GB/s). Its grid is Hkv × splits = 4 × 4 waves at 512 keys, too few
  to cover the latency. The F32 router reads at 84 GB/s, the norms run one
  256-thread block per token, and k and v are too small to reach bandwidth.
- **Q4_K experts are slow, not small.** The same trace on the Q4_K GGUF
  gives gate/up 4.58 ms at 126 GB/s and down 2.84 ms at 102 GB/s, so Q4_K
  experts take as long as F8 ones with half the bytes. Everything else is
  identical.
- At 8K, only full attention changes (0.78 -> 1.62 ms); at 32K the ten full
  layers read about 0.65 GB of f16 KV per token.

### Prefill

From the trace, F8, 8192-token cold prefill in 2048-token chunks, per chunk
(2.16 s): experts gate/up 39%, experts down 17%, o projection 12%, full
attention 7%, q projection 7%, sliding attention 4%, router matvec 4%, shared
expert 3%. The o projection (6144 -> 2560) takes 5.1 ms per call against
3.0 ms for q (2560 -> 6144), with the same weight count, which points at the
tile table. The F32 router matvec takes 1.7 ms per call for a 2560 × 384
matrix, and spends about 2.5 ms of host time per chunk.

## Ideas, ranked

Decode gains are from the trace: the time a call takes now minus what it
would take at ~200 GB/s or at a few microseconds of fixed cost. Together the
lossless F8 items 1-5 remove about 7.5 ms of 28.4 ms, roughly 36 -> 48
tok/s.

| # | Idea | Expected gain | Effort | Correctness risk |
|---:|---|---|---|---|
| 1 | ~~Decode attention with enough waves~~ done | decode +13% at 512, +14% at 32K | small-medium | float noise (summation order) |
| 2 | Router as BF16 at full bandwidth, top-k fused | decode -2.0 ms (+7%) | small | none (source is BF16) |
| 3 | Norm and reduce kernels across more than one block per token | decode -1.5 ms (+5%) | small | float noise (reduction order) |
| 4 | k and v (or q, k and v) in one launch | decode -0.6 ms (+2%) | small | none |
| 5 | LM head at full bandwidth | decode -0.8 ms (+3%) | small | none |
| 6 | Q4_K expert decode kernels at bandwidth | Q4_K decode -3.1 ms (+11%) | medium | none |
| 7 | F8-specific WMMA expert tile for prefill | prefill +30-50% | medium | none |
| 8 | Prefill tile tuning (o projection, router) and chunk size | prefill +10-15% | small | none |
| 9 | FP8 KV cache for the full-attention layers | decode at 32K +10% | medium | small, measurable |
| 10 | Prefill attention shared across all 12 heads of a KV group | prefill at 32K +10-20% | medium | none |
| 11 | Speculative decoding with n-gram drafts | 1.3-1.8x on repetitive output | medium-large | none (exact verify) |
| 12 | Q8_0 LM head | decode -2 ms (+7%) after 5 | small | small, measurable |

Dropped: GPU graph capture (host enqueue is 5% of the token; neutral to slower
on gfx1151 elsewhere). Folded into 2-4: the generic "fewer kernels per
layer" item.

1. **Decode attention.** 50 calls at ~70 us read 1-16 MB each. At T = 1 the
   GQA kernel launches Hkv × splits waves of 32 threads, with 128-key splits:
   16 waves at 512 keys. Smaller splits (32 keys) or one wave per (KV head,
   query head group, split) give the GPU hundreds of waves; the merge kernel
   then folds more partials. Target: ~15 us per call at 512 keys
   and bandwidth-bound at 8K and beyond.

   **Done:** a block per (KV head, 128-key split) of 12 waves, each scoring
   32-key tiles lane-per-key for 4 of the 12 query heads, and a merge that
   loads all split maxima at once. 71 -> 20 us per call at 512 context,
   full layers at 32K read K/V at 178 GB/s (was 105 at 8K). Teacher-forced
   decode: top-1 0.985, KL 0.0076 (was 0.975, 0.0082).
2. **Router.** The converter widens the BF16 router gate to F32, and the
   F32 matvec reads at 84 GB/s: 2.35 ms for 0.2 GB. As BF16 at 200 GB/s it
   would take 0.5 ms; fusing the top-k selection (0.59 ms) into the same
   launch or the next saves the rest. The converter and Kolibri's layout
   check (which requires an F32 router) change; old GGUFs keep working
   through the F32 path.
3. **Norms.** `norm_add` runs one 256-thread block per token, so at T = 1
   one block works alone on the GPU for 14-25 us. Splitting the row across
   blocks (partial sums, then a second pass or the next kernel's prologue)
   brings each to a few microseconds. The FFN reduce reads 8 partial rows
   and is the larger of the two.
4. **k and v together.** Each is 1.3 MB and reaches 121 GB/s; one launch over
   both matrices (they share the input) halves the fixed cost.
5. **LM head.** 0.66 GB of BF16 at 167 GB/s; the other large matvecs reach
   211. Probably the BF16 reader or the row split for a 128000-row output.
6. **Q4_K expert decode kernels.** #1070's per-row kernels do Q4_K
   dequantization at 102-126 GB/s. At 200 GB/s Q4_K experts (0.87 GB per
   token) would take 4.3 ms instead of 7.4 ms, and Q4_K would decode about
   10% faster than F8.
   Options: several rows per wave, so that independent loads are in flight;
   keep the SwiGLU output in LDS and run the down projection in the same
   block.
7. **F8-specific WMMA expert tile.** Prefill experts use #1070's generic
   `matrix_half_tile`, which decodes four F8 values per lane read. A tile that
   loads 16-byte F8 rows straight into f16 fragments (as the dense direct
   kernel does), with prefetch, should close much of the gap to the Q8 direct
   projection kernel's throughput. Experts are 56% of prefill at 8K.
8. **Prefill tuning.** The direct projection kernel's tile table was tuned
   for Qwen3.8's shapes; Kolibri's o projection (K = 6144, N = 2560) runs at
   60% of q's speed for the same weights, and the F32 router matvec has no
   prefill kernel worth the name. Kolibri sessions prefill in 2048-token
   chunks; 4096 or 8192 make the expert tiles fuller (2048 tokens give each
   of 384 experts about 32 rows).
9. **FP8 KV cache for the full layers.** Halves the 0.65 GB per token at 32K
   (and the 5 GiB at the native 262K context). The sliding layers' 513-row
   rings stay f16. A per-row scale is needed; check with the long-context
   teacher-forced comparison. Worth more after 1, once attention is
   bandwidth-bound.
10. **Prefill attention across a whole KV group.** The WMMA kernel processes
   16 queries × 4 query heads per block, so each K/V tile is loaded three times
   per KV head (12 query heads share it). A block of 12 waves, or a 4-head
   block that loops over the three head groups, loads it once. Also: 32-key
   tiles and double-buffered LDS loads.
11. **Speculative decoding with n-gram drafts.** Kolibri has no MTP head, but
   drafts can come from the prompt and the output so far (prompt lookup). The
   graph already runs T > 1 rows per forward and can rewind within the sliding
   rings' slack. Verifying 4 drafted tokens costs little more than one token,
   because decode reads the same weights either way. The win depends on
   acceptance, high for code edits and structured output and low for free
   prose and thinking, so measure acceptance on the Go tasks first. Exact:
   the verified tokens are the ones the model would have sampled.
12. **Q8_0 LM head.** The head is BF16 from the source, so anything smaller is
   lossy. Q8_0 halves 0.66 GB per token. Keep it only if the teacher-forced KL
   stays near the current 0.008.

## How changes are checked

- **Output.** `tests/kolibri/compare_golden.py` on the real model (greedy
  continuations identical on the five goldens) and `teacher_forced.py` on
  `tests/golden/kolibri/tf_text.txt`: mean KL against the fp32 reference near
  today's 0.008, and top-1 agreement near 0.975. Lossy items (4, 9) report both
  numbers.
- **Speed.** `tests/kolibri/bench.sh`, three runs, before and after, in the
  commit message.
- **Profiling.** `DS4_KOLIBRI_TRACE=1` prints GPU and host time per call of
  the forward at exit, separately for decode and prefill, with GB/s where
  the bytes are known. `DS4_KOLIBRI_TIMING=1` still gives synchronized stage
  times. `rocprofv3` is not packaged with Ubuntu's ROCm 7.1.
