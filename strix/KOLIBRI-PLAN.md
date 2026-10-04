# Kolibri-1 on Strix Halo: speed-up plan

Scope: ds4 ROCm backend on gfx1151 (Ryzen AI Max+ 395, 128 GB unified memory,
LPDDR5X with about 256 GB/s theoretical bandwidth), Kolibri-1 as the lossless
F8 GGUF (75.7 GiB) and the Q4_K-experts GGUF (42.8 GiB). How the model is
implemented, the correctness harness and the benchmark method are in
[docs/KOLIBRI.md](../docs/KOLIBRI.md). This file ranks what could make it faster.
The gains below are estimates from byte counts and stage timings, not
measurements; each item states how it would be checked.

## Where the time goes now

Measured on 2026-10-04 with `tests/kolibri/bench.sh` (median of 3 runs):

| GGUF | prefill 512 / 8K / 32K, tok/s | decode at 512 / 8K / 32K, tok/s |
|---|---|---|
| F8 | 674 / 922 / 776 | 36.5 / 36.1 / 31.8 |
| Q4_K experts | 611 / 918 / 773 | 36.8 / 35.6 / 31.4 |

### Decode

Bytes read per token come from the tensor shapes; stage times from
`DS4_KOLIBRI_TIMING=1` (which synchronises after every stage, so the times are
approximate; their sum, 28 ms, matches the unsynchronised 36 tok/s).

| Stage, per token (F8, 2K context) | Bytes read | Time | Effective bandwidth |
|---|---:|---:|---:|
| q/k/v projections (50 × 18.4M weights) | 0.95 GB | 4.8 ms | ~197 GB/s |
| output projection, sandwich norms | 0.81 GB | 4.6 ms | ~176 GB/s |
| router (F32) + 6 routed + shared expert | 1.62 GB | 10.6 ms | ~153 GB/s |
| LM head (BF16, 128000 × 2560) | 0.66 GB | 3.9 ms | ~168 GB/s |
| attention (QK prep, attention, merge) | ~0.08 GB KV | 4.2 ms | fixed costs |
| **total** | **~4.0 GB** | **~28 ms** | |

- The dense projections already run near the practical bandwidth limit.
  At ~200 GB/s for every byte the ceiling is about 20 ms per token, roughly
  50 tok/s for F8 and 55-60 tok/s for Q4_K experts (3.5 GB per token).
- The expert stage is not bandwidth-bound: Q4_K experts read 0.55 GB less per
  token than F8 but decode at the same speed.
- Attention at 2K context is mostly fixed per-launch cost. At 32K the ten
  full-attention layers read about 0.65 GB of f16 KV per token, which is why
  decode falls from 36 to 32 tok/s there.
- About 13 kernels run per layer, 650 per token.

### Prefill

For a 2048-token batch at 4K context (F8): routed experts ~1.43 s, attention
~0.31 s, output projection and norms ~0.28 s, q/k/v ~0.19 s. The expert stage is about 60%.
At 32K the full-attention layers' quadratic cost grows.

## Ideas, ranked

| # | Idea | Expected gain | Effort | Correctness risk |
|---:|---|---|---|---|
| 1 | Expert decode kernel rework | decode +10-15% | medium | none (same arithmetic) |
| 2 | Router weights as BF16 instead of F32 | decode +3% | small | none (source is BF16) |
| 3 | Fewer, fused kernels per layer | decode +5-10% | medium | none |
| 4 | FP8 KV cache for the full-attention layers | decode at 32K +10% | medium | small, measurable |
| 5 | Speculative decoding with n-gram drafts | 1.3-1.8x on repetitive output | medium-large | none (exact verify) |
| 6 | F8-specific WMMA expert tile for prefill | prefill +30-50% | medium | none |
| 7 | Prefill attention shared across all 12 heads of a KV group | prefill at 32K +10-20% | medium | none |
| 8 | Prefill chunk and tile-table tuning for Kolibri's shapes | prefill +5-15% | small | none |
| 9 | Q8_0 LM head | decode +7% | small | small, measurable |
| 10 | GPU graph capture of the decode step | decode +10-15% | large | none |

1. **Expert decode kernel rework.** The decode MoE runs #1070's per-row kernel:
   one wave per output row, two launches (gate/up with SwiGLU, then down), the
   shared expert as an extra slot. Waves for the down projection are short (one
   512-value block per row), so latency dominates. Options:
   - several rows per wave, so that independent loads are in flight;
   - one block per selected expert that keeps the 512-wide SwiGLU output in LDS
     and runs the down projection without a second launch;
   - folding the weighted reduce into the down kernel.

   Target: the stage at ~200 GB/s, 10.6 ms down to about 8 ms. Q4_K should then
   decode faster than F8 rather than at the same speed.
2. **Router as BF16.** The converter widens the BF16 router gate to F32, so
   each token reads 0.20 GB where 0.10 GB carries the same bits. The matvec
   readers already handle BF16; the converter and Kolibri's layout check
   (which requires an F32 router) change.
3. **Fewer kernels per layer.**
   - q, k and v in one matvec launch over adjacent weights;
   - QK norm and RoPE folded into the decode attention kernel's prologue;
   - the post-attention norm folded into the output projection's epilogue;
   - router logits and top-6 selection in one kernel.

   About 13 launches per layer would become 7 or 8.
4. **FP8 KV cache for the full layers.** Halves the 0.65 GB per token at 32K
   (and the 5 GiB at the native 262K context). The sliding layers' 513-row
   rings stay f16. A per-row scale is needed; check with the long-context
   teacher-forced comparison.
5. **Speculative decoding with n-gram drafts.** Kolibri has no MTP head, but
   drafts can come from the prompt and the output so far (prompt lookup). The
   graph already runs T > 1 rows per forward and can rewind within the sliding
   rings' slack. Verifying 4 drafted tokens costs little more than one token,
   because decode reads the same weights either way. The win depends on
   acceptance, high for code edits and structured output and low for free
   prose, so measure acceptance on the Go tasks first. Exact: the verified
   tokens are the ones the model would have sampled.
6. **F8-specific WMMA expert tile.** Prefill experts use #1070's generic
   `matrix_half_tile`, which decodes four F8 values per lane read. A tile that
   loads 16-byte F8 rows straight into f16 fragments (as the dense direct
   kernel does), with prefetch, should close much of the gap to the Q8 direct
   projection kernel's throughput.
7. **Prefill attention across a whole KV group.** The WMMA kernel processes
   16 queries × 4 query heads per block, so each K/V tile is loaded three times
   per KV head (12 query heads share it). A block of 12 waves, or a 4-head
   block that loops over the three head groups, loads it once. Also: 32-key
   tiles and double-buffered LDS loads.
8. **Prefill tuning.** Kolibri sessions prefill in 2048-token chunks; 4096 or
   8192 make the expert tiles fuller (2048 tokens give each of 384 experts
   about 32 rows). The direct projection kernel's tile table was tuned for
   Qwen3.8's shapes; Kolibri's are M = 6144 / 2560 / 512 and K = 2560 / 6144.
9. **Q8_0 LM head.** The head is BF16 from the source, so anything smaller is
   lossy. Q8_0 halves 0.66 GB per token. Keep it only if the teacher-forced KL
   stays near the current 0.008.
10. **GPU graph capture.** Replaying the whole decode step as one HIP graph
    removes per-launch cost. ds4's ROCm runtime launches on the legacy default
    stream, which cannot be captured, so this means moving the backend to an
    explicit stream first. That touches every model family, so it is the
    largest change here. It is worth more after 3, which removes most launches
    at less cost.

## How changes are checked

- **Output.** `tests/kolibri/compare_golden.py` on the real model (greedy
  continuations identical on the five goldens) and `teacher_forced.py` on
  `tests/golden/kolibri/tf_text.txt`: mean KL against the fp32 reference near
  today's 0.008, and top-1 agreement near 0.975. Lossy items (4, 9) report both
  numbers.
- **Speed.** `tests/kolibri/bench.sh`, three runs, before and after, in the
  commit message.
- **Profiling.** `DS4_KOLIBRI_TIMING=1` gives stage times; per-kernel times
  need `rocprofv3`, which is not installed on this machine yet.
