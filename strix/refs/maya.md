# Project Maya (GLM-5.3-Flash on consumer GPUs): study notes

Repo: https://github.com/mw00/project-maya (MIT, C++/CUDA/HIP, 379 commits up to
07c3dc9, 2026-10-15). It started from Strata and was rewritten for GLM-5.3-Flash.

**Clean-room.** Only prose was read: README, CHANGELOG, docs/*.md,
bench/results/*.md, commit messages, and the descriptions and comment threads
of 210 issues and PRs. No source, scripts or diffs were opened.

How to read the citations:
- `maya:path:line` is the clone.
- `#N` is an issue or PR.
- A bare hash is a Maya commit.
- ds4 citations are at 7b42067.

All numbers are Maya's own, and most of them are single runs on other
hardware.

## 0. TL;DR

- **Same model, different regime.**
  - Maya runs our GLM-5.3-Flash with its own 2-bit-class quants.
    - **Maya-S** (96.5 GB) holds all 12,096 experts in GPU memory on a 128 GB
      Strix Halo. Under Linux it decodes at **17.4-19.1 tok/s plain** and
      **20.9-24.0 with MTP**, with 340-390 tok/s prefill after hipBLASLt tuning
      (#202, #203).
    - Its quality: KL 0.428 to FP8 and 83% top-1 agreement.
  - ds4 streams Q4_K (178 GiB) from the NVMe at 3.2-3.6 tok/s.
  - ds4's `glm53-q2` is the same size class as Maya-S (90 GiB). It has never
    been measured resident on ROCm.
  - The disk now has room to fetch it again: 312 GB free, against 8.2 GB when
    PLAN item 1 was blocked.
- **GLM-5.3 is far more sensitive to lost experts in early layers than in late
  ones.**
  - Leaving out off-card experts from rank 5, one band at a time, costs KL
    0.040 in layers 3-7 and 0.0015-0.003 in layers 30-44 (#103).
  - Requantizing the dense weights shows the same gradient (#49).
  - This refines PLAN items 1 and 2, and it partly overturns "dropping is not
    worth it" for late layers.
- **Cache-aware routing (PLAN item 3) was built, measured and removed.** It cost
  KL 0.018-0.037 for +1-2% decode. With a drafter it cost about 40%, because
  acceptance fell from 35% to 10-21% (#146).
- **MTP:**
  - **Resident:** MTP pays. It gave +44-50% on a Gorgon Halo and +12-35% on a
    128 GB Strix (CHANGELOG.md:43-50, #96).
  - **Streaming:** it loses. Maya turns it off below a third of experts
    resident. Consecutive tokens share only about 31% of their experts.
  - **Verify kernels on gfx115x:** the per-token decode kernels, with each
    shared expert read once, beat the batched window by 26.7 against 17.2
    tok/s (870eb04).
- **Maya's whole resident token takes about 53 ms on this chip.** ds4's
  attention + router + readback phase alone takes about 162 ms. Maya hands
  routes over through mapped memory, with a system fence and a polling thread,
  and does no host sync. That supports PLAN item 5.
- **ds4 server bug found while comparing:**
  - User and tool text that contains literal special tags (`</think>`,
    `<|assistant|>`, `</tool_response>`, ...) is tokenized as real control
    tokens. The text is spliced raw into the rendered prompt
    (ds4_server.c:3279), and `special_token_at` (ds4.c:45722) matches tags
    anywhere in it.
  - Maya fixed the same class of bug in #146: 9 injected special tokens before
    the fix, 0 after.
  - Fixed in ds4 by fa50044: client text is escaped where it joins the
    template.

## 1. Model and quants

The architecture agrees with what ds4 validates (ds4.c:7200-7268):
- 45 trunk layers: 34 KDA and 11 DSA/MLA (latent 512, no RoPE).
- mHC with 4 streams and 20 Sinkhorn iterations.
- 288 experts, top-8, sigmoid routing with ×2.5 scaling.
- 3 dense lead layers and one NextN layer.

Three of Maya's observations are worth checking in ds4:
- **KDA layer 18's state grows ×68.** It does the same in llama.cpp, so this is
  the model's own behaviour, and it amplifies quantization noise downstream
  (maya:docs/GLM5-FLASH.md:485-487). Layer 18 is a candidate for extra bits.
- **The head reads the mean of the four hc streams** (maya:bench/results/MAYA-S-V1.md:10-11).
  Check ds4's output collapse against this.
- **Maya quantizes the NextN block's experts below the trunk's** (Q2_K/Q3_K in
  S). ds4 gives layer 45 the trunk's types.

Bytes per token at decode, estimated from the shapes:
- Routed experts: 8 × 42 × 25.2 M ≈ 8.5 B parameters, about 2.4 GB at Q2.
- Non-routed: about 8 B parameters.
  - KDA q/k/v/o: about 4.6 B.
  - DSA: about 1.3 B.
  - Shared experts: 1.06 B.
  - Dense FFN: 0.45 B.
  - Head: 0.63 B.
- At Q8_0 the non-routed weights are about 8.5 GB per token. **Once resident,
  dense precision is the main bandwidth lever.**

| Quant | File | Gate/up | Down | Attention / shared / dense | KL to FP8 | Top-1 agreement with FP8 |
|---|---:|---|---|---|---:|---:|
| Maya-S v1 | 90.1 GB | IQ2_XXS | IQ2_XXS, IQ2_S at the edges | Q6_K | 0.485 | - |
| **Maya-S** | 96.5 GB | IQ2_XXS | IQ2_S, IQ3_XXS in MoE layers 3-6 and 41-44 | Q6_K | 0.428 | 83.3% |
| Maya-S24 | 94.7 GB | = S | = S | Q4_K for KDA/MLA projections and shared experts | 0.444 | 83.1% |
| Maya-M | 116 GB | IQ2_S | IQ3_XXS (IQ3_S at the edges) | Q6_K | 0.329 → 0.291 (engine update) | 86.2-86.8% |
| Maya-L | 156.3 GB | IQ3_S | IQ4_XS (Q5_K at the edges) | Q6_K | 0.188 | 90.0% |
| ds4 glm53-q2 | 90 GiB | IQ2_XXS | Q2_K | Q8_0, kda_q/k Q4_K | not measured vs FP8; vs Q4: ppl ×1.135, top-1 81% | |

Sources: maya:bench/results/MAYA-*.md; gguf-tools/glm53_quantize.py:95-107.

How Maya makes and scores its quants:
- **Statistics from the real model.** The FP8 checkpoint runs in PyTorch one
  layer at a time over about 1 M calibration tokens. These cover chat, code,
  math, tool calls and long documents; M and L lean toward tool calls and
  front-end code.
- **Statistics kept per expert:** the imatrix diagonal and the full XᵀX.
- **GPTQ-style rounding.** Error feedback runs block by block, with llama.cpp's
  quantizer choosing the scales (maya:docs/QUANT-PLAN.md:64-117). No ablation
  of GPTQ against plain imatrix rounding was published.
- **v1 → v2 put the extra bits into down projections:** +6.4 GB, KL -12%, and
  -17% on tool-call text.
- **The yardstick:**
  - The FP8 model's top-64 log-probabilities, scored on 8 held-out texts
    (7,672 positions). The texts are chat, reasoning, web code, other code,
    tool calls and wikitext.
  - Zero-shot ARC/HellaSwag/WinoGrande/PIQA, 400 questions each. Maya itself
    says these don't separate quants.
  - A loop test and a needle test.
  - Tool calls and wikitext are the most quant-sensitive domains, at KL
    0.85-1.06 against 0.06-0.45 elsewhere.
- **Engine floor:** about KL 0.02 from Maya's own arithmetic, CPU against GPU.

ds4 lacks two of these:
- **An FP8 reference.** ds4 scores against its own Q4 (top-20, Go/Markdown,
  strix/pplcompare.py).
- **GPTQ rounding.** ds4's imatrix comes from its own quantized runtime.

## 2. Measured numbers on AMD APUs

| Machine | Model, mode | Decode tok/s | Prefill tok/s | Source |
|---|---|---:|---:|---|
| 8060S 128 GB, Linux, ROCm 7.2.2 | Maya-S resident, no MTP | 17.4-17.8 | 210-221 | #17, maya:docs/AMD_MAYA.md:158 |
| 8060S 128 GB, Linux, ROCm 10.2, v1.1.11 | Maya-S, MTP off / on | 17.4-19.1 / 20.9-24.0 (bench 30.6) | untuned 204/217, tuned 341/373 (2K/8K) | #202, #203 |
| 8060S, Windows, 96 GiB carve-out | Maya-S, 262K ctx | 18.3 | 320 (8K) | #102 |
| 8065S 192 GB (160 GiB carve-out), Windows | Maya-S, MTP, 32K | 19.7 → 28.4-29.6; 23.2-23.5 after 3-8K | 271-278 | CHANGELOG.md:43-57 |
| 8065S | Maya-L 156 GB, 134 GB pool | 15.9 | 345 | #69 |
| 8065S (README; CHANGELOG says Strix) | `--parallel` 2 / 4 | x1.67 / x2.02 aggregate | - | README:88-92 |
| **ds4**, 8060S 128 GB, ROCm 7.1 | Q4_K 178 GiB, streamed | 3.2-3.6 | 25-30 | strix/README.md, PLAN.md |

Notes on these numbers:
- **Ceiling:** on the 8060S, Maya-S reads about 9.4 GB per token at a measured
  240 GB/s, so its one-token ceiling is about 25 tok/s (#6). MTP and batching
  are how it goes past that.
- **None of Maya's Strix numbers stream from the SSD.**
- **The GTT limit is already 114 GiB here** (`ttm.pages_limit=29884416`,
  `mem_info_gtt_total`), the same as Maya's "112 GiB". There is no capacity to
  gain from raising it.

## 3. Techniques

### 3.1 Leaving out experts, by layer band (#103, #109)

`TRUNK_KEEP_MAP` takes one digit per MoE layer. For experts not on the card, it
keeps route ranks below n and leaves out the rest. Experts already on the card
always run.

KL against the same model with nothing left out:
- 1,135 tokens, CPU lane off, so a repeat gives KL exactly 0.
- RTX 3090 + 3060, 128 GB DDR4.

| Layers left at keep-5 | 3-7 | 8-12 | 13-17 | 18-22 | 23-29 | 30-34 | 35-39 | 40-44 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Maya-M | 0.040 | 0.034 | 0.035 | 0.026 | 0.012 | 0.003 | 0.0015 | 0.002 |
| Maya-L | 0.028 | 0.029 | 0.029 | 0.020 | 0.010 | 0.003 | 0.0015 | 0.003 |

The shipped map keeps 5 in layers 23-29 and 4 in layers 30-44:
- Maya-M: KL 0.018, top-1 agreement 95.3%, decode 20.0 → 24.6 tok/s.
- Maya-L: KL 0.0165, decode 11.7 → 14.75 tok/s.
- Code scores the same as prose.

The same budget spent uniformly costs more:
- keep-6 in every layer: KL 0.053;
- a global gate-weight threshold (`DISK_SKIP_W=0.12`): KL 0.12 and 88.4%
  top-1, for +62% decode on a RAM-starved 3090.

Leaving out only disk-resident experts, in every layer, raised KL from 0.017 to
0.040 with no speed gain. **The band matters more than the residency rule.**

### 3.2 MTP decode (#76, #96, #48, #145, 9812bee, 870eb04, 3cab7d5)

How a round works:
- **Draft.** The MTP block drafts up to 3 tokens in a chain, each draft fed
  back in for the next.
- **Verify.** One pass verifies all of them:
  - The dense, shared, router and head weights are read once per window.
  - The absorbed MLA `k_b`/`v_b` are read once per window: T=2 round 78.1 →
    75.6 ms, with byte-identical logits on gfx1151 (#117).
  - DSA attention runs one row at a time on the one-token kernels.
  - Routed experts run on the per-token kernels, every row in one gate/up
    launch and one down launch, so an expert two rows share is read once.
    That added +10%.
- **Rollback.** The verify writes no recurrent state. A commit replays the KDA
  recurrence for the kept rows, which takes about 60 MB of per-row history
  instead of about 0.5 GB of snapshots.
- **Draft length.** It maximises `(1 + a1 + a1·a2 + …) / round time`.
  - It runs in stints, and the first two rounds after a change are not timed.
  - It probes a neighbouring length for 4 rounds at a back-off gap that doubles
    up to 1024 rounds.
  - It is chosen by measured time, not by acceptance alone.
- **Cheaper drafts:**
  - a draft-vocabulary head;
  - `MTP_KEEP=2`: the draft computes only its 2 best-scored missing experts.
    Acceptance went from 77.6% to 87.1% and decode was +4.7% against fetching
    every miss.
  - Drafts never wait on the SSD.
- **Exact speculative sampling.** The draft is drawn from q and kept with
  probability min(1, p/q). Acceptance went from 68.6% to 74.7%, for +1.6-2.7%
  alone. It was checked with chi-square and PIT tests.

On a 128 GB Strix (resident):
- One verify round took 75-104 ms, against a 53 ms plain step.
- Acceptance was 62-70%, at 1.68-2.91 tokens per round.
- Chat, code, reasoning and prose sped up by +12%, +19%, +35% and +12%.

In ds4 today:
- GLM MTP drafts 1 token and commits at most 2.
- GLM-5.3 verifies through the prefill forward after copying the full KDA and
  indexer state (ds4.c:53649, 77832-77871).
- `glm_graph_verify_rows` rejects GLM-5.3 (ds4.c:54241). Its comment puts the
  indexed batch path at 1.4 ms per layer for n=2, against 0.79 ms for decode.
- On ROCm, 2-3 verify rows lose the in-place Q4_K path
  (`n_tokens == 1u`, rocm/ds4_rocm_moe_launch.cuh:719) and fall onto the
  sorted/tiled path.
- Exact sampling treats the GLM draft as a point mass (ds4.c:77760). The Qwen
  path already draws from q.
- ds4 already has the right verify kernels for Qwen: per-token rows that are
  bit-identical, with one read per shared expert (a37ef03, a9f8317).

### 3.3 Routing handoff (maya:docs/AMD_MAYA.md:288-295, maya:docs/AMD_UPSTREAM.md:10-19, #2, #101)

How Maya does it:
- **Route write.** The route kernel writes the expert ids and a sequence word
  into mapped host memory.
- **System fence.** On HIP, a system-scope fence follows the store. Without it
  the store can sit in L2 and the host never sees it (the doorbell bug in #2).
- **Host side.** A service thread polls the word, with no stream sync or driver
  query.
- **GPU side.** The decode stream waits on small device-side wait/fetch
  launches.
- **All-resident shortcut.** When every expert is resident, those launches are
  skipped, saving about 2.5 ms of a 54 ms token (+2.6-7%).
- **Windows.** HIP holds queued launches until the host waits, so Maya calls
  `cudaStreamQuery` after 1 ms without a route.

In ds4, the worker blocks in `cudaEventSynchronize` once per layer
(rocm/ds4_rocm_runtime.cuh:7098-7132). The routed step then waits on the host
for all pending reads before it queues kernels, so the GPU queue drains 42
times per token.

### 3.4 Memory sizing on an APU (#17, #101, #192, c12c69d, de82608)

The pool is `min(expert bytes, MemAvailable − 16 GiB − 1 GiB)`, measured after
the dense weights, KV and scratch are allocated:
- When everything fits, each layer gets one slot per expert plus 3 spares, and
  prompts borrow the pool's tail.
- On the 128 GB box: MemAvailable 109.69 GB → an 80.17 GiB pool holding all
  experts, warmed up in 17 s.

ds4 already does the equivalent (ds4_rocm_memory.h:20-60).

Lesson from de82608: don't touch-probe a whole APU pool. Writing a 157 GB
carve-out three times doubled the start time. Under `--parallel 2` the earlier
frees had not landed yet, so the third allocation got nothing and the start
failed.

### 3.5 Dense precision (maya:bench/results/MAYA-S24.md, #49)

| Change | KL | Decode |
|---|---:|---:|
| Attention and shared experts Q6_K → Q4_K, imatrix from FP8 (Maya-S24) | 0.421 → 0.444 to FP8; zero-shot 97.9 → 97.7% | +11-14%, about 1.5 GB freed |
| Load-time requant without imatrix, whole model, Q4_K | 0.066 | +10% |
| Load-time requant without imatrix, layers 29-44 only, Q4_K | 0.008 | - |

ds4's GLM recipe keeps the dense weights at Q8_0 (kda_q and kda_k at Q4_K in
Q2). Q4_K dense is already loadable (ds4.c:5240).

### 3.6 hipBLASLt tuning tables (#7, #162, #203, #6)

- **What it is.** An offline search per architecture, keyed on the exact
  library build. Version-matched IDs from another build were 43 of 43 wrong
  and fell back silently (#6).
- **Accuracy gate.** Absolute error is scaled by `sqrt(K/4096)`. The result is
  not bit-exact.
- **Gains on gfx1151:**
  - 3-7.5x per GEMM over `hipblasGemmEx`;
  - +67-72% end-to-end prefill (204/217 → 341/373 tok/s);
  - bigger sub-batches help only once a table is in place.
- **In ds4.** GLM prefill uses `cublasGemmEx` with the default algorithm
  (rocm/ds4_rocm_matmul.cuh:624). The fixed-index table is DeepSeek-only
  (`!g_glm_model`, :1014).

### 3.7 Smaller kernel ideas (gfx1151)

| Idea | What it does | Maya's result | Exact? | ds4 today |
|---|---|---|---|---|
| Packed-sign IQ2 arithmetic | One multiply spreads 4 sign bits into byte high bits; masks give the XOR and the corrections; plain word adds never carry because codebook bytes are positive | gate/up 1.124x on Strix, 17.8 → 18.0 tok/s; 7900 XT +8.6% | Bit-exact | `__vcmpne4`/`__vsub4` emulation in rocm/ds4_rocm_moe.cuh:39-86 |
| Dense GEMV for LPDDR5X | Two super-blocks' loads issued before the arithmetic (latency, not bandwidth, sets the pace); the Q6_K offset as a second `sudot4` | +3.1% on the 8065S (#95) | Bit-exact | GLM BF16 matvecs (rocm/ds4_rocm_glm.cuh:238, 263) not profiled |
| FP16 high + residual split in WMMA attention (wmma2) | FP32 Q and P each split into an FP16 high part and a residual | rel L2 3.4e-7 against 9.8e-5 for plain FP16; +4% prefill (#113) | Near-F32 | A way back onto WMMA for paths dropped for FP16 drift (e.g. ad27bf5) |
| Fused int8-WMMA prompt MoE | gate/up, the clamped SwiGLU and down in one pass | +5-13.5% on Strix, nothing on dGPUs (#98) | Not exact | The port first forgot GLM's SwiGLU clamp (rel L2 ~31): audit every fused epilogue for it |
| KDA prompt recurrence | Split over value columns (#199) | - | - | ds4 already splits wider (rocm/ds4_rocm_glm.cuh:725-769). Its KDA prepare kernel still walks tokens serially and shifts the conv state through global memory (:641-700). Maya's KDA conv rewrite was 6.5x, bitwise equal, on NVIDIA (abe3b45) |

### 3.8 Several requests at once (#115, #94, #204, #93, #144, #125)

- **One GPU.** Dense weights are read once through an up-to-8-column GEMV that
  is bit-exact to the one-column kernel. Attention and routed experts stay per
  request.
- **Gains:** x1.67 for 2 and x2.02 for 4 requests, fully resident. On a V100
  that streams experts it was only x1.41, and its VRAM hit rate fell from 95%
  to 82% at 4 requests.
- **ds4 status.** ds4's Qwen and Kolibri batching already groups experts across
  sessions and is ahead of this. GLM session batching on ROCm does not exist
  (ds4.c:83615-83646, Metal only).
- **Fair slices (#93).** A long answer is paused after a time slice. A short
  question asked during a 3,000-token answer got its first token at 13-17 s
  instead of 34 s.

### 3.9 Server and agent safeguards (#33, #34, #143, #146, #163)

- **Literal special tags in message and tool text.** Maya maps them to
  private-use marks, then encodes those spans as plain text. ds4 is open to
  the same problem (§0).
- **Loop guard.** It fires when the last W tokens are one pattern of at most 32
  tokens repeated.
  - W=256 while thinking: it closes the thinking.
  - W=1024 in the answer: it closes a tool argument or ends the turn.
  - A looping write call ended at 1,031 tokens as a valid tool call, instead of
    about 6,000 tokens and a `length` cut.
  - ds4 has no repetition guard. 2-bit quants make this more likely.
- **Thinking budget.** Clamped to `max_tokens − max(1024, max_tokens/4)`, so
  the answer always has room. ds4 applies `think_budget` unclamped
  (ds4_server.c:14282).
- **NaN guard.** One small kernel checks the head logits' exponent bits, at
  0.017 ms per token. On a hit, the engine drops the state, the checkpoints and
  the slots, and the server retries once.
  - ds4 only filters non-finite values in the sampler (ds4.c:46327).
  - A poisoned state could reach the disk KV cache.
- **Checkpoint at the start of the last turn.** An edited last message reused
  6,016 of 6,084 tokens. A recurrent KDA state cannot be truncated, so a
  missing checkpoint means a full re-read.
- **llama.cpp's `prompt_progress` stream chunks.** ds4 sends keepalives only
  (ds4_server.c:13421).

## 4. Ideas for ds4, ranked by expected gain

1. **Measure GLM-5.3 Q2 resident on Strix.** This is the largest lever, and the
   change is configuration, not code.
   - Fetch `glm53-q2` again (90 GiB; there is disk space now).
   - Run `strix/bench.sh` without `--ssd-streaming` at 8K and 32K, recording
     the GTT peak.
   - Score with pplcompare against the Q4 dumps.
   - Maya's numbers on this chip suggest 15-19 tok/s plain, against 3.4 tok/s
     streamed. ds4's own quality figures for Q2 against Q4 are ppl ×1.135 and
     81% top-1.
   - If it holds, Q2 resident becomes the fast mode and Q4 streamed the
     high-quality mode, and items 3, 5, 6 and 8 start to matter.
   - The same file also unblocks PLAN item 1 (the Q2 tier).
2. **A layer-band rule for misses on streamed Q4.**
   - Re-run PLAN items 1-2 with per-band settings (§3.1). Leave the early
     layers alone. In the late layers, loosen the Q2 threshold, or defer or
     drop misses at rank ≥4.
   - First take one `DS4_GLM_ROUTE_TRACE` run to see how misses split by layer
     and rank.
   - Scan the bands one at a time with `DS4_PPL_DUMP` and pplcompare, reporting
     KL, with Maya's table as the baseline.
   - Estimate: +5-9% decode for the drop or defer rule in layers 30-41 alone,
     with no Q2 file needed.
3. **GLM-5.3 MTP rebuilt for the resident case** (Q2 resident only; see item 9
   for streamed). In order:
   - a decode-style verify with the per-token expert kernels and one read per
     shared expert (ds4 has this pattern for Qwen, a37ef03);
   - `k_b`/`v_b` read once per window;
   - rollback by replaying the kept rows instead of copying the full state;
   - 2-3 chained drafts with a time-based length controller;
   - drawn drafts for exact sampling.

   Expected: +20-35%, up to +45%. Measure verify ms per round against step ms,
   tokens per round, and tok/s at short prompts and at 8K.
4. **Server correctness, cheap:**
   - escape literal special tags in user, tool and system text (done in
     fa50044);
   - clamp the thinking budget against `max_tokens`;
   - a loop guard;
   - a NaN guard before `kv_cache_maybe_store_continued`;
   - `prompt_progress`;
   - a checkpoint at the start of the last turn.
5. **Routing handoff without a host sync (PLAN item 5).**
   - First, `rocprofv3` on one decode layer, to split GPU time from idle gaps.
     Maya's 53 ms for a whole resident token on this chip says ds4's 162 ms
     phase is not bandwidth.
   - Then a fenced routing word in mapped memory, polled by the I/O worker,
     with a device-side wait ahead of the routed step.
6. **Dense weights at Q4_K, KDA projections first** (§3.5). On resident Q2 this
   reads about 20 ms less per token. On streamed Q4 it frees about 4 GiB of
   cache, roughly 300 experts or +3%. Expected quality cost: KL to FP8 +0.02,
   and less if the change stays in layers 22+.
7. **hipBLASLt timed search for GLM prefill shapes.** This is worth it only
   once prefill is not read-bound, that is, resident.
   - Microbenchmark first: `getAllAlgos`, timed, at T=2048/4096 for the
     projections and the MLA products.
   - Key the table on the library's version and git revision, as the DeepSeek
     table already does.
8. **GLM session batching on ROCm, for resident Q2.** x1.7-2.0 at 2-4 sessions
   in Maya. Reuse the Qwen/Kolibri batched kernels.
9. **PLAN item 4 (MTP while streaming) with Maya's guards.**
   - Draft with `KEEP=2` and no NVMe reads in the draft.
   - Record misses per token for the draft and the verify separately before
     anything else.
   - Expect a loss unless the union of misses stays under about 1.3x one
     token's.
10. **Quality yardstick.**
    - An FP8 teacher that streams one layer at a time on CPU or GPU, over
      about 8K tokens, scored on top-64. `glm53-fp8` is already packaged.
    - Add tool-call text and long-document text to the scored set.
    - A GPTQ pass in gguf-tools is plausible but unproven: Maya never ablated
      it.
11. **Small kernel ideas** (§3.7):
    - packed-sign IQ2 dots (bit-exact; matters for the Q2 tier and DeepSeek
      IQ2);
    - two blocks in flight in the GLM BF16 matvec;
    - the FP16 high + residual split for WMMA where FP16 drift blocked it;
    - the KDA prepare kernel's serial conv walk.

Changes to make to PLAN.md:
- **Item 3 (cache-aware routing):** move it to "not worth pursuing", citing
  #146.
- **Item 6 (prefetch):** Maya agrees. Only ranks 0-1 are reliable (96%/88%
  used, against 24-52% for ranks 4-7). As shipped, prefetch was -13.6%; tuned
  to rank-restricted, +2-5%; SSD look-ahead never won.
- **"Not worth pursuing: dropping experts":** qualify it per layer. Maya's
  measured GLM-5.3 numbers are much milder in late layers than the OLMoE and
  DeepSeek figures cited there.
- **Recheck the expert sizes.** Block-size arithmetic gives 13.5 MiB per Q4_K
  expert and 159.5 GiB for the trunk's experts. PLAN says 14.1 MiB and about
  145 GiB.

## 5. Negative results (Maya's)

- **Eviction policy is saturated** (#79, 3090 + 3060):
  - every rule lands within about 2%: aged counts 65.0%, LRU 64.7%, segmented
    LRU 64.7-65.0%;
  - reuse-interval EWMA was worse, at 58-60%;
  - Belady reached 72.8%;
  - near misses as soft counts made it worse;
  - LRU against aged counts in live runs: +2.6-7%.

  This agrees with PLAN's "size dominates policy".
- **Prefetch:**
  - as shipped: -13.6%;
  - on 2x 7900 XTX: -13-18%;
  - deeper prefetch: -1.8%;
  - SSD look-ahead: one third fewer reads, each about 45% slower (#84, #6,
    #168, #67).
- **Cache-aware routing:**
  - KL 0.018-0.037 for +1-2% decode;
  - about -40% with a drafter (#146).
- **Draft fetches:** fetching every draft miss was 3% slower than `KEEP=2`.
- **Speculation while streaming:**
  - MTP on a single GPU: 18.1 → 16.6 tok/s;
  - MTP on a RAM-bound 3090 + 3060: -17%;
  - MTP on an R9700 with 28% of experts resident: -4.6% to +12.6% (#76, #96);
  - DFlash2 on RAM-bound machines: -20% to -46%, even at 88-93% acceptance
    (#127, #157);
  - DFlash2's drafter is licensed CC BY-NC-ND.
- **On gfx1151, the batched MTP verify window lost to per-token kernels:** 17.2
  against 26.7 tok/s (870eb04).
- **CPU lane:** irrelevant on an APU, where the pool holds every expert. It
  also adds error: relative L2 2.89% against 1.63% on the GPU (#185). The lesson
  that transfers: spinning threads starved the disk readers (2.6 → 9.1 tok/s
  once fixed, #73).
- **Memory and KV:**
  - huge pages for the RAM tier: -9%, and 10+ minutes stuck in compaction on
    fragmented memory (#177, #44);
  - INT8 DSA latent: KL 0.027 per token, for at most 0.45 GiB freed at 64K;
  - KV streaming: no meaning on unified memory.
- **Quantization:**
  - output-matching scale refit: Maya-M KL 0.329 → 0.323, but held-out text
    got slightly worse (4 of 8 better, 4 worse);
  - REAP pruning of 60 of 288 experts: KL 0.45 (third party).
- **AMD toolchain:**
  - kernel rankings change between ROCm builds: wmma2 was slower, then faster
    (#38, #113);
  - v1.1.11 on Windows Strix: 8K prefill fell 159 → 62 tok/s while decode rose
    18.2 → 28.2 (#206, still open).

## 6. Open questions

- Does `glm53-q2` fit resident in this box's GTT at 32-64K context, and what
  does it decode at?
- Is the 162 ms non-read phase made of GPU kernels or host-sync gaps?
- Does Maya's band sensitivity, measured on IQ2/IQ3, hold for Q4_K experts?
- Does Ubuntu's libhipblaslt1 7.1.1 report the revision ds4's fixed-index
  table expects (`8d1ae90e`)? Does that path ever engage on this box?
- Was the x1.67/x2.02 parallel result measured on a Strix or a Gorgon Halo?
  The README and CHANGELOG disagree.
- Speculative: could the Q4_K tiled-prefill aperture fault be a visibility
  race on host-written tables, like Maya's #2 doorbell bug and its unresolved
  `RAM_ALL` fault? Both were hidden by serialization.
