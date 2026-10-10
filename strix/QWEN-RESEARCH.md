# Qwen3.8-Flash-Next on Strix Halo: outside research

Clean-room notes from 2026-10-10 for [QWEN-PLAN.md](QWEN-PLAN.md), which takes
what we act on. Sources are prose only: papers, READMEs, docs, changelogs,
issue and PR descriptions. No source code was read, and halogen's binary and
image were not touched (its EULA permits benchmarking and reading its
write-ups, not porting its kernels). Numbers are the authors' own, often
single runs on other models and GPUs; anything marked *estimate* is ours.

Gaps at the time (V3 file, warm runs): plain decode 33.0 / 32.1 tok/s at
512 / 8K against halogen's 35.9 (fully explained by 4.78 vs 4.27 GB per
token); MTP 49.6 code / 39.0 prose, 1.53x over plain against halogen's 1.65x;
a ~3-row verify pass 53.4 ms against 31.2 ms for one token (experts gate/up
11.8 vs 6.0, down 9.8 vs 4.0, GDN qkv 6.3 vs 4.4, hc up 3.8 vs 2.5); prefill
619 / 591 tok/s at 8K / 32K against 1,175-1,844.

## Ranked for ds4

1. **Measure the verify union before more expert-kernel work.** With 512
   experts and 10 routed per token, three verify rows rarely share experts,
   so a union of ~25-28 is likely. Down at 2.45x T=1 and gate/up at 1.97x
   then match bytes read, not a slow kernel, which would also explain why
   the multi-row kernel did not help down. Log unique experts per layer per
   verify and the achieved GB/s. *Estimate* of the floor: non-expert time x
   ~1.05 + expert time x union / 10, about 45-48 ms per verify.
2. **Dense verify work should cost ~1.05x T=1, not 1.4-1.5x.** GDN qkv and
   hc up read the same weights for every row; ~3 ms per verify is
   recoverable with all T columns held in registers per weight load
   (Marlin's design stays near the memory-bound ideal to batch 16-32).
3. **MTP policy, where halogen's lead most likely sits.** ds4 already
   engages a chained second draft adaptively (docs/SPECULATIVE_DECODING.md).
   Not yet in ds4:
   - n-gram prompt lookup: halogen +13-15% on coding turns, +9% on function
     calling (3-token n-grams, K = 3);
   - a pause when acceptance over 32 rounds drops below 0.35 (64 tokens);
   - expert-aware drafting: prefer draft tokens that reuse experts already
     in the union (EcoSpec: 1.22x -> 1.36x on Qwen3-235B, output-exact), or
     cap the union per layer (MoE-Spec, lossy: 10-30% over EAGLE-3);
   - Cascade's rule: stop or shorten drafting when token gain / verify cost
     < 1 (worst-case slowdown 5%, +7-14% over a fixed draft length).
4. **Fuse the hc norm and the GDN chain (6.6 ms of small kernels).** On
   gfx1151 with the same model family, a llama.cpp fork's fused HC stream
   mix removed ~11,155 dispatches and cut that op's busy time 17.1 -> 3.3
   ms; with the GDN norm/scale and output-gate folds, decode +9.6%. Fused
   MoE top-k: +2.4% on gfx1151. Measure gaps between kernels against time
   inside them first (rocprofv3 traces). Order: hc norm into its neighbour
   (2.0 ms), one kernel per GDN layer (1.6 ms; *estimate* ~0.8 ms saved),
   top-k into the router and the MoE reduce into the next residual.
5. **The last ~0.5 GB per token: 6-bit, not 4-bit.** halogen keeps its
   "mixing layers" at 6 bits (that this means hc is our reading) and says
   its 8-bit trunk cost ~2 GB per token in an earlier version. Per bit, KLD
   roughly halves (Llama-3-8B: Q8_0 0.0014, Q6_K 0.0055, Q5_K_M 0.0108,
   Q4_K_M+imatrix 0.0282).
   - hc at Q6_K with imatrix: -0.14 GiB, *estimate* +0.4-0.7% ppl.
   - hc at an IQ5_K-like non-linear 5.5 bpw: -0.21 GiB.
   - hc at ~4.25 bpw after an offline Hadamard rotation along K (HIGGS,
     QuIP#: near Q5 quality): -0.30 GiB, only if the rotation folds into the
     hc input's producer. Unproven for hc.
   - shared expert at Q6_K: -0.055 GiB, near-lossless; Q5_K gate/up with
     down at Q6_K: -0.08 GiB.
   - Trellis and codebook formats (QTIP, EXL3, AQLM): not worth it, ALU or
     LDS bound and weak on AMD.
   Low-rank factors quantize poorly unless the leading components get more
   bits (LoRAQuant); a 320-row output does not average errors out, which
   fits the rejected hc down Q4_K (+2.5%).
6. **Prefill (topic deferred; what halogen publishes).** Experts ordered by a
   GPU counting pass instead of a CPU sort (+4.6% at 8K, +8.1% at 32K); a
   shipped per-shape GEMM tuning table; 32,768-token chunks fastest (1,024
   halves long-context throughput, ~9 GB scratch); an optional 5.5 GiB
   pre-unpacked trunk for prompts. Its numbers assume IOMMU off (worth
   13-16% of prefill), platform_profile=performance and ~85 W; our local
   halogen measurements ran on this host's settings (IOMMU on), so the
   comparison here is fair.
7. **HIP graphs and megakernels: later, if at all.** Graphs remove host
   launch cost, which ds4 already overlaps: 1.03x on a bandwidth-bound L4
   against 1.26x on H100, "about the same" on an RX 6800, no gfx1151 A/B.
   ROCm bugs: kernarg slots leak on every `hipGraphExecUpdate` (gfx1100,
   gfx1201, ROCm 6.4-7.2), so capture once with per-token values read from
   device memory. Persistent kernels are proven on AMD only on CDNA (Fleet
   on MI350: 7.83 vs 10.51 ms per token on Qwen3-8B; Kog on MI300X, where a
   naive grid sync ate ~35% of token time and sentinel polling fixed it),
   nothing on RDNA 3.5. KTransformers' 1.23x from whole-token graphs could
   not be confirmed beyond its paper's abstract.

## halogen, in brief

Default v2 checkpoint: 62.1 GiB, ~4.16 bpw; experts and most weights 4-bit,
mixing layers 6-bit, draft-head dense projections 8-bit; the FP8 n-gram
table (47.7 GiB) paged from disk. The older w4b used per-column 4-bit groups,
an activation-aware sidecar over 723 non-expert tensors and the 12 o_proj at
8 bits; without the sidecar, perplexity 6-9% worse. Reported quality: KL
median 0.045, top-1 86.46%. Activations stay BF16. MTP: a head rewritten to
read four residual streams and draft two ahead (committed tokens per round
1.68 -> 2.47, decode +18%); depth 2 or 3 per conversation; a 4-row chain
verify; head at 8 bits lifted prose acceptance 51% -> 59%. Nothing published
on rotations, group sizes, the per-tensor map, verify kernels, fusion or
graphs. Weights are pinned (unpinned: 37 -> 6 tok/s). halogen's README now
describes 0.17.5 and says decode got faster in 0.17.x, so the 35.9 tok/s
baseline (0.15.3) may be stale.

## Sources

- halogen: https://github.com/peonist-ai/halogen-flash-server (README,
  docs/QUANT.md, docs/FLAGS.md, CHANGELOG.md, issues #125, #138, #149),
  https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next
- gfx1151 fusion: https://github.com/halo-box/strix-llama.cpp/pull/150,
  https://github.com/ggml-org/llama.cpp/pull/16715
- Graphs: https://github.com/ROCm/rocm-systems/issues/10713,
  https://github.com/ggml-org/llama.cpp/issues/29098,
  https://arxiv.org/pdf/2605.30571
- Megakernels: https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles,
  https://arxiv.org/html/2512.22219v1, https://arxiv.org/pdf/2604.15379,
  https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/
- Small-batch kernels: https://github.com/ggml-org/llama.cpp/pull/20905
  (MoE MMVQ; flat on the Radeon 8060S),
  https://github.com/ggml-org/llama.cpp/pull/23227,
  https://github.com/IST-DASLab/marlin
- Speculation on MoE: https://arxiv.org/abs/2506.20675 (Cascade),
  https://arxiv.org/html/2602.16052v1 (MoE-Spec),
  https://www.alphaxiv.org/abs/2607.12696 (EcoSpec),
  https://sleepingrobots.com/dreams/engramhalo-qwen38-flash-next-strix-halo/
- Formats: llama.cpp perplexity README (KLD table, via
  https://docs.rs/crate/llama-cpp-sys-4/0.4.4/source/llama.cpp/tools/perplexity/README.md),
  https://github.com/ikawrakow/ik_llama.cpp/pull/83,
  https://github.com/ikawrakow/ik_llama.cpp/discussions/8,
  https://arxiv.org/html/2411.17525 (HIGGS), https://arxiv.org/html/2402.04396
  (QuIP#), https://arxiv.org/pdf/2404.00456 (QuaRot),
  https://arxiv.org/html/2510.26690v2 (LoRAQuant),
  https://arxiv.org/html/2505.02390 (DeepSeek dynamic quants)
