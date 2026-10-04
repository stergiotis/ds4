# Kolibri 1

[README](../README.md) | [Models](MODELS.md) | [Strix Halo](STRIX_HALO.md)

[Aleph-Alpha/Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1)
(Apache-2.0, released 2026-10-03) on ROCm, tested on Strix Halo (`gfx1151`,
Radeon 8060S, 128 GB unified memory). Single GPU, resident weights; no SSD
streaming, tensor parallelism or MTP.

## Quick start

```sh
make strix-halo                      # see STRIX_HALO.md for rocWMMA headers
make -C gguf-tools quants-shared
hf download Aleph-Alpha/Kolibri-1    # 78.9 GB FP8 checkpoint
python3 gguf-tools/kolibri_convert.py --hf-dir <snapshot> \
    --dense f8_b128 --experts f8_b128 --out gguf/Kolibri-1-F8.gguf   # 75.7 GiB, lossless
# or, 42.8 GiB:  --dense f8_b128 --experts q4_k --out gguf/Kolibri-1-Q4_K.gguf
./ds4 --rocm -m gguf/Kolibri-1-F8.gguf --nothink -p "Was ist ein Kolibri?"
./ds4 --rocm -m gguf/Kolibri-1-F8.gguf --think-level 20 -p "..."    # low effort
./ds4-server --rocm -m gguf/Kolibri-1-F8.gguf --ctx 32768          # model id kolibri-1
```

The converter needs `numpy`; `tests/kolibri/pyproject.toml` has the harness
environment (`uv run --project tests/kolibri python gguf-tools/...`).
`--think-level` maps 1-33/34-66/67-100 to the template's low/medium/high
sentences and 0 to none; the server takes `reasoning_effort` (none, minimal,
low, medium, high, xhigh, max) or `chat_template_kwargs.enable_thinking`, and
the aliases `kolibri-1-nothink` / `kolibri-1-reasoner`.

## Model, as implemented by the reference

Ground truth is `aleph_alpha_inference/kolibri1.py` in
[aleph-alpha-inference](https://github.com/Aleph-Alpha/aleph-alpha-inference)
(a vLLM plugin built on vLLM's `qwen3_moe` blocks). Facts below were checked
against it and the checkpoint's `config.json`, `tokenizer.json` and
`tokenizer_config.json` at revision `e52eb46`.

| | |
| --- | --- |
| Layers | 50, every layer MoE |
| Hidden | 2560, RMSNorm eps 1e-6, weight multiplies directly (not `1 + w`) |
| Attention | GQA, 48 query / 4 KV heads, head_dim 128, no bias, scale `1/sqrt(128)` |
| QK norm | per-head RMSNorm on q and k (weights of size 128), before RoPE |
| Layer pattern | `layer_types`: 4 sliding, 1 full, repeated (full = layers 4, 9, ..., 49) |
| Sliding layers | window 513 = current token + 512 previous; RoPE neox (rotate halves), theta 10000, all 128 dims |
| Full layers | **NoPE**: no positional encoding at all ("RNoPE"); causal over the whole context |
| Norm layout | sandwich: `input_layernorm` → attn → `post_attn_norm` → +residual; `post_attention_layernorm` → MoE → `post_ffn_norm` → +residual; final `model.norm` |
| Router | fp32 logits from a BF16 `mlp.gate` [384, 2560]; **select** top-6 on `logits + expert_bias` (`moe.router.expert_bias`, fp32, magnitudes up to ~20); **weight** = `sigmoid(logits)` of the selected experts (unbiased), no renormalisation (`norm_topk_prob=false`), route scale 1.0 |
| Routed experts | 384 SwiGLU experts, intermediate 512 |
| Shared expert | 1 SwiGLU expert, intermediate 512, **ungated**, added to the routed sum |
| Vocab | 128000, untied LM head, `head_dtype` float32 |
| Special ids | `<|im_start|>` 127904, `<|im_end|>` 127906 (eos), `<|endoftext|>` 127901 (pad, also a stop id in `generation_config`), `<think>` 127907, `</think>` 127908, `<tool_call>` 127909, `</tool_call>` 127910, `<tool_response>` 127911, `</tool_response>` 127912; no BOS |
| Context | 262144 |
| Sampling defaults | temperature 1.0, top-p 0.97, top-k 128 |

Corrections to the task brief: full-attention layers are NoPE; routing adds a
selection-only expert bias; the shared expert has no gate; there are two extra
(sandwich) norms per layer.

### Weights (FP8 repo, 78.9 GB, 32 shards)

Every routed expert is stored separately (`mlp.experts.{e}.{gate,up,down}_proj`),
as are the shared expert and q/k/v/o projections. All of these are
`F8_E4M3` with a `weight_scale_inv` fp32 tensor of one scale per 128×128
block (`W = fp8 * scale_inv[o/128, i/128]`). Embeddings, LM head, all norms
and the router gate are BF16; `expert_bias` is fp32. vLLM quantizes
activations dynamically (per token, per 128-element group, e4m3) for the FP8
GEMMs; the numpy reference does not by default (see harness below).

Parameter budget: routed experts 75.5 B, attention 1.7 B, shared experts
0.2 B, embeddings + head 0.66 B.

### Tokenizer

Byte-level BPE (`byte_fallback: true`, no normalizer). The pre-tokenizer is
one `Split` regex followed by ByteLevel without its own regex:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

Digits split one at a time. 98 added tokens (ids 127900+); `<think>`,
`</think>`, `<tool_call>` and friends are added but not "special".

### Chat template

ChatML. A system block is always emitted, because the template always picks a
reasoning sentence (high by default; `reasoning_effort` low/minimal, medium,
high/xhigh/max, or `none`; `enable_thinking=false` is the same as `none`).
With thinking disabled the generation prompt ends with
`<|im_start|>assistant\n<think>\n\n</think>\n\n`; otherwise it ends with
`<|im_start|>assistant\n` and the model opens `<think>` itself. Earlier
assistant turns keep their reasoning only after the last real user query.
Tools are listed as JSON lines inside `<tools></tools>`; calls are Hermes
style `<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>`, and tool
results go back in a user turn as `<tool_response>...</tool_response>`.

## Design

### Base

The `kolibri` branch is antirez/ds4#1070 (Qwen3.8 Flash Next on ROCm, not
merged upstream) plus the Strix Halo `prefetch` work. #1070 provides the only
ROCm "island" for a GQA MoE model, whose type-templated projection and expert
kernels Kolibri reuses. Kolibri has its own family/variant/shape profile, one
graph (`ds4_kolibri_gpu_graph`, `forward_tokens(T)` for prefill and decode)
and family branches only at engine open, session sync/eval/rewind, payload
save/load and the server renderer.

### Weight formats

| GGUF | Size | Linear layers | Agreement with the reference (below) |
| --- | ---: | --- | --- |
| `--dense f8_b128 --experts f8_b128` | 75.7 GiB | the release's FP8, bit-exact | top-1 0.975, KL 0.008 |
| `--dense f8_b128 --experts q4_k` | 42.8 GiB | FP8 dense, Q4_K routed experts | top-1 0.903, KL 0.206 |
| `--dense q8_0 --experts q8_0` | 78.0 GiB | requantized Q8_0 | top-1 0.929, KL 0.097 |

`F8_B128` (GGUF type 200, ds4's own; no ggml equivalent) keeps the
checkpoint's e4m3 bytes and fp32 128×128 block scales row-local: per 512
values the four scales of its 128-wide source blocks, then 512 bytes (528
bytes, 8.25 bits per weight, payload 16-byte aligned). The converter writes
`kolibri1.f8_block = 512`; ds4 refuses F8 tensors without it. gfx1151 has no
FP8 instructions, but e4m3 → f16 is a shift and a mask into an f16 pattern
times 2^8, folded into the scale, so decoding costs about as much as Q8_0.

Q8_0 was the first plan and is supported, but requantization moves this
model visibly (each FP8 value gains up to half a Q8_0 step, 1% RMS): the raw
prompt "1, 2, 3, 4, 5," continues " 5, 5, 5" instead of " 6, 7, 8", and
numpy with Q8_0-rounded weights reproduces exactly that. Use F8_B128 for
fidelity and Q4_K for memory.

Embeddings and LM head stay BF16, norms, router and expert bias F32.

### Kernels

| Piece | Implementation |
| --- | --- |
| Dense projections | #1070 matvec (decode); prefill: the gfx1151 direct WMMA kernel templated on the weight format (F8 decode in the fragment load), F16-tile BLAS elsewhere |
| Routed experts | #1070 `moe_mv` (decode; the shared expert rides as an extra slot with weight 1) and the WMMA `matrix_half_tile` (prefill, F8 and Q4_K) |
| F8 reads | `value`/`value4` and a 16-bytes-per-lane `dot_f8` in the #1070 readers |
| QK prep (new) | weighted per-head RMSNorm, neox RoPE on sliding layers only, f16 K/V store |
| Attention (new) | prefill: WMMA flash attention, 16 queries × 4 heads per block sharing K/V tiles, S^T = K Q^T so softmax is lane-local; decode: one wave per (KV head, key split) serving all 12 query heads of the group |
| Router (new) | top-6 on `logits + bias`, weight `sigmoid(logit)`, no renormalisation |
| Sandwich norm (new) | fused `x += rms(h)·w_post; xn = rms(x)·w_next`, also folding the MoE reduce |

### KV cache

f16. Sliding layers keep a ring of `window - 1 + chunk` rows (position p in row
p % rows); full layers one row per position: 4 KV heads × 128 × 2 × 2 bytes =
2 KiB per token per full layer, 20 KiB per token for the 10 full layers,
5 GiB at the native 262144 context. Rewind stays on the live KV when it goes
back at most one prefill chunk (the sliding rings' slack), otherwise the
session re-prefills.

## Correctness

Harness (`tests/kolibri/`, fixtures in `tests/golden/kolibri*/`):

- `make_tokenizer_goldens.py`: HF `tokenizers` ids for 15 raw strings
  (digits, contractions, whitespace, code, combining marks, CJK, emoji) and 7
  rendered chats (thinking on/off, efforts, multi-turn, tool round trip).
  ds4 matches all 22 (`test_tokenizer.py`); the CLI renders the chats to the
  same ids, and `./ds4_test --server` checks the server renderer against
  HF's `apply_chat_template` text and parses Hermes calls.
- `kolibri_ref.py`: numpy forward from the FP8 safetensors (mmap, one layer at
  a time, fp32, optional vLLM-style FP8 activations, optional Q8_0-rounded
  weights), with prefill and cached greedy decode; `test_ref_consistency.py`
  checks cached decode against full recompute.
- `make_golden.py`: per-layer residuals, routing, logits and 16-token greedy
  continuations for 5 prompts (raw English, German and digits, an English chat
  with thinking, a German chat without). The reference's own text is sensible
  ("The capital of France is" → " Paris. The capital of Germany is Berlin.",
  the German chat → "Ein Kolibri ist ein kleiner, farbenprächtiger Vogel ...").
- `compare_golden.py`: runs `ds4 --first-token-test` with `DS4_KOLIBRI_*`
  settings on the C CPU reference or the ROCm graph.
- `make_tiny_checkpoint.py`: a random 10-layer Kolibri in the release format
  (window 9) for fast checks of every code path.

Results:

- C CPU reference vs numpy (tiny model, F32 or F8 weights): 2e-6 on every layer,
  identical routing and greedy output.
- ROCm vs numpy, real Kolibri-1, F8 GGUF: all 5 prompts give identical
  16-token greedy continuations, top-1 agreement 1.000 at every prompt position
  (0.982 on the chat with thinking), residual error ≤ 8e-3 except `count`
  (3e-2): a router near-tie at its position 5 decides differently with
  float-noise changes to the decode kernels, which moves the final residual
  between 4e-4 and 7e-2 (above the 0.05 threshold, a FAIL, with one of the
  kernel versions) while greedy output and top-1 still match.
- Teacher-forced over `tf_text.txt` (196 tokens, German, English, Python)
  against the fp32 reference (`teacher_forced.py`):

  | | top-1 | KL (nats) | perplexity |
  | --- | ---: | ---: | ---: |
  | reference, fp32 | - | - | 7.04 |
  | reference with vLLM-style FP8 activations | 0.903 | 0.157 | 7.00 |
  | ds4 F8, prefill paths | 0.975 | 0.008 | 7.11 |
  | ds4 F8, token-by-token decode | 0.975 | 0.008 | 7.10 |
  | ds4 Q8_0 | 0.929 | 0.097 | 6.93 |
  | ds4 Q4_K experts | 0.903 | 0.206 | 6.75 |

  ds4's F8 path sits 20× closer to the fp32 reference than the reference's
  own FP8-activation execution. Exact greedy agreement over long generations
  is not a meaningful target beyond that: the router has many near-ties among
  384 experts (a 6th/7th selection gap of 5e-4 flips under f16 rounding), so
  any two FP8 implementations, vLLM included, part ways on some prompts.
- Serving (`server_smoke.py`): German chat without reasoning ("Die Hauptstadt
  von Bayern ist München."), English with low reasoning (reasoning and
  content separate, also streamed), and a `get_weather` tool round trip.

The reference could not be cross-checked against vLLM itself on this machine
(vLLM's FP8 MoE targets CUDA and MI300-class ROCm).

## Performance

Strix Halo (Radeon 8060S, 128 GB), `tests/kolibri/bench.sh`: `ds4-bench`
on `speed-bench/promessi_sposi.txt`, cold prefill of the given context, then
128 greedy tokens at that context; median of three runs (spread under 2%
except the first, cold 512 run). Memory is the peak GPU allocation (GTT +
VRAM) above idle; host RSS stays under 0.8 GiB.

| GGUF | context | prefill t/s | decode t/s | GPU memory |
| --- | ---: | ---: | ---: | ---: |
| F8 (75.7 GiB) | 512 | 980 | 49.7 | 76.3 GiB |
| F8 | 8192 | 1373 | 46.6 | 77.0 GiB |
| F8 | 32768 | 1090 | 42.1 | 77.5 GiB |
| Q4_K experts (42.8 GiB) | 512 | 1125 | 55.2 | 43.3 GiB |
| Q4_K experts | 8192 | 1494 | 51.5 | 44.0 GiB |
| Q4_K experts | 32768 | 1161 | 46.1 | 44.5 GiB |

Starting point (first correct version, F8, 2K context): 210 t/s prefill,
21 t/s decode. Decode at 2K spends per token about 4.8 ms on QKV, 4.2 ms on
attention, 4.6 ms on the output projection and norms, 10.6 ms on the experts
and 3.9 ms on the LM head (`DS4_KOLIBRI_TIMING=1`, which syncs per stage).
Bandwidth would allow roughly 50 t/s for F8; see known gaps.
`DS4_KOLIBRI_TRACE=1` gives per-call times without the per-stage syncs.

Q4_K saves 33 GiB and decodes 10-12% faster than F8 (its experts read half
the bytes), so it fits next to other resident workloads (the machine's other
65 GB service, for example). It is lossy: see the teacher-forced table.

## Known gaps

The ranked ideas for making it faster are in
[strix/KOLIBRI-PLAN.md](../strix/KOLIBRI-PLAN.md).


- **FP8 KV cache for the full layers** is not implemented (f16; 5 GiB at
  262144 tokens fits next to the 76 GiB model).
- **Disk KV checkpoints** (`--kv-disk-dir`) and session payload save/load
  refuse Kolibri sessions with an error; live KV reuse works.
- **Decode loses time in small calls**, not in launches (the host enqueues a
  token in 1.4 ms): each remaining small kernel (norms ~10 us, k/v
  projections ~11 us) costs about its launch-to-completion floor, while the
  large matvecs already read at ~210 GB/s.
- **Router**: decode batches (T <= 8) use a BF16 copy of the F32 router
  weights (exact, checked when the copy is made; 98 MB of GPU memory) with
  logits and top-k in one launch. `DS4_KOLIBRI_ROUTER_F32=1` keeps the F32
  matvec; prefill always uses it.
  Q4_K experts read half the bytes of F8 but at half the bandwidth, so both
  GGUFs decode at the same speed. `DS4_KOLIBRI_TRACE=1` shows the per-call
  split.
- **Prefill**: F8 and Q4_K experts run Kolibri's WMMA tiles
  (`DS4_KOLIBRI_F8_TILE=0` / `DS4_KOLIBRI_Q4K_TILE=0` restore the generic
  #1070 tile). At 32K, attention dominates.
- **Thinking prefill**: the server prefills `<think>\n` when thinking is on
  (the template leaves it to the model) so Qwen's reasoning machinery applies;
  the reference's own first output in that position is `<think>\n`. The CLI
  follows the template exactly.
- **Streamed content** after reasoning starts with the template's `\n\n`
  separator (the non-streamed response trims it); shared with the Qwen path.
- **No top-k in the CLI** (the release recommends top-k 128); the server
  accepts `top_k`.
- **llama.cpp** has no Kolibri support yet (no issue or PR as of 2026-10-03),
  so there is no comparison.
- Batched multi-session decode runs sessions one at a time.
