# Kolibri 1

[README](../README.md) | [Models](MODELS.md) | [Strix Halo](STRIX_HALO.md)

Work-in-progress notes for running
[Aleph-Alpha/Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1)
(Apache-2.0, released 2026-10-03) on ROCm, first target Strix Halo
(`gfx1151`, 128 GB unified memory).

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

## Plan

### Base

The `kolibri` branch is antirez/ds4#1070 (Qwen3.8 Flash Next on ROCm, not yet
merged upstream) plus the Strix Halo `prefetch` work. #1070 matters because it
is the only ROCm "island" for a GQA model: f16 K/V cache, `attention<D>` with
key-split partials, type-templated decode MoE (`moe_mv<TYPE>`), expert prefill
(`matrix<TYPE>` and the WMMA `matrix_half_tile` for Q2/Q4/MXFP4), dense
matvec/BLAS paths, argmax. Qwen3.8 Flash Next also has hidden size 2560.

Kolibri follows the qwen4 pattern rather than threading through the
DeepSeek/GLM graph: its own family/variant/shape profile, one graph struct
with `forward_tokens(T)` serving prefill and decode, and family branches only
at engine open, session sync/eval, payload save/load and the server renderer.

### Weight format

ds4 has no FP8 weight type, and gfx1151 has no FP8 WMMA or FP8 conversion
instructions. The converter (`gguf-tools/kolibri_convert.py`) dequantizes the
FP8 blocks (`e4m3 * scale_inv`) and requantizes with `libds4quants`:

1. **Q8_0** for every linear layer including the experts (about 83 GB), BF16
   embeddings, F32 norms, router and expert bias, LM head Q8_0 or BF16
   (decided by the logit comparison). Q8_0 with 32-element blocks and an f16
   scale per block re-represents e4m3 values almost exactly; this is the
   fidelity baseline. #1070 decodes Q8_0 experts with `moe_mv<8>`; Q8_0
   prefill uses the generic `matrix<8>` until a WMMA tile is added.
2. **Q4_K experts** + Q8_0 dense (about 47 GB): fits next to other resident
   workloads and uses #1070's WMMA prefill tiles.

Native FP8 in GGUF stays an option for later: e4m3 → f16 is a shift, a mask
and one multiply folded into the block scale, and it would save ~4.5 GB over
Q8_0, but needs its own kernels in every path.

### Reused vs new

| Piece | Source |
| --- | --- |
| Q8_0 / Q4_K / BF16 matvec and prefill GEMM | #1070 `qwen4_rocm::matvec_dispatch`, `dense_blas` |
| Routed experts, decode and prefill | #1070 `moe_mv_dispatch`, `expert_lists`, `matrix_dispatch` (K=2560, M=512, top-6) |
| Shared expert | same kernels, as the (NS+1)-th expert slot, weight 1 |
| RMSNorm (input, sandwich, final) | existing ROCm `rms_norm_weight` / `add_rms_norm_weight` |
| Argmax / sampling | existing |
| **New:** QK prep | q/k/v split, weighted per-head RMSNorm, neox RoPE on sliding layers only, f16 K/V store (ring of 513+chunk rows for sliding layers, linear for full layers) |
| **New:** attention | GQA 48/4, D=128, sliding window or full causal, no output gate (#1070's kernel multiplies by `sigmoid(gate)`) |
| **New:** router | fp32 logits, top-6 on `logits + bias`, weight `sigmoid(logit)`, no renormalisation (#1070's router is softmax + renorm, GLM's renormalises) |
| **New:** tokenizer | the Kolibri split regex (close to qwen35's, without `\p{M}`), single-digit numbers |
| **New:** chat / server | ChatML renderer with reasoning sentences, Hermes JSON `<tool_call>` parser |

### KV cache

40 sliding layers need only 513 positions each; a ring buffer per layer (plus
room for one prefill chunk) keeps them at about 1 MB/layer in f16. The 10 full
layers grow with context: 4 KV heads × 128 × 2 (K, V) × 2 bytes = 4 KB per
token per layer, 40 KB per token, 10 GB at 256K. An FP8 (e4m3) KV cache for
the full layers halves that, after the f16 path is correct.

### Correctness harness

- `tests/kolibri/make_tokenizer_goldens.py` → `tests/golden/kolibri/tokenizer.json`:
  ids from HF `tokenizers` for raw strings and rendered chats (thinking on/off,
  reasoning efforts, multi-turn, a tool-call round trip).
- `tests/kolibri/kolibri_ref.py`: numpy forward pass from the FP8 safetensors
  (mmap, dequantized one layer at a time, fp32 math, optional vLLM-style FP8
  activation quantization), prefill plus cached greedy decode.
- `tests/kolibri/make_golden.py` → `tests/golden/kolibri/*.npz`: per-layer
  residuals, routing, top logits and greedy continuations for short prompts.
- The C side compares against these: the CPU reference first, then ROCm.

The reference cannot be cross-checked against vLLM here (vLLM's FP8 MoE needs
CUDA or MI300-class ROCm); the numpy code is checked by its own consistency
(cached decode = full recompute), sensible greedy text and low perplexity.
