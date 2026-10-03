#!/usr/bin/env python3
"""A small random Kolibri 1 checkpoint in the release's exact on-disk format
(FP8 e4m3 linears with 128x128 block scales, BF16 embeddings/head/norms/gate,
F32 expert bias), with the real tokenizer. It exercises the converter, the
numpy reference and the ds4 code paths without the 79 GB download.

    uv run make_tiny_checkpoint.py --tokenizer-dir HF_SNAPSHOT --out DIR

Shapes keep everything the kernels care about (head_dim 128, GQA 4:1 ... here
8:2, top-6 sigmoid routing, intermediate 256, hidden a multiple of 256) and
shrink the rest. The window is 9 so short prompts cross it.
"""

import argparse
import json
import os
import shutil

import ml_dtypes
import numpy as np
from safetensors.numpy import save_file

TINY = {
    "architectures": ["Kolibri1ForCausalLM"],
    "model_type": "kolibri1",
    "hidden_size": 512,
    "num_hidden_layers": 10,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "hidden_act": "silu",
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-06,
    "vocab_size": 128000,
    "rope_theta": 10000.0,
    "num_experts": 16,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 256,
    "shared_expert_intermediate_size": 256,
    "norm_topk_prob": False,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "tie_word_embeddings": False,
    "use_cache": True,
    "use_sliding_window": True,
    "sliding_window": 9,
    "layer_types": (["sliding_attention"] * 4 + ["full_attention"]) * 2,
    "bos_token_id": None,
    "eos_token_id": 127906,
    "pad_token_id": 127901,
    "dtype": "bfloat16",
    "head_dtype": "float32",
    "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic",
                            "weight_block_size": [128, 128]},
}


def fp8_block(w):
    """Quantize like the release: per 128x128 block, scale = amax / 448."""
    o, i = w.shape
    b = w.reshape(o // 128, 128, i // 128, 128)
    scale = np.maximum(np.abs(b).max(axis=(1, 3)), 1e-12) / 448.0
    q = (b / scale[:, None, :, None]).astype(ml_dtypes.float8_e4m3fn)
    return q.reshape(o, i), scale.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    c = TINY
    rng = np.random.default_rng(args.seed)
    H, nh, nkv, hd = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    E, ff, V = c["num_experts"], c["moe_intermediate_size"], c["vocab_size"]
    bf16 = ml_dtypes.bfloat16
    t = {}

    def lin(name, o, i):
        q, s = fp8_block(rng.standard_normal((o, i), np.float32) / np.sqrt(i))
        t[name + ".weight"], t[name + ".weight_scale_inv"] = q, s

    def norm(name, n):
        t[name] = (1.0 + 0.2 * rng.standard_normal(n)).astype(bf16)

    t["model.embed_tokens.weight"] = rng.standard_normal((V, H), np.float32).astype(bf16)
    for l in range(c["num_hidden_layers"]):
        p = f"model.layers.{l}"
        lin(p + ".self_attn.q_proj", nh * hd, H)
        lin(p + ".self_attn.k_proj", nkv * hd, H)
        lin(p + ".self_attn.v_proj", nkv * hd, H)
        lin(p + ".self_attn.o_proj", H, nh * hd)
        norm(p + ".self_attn.q_norm.weight", hd)
        norm(p + ".self_attn.k_norm.weight", hd)
        for n in ("input_layernorm", "post_attn_norm", "post_attention_layernorm", "post_ffn_norm"):
            norm(f"{p}.{n}.weight", H)
        t[p + ".mlp.gate.weight"] = (rng.standard_normal((E, H), np.float32) / np.sqrt(H)).astype(bf16)
        t[p + ".moe.router.expert_bias"] = (2.0 * rng.standard_normal(E)).astype(np.float32)
        for e in range(E):
            lin(f"{p}.mlp.experts.{e}.gate_proj", ff, H)
            lin(f"{p}.mlp.experts.{e}.up_proj", ff, H)
            lin(f"{p}.mlp.experts.{e}.down_proj", H, ff)
        lin(p + ".mlp.shared_experts.gate_proj", ff, H)
        lin(p + ".mlp.shared_experts.up_proj", ff, H)
        lin(p + ".mlp.shared_experts.down_proj", H, ff)
    norm("model.norm.weight", H)
    t["lm_head.weight"] = (rng.standard_normal((V, H), np.float32) / np.sqrt(H)).astype(bf16)

    os.makedirs(args.out, exist_ok=True)
    # Two shards, so the index path is exercised.
    names = sorted(t)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half],
              "model-00002-of-00002.safetensors": names[half:]}
    weight_map = {}
    for shard, keys in shards.items():
        save_file({k: t[k] for k in keys}, os.path.join(args.out, shard))
        weight_map.update({k: shard for k in keys})
    json.dump({"metadata": {}, "weight_map": weight_map},
              open(os.path.join(args.out, "model.safetensors.index.json"), "w"), indent=1)
    json.dump(c, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(args.tokenizer_dir, f), args.out)
    print(f"wrote {len(t)} tensors to {args.out}")


if __name__ == "__main__":
    main()
