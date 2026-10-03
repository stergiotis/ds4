#!/usr/bin/env python3
"""CPU reference forward pass for Kolibri 1, in numpy.

This follows aleph_alpha_inference/kolibri1.py (the vLLM plugin), which is the
ground truth:

  r = embed(tokens)
  per layer:
    h = attn(input_layernorm(r));      h = post_attn_norm(h);  r += h
    h = moe(post_attention_layernorm(r)); h = post_ffn_norm(h); r += h
  logits = lm_head(norm(r))            (float32 head)

  attn: q/k/v projections, per-head RMSNorm on q and k, RoPE (neox, theta
        10000, all 128 dims) only on sliding layers, full layers are NoPE.
        Sliding layers see the current token and the 512 before it
        (sliding_window = 513). GQA 48 query / 4 KV heads, scale 1/sqrt(128).
  moe:  fp32 router logits; top-6 selection on logits + expert_bias; weights
        are the unbiased sigmoid(logits) of the selected experts, no
        renormalisation, no scale. Plus one ungated shared SwiGLU expert.

Weights are read straight from the FP8 safetensors with mmap and dequantized
(e4m3 * per-128x128-block fp32 scale_inv) one layer at a time. Math is fp32;
activations are not FP8-quantized unless --act-quant is given (vLLM's
dynamic per-token, per-128-group e4m3 quantization), so the default is the
"ideal" dequantized model the kernels should approach.
"""

import json
import math
import os
import struct

import ml_dtypes
import numpy as np

F8 = ml_dtypes.float8_e4m3fn
BF16 = ml_dtypes.bfloat16
DTYPES = {"F8_E4M3": F8, "BF16": BF16, "F32": np.float32, "F16": np.float16}
F8_MAX = 448.0


class SafeTensors:
    """Minimal mmap reader for a sharded safetensors checkpoint."""

    def __init__(self, model_dir: str):
        self.dir = model_dir
        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
        self.where = idx["weight_map"]
        self.shards = {}

    def _shard(self, name):
        if name not in self.shards:
            path = os.path.join(self.dir, name)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            mm = np.memmap(path, dtype=np.uint8, mode="r")
            self.shards[name] = (header, mm, 8 + n)
        return self.shards[name]

    def get(self, key) -> np.ndarray:
        header, mm, base = self._shard(self.where[key])
        meta = header[key]
        a, b = meta["data_offsets"]
        return mm[base + a: base + b].view(DTYPES[meta["dtype"]]).reshape(meta["shape"])

    def f32(self, key) -> np.ndarray:
        return self.get(key).astype(np.float32)

    def linear(self, prefix) -> np.ndarray:
        """Dequantized [out, in] fp32 weight of an FP8 block-scaled linear."""
        w = self.get(prefix + ".weight")
        if w.dtype != F8:
            return w.astype(np.float32)
        s = self.get(prefix + ".weight_scale_inv").astype(np.float32)
        o, i = w.shape
        w = w.astype(np.float32).reshape(s.shape[0], 128, s.shape[1], 128)
        return (w * s[:, None, :, None]).reshape(o, i)


def act_quant(x: np.ndarray) -> np.ndarray:
    """vLLM dynamic FP8 activation quantization: per token, per 128 group."""
    t, d = x.shape
    g = x.reshape(t, d // 128, 128)
    scale = np.maximum(np.abs(g).max(-1, keepdims=True), 1e-10) / F8_MAX
    q = (g / scale).clip(-F8_MAX, F8_MAX).astype(F8).astype(np.float32)
    return (q * scale).reshape(t, d)


def rms_norm(x, w, eps):
    x = x.astype(np.float32)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


class Kolibri:
    def __init__(self, model_dir: str, act_quant: bool = False):
        self.cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.st = SafeTensors(model_dir)
        c = self.cfg
        self.L = c["num_hidden_layers"]
        self.H = c["hidden_size"]
        self.nh = c["num_attention_heads"]
        self.nkv = c["num_key_value_heads"]
        self.hd = c["head_dim"]
        self.eps = c["rms_norm_eps"]
        self.window = c["sliding_window"]
        self.topk = c["num_experts_per_tok"]
        self.sliding = [t == "sliding_attention" for t in c["layer_types"]]
        self.aq = act_quant
        half = self.hd // 2
        self.inv_freq = 1.0 / (c["rope_theta"] ** (np.arange(half, dtype=np.float64) * 2 / self.hd))

    def mm(self, x, w):
        """x @ w.T with optional FP8 activation quantization (FP8 linears only)."""
        return (act_quant(x) if self.aq else x) @ w.T

    def rope(self, x, pos):
        # x [T, heads, hd]; neox style: rotate the two halves.
        ang = np.outer(pos, self.inv_freq)
        cos = np.cos(ang).astype(np.float32)[:, None, :]
        sin = np.sin(ang).astype(np.float32)[:, None, :]
        h = self.hd // 2
        x1, x2 = x[..., :h], x[..., h:]
        return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

    def attention(self, l, x, pos, cache):
        """x [T, H] normed input, pos [T] absolute positions. cache: dict per
        layer with 'k','v' [S, nkv, hd] of earlier positions (appended here)."""
        st, p = self.st, f"model.layers.{l}.self_attn"
        T = x.shape[0]
        q = self.mm(x, st.linear(p + ".q_proj")).reshape(T, self.nh, self.hd)
        k = self.mm(x, st.linear(p + ".k_proj")).reshape(T, self.nkv, self.hd)
        v = self.mm(x, st.linear(p + ".v_proj")).reshape(T, self.nkv, self.hd)
        q = rms_norm(q, st.f32(p + ".q_norm.weight"), self.eps)
        k = rms_norm(k, st.f32(p + ".k_norm.weight"), self.eps)
        if self.sliding[l]:
            q, k = self.rope(q, pos), self.rope(k, pos)
        if cache is not None:
            if "k" in cache:
                k = np.concatenate([cache["k"], k]); v = np.concatenate([cache["v"], v])
                kpos = np.concatenate([cache["pos"], pos])
            else:
                kpos = pos
            cache["k"], cache["v"], cache["pos"] = k, v, kpos
        else:
            kpos = pos
        # Mask: causal, and for sliding layers only the last `window` positions
        # including the current one.
        allow = kpos[None, :] <= pos[:, None]
        if self.sliding[l]:
            allow &= kpos[None, :] > pos[:, None] - self.window
        rep = self.nh // self.nkv
        qg = q.reshape(T, self.nkv, rep, self.hd)
        s = np.einsum("tgrd,sgd->grts", qg, k) / math.sqrt(self.hd)
        s = np.where(allow[None, None], s, -np.inf)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = np.einsum("grts,sgd->tgrd", s, v).reshape(T, self.nh * self.hd)
        return self.mm(o, st.linear(p + ".o_proj"))

    def expert(self, prefix, x):
        st = self.st
        g = self.mm(x, st.linear(prefix + ".gate_proj"))
        u = self.mm(x, st.linear(prefix + ".up_proj"))
        return self.mm(silu(g) * u, st.linear(prefix + ".down_proj"))

    def moe(self, l, x):
        st, p = self.st, f"model.layers.{l}.mlp"
        logits = x @ st.f32(p + ".gate.weight").T                  # [T, E] fp32
        bias = st.f32(f"model.layers.{l}.moe.router.expert_bias")
        ids = np.argsort(-(logits + bias), axis=-1, kind="stable")[:, :self.topk]
        w = 1.0 / (1.0 + np.exp(-np.take_along_axis(logits, ids, -1)))
        out = self.expert(p + ".shared_experts", x)
        for e in np.unique(ids):
            rows, slot = np.nonzero(ids == e)
            out[rows] += w[rows, slot, None] * self.expert(f"{p}.experts.{e}", x[rows])
        return out, ids, w

    def forward(self, tokens, pos0=0, caches=None, trace=None):
        """Returns final-norm hidden [T, H]. `trace` (dict) collects the residual
        stream after embedding and after each layer, and routing choices."""
        st = self.st
        tokens = np.asarray(tokens)
        pos = np.arange(pos0, pos0 + len(tokens))
        r = st.get("model.embed_tokens.weight")[tokens].astype(np.float32)
        if trace is not None:
            trace.setdefault("resid", []).append(r.copy())
        for l in range(self.L):
            p = f"model.layers.{l}"
            h = self.attention(l, rms_norm(r, st.f32(p + ".input_layernorm.weight"), self.eps),
                               pos, None if caches is None else caches[l])
            r = r + rms_norm(h, st.f32(p + ".post_attn_norm.weight"), self.eps)
            h, ids, w = self.moe(l, rms_norm(r, st.f32(p + ".post_attention_layernorm.weight"), self.eps))
            r = r + rms_norm(h, st.f32(p + ".post_ffn_norm.weight"), self.eps)
            if trace is not None:
                trace["resid"].append(r.copy())
                trace.setdefault("experts", []).append(ids)
                trace.setdefault("weights", []).append(w)
        return rms_norm(r, st.f32("model.norm.weight"), self.eps)

    def logits(self, h):
        return h.astype(np.float32) @ self.st.f32("lm_head.weight").T

    def greedy(self, prompt, n, trace=None):
        """Prefill the prompt, then decode n tokens greedily with a KV cache."""
        caches = [dict() for _ in range(self.L)]
        h = self.forward(prompt, 0, caches, trace)
        lg = self.logits(h)
        out, step_logits = [], [lg[-1]]
        tok = int(lg[-1].argmax())
        for i in range(n):
            out.append(tok)
            if i == n - 1:
                break
            h = self.forward([tok], len(prompt) + i, caches)
            lg1 = self.logits(h)[-1]
            step_logits.append(lg1)
            tok = int(lg1.argmax())
        return out, lg, step_logits
