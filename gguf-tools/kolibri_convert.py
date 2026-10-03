#!/usr/bin/env python3
"""Convert Aleph-Alpha/Kolibri-1 (FP8 e4m3, 128x128 block scales) to a ds4 GGUF.

    make -C gguf-tools quants-shared
    python3 gguf-tools/kolibri_convert.py --hf-dir SNAPSHOT --out gguf/Kolibri-1-Q8_0.gguf
    python3 gguf-tools/kolibri_convert.py --hf-dir SNAPSHOT --experts q4_k \
        --out gguf/Kolibri-1-Q4_K.gguf

FP8 linears are dequantized (e4m3 * scale_inv of their 128x128 block) to fp32
and requantized with libds4quants: Q8_0 for attention and shared experts,
Q8_0 or Q4_K for routed experts. Embeddings and the LM head keep their BF16
source values; norms, router and expert bias are F32.

Tensor names follow llama.cpp conventions (blk.N.attn_q, ffn_gate_exps, ...),
with the sandwich norms as post_attention_norm / post_ffw_norm (Gemma 2) and
the selection bias as exp_probs_b. Routed experts are stacked into one
[n_expert, out, in] tensor per projection.
"""

import argparse
import concurrent.futures as cf
import ctypes
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm53_quantize import (GGUF_ALIGNMENT, GGUF_ARRAY, GGUF_STRING, GGUF_VERSION,  # noqa: E402
                            align, kv_bool, kv_f32, kv_string, kv_u32, kv_u32_array,
                            pack_string)

ARCH = "kolibri1"
# F8_B128 is ds4's own type (no ggml equivalent): the release's FP8 weights
# kept bit-exact, row-local.  Each 128-element block is the fp32 scale of
# its 128x128 source block followed by the 128 e4m3 bytes (132 bytes).
QTYPES = {"F32": 0, "Q8_0": 8, "Q4_K": 12, "BF16": 30, "F8_B128": 200}
SRC_DTYPES = {"F8_E4M3": 1, "BF16": 2, "F32": 4}


def fail(msg):
    raise SystemExit(f"kolibri_convert: {msg}")


def fp8_e4m3_lut():
    """All 256 e4m3fn codes as float32 (no infinities; 0x7f/0xff are NaN)."""
    lut = np.zeros(256, np.float32)
    for c in range(256):
        s = -1.0 if c & 0x80 else 1.0
        e, m = (c >> 3) & 0xF, c & 0x7
        if e == 0xF and m == 0x7:
            lut[c] = np.nan
        elif e == 0:
            lut[c] = s * m / 8.0 * 2.0 ** -6
        else:
            lut[c] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return lut


class Source:
    """mmap view of the sharded safetensors checkpoint."""

    def __init__(self, hf_dir):
        self.dir = Path(hf_dir)
        self.where = json.load(open(self.dir / "model.safetensors.index.json"))["weight_map"]
        self.shards = {}
        self.lut = fp8_e4m3_lut()
        self.used = set()

    def _shard(self, name):
        if name not in self.shards:
            path = self.dir / name
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            self.shards[name] = (header, np.memmap(path, np.uint8, "r"), 8 + n)
        return self.shards[name]

    def f8_b128(self, key):
        """F8_B128 bytes of an FP8 block-scaled [out, in] weight, rows in order."""
        dtype, shape, data = self.raw(key)
        if dtype != "F8_E4M3":
            fail(f"{key}: F8_B128 needs an F8_E4M3 source, got {dtype}")
        _, sshape, sdata = self.raw(key + "_scale_inv")
        o, i = shape
        s = sdata.view("<f4").reshape(sshape)
        out = np.empty((o, i // 128, 132), np.uint8)
        out[:, :, :4] = np.repeat(s, 128, axis=0).reshape(o, i // 128, 1).view(np.uint8)
        out[:, :, 4:] = data.reshape(o, i // 128, 128)
        return out.tobytes()

    def raw(self, key):
        if key not in self.where:
            fail(f"missing source tensor {key}")
        self.used.add(key)
        header, mm, base = self._shard(self.where[key])
        meta = header[key]
        a, b = meta["data_offsets"]
        return meta["dtype"], meta["shape"], mm[base + a: base + b]

    def f32(self, key):
        dtype, shape, data = self.raw(key)
        if dtype == "F32":
            return data.view("<f4").reshape(shape)
        if dtype == "BF16":
            return (data.view("<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)
        if dtype == "F8_E4M3":
            dtype_s, sshape, sdata = self.raw(key + "_scale_inv")
            if dtype_s != "F32":
                fail(f"{key}_scale_inv is {dtype_s}, expected F32")
            o, i = shape
            if sshape != [(o + 127) // 128, (i + 127) // 128] or o % 128 or i % 128:
                fail(f"{key}: unexpected block-scale shape {sshape} for {shape}")
            s = sdata.view("<f4").reshape(sshape)
            w = self.lut[data.reshape(shape)].reshape(sshape[0], 128, sshape[1], 128)
            return (w * s[:, None, :, None]).reshape(o, i)
        fail(f"{key}: unsupported source dtype {dtype}")

    def bf16_bytes(self, key):
        dtype, shape, data = self.raw(key)
        if dtype != "BF16":
            fail(f"{key}: expected BF16, got {dtype}")
        return shape, data


class Quantizer:
    def __init__(self, path):
        if not Path(path).is_file():
            fail(f"quantizer library not found: {path}; run make -C gguf-tools quants-shared")
        lib = ctypes.CDLL(str(path))
        lib.ds4q_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        lib.ds4q_row_size.restype = ctypes.c_size_t
        lib.ds4q_quantize_init.argtypes = [ctypes.c_int]
        lib.ds4q_quantize_chunk.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                            ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                            ctypes.c_int64, ctypes.POINTER(ctypes.c_float)]
        lib.ds4q_quantize_chunk.restype = ctypes.c_size_t
        for q in ("Q8_0", "Q4_K"):
            lib.ds4q_quantize_init(QTYPES[q])
        self.lib = lib

    def row_size(self, qtype, n):
        if qtype == "F8_B128":
            return n // 128 * 132
        if qtype == "F32":
            return 4 * n
        if qtype == "BF16":
            return 2 * n
        return self.lib.ds4q_row_size(QTYPES[qtype], n)

    def encode(self, x, qtype):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if qtype == "F32":
            return x.tobytes()
        if not np.all(np.isfinite(x)):
            fail("non-finite weights")
        rows, cols = x.shape
        out = np.empty(rows * self.row_size(qtype, cols), np.uint8)
        n = self.lib.ds4q_quantize_chunk(QTYPES[qtype], x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                         out.ctypes.data_as(ctypes.c_void_p), 0, rows, cols, None)
        if n != out.nbytes:
            fail(f"{qtype} quantization wrote {n} bytes, expected {out.nbytes}")
        return out.tobytes()


class Tensor:
    """One GGUF tensor. `ne` is ggml order (innermost first). `parts` yields the
    source rows to encode, as (callable returning a [rows, ne0] fp32 array) or
    raw bytes; each part is encoded independently so experts stream."""

    def __init__(self, name, qtype, ne, parts):
        self.name, self.qtype, self.ne, self.parts = name, qtype, list(ne), parts
        self.offset = 0
        self.nbytes = 0


def tokenizer_records(hf_dir, vocab_size):
    tok = json.load(open(Path(hf_dir) / "tokenizer.json"))
    cfg = json.load(open(Path(hf_dir) / "tokenizer_config.json"))
    model = tok["model"]
    if model["type"] != "BPE":
        fail("tokenizer is not BPE")
    tokens = [None] * vocab_size
    types = [1] * vocab_size              # 1 normal, 3 control, 4 user-defined, 5 unused
    for s, i in model["vocab"].items():
        tokens[i] = s
    for a in tok["added_tokens"]:
        tokens[a["id"]] = a["content"]
        types[a["id"]] = 3 if a["special"] else 4
    for i, s in enumerate(tokens):
        if s is None:
            tokens[i] = f"[PAD{i}]"
            types[i] = 5
    merges = [" ".join(m) if isinstance(m, list) else m for m in model["merges"]]

    def str_array(key, values):
        out = pack_string(key) + struct.pack("<IIQ", GGUF_ARRAY, GGUF_STRING, len(values))
        return out + b"".join(pack_string(v) for v in values)

    def i32_array(key, values):
        return (pack_string(key) + struct.pack("<IIQ", GGUF_ARRAY, 5, len(values))
                + struct.pack(f"<{len(values)}i", *values))

    ids = {s: i for i, s in enumerate(tokens)}
    return [
        kv_string("tokenizer.ggml.model", "gpt2"),
        kv_string("tokenizer.ggml.pre", "kolibri1"),
        str_array("tokenizer.ggml.tokens", tokens),
        i32_array("tokenizer.ggml.token_type", types),
        str_array("tokenizer.ggml.merges", merges),
        kv_u32("tokenizer.ggml.eos_token_id", ids[cfg["eos_token"]]),
        kv_u32("tokenizer.ggml.padding_token_id", ids[cfg["pad_token"]]),
        kv_bool("tokenizer.ggml.add_bos_token", False),
        kv_string("tokenizer.chat_template", cfg["chat_template"]),
    ]


def model_records(cfg, args, n_tensors):
    sliding = [1 if t == "sliding_attention" else 0 for t in cfg["layer_types"]]
    p = ARCH + "."
    return [
        kv_string("general.architecture", ARCH),
        kv_string("general.name", "Kolibri-1"),
        kv_string("general.source.huggingface.repository", "Aleph-Alpha/Kolibri-1"),
        kv_string("general.source.revision", args.revision),
        kv_string("general.license", "apache-2.0"),
        kv_string("general.file_type.experts", args.experts.upper()),
        kv_u32(p + "block_count", cfg["num_hidden_layers"]),
        kv_u32(p + "context_length", cfg["max_position_embeddings"]),
        kv_u32(p + "embedding_length", cfg["hidden_size"]),
        kv_u32(p + "vocab_size", cfg["vocab_size"]),
        kv_u32(p + "attention.head_count", cfg["num_attention_heads"]),
        kv_u32(p + "attention.head_count_kv", cfg["num_key_value_heads"]),
        kv_u32(p + "attention.key_length", cfg["head_dim"]),
        kv_u32(p + "attention.value_length", cfg["head_dim"]),
        kv_f32(p + "attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"]),
        kv_u32(p + "attention.sliding_window", cfg["sliding_window"]),
        kv_u32_array(p + "attention.sliding_window_pattern", sliding),
        kv_f32(p + "rope.freq_base", cfg["rope_theta"]),
        kv_u32(p + "rope.dimension_count", cfg["head_dim"]),
        kv_u32(p + "expert_count", cfg["num_experts"]),
        kv_u32(p + "expert_used_count", cfg["num_experts_per_tok"]),
        kv_u32(p + "expert_shared_count", 1),
        kv_u32(p + "expert_feed_forward_length", cfg["moe_intermediate_size"]),
        kv_u32(p + "expert_shared_feed_forward_length", cfg["shared_expert_intermediate_size"]),
        kv_bool(p + "expert_weights_norm", bool(cfg["norm_topk_prob"])),
        kv_f32(p + "expert_weights_scale", 1.0),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
    ]


def build_plan(src, cfg, experts_q, dense_q="Q8_0"):
    L, E = cfg["num_hidden_layers"], cfg["num_experts"]
    H, ff = cfg["hidden_size"], cfg["moe_intermediate_size"]
    tensors = []

    def bf16(name, key):
        shape, data = src.bf16_bytes(key)
        tensors.append(Tensor(name, "BF16", shape[::-1], [lambda d=data: bytes(d)]))

    def f32(name, key):
        shape = src.raw(key)[1]
        tensors.append(Tensor(name, "F32", shape[::-1], [lambda k=key: src.f32(k).reshape(-1, shape[-1])]))

    def lin(name, key, q=dense_q):
        shape = src.raw(key + ".weight")[1]
        src.raw(key + ".weight_scale_inv")
        part = (lambda k=key: src.f8_b128(k + ".weight")) if q == "F8_B128" else \
               (lambda k=key: src.f32(k + ".weight"))
        tensors.append(Tensor(name, q, shape[::-1], [part]))

    bf16("token_embd.weight", "model.embed_tokens.weight")
    for l in range(L):
        m, b = f"model.layers.{l}", f"blk.{l}"
        f32(b + ".attn_norm.weight", m + ".input_layernorm.weight")
        lin(b + ".attn_q.weight", m + ".self_attn.q_proj")
        lin(b + ".attn_k.weight", m + ".self_attn.k_proj")
        lin(b + ".attn_v.weight", m + ".self_attn.v_proj")
        f32(b + ".attn_q_norm.weight", m + ".self_attn.q_norm.weight")
        f32(b + ".attn_k_norm.weight", m + ".self_attn.k_norm.weight")
        lin(b + ".attn_output.weight", m + ".self_attn.o_proj")
        f32(b + ".post_attention_norm.weight", m + ".post_attn_norm.weight")
        f32(b + ".ffn_norm.weight", m + ".post_attention_layernorm.weight")
        f32(b + ".ffn_gate_inp.weight", m + ".mlp.gate.weight")
        f32(b + ".exp_probs_b.bias", m + ".moe.router.expert_bias")
        for proj, short, ne in (("gate_proj", "gate", (H, ff)), ("up_proj", "up", (H, ff)),
                                ("down_proj", "down", (ff, H))):
            keys = [f"{m}.mlp.experts.{e}.{proj}.weight" for e in range(E)]
            for k in keys:
                src.raw(k)
                src.raw(k + "_scale_inv")
            if experts_q == "F8_B128":
                parts = [lambda k=k: src.f8_b128(k) for k in keys]
            else:
                parts = [lambda k=k: src.f32(k) for k in keys]
            tensors.append(Tensor(f"{b}.ffn_{short}_exps.weight", experts_q, (*ne, E), parts))
            lin(f"{b}.ffn_{short}_shexp.weight", f"{m}.mlp.shared_experts.{proj}")
        f32(b + ".post_ffw_norm.weight", m + ".post_ffn_norm.weight")
    f32("output_norm.weight", "model.norm.weight")
    bf16("output.weight", "lm_head.weight")
    return tensors


def tensor_bytes(q, t):
    rows = 1
    for d in t.ne[1:]:
        rows *= d
    return rows * q.row_size(t.qtype, t.ne[0])


def write(args, src, cfg, q):
    tensors = build_plan(src, cfg, args.experts.upper(), args.dense.upper())
    # Every source tensor must be consumed: catches a missed rename.
    unused = sorted(k for k in src.where if k not in src.used and
                    not (k.endswith("_scale_inv") and k[:-len("_scale_inv")] in src.used))
    if unused:
        fail(f"{len(unused)} source tensors not converted, first {unused[0]}")
    off = 0
    for t in tensors:
        t.nbytes = tensor_bytes(q, t)
        t.offset = off
        off = align(off + t.nbytes)
    kv = model_records(cfg, args, len(tensors)) + tokenizer_records(args.hf_dir, cfg["vocab_size"])
    header = struct.pack("<4sIQQ", b"GGUF", GGUF_VERSION, len(tensors), len(kv)) + b"".join(kv)
    for t in tensors:
        header += pack_string(t.name) + struct.pack("<I", len(t.ne))
        header += struct.pack(f"<{len(t.ne)}Q", *t.ne) + struct.pack("<IQ", QTYPES[t.qtype], t.offset)
    data_start = align(len(header))
    total = data_start + off
    print(f"{len(tensors)} tensors, {total / 2**30:.2f} GiB -> {args.out}", flush=True)
    tmp = args.out + ".part"
    t0, done = time.time(), 0
    with open(tmp, "wb") as f, cf.ThreadPoolExecutor(args.threads) as pool:
        f.write(header + b"\0" * (data_start - len(header)))
        for i, t in enumerate(tensors):
            assert f.tell() == data_start + t.offset, t.name

            def enc(part, t=t):
                x = part()
                return x if isinstance(x, (bytes, bytearray)) else q.encode(x, t.qtype)
            written = 0
            for blob in pool.map(enc, t.parts):
                f.write(blob)
                written += len(blob)
            if written != t.nbytes:
                fail(f"{t.name}: wrote {written} bytes, planned {t.nbytes}")
            f.write(b"\0" * (align(f.tell() - data_start) - (f.tell() - data_start)))
            done += t.nbytes
            if i % 25 == 0 or i == len(tensors) - 1:
                el = time.time() - t0
                print(f"  [{i + 1}/{len(tensors)}] {t.name} {done / 2**30:.1f} GiB "
                      f"{done / 2**20 / max(el, 1e-9):.0f} MiB/s", flush=True)
    os.replace(tmp, args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-dir", required=True, help="Kolibri-1 snapshot directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--experts", choices=("q8_0", "q4_k", "f8_b128", "f32"), default="q8_0")
    ap.add_argument("--dense", choices=("q8_0", "f8_b128", "f32"), default="q8_0",
                    help="attention and shared-expert projections (f32 is for tests)")
    ap.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument("--quants-lib", default=str(Path(__file__).with_name("libds4quants.so")))
    ap.add_argument("--revision", default="unknown")
    args = ap.parse_args()
    cfg = json.load(open(Path(args.hf_dir) / "config.json"))
    if cfg.get("model_type") != ARCH:
        fail(f"model_type is {cfg.get('model_type')!r}, expected {ARCH}")
    if cfg.get("quantization_config", {}).get("weight_block_size") != [128, 128]:
        fail("expected FP8 weights with 128x128 block scales")
    write(args, Source(args.hf_dir), cfg, Quantizer(args.quants_lib))


if __name__ == "__main__":
    main()
