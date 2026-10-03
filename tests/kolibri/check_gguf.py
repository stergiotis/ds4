#!/usr/bin/env python3
"""Read a Kolibri GGUF back and compare every F32/BF16/Q8_0 tensor with the
dequantized FP8 source. Checks names, shapes, expert stacking and the
requantization error.

    uv run check_gguf.py --hf-dir SNAPSHOT --gguf FILE [--sample N]
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-tools"))
from kolibri_convert import Source  # noqa: E402

TYPE_SIZES = {0: (1, 4), 8: (32, 34), 12: (256, 144), 30: (1, 2), 200: (128, 132)}


def read_gguf(path):
    f = open(path, "rb")

    def rd(fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))

    def rstr():
        return f.read(rd("Q")[0]).decode()

    def rval(t):
        scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if t in scalar:
            return rd(scalar[t])[0]
        if t == 8:
            return rstr()
        if t == 9:
            et, n = rd("IQ")
            return [rval(et) for _ in range(n)]
        raise ValueError(t)

    magic, ver, nt, nkv = rd("4sIQQ")
    assert magic == b"GGUF"
    kv = {}
    for _ in range(nkv):
        k = rstr()
        kv[k] = rval(rd("I")[0])
    tensors = {}
    for _ in range(nt):
        name = rstr()
        nd = rd("I")[0]
        ne = list(rd(f"{nd}Q"))
        typ, off = rd("IQ")
        tensors[name] = (ne, typ, off)
    align = kv.get("general.alignment", 32)
    start = (f.tell() + align - 1) // align * align
    mm = np.memmap(path, np.uint8, "r")
    return kv, tensors, mm, start


def dequant(mm, start, ne, typ, rows=None):
    blk, size = TYPE_SIZES[typ]
    n0 = ne[0]
    nrows = int(np.prod(ne[1:]))
    rb = n0 // blk * size
    return rb, nrows, lambda r0, r1, off: _deq(mm[start + off + r0 * rb: start + off + r1 * rb], typ, n0, r1 - r0)


def _deq(raw, typ, n0, nr):
    if typ == 0:
        return raw.view("<f4").reshape(nr, n0)
    if typ == 30:
        return (raw.view("<u2").astype(np.uint32) << 16).view(np.float32).reshape(nr, n0)
    if typ == 8:
        b = raw.reshape(nr, n0 // 32, 34)
        d = b[:, :, :2].copy().view("<f2").astype(np.float32)
        q = b[:, :, 2:].view(np.int8).astype(np.float32)
        return (q * d).reshape(nr, n0)
    if typ == 200:
        import ml_dtypes
        b = raw.reshape(nr, n0 // 128, 132)
        d = b[:, :, :4].copy().view("<f4")
        q = b[:, :, 4:].copy().view(ml_dtypes.float8_e4m3fn).astype(np.float32)
        return (q * d).reshape(nr, n0)
    raise ValueError(typ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--sample", type=int, default=0, help="check only N experts per stacked tensor")
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(args.hf_dir, "config.json")))
    src = Source(args.hf_dir)
    kv, tensors, mm, start = read_gguf(args.gguf)
    assert kv["general.architecture"] == "kolibri1"
    assert len(kv["tokenizer.ggml.tokens"]) == cfg["vocab_size"]
    E = cfg["num_experts"]
    m = {"attn_norm": "input_layernorm.weight", "attn_q": "self_attn.q_proj.weight",
         "attn_k": "self_attn.k_proj.weight", "attn_v": "self_attn.v_proj.weight",
         "attn_q_norm": "self_attn.q_norm.weight", "attn_k_norm": "self_attn.k_norm.weight",
         "attn_output": "self_attn.o_proj.weight", "post_attention_norm": "post_attn_norm.weight",
         "ffn_norm": "post_attention_layernorm.weight", "ffn_gate_inp": "mlp.gate.weight",
         "exp_probs_b": "moe.router.expert_bias", "post_ffw_norm": "post_ffn_norm.weight",
         "ffn_gate_shexp": "mlp.shared_experts.gate_proj.weight",
         "ffn_up_shexp": "mlp.shared_experts.up_proj.weight",
         "ffn_down_shexp": "mlp.shared_experts.down_proj.weight"}
    worst, rms = {}, {}
    skipped = 0
    for name, (ne, typ, off) in tensors.items():
        if typ not in TYPE_SIZES or typ == 12:
            skipped += 1
            continue
        parts = name.split(".")
        if parts[0] == "blk":
            l, kind = int(parts[1]), parts[2]
            if kind.endswith("_exps"):
                proj = kind[4:-5] + "_proj"
                rb, nrows, get = dequant(mm, start, ne, typ)
                per = ne[1]
                experts = range(E) if not args.sample else np.linspace(0, E - 1, args.sample).astype(int)
                for e in experts:
                    ref = src.f32(f"model.layers.{l}.mlp.experts.{e}.{proj}.weight")
                    got = get(e * per, (e + 1) * per, off)
                    err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30)
                    worst[kind] = max(worst.get(kind, 0), err)
                    rms[kind] = max(rms.get(kind, 0), float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30)))
                continue
            ref = src.f32(f"model.layers.{l}.{m[kind]}")
        else:
            ref = src.f32({"token_embd.weight": "model.embed_tokens.weight",
                           "output.weight": "lm_head.weight",
                           "output_norm.weight": "model.norm.weight"}[name])
        ref = ref.reshape(-1, ref.shape[-1]) if ref.ndim > 1 else ref.reshape(1, -1)
        assert list(ref.shape[::-1]) == (ne if len(ne) > 1 else ne + [1]), (name, ne, ref.shape)
        rb, nrows, get = dequant(mm, start, ne, typ)
        got = get(0, nrows, off)
        err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30)
        key = parts[2] if parts[0] == "blk" else name
        worst[key] = max(worst.get(key, 0), err)
        rms[key] = max(rms.get(key, 0), float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30)))
    for k, v in sorted(worst.items()):
        print(f"  {k:28s} max |err| / max |w| = {v:.2e}   |err| / |w| (rms) = {rms[k]:.2e}")
    print(f"checked {len(tensors) - skipped} tensors, skipped {skipped} (Q4_K)")
    bad = {k: v for k, v in worst.items() if v > 5e-3}
    if bad:
        raise SystemExit(f"FAIL: {bad}")
    print("OK")


if __name__ == "__main__":
    main()
