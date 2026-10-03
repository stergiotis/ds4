#!/usr/bin/env python3
"""Minimal GGUF (v2/v3) tensor-table reader: name, type, dims, bytes."""
import struct, sys
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84),
         11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
         14: ("Q6_K", 256, 210), 30: ("BF16", 1, 2)}
def rd(f, fmt): return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))
def rstr(f): n, = rd(f, "Q"); return f.read(n).decode("utf-8", "replace")
def skip_val(f, t):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in sizes: f.read(sizes[t])
    elif t == 8: rstr(f)
    elif t == 9:
        at, n = rd(f, "IQ")
        for _ in range(n): skip_val(f, at)
    else: raise ValueError(f"kv type {t}")
def tensors(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        ver, nt, nkv = rd(f, "IQQ")
        for _ in range(nkv):
            rstr(f); t, = rd(f, "I"); skip_val(f, t)
        out = []
        for _ in range(nt):
            name = rstr(f); nd, = rd(f, "I"); dims = rd(f, "Q" * nd); typ, off = rd(f, "IQ")
            n = 1
            for d in dims: n *= d
            tn, blk, bb = TYPES.get(typ, (f"T{typ}", 0, 0))
            out.append((name, tn, dims, off, n // blk * bb if blk else -1))
        return out
if __name__ == "__main__":
    for name, tn, dims, off, b in tensors(sys.argv[1]):
        print(f"{name}\t{tn}\t{'x'.join(map(str, dims))}\t{off}\t{b}")
