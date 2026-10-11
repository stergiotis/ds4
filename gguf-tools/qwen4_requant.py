#!/usr/bin/env python3
"""Requantize selected Qwen3.8 tensors and leave the n-gram table out.

Every tensor matching a --set GLOB=TYPE rule is dequantized and quantized to
TYPE (q8_0, q4_k, bf16, f16); the rest is copied unchanged. The output names
the input as its n-gram source (qwen4exp.ple.ngram_file), so it does not copy
the 95 GiB BF16 table.

--imatrix FILE (llama.cpp legacy .dat, e.g. from DS4_QWEN4_DENSE_IMATRIX)
weights each Q4_K block by the mean squared input of its columns.

--simulate stores each rounded tensor back in a type the runtime already
runs: q4_k values as Q8_0 (or the original F16/F32), bf16 values as F32. Q8_0
re-encodes a Q4_K block with ~1/8 of the Q4_K step, so the quality of a
format can be measured before the runtime has kernels for it.

Two types exist only for --simulate (no GGUF encoding yet):
  r4   rotated 4-bit: each 128-column block of a row is multiplied by fixed
       random signs (seeded by the tensor name) and a 128-point Hadamard
       matrix, rounded to codes -8..7 with one f16 scale per block (scale
       chosen by mean squared error), and rotated back. 4.125 bits a weight.
  q6a  6-bit affine: groups of 64 with an f16 scale and minimum. 6.5 bits.

--hgn FILE (with --simulate) takes the values of every trunk ht matrix from a
halogen v2 checkpoint (attention, indexer, GDN projections, shared expert,
lm_head; decoded with strix/hgn, value heads reordered to GGUF order) and
stores them as Q8_0, to measure that checkpoint's dense quality in ds4.

--hgn-ht4 FILE copies the same matrices (shared expert excepted) into the
output unchanged as ds4 type HT4 (210), with their F16 rotation vectors as
NAME.suh / NAME.svh and, for the GDN projections, I32 NAME.rperm / NAME.cperm
mapping halogen's grouped value heads to GGUF's tiled order.

  qwen4_requant.py --model Q4.gguf --output out.gguf \\
      --set 'blk.*.hc_*.weight=q8_0' --set 'blk.*.ffn_gate_inp.weight=bf16' [--simulate]
"""

import argparse
import sys
import ctypes
import fnmatch
import math
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import struct

import numpy as np

from qwen4_pack_to_qwen4exp import Reader, kv_bytes, w_str

NGRAM = 'per_layer_token_embd.weight'
F32, F16, Q8_0, Q4_K, BF16, I32, HT4 = 0, 1, 8, 12, 30, 26, 210
R4, Q6A = 1000, 1001   # simulation-only rounding types
SIM_ONLY = (R4, Q6A)
NAMES = {'f32': F32, 'f16': F16, 'q8_0': Q8_0, 'q4_k': Q4_K, 'bf16': BF16, 'r4': R4, 'q6a': Q6A}
# type -> (block elements, block bytes)
BLOCKS = {0: (1, 4), 1: (1, 2), 2: (32, 18), 8: (32, 34), 12: (256, 144),
          26: (1, 4), 30: (1, 2), 39: (32, 17), R4: (128, 66), Q6A: (64, 52), HT4: (256, 128)}


def nbytes(kind, dims):
    if kind == HT4:   # 16 x 16 tiles: any K and N that are multiples of 16 (ds4 needs 128)
        return math.prod(dims) // 2
    block, size = BLOCKS[kind]
    if dims[0] % block:
        raise ValueError(f'row length {dims[0]} is not a multiple of {block}')
    return math.prod(dims) // block * size


def bf16_to_f32(raw):
    return (raw.view('<u2').astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(x):
    bits = x.astype(np.float32).view(np.uint32)
    bits = bits + 0x7FFF + ((bits >> 16) & 1)  # round to nearest even
    return (bits >> 16).astype(np.uint16)


def dequant(kind, raw, n):
    if kind == F32:
        return raw.view('<f4').astype(np.float32)
    if kind == F16:
        return raw.view('<f2').astype(np.float32)
    if kind == BF16:
        return bf16_to_f32(raw)
    if kind == Q8_0:
        b = raw.reshape(-1, 34)
        d = b[:, :2].copy().view('<f2').astype(np.float32)
        return (b[:, 2:].view(np.int8).astype(np.float32) * d).reshape(-1)
    if kind == Q4_K:
        b = raw.reshape(-1, 144)
        d = b[:, 0:2].copy().view('<f2').astype(np.float32)
        dmin = b[:, 2:4].copy().view('<f2').astype(np.float32)
        q = b[:, 4:16].astype(np.uint8)
        sc = np.empty((len(b), 8), np.float32)
        mn = np.empty((len(b), 8), np.float32)
        for j in range(4):
            sc[:, j] = q[:, j] & 63
            mn[:, j] = q[:, j + 4] & 63
        for j in range(4, 8):
            sc[:, j] = (q[:, j + 4] & 0xF) | ((q[:, j - 4] >> 6) << 4)
            mn[:, j] = (q[:, j + 4] >> 4) | ((q[:, j] >> 6) << 4)
        qs = b[:, 16:].reshape(-1, 4, 32)
        nib = np.stack([qs & 0xF, qs >> 4], axis=2).reshape(-1, 8, 32).astype(np.float32)
        y = nib * (d * sc)[:, :, None] - (dmin * mn)[:, :, None]
        return y.reshape(-1)
    raise ValueError(f'cannot dequantize type {kind}')


def _hadamard128():
    from scipy.linalg import hadamard
    return (hadamard(128) / np.sqrt(128)).astype(np.float32)


def sim_r4(x, ncols, name):
    """Rotated 4-bit round trip of float32 rows of ncols (ncols % 128 == 0)."""
    w = x.reshape(-1, ncols)
    signs = np.random.default_rng(zlib.crc32(name.encode())).choice(np.array([-1.0, 1.0], np.float32), ncols)
    h = _hadamard128()
    out = np.empty_like(w)
    for r0 in range(0, w.shape[0], 4096):
        b = ((w[r0:r0 + 4096] * signs).reshape(-1, ncols // 128, 128) @ h).astype(np.float32)
        amax = np.abs(b).max(axis=2, keepdims=True)
        best = np.zeros_like(b)
        err = np.full(amax.shape, np.inf, np.float32)
        for c in np.linspace(6.0, 8.0, 9):
            d = (amax / c).astype(np.float16).astype(np.float32)
            d[d == 0] = 1.0
            y = np.clip(np.rint(b / d), -8, 7) * d
            e = ((y - b) ** 2).sum(axis=2, keepdims=True)
            better = e < err
            best = np.where(better, y, best)
            err = np.where(better, e, err)
        out[r0:r0 + 4096] = (best @ h.T).reshape(-1, ncols) * signs
    return out.reshape(-1)


def sim_q6a(x, ncols):
    """6-bit affine round trip, groups of 64 (ncols % 64 == 0)."""
    g = x.reshape(-1, 64)
    lo = g.min(axis=1, keepdims=True).astype(np.float16).astype(np.float32)
    d = ((g.max(axis=1, keepdims=True) - lo) / 63).astype(np.float16).astype(np.float32)
    d[d == 0] = 1.0
    return (lo + np.clip(np.rint((g - lo) / d), 0, 63) * d).reshape(-1)


class Quantizer:
    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path))
        self.lib.ds4q_quantize_init.argtypes = [ctypes.c_int]
        self.lib.ds4q_quantize_chunk.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_float)]
        self.lib.ds4q_quantize_chunk.restype = ctypes.c_size_t
        for kind in (Q8_0, Q4_K):
            self.lib.ds4q_quantize_init(kind)
        self.pool = ThreadPoolExecutor(os.cpu_count() or 8)

    def quantize(self, kind, x, ncols, weights=None):
        """Quantize float32 rows of ncols to kind (weights: per-column importance
        for Q4_K); returns the packed bytes."""
        if kind == F32:
            return x.astype('<f4').view(np.uint8)
        if kind == F16:
            return x.astype('<f2').view(np.uint8)
        if kind == BF16:
            return f32_to_bf16_bits(x).view(np.uint8)
        x = np.ascontiguousarray(x, dtype=np.float32)
        nrows = x.size // ncols
        out = np.empty(nbytes(kind, [ncols, nrows]), np.uint8)
        src = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        imp = None
        if weights is not None and kind == Q4_K:
            weights = np.ascontiguousarray(weights, dtype=np.float32)
            imp = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        step = max(1, nrows // 64)

        def part(r0):
            r1 = min(nrows, r0 + step)
            self.lib.ds4q_quantize_chunk(kind, src, out.ctypes.data, r0 * ncols, r1 - r0, ncols, imp)

        list(self.pool.map(part, range(0, nrows, step)))
        return out


def read_imatrix(path):
    """name -> per-column mean squared input from a legacy .dat imatrix."""
    out = {}
    with open(path, 'rb') as f:
        n, = struct.unpack('<i', f.read(4))
        for _ in range(n):
            size, = struct.unpack('<i', f.read(4))
            name = f.read(size).decode()
            _, nval = struct.unpack('<ii', f.read(8))
            out[name] = np.frombuffer(f.read(4 * nval), '<f4').astype(np.float32)
    return out


def plan_tensor(name, kind, dims, rules, simulate):
    """(stored type, rounding type or None) for one tensor; vectors stay as they are."""
    if len(dims) < 2:
        return kind, None
    target = None
    for pattern, t in rules:
        if fnmatch.fnmatchcase(name, pattern):
            target = t
    if target is None or target == kind:
        return kind, None
    if target in SIM_ONLY:
        if not simulate:
            raise ValueError(f'{name}: r4 and q6a exist only with --simulate')
        return (kind if kind in (F16, F32) else Q8_0), target
    if not simulate:
        return target, target
    if target == Q4_K:
        return (kind if kind in (F16, F32) else Q8_0), Q4_K
    if target == BF16:
        return F32, BF16
    return target, target  # q8_0 and f16 already run


def requant(args):
    model = Reader(str(args.model))
    rules = []
    for rule in args.set:
        pattern, _, t = rule.rpartition('=')
        if not pattern or t not in NAMES:
            raise ValueError(f'bad --set {rule!r}; TYPE is one of {", ".join(NAMES)}')
        rules.append((pattern, NAMES[t]))
    if model.kv.get('general.architecture') != 'qwen4exp' or NGRAM not in model.tensors:
        raise ValueError('expected a Qwen GGUF that carries its n-gram table')
    out = Path(args.output)
    pending = Path(str(out) + '.incomplete')
    if out.exists() or pending.exists():
        raise ValueError(f'{out} or its .incomplete file already exists')
    q = Quantizer(args.lib)
    imatrix = read_imatrix(args.imatrix) if args.imatrix else {}
    hgn_over, hgn_get = {}, None
    if args.hgn:
        if not args.simulate:
            raise ValueError('--hgn needs --simulate')
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'strix' / 'hgn'))
        import hgn, ht, hmap
        ht_table, _ = hgn.table(args.hgn)
        f16 = lambda n: np.frombuffer(bytes(hgn.payload(args.hgn, ht_table[n])), '<f2').astype(np.float32)
        hgn_over = hmap.gguf_overrides(ht_table)
        def hgn_get(name):
            base, fix, rows = hgn_over[name]
            W = hmap.ht_decode(args.hgn, ht_table, base, f16, ht)
            if rows: W = W[rows[0]:rows[1]]
            return (fix(W) if fix else W).reshape(-1)
        print(f'{len(hgn_over)} tensors from {args.hgn}', flush=True)

    alignment = model.kv.get('general.alignment', 32)
    align = lambda n: (n + alignment - 1) // alignment * alignment
    native = {}
    if args.hgn_ht4:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'strix' / 'hgn'))
        import hgn, hmap
        nt, _ = hgn.table(args.hgn_ht4)
        # the shared expert stays: as HT4 it measured slower (separate launches
        # instead of the routed kernels' slot) and less accurate than Q8_0
        native = {k: v for k, v in hmap.gguf_overrides(nt).items() if '_shexp' not in k}
        def nat_bytes(base, rows=None):
            e = nt[base + '.weight']; N, K = e['shape']
            b = bytes(hgn.payload(args.hgn_ht4, e))
            return b if rows is None else b[rows[0] * K // 2:rows[1] * K // 2]
        def vec(base, which, rows=None):
            v = bytes(hgn.payload(args.hgn_ht4, nt[base + '.' + which]))
            return v if rows is None or which == 'suh' else v[rows[0] * 2:rows[1] * 2]
        g2t = hmap.G2T
        print(f'{len(native)} tensors as HT4 from {args.hgn_ht4}', flush=True)
    plan, offset = [], 0
    for name, (kind, dims, old) in model.tensors.items():
        if name == NGRAM:
            continue
        if name in native:
            base, fix, rows = native[name]
            K, N = dims
            stem = name[:-len('.weight')]
            items = [(name, HT4, dims, lambda b=base, r=rows: nat_bytes(b, r)),
                     (stem + '.suh', F16, [K], lambda b=base, r=rows: vec(b, 'suh', r)),
                     (stem + '.svh', F16, [N], lambda b=base, r=rows: vec(b, 'svh', r))]
            gg = name.split('.')[-2]
            if gg == 'attn_qkv':   # q, k bands as they are; the 48 v bands grouped -> tiled
                rp = np.arange(N // 128, dtype='<i4'); vb = rp.size - 48
                rp[vb:] = vb + g2t
                items.append((stem + '.rperm', I32, [rp.size], lambda a=rp: a.tobytes()))
            elif gg == 'attn_gate':
                items.append((stem + '.rperm', I32, [48], lambda: g2t.astype('<i4').tobytes()))
            elif gg == 'ssm_out':
                items.append((stem + '.cperm', I32, [48], lambda: g2t.astype('<i4').tobytes()))
            for iname, itype, idims, getter in items:
                size = nbytes(itype, idims)
                plan.append((iname, kind, idims, old, itype, None, offset, size, getter))
                offset = align(offset + size)
            continue
        stored, rounding = (Q8_0, Q8_0) if name in hgn_over else plan_tensor(name, kind, dims, rules, args.simulate)
        if rounding is not None:
            nbytes(rounding, dims)  # the rounding type must tile the rows
        size = nbytes(stored, dims)
        plan.append((name, kind, dims, old, stored, rounding, offset, size, None))
        offset = align(offset + size)

    metadata = {k: (model.kv_types[k], v) for k, v in model.kv.items()}
    src = Path(args.model).resolve()
    ref = os.path.relpath(src, out.resolve().parent) if src.parent == out.resolve().parent else str(src)
    metadata['qwen4exp.ple.ngram_file'] = (8, ref)
    metadata['ds4.qwen4.requant'] = (8, ('simulate ' if args.simulate else '') +
                                     (f'hgn-ht4 {Path(args.hgn_ht4).name} ' if args.hgn_ht4 else '') +
                                     (f'imatrix {Path(args.imatrix).name} ' if args.imatrix else '') + ' '.join(args.set))
    head = bytearray(b'GGUF' + struct.pack('<IQQ', 3, len(plan), len(metadata)))
    head += b''.join(kv_bytes(k, t, v) for k, (t, v) in metadata.items())
    for name, _, dims, _, stored, _, pos, _, _ in plan:
        head += w_str(name) + struct.pack('<I', len(dims))
        head += struct.pack('<' + 'Q' * len(dims), *dims) + struct.pack('<IQ', stored, pos)
    data_start = align(len(head))

    changed = before = after = 0
    with pending.open('xb') as dst:
        dst.write(head)
        for name, kind, dims, old, stored, rounding, pos, size, getter in plan:
            if getter is not None:
                b = getter()
                if len(b) != size:
                    raise ValueError(f'{name}: {len(b)} bytes, planned {size}')
                dst.seek(data_start + pos)
                dst.write(b)
                continue
            src_bytes = nbytes(kind, dims)
            model.f.seek(model.data_start + old)
            raw = np.frombuffer(model.f.read(src_bytes), np.uint8)
            if len(raw) != src_bytes:
                raise ValueError(f'truncated tensor {name}')
            dst.seek(data_start + pos)
            if rounding is None:
                dst.write(raw.tobytes())
                continue
            x = hgn_get(name) if name in hgn_over else dequant(kind, raw, math.prod(dims))
            weights = imatrix.get(name)
            if weights is not None and (len(weights) != dims[0] or not np.all(np.isfinite(weights))):
                raise ValueError(f'{name}: imatrix has {len(weights)} columns, tensor {dims[0]}')
            if rounding == Q4_K and imatrix and weights is None:
                print(f'warning: no imatrix entry for {name}', flush=True)
            if rounding in SIM_ONLY:
                y = q.quantize(stored, sim_r4(x, dims[0], name) if rounding == R4 else sim_q6a(x, dims[0]), dims[0])
            else:
                y = q.quantize(rounding, x, dims[0], weights)
                if stored != rounding:
                    y = q.quantize(stored, dequant(rounding, y, x.size), dims[0])
            if len(y) != size:
                raise ValueError(f'{name}: wrote {len(y)} bytes, planned {size}')
            dst.write(y.tobytes())
            changed += 1
            before += src_bytes
            after += nbytes(rounding, dims)
        dst.truncate(data_start + offset)
        dst.flush()
        os.fsync(dst.fileno())
    pending.rename(out)
    print(f'{out}: {changed} tensors rounded, {before / 2**30:.2f} -> {after / 2**30:.2f} GiB '
          f'in the rounding types; file {(data_start + offset) / 2**30:.2f} GiB; n-grams from {ref}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--set', action='append', default=[], metavar='GLOB=TYPE')
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--imatrix')
    parser.add_argument('--hgn')
    parser.add_argument('--hgn-ht4')
    parser.add_argument('--lib', default=Path(__file__).resolve().with_name('libds4quants.so'))
    args = parser.parse_args()
    try:
        requant(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
