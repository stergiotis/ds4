"""ht decode (hgn-spec STORAGE.md: tiles, trellis state, codebook 1, 128-point rotation)."""
import numpy as np
from scipy.linalg import hadamard
def codebook():
    s = np.arange(65536, dtype=np.uint64)
    x = (s * 0x83DCD12D) & 0xFFFFFFFF
    a = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
    return ((1774.0 * a.astype(np.float64) - 905216.0) / 2**18).astype(np.float16).astype(np.float32)
CB = codebook()
def decode_M(payload, N, K, b=4):
    """stored numbers M[N][K] of a 2-D ht tensor"""
    nt = N * K // 256
    words = np.frombuffer(payload, '<u4', count=nt * 8 * b).reshape(nt, 8 * b)
    bits = np.unpackbits(words.astype('>u4').view(np.uint8), axis=1)          # MSB first per word
    L = 256 * b
    i = np.arange(256)
    idx = (b * (i[:, None] + 1) - 16 + np.arange(16)[None, :]) % L              # 16 state bits per value
    st = np.zeros((nt, 256), np.int64)
    for k in range(16):
        st = (st << 1) | bits[:, idx[:, k]]
    vals = CB[st]                                                               # (tiles, 256)
    trow = (i // 8) % 16; tcol = 8 * (i // 128) + (i % 8)
    tile = np.zeros((nt, 16, 16), np.float32); tile[:, trow, tcol] = vals
    # tile t = ((r/128)*(K/16) + c/16)*8 + (r%128)/16
    t = tile.reshape(N // 128, K // 16, 8, 16, 16)          # band, coltile, rowtile, r, c
    return t.transpose(0, 2, 3, 1, 4).reshape(N, K)
def rotate_back(M, su, sv):
    H = (hadamard(128) / np.sqrt(128)).astype(np.float32)
    N, K = M.shape
    B = M.reshape(N // 128, 128, K // 128, 128)
    W = np.einsum('ij,bjck,kl->bicl', H, B, H, optimize=True).reshape(N, K)
    return sv[:, None] * W * su[None, :]
