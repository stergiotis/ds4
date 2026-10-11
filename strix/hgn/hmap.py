"""HGN (v2) -> qwen4exp GGUF name mapping for the ht matrices, with the GDN value-head reorder."""
import numpy as np
NVH, NKH, HD = 48, 16, 128
# grouped HGN value head g -> tiled GGUF head
G2T = np.array([(g % 3) * NKH + g // 3 for g in range(NVH)])
def vperm_rows(W, start, count=NVH):
    """reorder `count` head blocks of HD rows starting at row `start` from grouped to tiled order"""
    W = W.copy(); blk = W[start:start + count * HD].reshape(count, HD, -1)
    out = np.empty_like(blk); out[G2T] = blk
    W[start:start + count * HD] = out.reshape(count * HD, -1); return W
def vperm_cols(W):
    blk = W.reshape(W.shape[0], NVH, HD); out = np.empty_like(blk); out[:, G2T] = blk
    return out.reshape(W.shape[0], NVH * HD)
# (hgn base, gguf name, fix) per layer kind; fix maps the decoded [N, K] matrix to GGUF order
GDN = [('linear_attn.in_proj_qkv', 'attn_qkv', lambda W: vperm_rows(W, 2 * NKH * HD)),
       ('linear_attn.in_proj_z', 'attn_gate', lambda W: vperm_rows(W, 0)),
       ('linear_attn.out_proj', 'ssm_out', vperm_cols)]
ATT = [('self_attn.q_proj', 'attn_q', None), ('self_attn.k_proj', 'attn_k', None),
       ('self_attn.v_proj', 'attn_v', None), ('self_attn.o_proj', 'attn_output', None)]
SHX = [('mlp.shared_expert.gate_proj', 'ffn_gate_shexp', None), ('mlp.shared_expert.up_proj', 'ffn_up_shexp', None),
       ('mlp.shared_expert.down_proj', 'ffn_down_shexp', None)]
def layer_maps(L, is_attn):
    return [(f'layers.{L}.{h}', f'blk.{L}.{g}', f) for h, g, f in (ATT if is_attn else GDN) + SHX]


def ht_decode(P, t, base, f16, ht, chunk_bands=64):
    """dense values [N, K] of a 2-D ht matrix, decoded band chunk by band chunk"""
    import hgn
    e = t[base + '.weight']; N, K = e['shape'][-2:]
    su, sv = f16(base + '.suh'), f16(base + '.svh')
    pay = hgn.payload(P, e); band_bytes = 128 * K // 2
    out = np.empty((N, K), np.float32)
    for b0 in range(0, N // 128, chunk_bands):
        nb = min(chunk_bands, N // 128 - b0)
        M = ht.decode_M(bytes(pay[b0 * band_bytes:(b0 + nb) * band_bytes]), nb * 128, K)
        out[b0 * 128:(b0 + nb) * 128] = ht.rotate_back(M, su, sv[b0 * 128:(b0 + nb) * 128])
    return out

def gguf_overrides(t, n_layers=48, attn_every=4):
    """{gguf tensor name: (hgn base, fix, row slice)} for every ht matrix of the trunk"""
    out = {}
    for L in range(n_layers):
        att = L % attn_every == attn_every - 1
        for h, g, f in layer_maps(L, att):
            out[g + '.weight'] = (h, f, None)
        if att:
            out[f'blk.{L}.indexer.q_proj.weight'] = (f'layers.{L}.self_attn.indexer.index_qk_proj', None, (0, 512))
            out[f'blk.{L}.indexer.k_proj.weight'] = (f'layers.{L}.self_attn.indexer.index_qk_proj', None, (512, 640))
    out['output.weight'] = ('lm_head', None, None)
    return {k: v for k, v in out.items() if v[0] + '.weight' in t}


def q6g64_decode(payload, N, K):
    """values [N, K] of a q6g64 matrix (hgn-spec STORAGE.md)"""
    S = (13 * K // 16 + 15) // 16 * 16
    rows = np.frombuffer(payload, np.uint8, count=N * S).reshape(N, S)
    lo = rows[:, :K // 2]
    low = np.empty((N, K), np.uint8); low[:, 0::2] = lo & 15; low[:, 1::2] = lo >> 4
    hb = rows[:, K // 2:3 * K // 4]
    high = np.stack([(hb >> (2 * j)) & 3 for j in range(4)], axis=2).reshape(N, K)
    rec = rows[:, 3 * K // 4:13 * K // 16].copy().view('<f2').reshape(N, K // 64, 2).astype(np.float32)
    c = (low | (high << 4)).astype(np.float32).reshape(N, K // 64, 64)
    return (c * rec[:, :, :1] + rec[:, :, 1:]).reshape(N, K)

def hc_overrides(t, n_layers=48):
    """{gguf name: hgn name} for the q6g64 hyper-connection matrices of the trunk and the final mixer"""
    out = {}
    for L in range(n_layers):
        for side, g in (('attn', 'attn'), ('mlp', 'ffn')):
            out[f'blk.{L}.hc_{g}_down.weight'] = f'layers.{L}.{side}_hyper_connection.input_mix_weight_down.weight'
            out[f'blk.{L}.hc_{g}_up.weight'] = f'layers.{L}.{side}_hyper_connection.input_mix_weight_up.weight'
    out['output_hc_down.weight'] = 'hyper_connection_mixer.input_mix_weight_down.weight'
    out['output_hc_up.weight'] = 'hyper_connection_mixer.input_mix_weight_up.weight'
    return {k: v for k, v in out.items() if v in t and t[v]['store'] == 24}
