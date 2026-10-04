/* Kolibri 1 on ROCm: the operators the Qwen3.8 kernels do not cover.
 *
 * - qk_prep: per-head RMSNorm of q and k with their weights, neox RoPE on
 *   sliding layers only (full layers are NoPE), and the f16 K/V store.
 *   Each layer's cache has cache_rows rows and position p lives in row
 *   p % cache_rows: sliding layers use a ring of window - 1 + chunk rows,
 *   full layers one row per context position.
 * - attention: GQA, online softmax over the keys a query may see (the last
 *   `window` positions, or all of them), optional key splits merged
 *   afterwards.  No output gate.  Prefill batches use WMMA tiles, decode
 *   batches a block per KV head and key split that scores 32-key tiles
 *   lane-per-key.
 * - router: top-k of logits + selection bias, weight sigmoid(logit), no
 *   renormalisation; an extra weight slot of 1.0 lets the shared expert ride
 *   as the last slot of the Qwen expert kernels.
 * - norm_add: the sandwich-norm residual update, x += rms(h) * w_post, then
 *   xn = rms(x) * w_next for the next sublayer.  h is either a tensor or the
 *   weighted sum of the expert partials (plus an optional dense shared
 *   expert row), so the MoE reduce folds in.
 *
 * Included by ds4_rocm.cu after the Qwen3.8 file, whose helpers it uses. */

namespace kolibri_rocm {

using qwen4_rocm::sum;
using qwen4_rocm::block_sum;
using qwen4_rocm::sigmoid;
using qwen4_rocm::tensor;
using qwen4_rocm::weight;

static int launched(void) { return cuda_ok(cudaGetLastError(), "Kolibri kernel"); }

/* One wave per (token, head): heads [0, H) are queries, [H, H + Hkv) keys,
 * whose value row is stored alongside.  Lane l owns dims l + 32 j, so the
 * RoPE pair (i, i + D/2) sits in the same lane. */
template<unsigned D>
__global__ void qk_prep(float *q, const float *k, const float *v, __half *kc, __half *vc,
        const float *qn, const float *kn, unsigned H, unsigned Hkv, unsigned pos0,
        unsigned cache_rows, bool rope, float freq_base, float eps) {
    constexpr unsigned J = D / 32;
    const unsigned head = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const unsigned t = blockIdx.y, lane = threadIdx.x & 31;
    if (head >= H + Hkv) return;
    const bool is_q = head < H;
    const unsigned h = is_q ? head : head - H;
    const float *src = is_q ? q + ((uint64_t)t * H + h) * D : k + ((uint64_t)t * Hkv + h) * D;
    const float *g = is_q ? qn : kn;
    float x[J];
    float ss = 0;
    for (unsigned j = 0; j < J; j++) { x[j] = src[lane + 32 * j]; ss += x[j] * x[j]; }
    const float r = rsqrtf(sum(ss) / D + eps);
    for (unsigned j = 0; j < J; j++) x[j] *= r * g[lane + 32 * j];
    const unsigned pos = pos0 + t;
    if (rope) {
        /* Float angles, as vLLM builds its cos/sin cache. */
        for (unsigned j = 0; j < J / 2; j++) {
            const unsigned i = lane + 32 * j;
            const float inv = powf(freq_base, -2.0f * (float)i / (float)D);
            const float a = (float)pos * inv, c = cosf(a), s = sinf(a);
            const float x0 = x[j], x1 = x[j + J / 2];
            x[j] = x0 * c - x1 * s;
            x[j + J / 2] = x1 * c + x0 * s;
        }
    }
    if (is_q) {
        float *dst = q + ((uint64_t)t * H + h) * D;
        for (unsigned j = 0; j < J; j++) dst[lane + 32 * j] = x[j];
    } else {
        const uint64_t row = ((uint64_t)(pos % cache_rows) * Hkv + h) * D;
        const float *vs = v + ((uint64_t)t * Hkv + h) * D;
        for (unsigned j = 0; j < J; j++) {
            kc[row + lane + 32 * j] = __float2half(x[j]);
            vc[row + lane + 32 * j] = __float2half(vs[lane + 32 * j]);
        }
    }
}

/* grid (ceil(H/4), T, splits), 4 waves per block, one head per wave. */
template<unsigned D>
__global__ void attention(float *out, float *partial, const float *q, const __half *kc,
        const __half *vc, unsigned H, unsigned Hkv, unsigned pos0, unsigned cache_rows,
        unsigned window, unsigned splits, unsigned per, float scale) {
    constexpr unsigned J = D / 32;
    const unsigned h = blockIdx.x * 4 + threadIdx.x / 32, t = blockIdx.y, split = blockIdx.z;
    if (h >= H) return;
    const unsigned lane = threadIdx.x & 31, kh = h / (H / Hkv), pos = pos0 + t;
    const unsigned lo = window && pos + 1 > window ? pos + 1 - window : 0;
    const unsigned begin = lo + split * per;
    const unsigned end = min(pos + 1, begin + per);
    float qv[J], acc[J] = {}, m = -3e38f, denom = 0;
    for (unsigned j = 0; j < J; j++) qv[j] = q[((uint64_t)t * H + h) * D + lane + 32 * j] * scale;
    for (unsigned p = begin; p < end; p++) {
        const uint64_t row = ((uint64_t)(p % cache_rows) * Hkv + kh) * D;
        float s = 0;
        for (unsigned j = 0; j < J; j++) s += qv[j] * __half2float(kc[row + lane + 32 * j]);
        s = sum(s);
        const float nm = fmaxf(m, s), corr = expf(m - nm), w = expf(s - nm);
        denom = denom * corr + w;
        for (unsigned j = 0; j < J; j++) acc[j] = acc[j] * corr + w * __half2float(vc[row + lane + 32 * j]);
        m = nm;
    }
    if (splits == 1) {
        for (unsigned j = 0; j < J; j++)
            out[((uint64_t)t * H + h) * D + lane + 32 * j] = denom > 0 ? acc[j] / denom : 0;
    } else {
        float *dst = partial + (((uint64_t)t * H + h) * splits + split) * (D + 2);
        if (!lane) { dst[0] = m; dst[1] = denom; }
        for (unsigned j = 0; j < J; j++) dst[2 + lane + 32 * j] = acc[j];
    }
}

__device__ __forceinline__ float wave_max(float x) {
#if defined(__gfx1151__)
    x = fmaxf(x, __int_as_float(__builtin_amdgcn_permlanex16(
        __float_as_int(x), __float_as_int(x), 0x76543210, 0xFEDCBA98, true, false)));
    x = fmaxf(x, qwen4_rocm::qwen_dpp_xor<8>(x));
    x = fmaxf(x, qwen4_rocm::qwen_dpp_xor<4>(x));
    x = fmaxf(x, qwen4_rocm::qwen_dpp_xor<2>(x));
    x = fmaxf(x, qwen4_rocm::qwen_dpp_xor<1>(x));
#else
    for (int d = 16; d; d >>= 1) x = fmaxf(x, __shfl_xor(x, d, 32));
#endif
    return x;
}

__device__ __forceinline__ unsigned wave_min_u32(unsigned x) {
#if defined(__gfx1151__)
    x = min(x, (unsigned)__builtin_amdgcn_permlanex16((int)x, (int)x, 0x76543210, 0xFEDCBA98, true, false));
    x = min(x, (unsigned)__builtin_amdgcn_update_dpp(0, (int)x, 0x160 | 8, 0xf, 0xf, true));
    x = min(x, (unsigned)__builtin_amdgcn_update_dpp(0, (int)x, 0x160 | 4, 0xf, 0xf, true));
    x = min(x, (unsigned)__builtin_amdgcn_update_dpp(0, (int)x, 0x160 | 2, 0xf, 0xf, true));
    x = min(x, (unsigned)__builtin_amdgcn_update_dpp(0, (int)x, 0x160 | 1, 0xf, 0xf, true));
#else
    for (int d = 16; d; d >>= 1) x = min(x, (unsigned)__shfl_xor((int)x, d, 32));
#endif
    return x;
}

/* Decode-sized batches: one block per (KV head, key split, row).  The
 * block's waves form W tile slots × G/GH head groups: each wave takes 32-key
 * tiles in turn and scores them against its GH query heads, lane i holding
 * key i (q sits in LDS and is read as broadcasts).  The online softmax then
 * needs one max and one sum per head and tile instead of a cross-lane
 * reduction per key and head, and lane l accumulates dims 4l..4l+3 from the
 * tile's V rows.  The tile slots' states are merged in LDS and written as
 * one split partial in attention_merge's layout, or as the output when there
 * is one split. */
template<unsigned D, unsigned G, unsigned GH>
__launch_bounds__(32 * 4 * (G / GH))
__global__ void attention_decode(float *out, float *partial, const float *q, const __half *kc,
        const __half *vc, unsigned H, unsigned Hkv, unsigned pos0, unsigned cache_rows,
        unsigned window, unsigned splits, unsigned per, float scale) {
    static_assert(D == 128 && G % GH == 0, "each lane owns four dims");
    constexpr unsigned W = 4, TILE = 32;
    __shared__ float4 qs[G][D / 4];
    __shared__ float ps[W][G][TILE];
    __shared__ float ms[W][G], ls[W][G];
    __shared__ float accs[W][G][D];
    const unsigned kh = blockIdx.x, split = blockIdx.y, t = blockIdx.z;
    const unsigned wave = threadIdx.x / 32, lane = threadIdx.x & 31;
    const unsigned slot = wave % W, g0 = wave / W * GH;
    const unsigned pos = pos0 + t, lo = window && pos + 1 > window ? pos + 1 - window : 0;
    const unsigned begin = lo + split * per, end = min(pos + 1, begin + per);
    const float *qr = q + ((uint64_t)t * H + kh * G) * D;
    for (unsigned i = threadIdx.x; i < G * D; i += blockDim.x) ((float *)qs)[i] = qr[i] * scale;
    __syncthreads();
    float m[GH], l[GH], acc[GH][4];
    #pragma unroll
    for (unsigned g = 0; g < GH; g++) {
        m[g] = -3e38f; l[g] = 0;
        #pragma unroll
        for (unsigned c = 0; c < 4; c++) acc[g][c] = 0;
    }
    /* Uniform trip count so the block can synchronize; a wave whose tile
     * starts past the end only takes part in the barriers. */
    for (unsigned tile0 = begin; tile0 < end; tile0 += W * TILE) {
        const unsigned base = tile0 + slot * TILE;
        const bool active = base < end;
        const unsigned p = base + lane;
        const bool valid = p < end;
        if (active) {
            float s[GH];
            #pragma unroll
            for (unsigned g = 0; g < GH; g++) s[g] = 0;
            if (valid) {
                const uint4 *kr = (const uint4 *)(kc + ((uint64_t)(p % cache_rows) * Hkv + kh) * D);
                #pragma unroll 4
                for (unsigned c = 0; c < D / 8; c++) {
                    const uint4 raw = kr[c];
                    const float2 k0 = __half22float2(*(const __half2 *)&raw.x);
                    const float2 k1 = __half22float2(*(const __half2 *)&raw.y);
                    const float2 k2 = __half22float2(*(const __half2 *)&raw.z);
                    const float2 k3 = __half22float2(*(const __half2 *)&raw.w);
                    #pragma unroll
                    for (unsigned g = 0; g < GH; g++) {
                        const float4 a = qs[g0 + g][2 * c], b = qs[g0 + g][2 * c + 1];
                        s[g] += a.x * k0.x + a.y * k0.y + a.z * k1.x + a.w * k1.y +
                                b.x * k2.x + b.y * k2.y + b.z * k3.x + b.w * k3.y;
                    }
                }
            }
            #pragma unroll
            for (unsigned g = 0; g < GH; g++) {
                const float sv = valid ? s[g] : -3e38f;
                const float nm = fmaxf(m[g], wave_max(sv));
                const float corr = expf(m[g] - nm);
                const float pg = valid ? expf(sv - nm) : 0.0f;
                l[g] = l[g] * corr + sum(pg);
                #pragma unroll
                for (unsigned c = 0; c < 4; c++) acc[g][c] *= corr;
                m[g] = nm;
                ps[slot][g0 + g][lane] = pg;
            }
        }
        __syncthreads();
        if (active) {
            /* Keys past the end have weight 0 and read the last valid row. */
            const unsigned n = min(TILE, end - base);
            #pragma unroll 2
            for (unsigned i0 = 0; i0 < TILE; i0 += 4) {
                float4 w4[GH];
                #pragma unroll
                for (unsigned g = 0; g < GH; g++) w4[g] = *(const float4 *)&ps[slot][g0 + g][i0];
                uint2 raw[4];
                #pragma unroll
                for (unsigned u = 0; u < 4; u++)
                    raw[u] = *(const uint2 *)(vc + ((uint64_t)((base + min(i0 + u, n - 1)) % cache_rows) * Hkv + kh) * D + 4 * lane);
                #pragma unroll
                for (unsigned u = 0; u < 4; u++) {
                    const float2 v0 = __half22float2(*(const __half2 *)&raw[u].x);
                    const float2 v1 = __half22float2(*(const __half2 *)&raw[u].y);
                    #pragma unroll
                    for (unsigned g = 0; g < GH; g++) {
                        const float w = u == 0 ? w4[g].x : u == 1 ? w4[g].y : u == 2 ? w4[g].z : w4[g].w;
                        acc[g][0] += w * v0.x; acc[g][1] += w * v0.y;
                        acc[g][2] += w * v1.x; acc[g][3] += w * v1.y;
                    }
                }
            }
        }
        __syncthreads();
    }
    /* Merge the tile slots: rescale each to the block's max per head. */
    if (!lane) {
        #pragma unroll
        for (unsigned g = 0; g < GH; g++) { ms[slot][g0 + g] = m[g]; ls[slot][g0 + g] = l[g]; }
    }
    __syncthreads();
    #pragma unroll
    for (unsigned g = 0; g < GH; g++) {
        float bm = -3e38f;
        for (unsigned w = 0; w < W; w++) if (ls[w][g0 + g] > 0) bm = fmaxf(bm, ms[w][g0 + g]);
        const float f = l[g] > 0 ? expf(m[g] - bm) : 0.0f;
        #pragma unroll
        for (unsigned c = 0; c < 4; c++) accs[slot][g0 + g][4 * lane + c] = acc[g][c] * f;
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < G * D; i += blockDim.x) {
        const unsigned g = i / D, d = i % D;
        float bm = -3e38f, den = 0, a = 0;
        for (unsigned w = 0; w < W; w++) if (ls[w][g] > 0) bm = fmaxf(bm, ms[w][g]);
        for (unsigned w = 0; w < W; w++) {
            if (ls[w][g] > 0) den += ls[w][g] * expf(ms[w][g] - bm);
            a += accs[w][g][d];
        }
        const uint64_t hq = (uint64_t)t * H + kh * G + g;
        if (splits == 1) {
            out[hq * D + d] = den > 0 ? a / den : 0;
        } else {
            float *dst = partial + (hq * splits + split) * (D + 2);
            if (!d) { dst[0] = bm; dst[1] = den; }
            dst[2 + d] = a;
        }
    }
}

/* One block of D threads per (head, row).  The splits' maxima and sums are
 * loaded at once into LDS (splits <= 256), so only the accumulator column
 * is read in a loop, with independent loads. */
__global__ void attention_merge(float *out, const float *partial, unsigned H, unsigned D, unsigned splits) {
    __shared__ float wsh[256], dsh[256];
    const unsigned h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const float *p = partial + ((uint64_t)t * H + h) * splits * (D + 2);
    for (unsigned s = d; s < splits; s += blockDim.x) { wsh[s] = p[s * (D + 2)]; dsh[s] = p[s * (D + 2) + 1]; }
    __syncthreads();
    float m = -3e38f;
    for (unsigned s = 0; s < splits; s++) if (dsh[s] > 0) m = fmaxf(m, wsh[s]);
    __syncthreads();
    for (unsigned s = d; s < splits; s += blockDim.x) wsh[s] = dsh[s] > 0 ? expf(wsh[s] - m) : 0;
    __syncthreads();
    if (d >= D) return;
    float denom = 0, acc = 0;
    #pragma unroll 8
    for (unsigned s = 0; s < splits; s++) {
        denom += dsh[s] * wsh[s];
        acc += p[s * (D + 2) + 2 + d] * wsh[s];
    }
    out[((uint64_t)t * H + h) * D + d] = denom > 0 ? acc / denom : 0;
}

/* One block per token.  Ties keep the lower expert id. */
__global__ void router(int *sel, float *weights, const float *logits, const float *bias,
        unsigned NE, unsigned NS, unsigned stride, float scale) {
    const unsigned t = blockIdx.x, tid = threadIdx.x;
    __shared__ float score[512];
    __shared__ float best_v[256];
    __shared__ unsigned best_i[256];
    for (unsigned e = tid; e < NE; e += blockDim.x) score[e] = logits[(uint64_t)t * NE + e] + bias[e];
    __syncthreads();
    for (unsigned s = 0; s < NS; s++) {
        float bv = -INFINITY;
        unsigned bi = UINT_MAX;
        for (unsigned e = tid; e < NE; e += blockDim.x)
            if (score[e] > bv || (score[e] == bv && e < bi)) { bv = score[e]; bi = e; }
        best_v[tid] = bv; best_i[tid] = bi;
        __syncthreads();
        for (unsigned w = blockDim.x / 2; w; w /= 2) {
            if (tid < w && (best_v[tid + w] > best_v[tid] ||
                            (best_v[tid + w] == best_v[tid] && best_i[tid + w] < best_i[tid]))) {
                best_v[tid] = best_v[tid + w]; best_i[tid] = best_i[tid + w];
            }
            __syncthreads();
        }
        if (!tid) {
            const unsigned e = best_i[0];
            sel[(uint64_t)t * NS + s] = (int)e;
            weights[(uint64_t)t * stride + s] = sigmoid(logits[(uint64_t)t * NE + e]) * scale;
            score[e] = -INFINITY;
        }
        __syncthreads();
    }
    if (!tid) for (unsigned s = NS; s < stride; s++) weights[(uint64_t)t * stride + s] = 1.0f;
}

/* Router weights as BF16: the GGUF stores them as F32 widened from the
 * source's BF16, so dropping the low half is exact.  pack also counts the
 * values whose low half is not zero (then the copy is not used). */
__global__ void router_pack(uint16_t *dst, const float *src, uint64_t n, unsigned *inexact) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned bits = __float_as_uint(src[i]);
    dst[i] = (uint16_t)(bits >> 16);
    if (bits & 0xffffu) atomicAdd(inexact, 1u);
}

__device__ unsigned router_ticket;

/* Decode-sized router: logits for T <= 8 rows, then top-k selection, in one
 * launch.  One wave per expert row loads its K BF16 weights (K % 256 == 0,
 * K <= 4096) up front and applies them to every row; the block that
 * finishes last (ticket) selects the experts, one wave per row with each
 * lane holding experts lane + 32 j.  Selection matches router: highest
 * logit + bias first, ties to the lower id, weight sigmoid(logit) * scale,
 * slots NS..stride-1 get 1.0. */
template<unsigned KC>
__global__ __launch_bounds__(128) void router_decode(int *sel, float *weights, float *logits,
        const float *x, const uint16_t *w, const float *bias, unsigned T, unsigned K, unsigned NE,
        unsigned NS, unsigned stride, float scale) {
    const unsigned wave = threadIdx.x / 32, lane = threadIdx.x & 31;
    const unsigned e = blockIdx.x * 4 + wave;
    __shared__ float4 xs[KC * 64];
    {
        const uint4 *row = (const uint4 *)(w + (uint64_t)min(e, NE - 1) * K);
        uint4 wv[KC];
        #pragma unroll
        for (unsigned c = 0; c < KC; c++) wv[c] = row[c * 32 + lane];
        for (unsigned t = 0; t < T; t++) {
            /* The block's four waves share each x row through LDS. */
            if (t) __syncthreads();
            const float4 *xg = (const float4 *)(x + (uint64_t)t * K);
            for (unsigned i = threadIdx.x; i < K / 4; i += blockDim.x) xs[i] = xg[i];
            __syncthreads();
            const float4 *xr = xs;
            float acc = 0;
            #pragma unroll
            for (unsigned c = 0; c < KC; c++) {
                const float4 a = xr[(c * 32 + lane) * 2], b = xr[(c * 32 + lane) * 2 + 1];
                const uint4 q = wv[c];
                acc += a.x * __uint_as_float(q.x << 16) + a.y * __uint_as_float(q.x & 0xffff0000u) +
                       a.z * __uint_as_float(q.y << 16) + a.w * __uint_as_float(q.y & 0xffff0000u) +
                       b.x * __uint_as_float(q.z << 16) + b.y * __uint_as_float(q.z & 0xffff0000u) +
                       b.z * __uint_as_float(q.w << 16) + b.w * __uint_as_float(q.w & 0xffff0000u);
            }
            acc = sum(acc);
            if (!lane && e < NE) logits[(uint64_t)t * NE + e] = acc;
        }
    }
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (!threadIdx.x) {
        last = atomicAdd(&router_ticket, 1u) == gridDim.x - 1;
        if (last) router_ticket = 0;
    }
    __syncthreads();
    if (!last) return;
    /* Acquire: the agent-scope fence invalidates this CU's caches, so plain
     * loads see the other blocks' logits. */
    __threadfence();
    constexpr unsigned J = 16;   /* NE <= 512 */
    for (unsigned t = wave; t < T; t += 4) {
        const float *lr = logits + (uint64_t)t * NE;
        float sc[J];
        #pragma unroll
        for (unsigned j = 0; j < J; j++) {
            const unsigned ej = lane + 32 * j;
            sc[j] = ej < NE ? lr[ej] + bias[ej] : -INFINITY;
        }
        for (unsigned s = 0; s < NS; s++) {
            float bv = -INFINITY;
            unsigned bi = UINT_MAX;
            #pragma unroll
            for (unsigned j = 0; j < J; j++)
                if (sc[j] > bv) { bv = sc[j]; bi = lane + 32 * j; }
            /* Highest score, then the lowest id among the lanes holding it. */
            const float top = wave_max(bv);
            bi = wave_min_u32(bv == top ? bi : UINT_MAX);
            #pragma unroll
            for (unsigned j = 0; j < J; j++) if (lane + 32 * j == bi) sc[j] = -INFINITY;
            if (!lane) {
                sel[(uint64_t)t * NS + s] = (int)bi;
                weights[(uint64_t)t * stride + s] = sigmoid(lr[bi]) * scale;
            }
        }
        if (!lane) for (unsigned s = NS; s < stride; s++) weights[(uint64_t)t * stride + s] = 1.0f;
    }
}

/* LM head for one row from BF16 weights [M][K], K = 256 * KC.  Lane l holds
 * x at (c * 32 + l) * 8 .. + 8 for c < KC in registers; each wave streams
 * ROWS consecutive rows with 16-byte loads, the next row's loads issued
 * before the current row is reduced. */
template<unsigned KC, unsigned ROWS>
__global__ __launch_bounds__(128) void head_bf16(float *out, const uint16_t *w, const float *x, unsigned M) {
    constexpr unsigned K = KC * 256;
    const unsigned wave = threadIdx.x / 32, lane = threadIdx.x & 31;
    const unsigned row0 = (blockIdx.x * 4 + wave) * ROWS;
    if (row0 >= M) return;
    float4 xa[KC], xb[KC];
    #pragma unroll
    for (unsigned c = 0; c < KC; c++) {
        xa[c] = ((const float4 *)x)[(c * 32 + lane) * 2];
        xb[c] = ((const float4 *)x)[(c * 32 + lane) * 2 + 1];
    }
    uint4 wv[2][KC];
    #pragma unroll
    for (unsigned c = 0; c < KC; c++) wv[0][c] = ((const uint4 *)(w + (uint64_t)row0 * K))[c * 32 + lane];
    #pragma unroll
    for (unsigned r = 0; r < ROWS; r++) {
        const unsigned row = row0 + r;
        if (row >= M) break;
        if (r + 1 < ROWS && row + 1 < M) {
            #pragma unroll
            for (unsigned c = 0; c < KC; c++)
                wv[(r + 1) & 1][c] = ((const uint4 *)(w + (uint64_t)(row + 1) * K))[c * 32 + lane];
        }
        float acc = 0;
        #pragma unroll
        for (unsigned c = 0; c < KC; c++) {
            const uint4 q = wv[r & 1][c];
            const float4 a = xa[c], b = xb[c];
            acc += a.x * __uint_as_float(q.x << 16) + a.y * __uint_as_float(q.x & 0xffff0000u) +
                   a.z * __uint_as_float(q.y << 16) + a.w * __uint_as_float(q.y & 0xffff0000u) +
                   b.x * __uint_as_float(q.z << 16) + b.y * __uint_as_float(q.z & 0xffff0000u) +
                   b.z * __uint_as_float(q.w << 16) + b.w * __uint_as_float(q.w & 0xffff0000u);
        }
        acc = sum(acc);
        if (!lane) out[row] = acc;
    }
}

/* Q4_K expert rows for decode-sized batches.  A Q4_K block is a 16-byte
 * header (d, dmin, 12 bytes of 6-bit scales and mins) and 128 bytes of
 * nibbles: bytes c * 32 .. c * 32 + 31 hold values c * 64 + 0..31 (low
 * nibbles, group 2c) and c * 64 + 32..63 (high nibbles, group 2c + 1).
 * Lane l reads the 4 bytes at c * 32 + (l % 8) * 4 with c = l / 8, i.e.
 * eight values of two groups, plus the header as a broadcast; all KB
 * blocks of a row are unrolled so the loads go out together. */
/* Byte k of the 12 scale bytes, held in three words (no indexed memory). */
__device__ __forceinline__ unsigned q4k_byte(uint4 h, unsigned k) {
    const unsigned w = k < 4 ? h.y : k < 8 ? h.z : h.w;
    return (w >> (8 * (k & 3))) & 255;
}

__device__ __forceinline__ void q4k_scale_min(uint4 h, unsigned g, float &s, float &m) {
    if (g < 4) { s = (float)(q4k_byte(h, g) & 63); m = (float)(q4k_byte(h, g + 4) & 63); }
    else {
        const unsigned hi = q4k_byte(h, g + 4);
        s = (float)((hi & 15) | ((q4k_byte(h, g - 4) >> 6) << 4));
        m = (float)((hi >> 4) | ((q4k_byte(h, g) >> 6) << 4));
    }
}

/* One block's contribution for this lane: nibble word q, header h. */
__device__ __forceinline__ float q4k_block(uint4 h, unsigned q, float4 xa, float4 xb, unsigned c) {
    const float d = __half2float(__ushort_as_half((unsigned short)(h.x & 0xffff)));
    const float dm = __half2float(__ushort_as_half((unsigned short)(h.x >> 16)));
    float s0, m0, s1, m1;
    q4k_scale_min(h, 2 * c, s0, m0);
    q4k_scale_min(h, 2 * c + 1, s1, m1);
    const float lo = (float)(q & 15) * xa.x + (float)((q >> 8) & 15) * xa.y +
                     (float)((q >> 16) & 15) * xa.z + (float)((q >> 24) & 15) * xa.w;
    const float hi = (float)((q >> 4) & 15) * xb.x + (float)((q >> 12) & 15) * xb.y +
                     (float)((q >> 20) & 15) * xb.z + (float)(q >> 28) * xb.w;
    return d * (s0 * lo + s1 * hi) - dm * (m0 * (xa.x + xa.y + xa.z + xa.w) + m1 * (xb.x + xb.y + xb.z + xb.w));
}

/* Row dots for one row (r1 == NULL) or two rows sharing x, KB blocks in
 * chunks of CH whose loads all go out before the chunk is used. */
template<unsigned KB, unsigned CH>
__device__ __forceinline__ void q4k_rows_dot(const char *r0, const char *r1, const float *x, unsigned lane,
        float &a, float &b) {
    static_assert(KB % CH == 0, "chunked blocks");
    const unsigned c = lane / 8, o = (lane % 8) * 4;
    float acc0 = 0, acc1 = 0;
    #pragma unroll
    for (unsigned b0 = 0; b0 < KB; b0 += CH) {
        uint4 h0[CH], h1[CH];
        unsigned q0[CH], q1[CH];
        float4 xa[CH], xb[CH];
        #pragma unroll
        for (unsigned k = 0; k < CH; k++) {
            const unsigned bk = b0 + k;
            h0[k] = *(const uint4 *)(r0 + bk * 144);
            q0[k] = *(const unsigned *)(r0 + bk * 144 + 16 + c * 32 + o);
            if (r1) {
                h1[k] = *(const uint4 *)(r1 + bk * 144);
                q1[k] = *(const unsigned *)(r1 + bk * 144 + 16 + c * 32 + o);
            }
            xa[k] = *(const float4 *)(x + bk * 256 + c * 64 + o);
            xb[k] = *(const float4 *)(x + bk * 256 + c * 64 + 32 + o);
        }
        #pragma unroll
        for (unsigned k = 0; k < CH; k++) {
            acc0 += q4k_block(h0[k], q0[k], xa[k], xb[k], c);
            if (r1) acc1 += q4k_block(h1[k], q1[k], xa[k], xb[k], c);
        }
    }
    a = sum(acc0);
    b = r1 ? sum(acc1) : 0.0f;
}

/* grid (ceil(M / 4), NS + 1, T), 4 waves per block, one output row per
 * wave; slot NS is the F8 shared expert.  Same layout and outputs as the
 * Qwen moe_mv: gate/up write silu(gate) * up, down writes the projection. */
template<unsigned KB, bool DOWN>
__global__ __launch_bounds__(128) void moe_q4k(float *out, const float *x, const int *selected,
        const char *w0, const char *w1, const char *sh0, const char *sh1,
        unsigned NE, unsigned NS, unsigned M, uint64_t rb, uint64_t srb) {
    constexpr unsigned K = KB * 256;
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 32, slot = blockIdx.y, t = blockIdx.z;
    const unsigned lane = threadIdx.x & 31;
    if (row >= M) return;
    const uint64_t pair = (uint64_t)t * (NS + 1) + slot;
    const float *xt = x + (DOWN ? pair : t) * K;
    float a = 0, b = 0;
    if (slot == NS) {
        a = qwen4_rocm::dot<200>(sh0 + row * srb, xt, K);
        if (!DOWN) b = qwen4_rocm::dot<200>(sh1 + row * srb, xt, K);
    } else {
        const int e = selected[(uint64_t)t * NS + slot];
        if (e >= 0 && (unsigned)e < NE) {
            const uint64_t off = ((uint64_t)e * M + row) * rb;
            q4k_rows_dot<KB, (KB % 5 ? KB : 5)>(w0 + off, DOWN ? NULL : w1 + off, xt, lane, a, b);
        }
    }
    if (!lane) out[pair * M + row] = DOWN ? a : qwen4_rocm::silu(a) * b;
}

/* One block of THREADS threads per token row; D <= THREADS * MAXV.  NSC is
 * the slot count when known at compile time (decode: K routed + shared), so
 * the partial loads unroll and go out together; 0 loops over NS.  x is read
 * once and kept in registers for both reductions. */
template<unsigned THREADS, unsigned MAXV, unsigned NSC>
__launch_bounds__(THREADS)
__global__ void norm_add(float *x, float *xn, const float *h, const float *part, const float *weights,
        const float *shared, unsigned NS, unsigned pstride, unsigned wstride, const float *w_post,
        const float *w_next, unsigned D, float eps) {
    __shared__ float red[THREADS / 32];
    const unsigned t = blockIdx.x, tid = threadIdx.x;
    float *xr = x + (uint64_t)t * D;
    float xv[MAXV];
    #pragma unroll
    for (unsigned i = 0; i < MAXV; i++) {
        const unsigned d = tid + THREADS * i;
        xv[i] = d < D ? xr[d] : 0;
    }
    if (h || part) {
        float v[MAXV], ss = 0;
        if (part && NSC) {
            float ws[NSC ? NSC : 1];
            #pragma unroll
            for (unsigned s = 0; s < NSC; s++) ws[s] = weights[(uint64_t)t * wstride + s];
            #pragma unroll
            for (unsigned i = 0; i < MAXV; i++) {
                const unsigned d = tid + THREADS * i;
                float a = 0;
                if (d < D) {
                    #pragma unroll
                    for (unsigned s = 0; s < NSC; s++) a += ws[s] * part[((uint64_t)t * pstride + s) * D + d];
                    if (shared) a += shared[(uint64_t)t * D + d];
                }
                v[i] = a;
                ss += a * a;
            }
        } else {
            #pragma unroll
            for (unsigned i = 0; i < MAXV; i++) {
                const unsigned d = tid + THREADS * i;
                float a = 0;
                if (d < D) {
                    if (part) {
                        for (unsigned s = 0; s < NS; s++)
                            a += weights[(uint64_t)t * wstride + s] * part[((uint64_t)t * pstride + s) * D + d];
                        if (shared) a += shared[(uint64_t)t * D + d];
                    } else {
                        a = h[(uint64_t)t * D + d];
                    }
                }
                v[i] = a;
                ss += a * a;
            }
        }
        const float r = rsqrtf(block_sum(ss, red) / D + eps);
        #pragma unroll
        for (unsigned i = 0; i < MAXV; i++) {
            const unsigned d = tid + THREADS * i;
            if (d < D) {
                xv[i] += v[i] * r * w_post[d];
                xr[d] = xv[i];
            }
        }
    }
    float ss = 0;
    #pragma unroll
    for (unsigned i = 0; i < MAXV; i++) ss += xv[i] * xv[i];
    const float r = rsqrtf(block_sum(ss, red) / D + eps);
    #pragma unroll
    for (unsigned i = 0; i < MAXV; i++) {
        const unsigned d = tid + THREADS * i;
        if (d < D) xn[(uint64_t)t * D + d] = xv[i] * r * w_next[d];
    }
}

/* Prefill attention on WMMA tiles (f16 operands, f32 accumulation).
 *
 * A block covers 16 query positions x 4 query heads of one KV head; its four
 * waves (one head each) share every 16-key K/V tile through LDS.  Each wave
 * computes S^T = K Q^T so the accumulator gives every lane one query column
 * (lane % 16) and eight of its sixteen keys (the other eight sit in lane^16):
 * the online-softmax max and sum need one cross-lane exchange.  O^T = V^T P^T
 * keeps the per-query rescale lane-local too.  WMMA inputs are replicated
 * across both half-waves, outputs hold rows 2j + lane/16, column lane % 16. */
typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

constexpr unsigned FA_D = 128, FA_LD = FA_D + 8;

__launch_bounds__(128, 2)
__global__ void attention_wmma(float *out, const float *q, const __half *kc, const __half *vc,
        unsigned T, unsigned H, unsigned Hkv, unsigned pos0, unsigned cache_rows,
        unsigned window, float scale_log2) {
    __shared__ _Float16 ks[16][FA_LD], vs[16][FA_LD];
    const unsigned tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, col = lane & 15, half = lane >> 4;
    const unsigned t0 = blockIdx.x * 16, h = blockIdx.y * 4 + wave, kh = h / (H / Hkv);
    const unsigned tq = t0 + col, qpos = pos0 + (tq < T ? tq : T - 1);
    /* Q^T fragments: this lane's query column, sixteen dims per step. */
    half16 qf[FA_D / 16];
    const float *qr = q + ((uint64_t)(tq < T ? tq : T - 1) * H + h) * FA_D;
    #pragma unroll
    for (unsigned dt = 0; dt < FA_D / 16; dt++)
        #pragma unroll
        for (unsigned i = 0; i < 16; i++) qf[dt][i] = (_Float16)(qr[dt * 16 + i] * scale_log2);
    float8 o[FA_D / 16] = {};
    float m = -INFINITY, l = 0;
    const unsigned first_q = pos0 + t0, last_q = pos0 + min(t0 + 16, T) - 1;
    const unsigned lo = window && first_q + 1 > window ? first_q + 1 - window : 0;
    for (unsigned kb = lo & ~15u; kb <= last_q; kb += 16) {
        __syncthreads();
        for (unsigned i = tid; i < 16 * (FA_D / 8); i += 128) {
            const unsigned r = i / (FA_D / 8), c = (i % (FA_D / 8)) * 8, p = kb + r;
            uint4 kv = {}, vv = {};
            if (p <= last_q) {
                const uint64_t base = ((uint64_t)(p % cache_rows) * Hkv + kh) * FA_D + c;
                kv = *(const uint4 *)(kc + base);
                vv = *(const uint4 *)(vc + base);
            }
            *(uint4 *)&ks[r][c] = kv;
            *(uint4 *)&vs[r][c] = vv;
        }
        __syncthreads();
        float8 s = {};
        #pragma unroll
        for (unsigned dt = 0; dt < FA_D / 16; dt++) {
            const half16 a = *(const half16 *)&ks[col][dt * 16];
            s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, qf[dt], s);
        }
        /* this lane: query qpos, keys kb + 2j + half */
        float mx = -INFINITY;
        #pragma unroll
        for (unsigned j = 0; j < 8; j++) {
            const unsigned p = kb + 2 * j + half;
            const bool ok = p <= qpos && (!window || p + window > qpos);
            s[j] = ok ? s[j] : -INFINITY;
            mx = fmaxf(mx, s[j]);
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        const float mn = fmaxf(m, mx);
        const float corr = mn == -INFINITY ? 1.0f : exp2f(m - mn);
        float ps[8], sum = 0;
        #pragma unroll
        for (unsigned j = 0; j < 8; j++) {
            ps[j] = mn == -INFINITY ? 0.0f : exp2f(s[j] - mn);
            sum += ps[j];
        }
        sum += __shfl_xor(sum, 16, 32);
        l = l * corr + sum;
        m = mn;
        /* P^T as a B operand: all sixteen keys of this lane's query column */
        half16 pb;
        #pragma unroll
        for (unsigned j = 0; j < 8; j++) {
            const float other = __shfl_xor(ps[j], 16, 32);
            pb[2 * j + half] = (_Float16)ps[j];
            pb[2 * j + 1 - half] = (_Float16)other;
        }
        #pragma unroll
        for (unsigned dt = 0; dt < FA_D / 16; dt++) {
            half16 a;
            #pragma unroll
            for (unsigned k = 0; k < 16; k++) a[k] = vs[k][dt * 16 + col];
            o[dt] *= corr;
            o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, pb, o[dt]);
        }
    }
    if (tq >= T) return;
    const float inv = l > 0 ? 1.0f / l : 0.0f;
    float *dst = out + ((uint64_t)tq * H + h) * FA_D;
    #pragma unroll
    for (unsigned dt = 0; dt < FA_D / 16; dt++)
        #pragma unroll
        for (unsigned j = 0; j < 8; j++) dst[dt * 16 + 2 * j + half] = o[dt][j] * inv;
}

static const float *f32_weight(const void *map, uint64_t size, uint64_t off, uint64_t n) {
    return (const float *)weight(map, size, off, n * 4);
}

} // namespace kolibri_rocm

extern "C" int ds4_gpu_kolibri_qk_prep_tensor(ds4_gpu_tensor *q, const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v, ds4_gpu_tensor *kc, ds4_gpu_tensor *vc,
        const void *map, uint64_t size, uint64_t qn_off, uint64_t kn_off,
        uint32_t T, uint32_t H, uint32_t Hkv, uint32_t D, uint32_t pos0, uint32_t cache_rows,
        int rope, float freq_base, float eps) {
    using namespace kolibri_rocm;
    if (!T || !H || !Hkv || D != 128 || !cache_rows || T > cache_rows ||
        !tensor(q, (uint64_t)T * H * D * 4) || !tensor(k, (uint64_t)T * Hkv * D * 4) ||
        !tensor(v, (uint64_t)T * Hkv * D * 4) || !tensor(kc, (uint64_t)cache_rows * Hkv * D * 2) ||
        !tensor(vc, (uint64_t)cache_rows * Hkv * D * 2)) return 0;
    const float *qn = f32_weight(map, size, qn_off, D), *kn = f32_weight(map, size, kn_off, D);
    if (!qn || !kn) return 0;
    const dim3 grid((H + Hkv + 3) / 4, T);
    qk_prep<128><<<grid, 128, 0, 0>>>((float *)q->ptr, (const float *)k->ptr, (const float *)v->ptr,
        (__half *)kc->ptr, (__half *)vc->ptr, qn, kn, H, Hkv, pos0, cache_rows, rope != 0, freq_base, eps);
    return launched();
}

extern "C" int ds4_gpu_kolibri_attention_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *kc, const ds4_gpu_tensor *vc, uint32_t T, uint32_t H, uint32_t Hkv,
        uint32_t D, uint32_t pos0, uint32_t cache_rows, uint32_t window) {
    using namespace kolibri_rocm;
    if (!T || !H || !Hkv || H % Hkv || D != 128 || !cache_rows ||
        !tensor(out, (uint64_t)T * H * D * 4) || !tensor(q, (uint64_t)T * H * D * 4) ||
        !tensor(kc, (uint64_t)cache_rows * Hkv * D * 2) || !tensor(vc, (uint64_t)cache_rows * Hkv * D * 2)) return 0;
    /* Prefill batches: WMMA tiles (heads per block must share a KV head). */
    static int no_wmma = -1;
    if (no_wmma < 0) no_wmma = getenv("DS4_KOLIBRI_ATTN_SCALAR") != NULL;
    if (T >= 16 && !no_wmma && ds4_rocm_is_gfx1151() && (H / Hkv) % 4 == 0) {
        attention_wmma<<<dim3((T + 15) / 16, H / 4), 128, 0, 0>>>((float *)out->ptr,
            (const float *)q->ptr, (const __half *)kc->ptr, (const __half *)vc->ptr,
            T, H, Hkv, pos0, cache_rows, window, 1.4426950408889634f / sqrtf((float)D));
        return launched();
    }
    /* Decode-sized batches read each KV head's cache once for all its query
     * heads; splits of 128 keys give the GPU enough waves. */
    const unsigned G = H / Hkv;
    if (T < 16 && (G == 12 || G == 4)) {
        const unsigned last = pos0 + T;
        const unsigned span = window && last > window ? window : last;
        const unsigned splits = std::min(64u, std::max(1u, (span + 127) / 128));
        const unsigned per = (span + splits - 1) / splits;
        float *partial = NULL;
        if (splits > 1) {
            partial = (float *)cuda_tmp_alloc((uint64_t)T * H * splits * (D + 2) * 4, "Kolibri attention partials");
            if (!partial) return 0;
        }
        const dim3 grid(Hkv, splits, T);
        const float scale = 1.0f / sqrtf((float)D);
        if (G == 12) attention_decode<128, 12, 4><<<grid, 384, 0, 0>>>((float *)out->ptr, partial, (const float *)q->ptr,
            (const __half *)kc->ptr, (const __half *)vc->ptr, H, Hkv, pos0, cache_rows, window, splits, per, scale);
        else attention_decode<128, 4, 4><<<grid, 128, 0, 0>>>((float *)out->ptr, partial, (const float *)q->ptr,
            (const __half *)kc->ptr, (const __half *)vc->ptr, H, Hkv, pos0, cache_rows, window, splits, per, scale);
        if (!launched()) return 0;
        if (splits > 1) {
            attention_merge<<<dim3(H, T), D, 0, 0>>>((float *)out->ptr, partial, H, D, splits);
            if (!launched()) return 0;
        }
        return 1;
    }
    /* The longest key range of the batch decides the split: short batches
     * over long contexts split keys so the GPU has enough waves. */
    const unsigned last = pos0 + T;
    const unsigned span = window && last > window ? window : last;
    unsigned splits = 1, per = span;
    if (T <= 16 && span > 512) {
        splits = std::min(32u, (span + 511) / 512);
        per = (span + splits - 1) / splits;
    }
    float *partial = NULL;
    if (splits > 1) {
        partial = (float *)cuda_tmp_alloc((uint64_t)T * H * splits * (D + 2) * 4, "Kolibri attention partials");
        if (!partial) return 0;
    }
    const dim3 grid((H + 3) / 4, T, splits);
    attention<128><<<grid, 128, 0, 0>>>((float *)out->ptr, partial, (const float *)q->ptr,
        (const __half *)kc->ptr, (const __half *)vc->ptr, H, Hkv, pos0, cache_rows, window,
        splits, per, 1.0f / sqrtf((float)D));
    if (!launched()) return 0;
    if (splits > 1) {
        attention_merge<<<dim3(H, T), D, 0, 0>>>((float *)out->ptr, partial, H, D, splits);
        if (!launched()) return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_kolibri_router_tensor(ds4_gpu_tensor *sel, ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *logits, const void *map, uint64_t size, uint64_t bias_off,
        uint32_t T, uint32_t NE, uint32_t NS, uint32_t stride, float scale) {
    using namespace kolibri_rocm;
    if (!T || !NE || NE > 512 || !NS || NS > NE || stride < NS ||
        !tensor(sel, (uint64_t)T * NS * 4) || !tensor(weights, (uint64_t)T * stride * 4) ||
        !tensor(logits, (uint64_t)T * NE * 4)) return 0;
    const float *bias = f32_weight(map, size, bias_off, NE);
    if (!bias) return 0;
    router<<<T, 256, 0, 0>>>((int *)sel->ptr, (float *)weights->ptr, (const float *)logits->ptr,
        bias, NE, NS, stride, scale);
    return launched();
}

/* BF16 copies of the router weights, one per weight offset, made on first
 * use and kept for the process.  inexact marks an F32 router that is not
 * BF16-exact; it keeps the F32 path. */
namespace kolibri_rocm {
struct router_copy { const void *map; uint64_t off; uint16_t *w; bool inexact; };
static router_copy router_copies[256];
static unsigned n_router_copies;

static const router_copy *router_bf16(const void *map, uint64_t size, uint64_t off, uint64_t n) {
    for (unsigned i = 0; i < n_router_copies; i++)
        if (router_copies[i].map == map && router_copies[i].off == off) return &router_copies[i];
    if (n_router_copies == sizeof(router_copies) / sizeof(router_copies[0])) return NULL;
    const float *src = f32_weight(map, size, off, n);
    if (!src) return NULL;
    router_copy c = {map, off, NULL, false};
    unsigned *inexact = NULL, h_inexact = 0;
    if (!cuda_ok(cudaMalloc((void **)&c.w, n * 2), "Kolibri router copy") ||
        !cuda_ok(cudaMalloc((void **)&inexact, 4), "Kolibri router copy") ||
        !cuda_ok(cudaMemset(inexact, 0, 4), "Kolibri router copy")) return NULL;
    router_pack<<<(unsigned)((n + 255) / 256), 256, 0, 0>>>(c.w, src, n, inexact);
    if (!launched() || !cuda_ok(cudaMemcpy(&h_inexact, inexact, 4, cudaMemcpyDeviceToHost), "Kolibri router copy"))
        return NULL;
    (void)cudaFree(inexact);
    if (h_inexact) {
        fprintf(stderr, "ds4: Kolibri router at offset %llu has %u values that are not BF16; it stays F32\n",
                (unsigned long long)off, h_inexact);
        (void)cudaFree(c.w);
        c.w = NULL;
        c.inexact = true;
    }
    router_copies[n_router_copies] = c;
    return &router_copies[n_router_copies++];
}
}  // namespace kolibri_rocm

/* Router for decode-sized batches.  Returns -1 when the fused path does not
 * apply (shape, an inexact router, DS4_KOLIBRI_ROUTER_F32=1), so the caller
 * runs the F32 matvec and ds4_gpu_kolibri_router_tensor instead. */
extern "C" int ds4_gpu_kolibri_router_decode_tensor(ds4_gpu_tensor *sel, ds4_gpu_tensor *weights,
        ds4_gpu_tensor *logits, const ds4_gpu_tensor *x, const void *map, uint64_t size,
        uint64_t w_off, uint64_t bias_off, uint32_t T, uint32_t K, uint32_t NE, uint32_t NS,
        uint32_t stride, float scale) {
    using namespace kolibri_rocm;
    static int f32 = -1;
    if (f32 < 0) f32 = getenv("DS4_KOLIBRI_ROUTER_F32") != NULL;
    if (f32 || !T || T > 8 || K % 256 || K > 4096 || !NE || NE > 512 || !NS || NS > NE || stride < NS)
        return -1;
    if (!tensor(sel, (uint64_t)T * NS * 4) || !tensor(weights, (uint64_t)T * stride * 4) ||
        !tensor(logits, (uint64_t)T * NE * 4) || !tensor(x, (uint64_t)T * K * 4)) return 0;
    const router_copy *c = router_bf16(map, size, w_off, (uint64_t)K * NE);
    if (!c) return 0;
    if (c->inexact) return -1;
    const float *bias = f32_weight(map, size, bias_off, NE);
    if (!bias) return 0;
    const unsigned blocks = (NE + 3) / 4;
#define KOLIBRI_ROUTER(KC) router_decode<KC><<<blocks, 128, 0, 0>>>((int *)sel->ptr, (float *)weights->ptr, \
        (float *)logits->ptr, (const float *)x->ptr, c->w, bias, T, K, NE, NS, stride, scale)
    switch (K / 256) {
    case 10: KOLIBRI_ROUTER(10); break;
    case 16: KOLIBRI_ROUTER(16); break;
    default: return -1;
    }
#undef KOLIBRI_ROUTER
    return launched();
}

/* LM head for one row from a BF16 matrix.  Returns -1 when the shape does
 * not fit (the caller runs the generic matvec). */
extern "C" int ds4_gpu_kolibri_head_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t w_off, uint32_t K, uint32_t M) {
    using namespace kolibri_rocm;
    if (K != 2560 || !M) return -1;
    if (!tensor(out, (uint64_t)M * 4) || !tensor(x, (uint64_t)K * 4)) return 0;
    const uint16_t *w = (const uint16_t *)weight(map, size, w_off, (uint64_t)M * K * 2);
    if (!w || ((uintptr_t)w & 15)) return w ? -1 : 0;
    constexpr unsigned ROWS = 8;
    head_bf16<10, ROWS><<<(M + 4 * ROWS - 1) / (4 * ROWS), 128, 0, 0>>>((float *)out->ptr, w,
        (const float *)x->ptr, M);
    return launched();
}

/* Q4_K routed experts with an F8 shared expert, decode-sized batches.
 * Returns -1 when the shapes or types do not fit (the caller runs the Qwen
 * kernels). */
extern "C" int ds4_gpu_kolibri_moe_q4k_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *sel, const void *map, uint64_t size, uint64_t o0, uint64_t o1,
        uint64_t so0, uint64_t so1, uint32_t type, uint32_t shared_type, uint32_t NE, uint32_t T,
        uint32_t NS, uint32_t K, uint32_t M, int down) {
    using namespace kolibri_rocm;
    static int off = -1;
    if (off < 0) off = getenv("DS4_KOLIBRI_Q4K_QWEN") != NULL;
    if (off || type != 12 || shared_type != 200 || !T || T > 16 || !NS || K % 512 ||
        (K != 2560 && K != 512)) return -1;
    const unsigned NO = NS + 1;
    if (!tensor(out, (uint64_t)T * NO * M * 4) || !tensor(sel, (uint64_t)T * NS * 4) ||
        !tensor(x, (uint64_t)T * (down ? NO : 1) * K * 4)) return 0;
    const uint64_t rb = (uint64_t)K / 256 * 144, srb = (uint64_t)K / 512 * 528;
    const char *w0 = weight(map, size, o0, rb * M * NE), *s0 = weight(map, size, so0, srb * M);
    const char *w1 = down ? NULL : weight(map, size, o1, rb * M * NE);
    const char *s1 = down ? NULL : weight(map, size, so1, srb * M);
    if (!w0 || !s0 || (!down && (!w1 || !s1))) return 0;
    if (((uintptr_t)w0 | (uintptr_t)(w1 ? w1 : w0) | (uintptr_t)x->ptr) & 15) return -1;
    const dim3 grid((M + 3) / 4, NO, T);
    if (down && K == 512)
        moe_q4k<2, true><<<grid, 128, 0, 0>>>((float *)out->ptr, (const float *)x->ptr, (const int *)sel->ptr,
            w0, NULL, s0, NULL, NE, NS, M, rb, srb);
    else if (!down && K == 2560)
        moe_q4k<10, false><<<grid, 128, 0, 0>>>((float *)out->ptr, (const float *)x->ptr, (const int *)sel->ptr,
            w0, w1, s0, s1, NE, NS, M, rb, srb);
    else return -1;
    return launched();
}

/* h == part == NULL only normalizes x into xn (the first layer's input). */
extern "C" int ds4_gpu_kolibri_norm_add_tensor(ds4_gpu_tensor *x, ds4_gpu_tensor *xn,
        const ds4_gpu_tensor *h, const ds4_gpu_tensor *part, const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *shared, uint32_t NS, uint32_t pstride, uint32_t wstride,
        const void *map, uint64_t size, uint64_t post_off, uint64_t next_off,
        uint32_t T, uint32_t D, float eps) {
    using namespace kolibri_rocm;
    if (!T || !D || D > 256 * 16 || !tensor(x, (uint64_t)T * D * 4) || !tensor(xn, (uint64_t)T * D * 4) ||
        (h && !tensor(h, (uint64_t)T * D * 4)) ||
        (part && (!NS || pstride < NS || wstride < NS || !tensor(part, (uint64_t)T * pstride * D * 4) ||
                  !tensor(weights, (uint64_t)T * wstride * 4))) ||
        (shared && !tensor(shared, (uint64_t)T * D * 4))) return 0;
    const float *w_post = (h || part) ? f32_weight(map, size, post_off, D) : NULL;
    const float *w_next = f32_weight(map, size, next_off, D);
    if (!w_next || ((h || part) && !w_post)) return 0;
#define KOLIBRI_NORM_ADD(THREADS, MAXV, NSC) norm_add<THREADS, MAXV, NSC><<<T, THREADS, 0, 0>>>( \
        (float *)x->ptr, (float *)xn->ptr, h ? (const float *)h->ptr : NULL, \
        part ? (const float *)part->ptr : NULL, part ? (const float *)weights->ptr : NULL, \
        shared ? (const float *)shared->ptr : NULL, NS, pstride, wstride, w_post, w_next, D, eps)
    /* Decode-sized batches have a block per row and little else to run:
     * 1024 threads keep each thread's loads few and in flight. */
    if (T <= 16 && D <= 1024 * 3) {
        if (part && NS == 7) KOLIBRI_NORM_ADD(1024, 3, 7);
        else KOLIBRI_NORM_ADD(1024, 3, 0);
    } else if (D <= 256 * 4) KOLIBRI_NORM_ADD(256, 4, 0);
    else if (D <= 256 * 10) KOLIBRI_NORM_ADD(256, 10, 0);
    else KOLIBRI_NORM_ADD(256, 16, 0);
#undef KOLIBRI_NORM_ADD
    return launched();
}

/* DS4_KOLIBRI_TRACE=1.  Each interval is stream time between two timing
 * events, so it covers the kernels enqueued in between plus any gap where
 * the GPU waited for the host to launch them; the host column (enqueue time
 * per call) shows when that happens.  Events and records are preallocated;
 * nothing synchronizes before the forward's own end_commands. */
namespace kolibri_trace {

enum { MAX_MARKS = 4096, MAX_LABELS = 48 };

struct label_stat {
    const char *label;
    uint64_t calls;
    double gpu_ms, host_ms, bytes;
};

struct bucket {
    label_stat stat[MAX_LABELS];
    unsigned n_stat;
    uint64_t forwards, tokens, overflow;
    double wall_ms, gpu_ms, host_ms;
};

static int enabled = -1;
static cudaEvent_t ev[MAX_MARKS];
static const char *label[MAX_MARKS];
static uint64_t bytes[MAX_MARKS];
static double host[MAX_MARKS];
static unsigned n_ev, n_marks;
static bool overflow;
static bucket bk[2];   /* 0 decode, 1 prefill */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static label_stat *find(bucket *b, const char *name) {
    for (unsigned i = 0; i < b->n_stat; i++)
        if (strcmp(b->stat[i].label, name) == 0) return &b->stat[i];
    if (b->n_stat == MAX_LABELS) return NULL;
    label_stat *s = &b->stat[b->n_stat++];
    memset(s, 0, sizeof(*s));
    s->label = name;
    return s;
}

static int by_gpu(const void *a, const void *b) {
    const double x = ((const label_stat *)a)->gpu_ms, y = ((const label_stat *)b)->gpu_ms;
    return x < y ? 1 : x > y ? -1 : 0;
}

static void report(void) {
    static const char *name[2] = {"decode", "prefill"};
    for (int k = 0; k < 2; k++) {
        bucket *b = &bk[k];
        if (!b->forwards) continue;
        const double f = (double)b->forwards;
        fprintf(stderr, "ds4: Kolibri trace, %s: %llu forwards, %llu tokens; per forward: "
                "wall %.2f ms, GPU %.2f ms, host enqueue %.2f ms%s\n",
                name[k], (unsigned long long)b->forwards, (unsigned long long)b->tokens,
                b->wall_ms / f, b->gpu_ms / f, b->host_ms / f,
                b->overflow ? " (some forwards had too many marks and were skipped)" : "");
        fprintf(stderr, "ds4:   %-22s %8s %10s %6s %10s %8s %8s\n",
                "call", "calls/fw", "GPU us/fw", "%", "host us/fw", "us/call", "GB/s");
        qsort(b->stat, b->n_stat, sizeof(b->stat[0]), by_gpu);
        for (unsigned i = 0; i < b->n_stat; i++) {
            const label_stat *s = &b->stat[i];
            char bw[16] = "";
            if (s->bytes > 0 && s->gpu_ms > 0) snprintf(bw, sizeof(bw), "%.0f", s->bytes / s->gpu_ms * 1e-6);
            fprintf(stderr, "ds4:   %-22s %8.1f %10.1f %6.1f %10.1f %8.1f %8s\n",
                    s->label, s->calls / f, 1e3 * s->gpu_ms / f, 100.0 * s->gpu_ms / b->gpu_ms,
                    1e3 * s->host_ms / f, 1e3 * s->gpu_ms / s->calls, bw);
        }
    }
    for (unsigned i = 0; i < n_ev; i++) (void)cudaEventDestroy(ev[i]);
    n_ev = n_marks = 0;
}

static bool on(void) {
    if (enabled < 0) {
        const char *v = getenv("DS4_KOLIBRI_TRACE");
        enabled = v && v[0] && strcmp(v, "0") != 0;
        if (enabled) atexit(report);
    }
    return enabled != 0;
}

}  // namespace kolibri_trace

extern "C" int ds4_gpu_kolibri_trace_mark(const char *name, uint64_t nbytes) {
    using namespace kolibri_trace;
    if (!on()) return 1;
    if (!name) n_marks = 0, overflow = false;
    if (n_marks == MAX_MARKS) {
        overflow = true;
        return 1;
    }
    if (n_marks == n_ev) {
        if (!cuda_ok(cudaEventCreate(&ev[n_ev]), "Kolibri trace event")) return 0;
        n_ev++;
    }
    label[n_marks] = name;
    bytes[n_marks] = nbytes;
    host[n_marks] = now_ms();
    if (!cuda_ok(cudaEventRecord(ev[n_marks], 0), "Kolibri trace record")) return 0;
    n_marks++;
    return 1;
}

extern "C" int ds4_gpu_kolibri_trace_end(uint32_t T) {
    using namespace kolibri_trace;
    if (!on() || n_marks < 2 || label[0]) return 1;
    bucket *b = &bk[T > 1];
    if (overflow) {
        b->overflow++;
        n_marks = 0;
        return 1;
    }
    for (unsigned i = 1; i < n_marks; i++) {
        float ms = 0.0f;
        if (!cuda_ok(cudaEventElapsedTime(&ms, ev[i - 1], ev[i]), "Kolibri trace elapsed")) return 0;
        label_stat *s = find(b, label[i]);
        if (!s) continue;
        s->calls++;
        s->gpu_ms += ms;
        s->host_ms += host[i] - host[i - 1];
        s->bytes += (double)bytes[i];
        b->gpu_ms += ms;
    }
    b->host_ms += host[n_marks - 1] - host[0];
    b->wall_ms += now_ms() - host[0];
    b->forwards++;
    b->tokens += T;
    n_marks = 0;
    return 1;
}
