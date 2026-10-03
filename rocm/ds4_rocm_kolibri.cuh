/* Kolibri 1 on ROCm: the operators the Qwen3.8 kernels do not cover.
 *
 * - qk_prep: per-head RMSNorm of q and k with their weights, neox RoPE on
 *   sliding layers only (full layers are NoPE), and the f16 K/V store.
 *   Each layer's cache has cache_rows rows and position p lives in row
 *   p % cache_rows: sliding layers use a ring of window - 1 + chunk rows,
 *   full layers one row per context position.
 * - attention: GQA, one wave per query head, online softmax over the keys a
 *   query may see (the last `window` positions, or all of them), optional
 *   key splits merged afterwards.  No output gate.
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

__global__ void attention_merge(float *out, const float *partial, unsigned H, unsigned D, unsigned splits) {
    const unsigned h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    if (d >= D) return;
    const float *p = partial + ((uint64_t)t * H + h) * splits * (D + 2);
    float m = -3e38f, denom = 0, acc = 0;
    for (unsigned s = 0; s < splits; s++) if (p[s * (D + 2) + 1] > 0) m = fmaxf(m, p[s * (D + 2)]);
    for (unsigned s = 0; s < splits; s++) {
        const float *row = p + s * (D + 2);
        const float w = row[1] > 0 ? expf(row[0] - m) : 0;
        denom += row[1] * w;
        acc += row[2 + d] * w;
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

/* One block of 256 threads per token row; D <= 256 * MAXV. */
template<unsigned MAXV>
__global__ void norm_add(float *x, float *xn, const float *h, const float *part, const float *weights,
        const float *shared, unsigned NS, unsigned pstride, unsigned wstride, const float *w_post,
        const float *w_next, unsigned D, float eps) {
    __shared__ float red[8];
    const unsigned t = blockIdx.x, tid = threadIdx.x;
    float *xr = x + (uint64_t)t * D;
    float v[MAXV];
    if (h || part) {
        float ss = 0;
        for (unsigned i = 0; i < MAXV; i++) {
            const unsigned d = tid + 256 * i;
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
        const float r = rsqrtf(block_sum(ss, red) / D + eps);
        for (unsigned i = 0; i < MAXV; i++) {
            const unsigned d = tid + 256 * i;
            if (d < D) xr[d] += v[i] * r * w_post[d];
        }
        __syncthreads();
    }
    float ss = 0;
    for (unsigned i = 0; i < MAXV; i++) {
        const unsigned d = tid + 256 * i;
        v[i] = d < D ? xr[d] : 0;
        ss += v[i] * v[i];
    }
    const float r = rsqrtf(block_sum(ss, red) / D + eps);
    for (unsigned i = 0; i < MAXV; i++) {
        const unsigned d = tid + 256 * i;
        if (d < D) xn[(uint64_t)t * D + d] = v[i] * r * w_next[d];
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
#define KOLIBRI_NORM_ADD(MAXV) norm_add<MAXV><<<T, 256, 0, 0>>>((float *)x->ptr, (float *)xn->ptr, \
        h ? (const float *)h->ptr : NULL, part ? (const float *)part->ptr : NULL, \
        part ? (const float *)weights->ptr : NULL, shared ? (const float *)shared->ptr : NULL, \
        NS, pstride, wstride, w_post, w_next, D, eps)
    if (D <= 256 * 4) KOLIBRI_NORM_ADD(4);
    else if (D <= 256 * 10) KOLIBRI_NORM_ADD(10);
    else KOLIBRI_NORM_ADD(16);
#undef KOLIBRI_NORM_ADD
    return launched();
}
