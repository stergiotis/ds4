// Microbenchmark: ht (hgn-spec, b = 4) matvec y = diag(sv) Hn M Hn diag(su) x on gfx1151.
// Usage: htmv payload.bin N K su.f32 sv.f32 x.f32 y_out.f32   (correctness, one matrix)
//        htmv --bench N K copies                            (bandwidth, random payload)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <chrono>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); exit(1); } } while (0)

__device__ __forceinline__ float cb1(unsigned s) {
    const unsigned x = s * 0x83DCD12Du;
    const unsigned a = __builtin_amdgcn_udot4(x, 0x01010101u, 0u, false);   // sum of the four bytes
    return __half2float(__float2half_rn(((float)a * 1774.0f - 905216.0f) * (1.0f / 262144.0f)));
}

// x' = Hn (su * x), one block of 128 per workgroup (128 threads)
__global__ void rot_in(float *xr, const float *x, const __half *su, unsigned K) {
    __shared__ float v[128];
    const unsigned b = blockIdx.x, t = threadIdx.x;
    v[t] = x[b * 128 + t] * __half2float(su[b * 128 + t]);
    __syncthreads();
    for (unsigned h = 1; h < 128; h <<= 1) {               // Sylvester FWHT
        const unsigned j = t ^ h;
        const float a = v[t], c = v[j];
        __syncthreads();
        v[t] = (t & h) ? c - a : a + c;
        __syncthreads();
    }
    xr[b * 128 + t] = v[t] * 0.08838834764831845f;          // 1 / sqrt(128)
}

// one workgroup = one band (8 waves = 8 row tiles = 128 rows) x one K chunk of CT column tiles
template<unsigned CT>
__global__ __launch_bounds__(256) void ht_mv(float *y, const unsigned *w, const float *xr, const __half *sv,
                                             unsigned N, unsigned K) {
    const unsigned band = blockIdx.x, chunk = blockIdx.y, wave = threadIdx.x / 32, lane = threadIdx.x & 31;
    const unsigned ctiles = K / 16;
    const unsigned c0 = chunk * CT;
    // tile t = ((band)*(K/16) + ct)*8 + rowtile; 32 words per tile
    const unsigned *wb = w + ((uint64_t)band * ctiles * 8) * 32;
    float acc = 0.f;
    const unsigned col_half = 8 * (lane / 16);
    #pragma unroll 4
    for (unsigned ct = c0; ct < c0 + CT && ct < ctiles; ct++) {
        const unsigned cur = wb[((uint64_t)ct * 8 + wave) * 32 + lane];
        const unsigned prev = __shfl(cur, (lane + 31) & 31, 32);
        const unsigned long long C = ((unsigned long long)prev << 32) | cur;
        const float *xc = xr + ct * 16 + col_half;
        const float4 xa = *(const float4 *)xc, xb = *(const float4 *)(xc + 4);
        const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        #pragma unroll
        for (unsigned j = 0; j < 8; j++) acc += cb1((unsigned)(C >> (28 - 4 * j)) & 0xFFFFu) * xv[j];
    }
    acc += __shfl_xor(acc, 16, 32);                           // lanes l and l+16 hold the same row
    __shared__ float z[128];
    if (lane < 16) z[wave * 16 + lane] = acc;
    __syncthreads();
    // output rotation of the band's 128 partial sums, then sv, then accumulate
    const unsigned t = threadIdx.x;
    for (unsigned h = 1; h < 128; h <<= 1) {
        float a = 0.f, c = 0.f;
        if (t < 128) { a = z[t]; c = z[t ^ h]; }
        __syncthreads();
        if (t < 128) z[t] = (t & h) ? c - a : a + c;
        __syncthreads();
    }
    if (t < 128) atomicAdd(&y[band * 128 + t], z[t] * 0.08838834764831845f * __half2float(sv[band * 128 + t]));
}


typedef _Float16 h2v __attribute__((ext_vector_type(2)));
// x' in f16 for the packed path
__global__ void rot_in16(_Float16 *xr, const float *x, const __half *su, unsigned K) {
    __shared__ float v[128];
    const unsigned b = blockIdx.x, t = threadIdx.x;
    v[t] = x[b * 128 + t] * __half2float(su[b * 128 + t]);
    __syncthreads();
    for (unsigned h = 1; h < 128; h <<= 1) {
        const unsigned j = t ^ h;
        const float a = v[t], c = v[j];
        __syncthreads();
        v[t] = (t & h) ? c - a : a + c;
        __syncthreads();
    }
    xr[b * 128 + t] = (_Float16)(v[t] * 0.08838834764831845f);
}

// two codebook values as a packed f16 pair: (1024 + a) is f16 bits 0x6400 | a, then one packed FMA
__device__ __forceinline__ h2v cb1x2(unsigned s0, unsigned s1) {
    const unsigned a0 = __builtin_amdgcn_udot4(s0 * 0x83DCD12Du, 0x01010101u, 0u, false);
    const unsigned a1 = __builtin_amdgcn_udot4(s1 * 0x83DCD12Du, 0x01010101u, 0u, false);
    const unsigned bits = (0x6400u | a0) | ((0x6400u | a1) << 16);
    h2v v; __builtin_memcpy(&v, &bits, 4);
    const unsigned short m16 = 0x1EEE, c16 = 0xC931;
    _Float16 m, c; __builtin_memcpy(&m, &m16, 2); __builtin_memcpy(&c, &c16, 2);
    return v * (h2v){m, m} + (h2v){c, c};
}

template<unsigned CT>
__global__ __launch_bounds__(256) void ht_mv2(float *y, const unsigned *w, const _Float16 *xr, const __half *sv,
                                              unsigned N, unsigned K) {
    const unsigned band = blockIdx.x, chunk = blockIdx.y, wave = threadIdx.x / 32, lane = threadIdx.x & 31;
    const unsigned ctiles = K / 16, c0 = chunk * CT;
    const unsigned *wb = w + ((uint64_t)band * ctiles * 8) * 32;
    float acc = 0.f;
    const unsigned col_half = 8 * (lane / 16);
    #pragma unroll 4
    for (unsigned ct = c0; ct < c0 + CT && ct < ctiles; ct++) {
        const unsigned cur = wb[((uint64_t)ct * 8 + wave) * 32 + lane];
        const unsigned prev = __shfl(cur, (lane + 31) & 31, 32);
        const uint4 xq = *(const uint4 *)(xr + ct * 16 + col_half);   // 8 f16 x values
        const unsigned xw[4] = {xq.x, xq.y, xq.z, xq.w};
        #pragma unroll
        for (unsigned j = 0; j < 8; j += 2) {
            // state j: bits (28 - 4j) .. of prev:cur  ->  alignbit(prev, cur, 28 - 4j) & 0xffff
            const unsigned s0 = __builtin_amdgcn_alignbit(prev, cur, 28 - 4 * j) & 0xFFFFu;
            const unsigned s1 = __builtin_amdgcn_alignbit(prev, cur, 24 - 4 * j) & 0xFFFFu;
            h2v xv; __builtin_memcpy(&xv, &xw[j / 2], 4);
            acc = __builtin_amdgcn_fdot2(cb1x2(s0, s1), xv, acc, false);
        }
    }
    acc += __shfl_xor(acc, 16, 32);
    __shared__ float z[128];
    if (lane < 16) z[wave * 16 + lane] = acc;
    __syncthreads();
    const unsigned t = threadIdx.x;
    for (unsigned h = 1; h < 128; h <<= 1) {
        float a = 0.f, c = 0.f;
        if (t < 128) { a = z[t]; c = z[t ^ h]; }
        __syncthreads();
        if (t < 128) z[t] = (t & h) ? c - a : a + c;
        __syncthreads();
    }
    if (t < 128) atomicAdd(&y[band * 128 + t], z[t] * 0.08838834764831845f * __half2float(sv[band * 128 + t]));
}

// ht_mv2 with the words of U column tiles loaded before any is decoded
template<unsigned CT, unsigned U>
__global__ __launch_bounds__(256) void ht_mv3(float *y, const unsigned *w, const _Float16 *xr, const __half *sv,
                                              unsigned N, unsigned K) {
    const unsigned band = blockIdx.x, chunk = blockIdx.y, wave = threadIdx.x / 32, lane = threadIdx.x & 31;
    const unsigned ctiles = K / 16, c0 = chunk * CT;
    const unsigned *wb = w + ((uint64_t)band * ctiles * 8) * 32 + wave * 32 + lane;
    float acc = 0.f;
    const unsigned col_half = 8 * (lane / 16);
    for (unsigned ct = c0; ct < c0 + CT; ct += U) {
        unsigned cur[U];
        uint4 xq[U];
        #pragma unroll
        for (unsigned u = 0; u < U; u++) {
            cur[u] = wb[(uint64_t)(ct + u) * 256];
            xq[u] = *(const uint4 *)(xr + (ct + u) * 16 + col_half);
        }
        #pragma unroll
        for (unsigned u = 0; u < U; u++) {
            const unsigned prev = __shfl(cur[u], (lane + 31) & 31, 32);
            const unsigned xw[4] = {xq[u].x, xq[u].y, xq[u].z, xq[u].w};
            #pragma unroll
            for (unsigned j = 0; j < 8; j += 2) {
                const unsigned s0 = __builtin_amdgcn_alignbit(prev, cur[u], 28 - 4 * j) & 0xFFFFu;
                const unsigned s1 = __builtin_amdgcn_alignbit(prev, cur[u], 24 - 4 * j) & 0xFFFFu;
                h2v xv; __builtin_memcpy(&xv, &xw[j / 2], 4);
                acc = __builtin_amdgcn_fdot2(cb1x2(s0, s1), xv, acc, false);
            }
        }
    }
    acc += __shfl_xor(acc, 16, 32);
    __shared__ float z[128];
    if (lane < 16) z[wave * 16 + lane] = acc;
    __syncthreads();
    const unsigned t = threadIdx.x;
    for (unsigned h = 1; h < 128; h <<= 1) {
        float a = 0.f, c = 0.f;
        if (t < 128) { a = z[t]; c = z[t ^ h]; }
        __syncthreads();
        if (t < 128) z[t] = (t & h) ? c - a : a + c;
        __syncthreads();
    }
    if (t < 128) atomicAdd(&y[band * 128 + t], z[t] * 0.08838834764831845f * __half2float(sv[band * 128 + t]));
}

template<typename T> std::vector<T> load(const char *p, size_t n) {
    std::vector<T> v(n); FILE *f = fopen(p, "rb"); if (!f || fread(v.data(), sizeof(T), n, f) != n) { fprintf(stderr, "read %s\n", p); exit(1); } fclose(f); return v;
}

int main(int argc, char **argv) {
    constexpr unsigned CT = 32;
    if (argc >= 5 && !strcmp(argv[1], "--bench")) {
        const unsigned N = atoi(argv[2]), K = atoi(argv[3]), copies = atoi(argv[4]);
        const bool norot = argc > 5 && atoi(argv[5]);
        const unsigned ctv = argc > 6 ? atoi(argv[6]) : 32;
        const int ver = argc > 7 ? atoi(argv[7]) : 1;
        const bool v2 = ver == 2;
        const bool v3 = ver >= 3;
        _Float16 *xr16; CK(hipMalloc(&xr16, K * 2));
        const size_t bytes = (size_t)N * K / 2;
        unsigned *w; float *x, *xr, *y; __half *su, *sv;
        CK(hipMalloc(&w, bytes * copies)); CK(hipMalloc(&x, K * 4)); CK(hipMalloc(&xr, K * 4));
        CK(hipMalloc(&y, N * 4 * copies)); CK(hipMalloc(&su, K * 2)); CK(hipMalloc(&sv, N * 2));
        std::vector<unsigned> h(bytes / 4); srand(1); for (auto &v : h) v = (unsigned)rand() * 2654435761u ^ rand();
        for (unsigned c = 0; c < copies; c++) CK(hipMemcpy((char *)w + c * bytes, h.data(), bytes, hipMemcpyHostToDevice));
        std::vector<float> hx(K, 0.01f); CK(hipMemcpy(x, hx.data(), K * 4, hipMemcpyHostToDevice));
        std::vector<__half> hs(std::max(N, K), __float2half(1.f));
        CK(hipMemcpy(su, hs.data(), K * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(sv, hs.data(), N * 2, hipMemcpyHostToDevice));
        const unsigned ct = ctv;
        const dim3 g(N / 128, (K / 16 + ct - 1) / ct);
#define LAUNCH(c) do { const unsigned *wc = (const unsigned *)((char *)w + (c) * bytes); \
            if (v3) { if (!norot) rot_in16<<<K / 128, 128>>>(xr16, x, su, K); \
                      if (ver == 3) ht_mv3<16, 4><<<g, 256>>>(y + (c) * N, wc, xr16, sv, N, K); \
                      else if (ver == 4) ht_mv3<16, 8><<<g, 256>>>(y + (c) * N, wc, xr16, sv, N, K); \
                      else ht_mv3<32, 8><<<dim3(N / 128, K / 16 / 32), 256>>>(y + (c) * N, wc, xr16, sv, N, K); break; } \
            if (v2) { if (!norot) rot_in16<<<K / 128, 128>>>(xr16, x, su, K); \
                      if (ct == 16) ht_mv2<16><<<g, 256>>>(y + (c) * N, wc, xr16, sv, N, K); \
                      else ht_mv2<64><<<g, 256>>>(y + (c) * N, wc, xr16, sv, N, K); break; } \
            if (!norot) rot_in<<<K / 128, 128>>>(xr, x, su, K); \
            if (ct == 8) ht_mv<8><<<g, 256>>>(y + (c) * N, wc, xr, sv, N, K); \
            else if (ct == 16) ht_mv<16><<<g, 256>>>(y + (c) * N, wc, xr, sv, N, K); \
            else if (ct == 64) ht_mv<64><<<g, 256>>>(y + (c) * N, wc, xr, sv, N, K); \
            else ht_mv<32><<<g, 256>>>(y + (c) * N, wc, xr, sv, N, K); } while (0)
        for (int it = 0; it < 3; it++)
            for (unsigned c = 0; c < copies; c++) LAUNCH(c);
        CK(hipDeviceSynchronize());
        const int reps = 10;
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < reps; it++)
            for (unsigned c = 0; c < copies; c++) LAUNCH(c);
        CK(hipDeviceSynchronize());
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("ht matvec N=%u K=%u x%u rot=%d ct=%u: %.3f ms per pass of %.1f MB, %.1f GB/s\n", N, K, copies, !norot, ct, 1e3 * s / reps,
               bytes * copies / 1e6, bytes * copies * reps / s / 1e9);
        return 0;
    }
    if (argc < 8) { fprintf(stderr, "usage\n"); return 2; }
    const unsigned N = atoi(argv[2]), K = atoi(argv[3]);
    auto hw = load<unsigned>(argv[1], (size_t)N * K / 8);
    auto hsu = load<float>(argv[4], K), hsv = load<float>(argv[5], N), hx = load<float>(argv[6], K);
    std::vector<__half> su16(K), sv16(N); for (unsigned i = 0; i < K; i++) su16[i] = __float2half(hsu[i]); for (unsigned i = 0; i < N; i++) sv16[i] = __float2half(hsv[i]);
    unsigned *w; float *x, *xr, *y; __half *su, *sv;
    CK(hipMalloc(&w, hw.size() * 4)); CK(hipMalloc(&x, K * 4)); CK(hipMalloc(&xr, K * 4)); CK(hipMalloc(&y, N * 4));
    CK(hipMalloc(&su, K * 2)); CK(hipMalloc(&sv, N * 2));
    CK(hipMemcpy(w, hw.data(), hw.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(x, hx.data(), K * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(su, su16.data(), K * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(sv, sv16.data(), N * 2, hipMemcpyHostToDevice));
    CK(hipMemset(y, 0, N * 4));
    if (getenv("HTV2")) {
        _Float16 *xr16; CK(hipMalloc(&xr16, K * 2));
        rot_in16<<<K / 128, 128>>>(xr16, x, su, K);
        ht_mv3<16, 8><<<dim3(N / 128, (K / 16 + 15) / 16), 256>>>(y, w, xr16, sv, N, K);
    } else {
        rot_in<<<K / 128, 128>>>(xr, x, su, K);
        ht_mv<CT><<<dim3(N / 128, (K / 16 + CT - 1) / CT), 256>>>(y, w, xr, sv, N, K);
    }
    CK(hipDeviceSynchronize());
    std::vector<float> hy(N); CK(hipMemcpy(hy.data(), y, N * 4, hipMemcpyDeviceToHost));
    FILE *f = fopen(argv[7], "wb"); fwrite(hy.data(), 4, N, f); fclose(f);
    printf("wrote %u outputs\n", N);
    return 0;
}
