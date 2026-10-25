/* GLM-5.3 one-token routed-MoE decode benchmark (ROCm).
 *
 * Times ds4_gpu_routed_moe_one_tensor, the call the GLM resident decode makes
 * once per MoE layer, at the Q2 file's shapes: 288 experts, top-8, IQ2_XXS
 * gate/up 4096 -> 2048, Q2_K down 2048 -> 4096, SwiGLU clamp 10. Each call
 * takes the next of 64 fixed pseudo-random expert selections, so the 2 GiB of
 * synthetic experts are read the way decode reads them, mostly cold.
 *
 * It also prints an FNV-1a hash over every call's output: two builds or
 * kernel variants produce bit-identical results exactly when the hashes match.
 *
 * Usage: tests/bench_glm_moe_rocm [rounds]
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "ds4_gpu.h"

#define TYPE_Q2_K 10u
#define TYPE_IQ2_XXS 16u
#define N_TOTAL_EXPERT 288u
#define N_EXPERT 8u
#define IN_DIM 4096u
#define MID_DIM 2048u
#define CLAMP 10.0f
#define N_SELECTIONS 64u

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t rng_next(uint64_t *s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return *s;
}

/* Random quants; fp16 scales at the given offsets of each block are set to
 * small finite values so no call meets NaN or denormal scales. */
static void fill_blocks(uint8_t *p, uint64_t bytes, uint64_t block,
                        const uint32_t *scale_off, uint32_t n_scale, uint64_t *rng) {
    for (uint64_t i = 0; i < bytes; i += 8u) {
        const uint64_t v = rng_next(rng);
        memcpy(p + i, &v, bytes - i < 8u ? bytes - i : 8u);
    }
    for (uint64_t b = 0; b + block <= bytes; b += block) {
        for (uint32_t k = 0; k < n_scale; k++) {
            const uint16_t h = (uint16_t)(0x1000u + (rng_next(rng) >> 57u));
            memcpy(p + b + scale_off[k], &h, 2u);
        }
    }
}

static uint64_t fnv1a(uint64_t h, const void *p, uint64_t n) {
    const uint8_t *b = (const uint8_t *)p;
    for (uint64_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

int main(int argc, char **argv) {
    const uint32_t rounds = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 8u;
    const uint64_t gate_row_bytes = IN_DIM / 256u * 66u;   /* IQ2_XXS */
    const uint64_t down_row_bytes = MID_DIM / 256u * 84u;  /* Q2_K */
    const uint64_t gate_expert_bytes = MID_DIM * gate_row_bytes;
    const uint64_t down_expert_bytes = IN_DIM * down_row_bytes;
    const uint64_t gate_bytes = N_TOTAL_EXPERT * gate_expert_bytes;
    const uint64_t down_bytes = N_TOTAL_EXPERT * down_expert_bytes;
    const uint64_t gate_off = 0u;
    const uint64_t up_off = (gate_bytes + 4095u) & ~4095ull;
    const uint64_t down_off = (up_off + gate_bytes + 4095u) & ~4095ull;
    const uint64_t model_size = (down_off + down_bytes + 4095u) & ~4095ull;
    const double call_bytes = (double)N_EXPERT * (2.0 * gate_expert_bytes + down_expert_bytes);

    FILE *model_file = tmpfile();
    void *model = MAP_FAILED;
    if (model_file && ftruncate(fileno(model_file), (off_t)model_size) == 0) {
        model = mmap(NULL, (size_t)model_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED, fileno(model_file), 0);
    }
    if (model == MAP_FAILED) {
        fprintf(stderr, "bench: model image allocation failed\n");
        return 1;
    }
    uint64_t rng = 0x243f6a8885a308d3ull;
    const uint32_t iq2_scale[] = { 0u };          /* block_iq2_xxs: d, qs[32] */
    const uint32_t q2k_scale[] = { 80u, 82u };    /* block_q2_K: scales, qs, d, dmin */
    fill_blocks((uint8_t *)model + gate_off, gate_bytes, 66u, iq2_scale, 1u, &rng);
    fill_blocks((uint8_t *)model + up_off, gate_bytes, 66u, iq2_scale, 1u, &rng);
    fill_blocks((uint8_t *)model + down_off, down_bytes, 84u, q2k_scale, 2u, &rng);

    if (!ds4_gpu_init()) {
        fprintf(stderr, "bench: ds4_gpu_init failed\n");
        return 1;
    }
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    const uint64_t offsets[] = { gate_off, up_off, down_off };
    const uint64_t sizes[] = { gate_bytes, gate_bytes, down_bytes };
    if (!ds4_gpu_set_model_map(model, model_size) ||
        !ds4_gpu_set_model_fd(fileno(model_file)) ||
        !ds4_gpu_set_model_map_spans(model, model_size, offsets, sizes, 3u, down_bytes)) {
        fprintf(stderr, "bench: model cache setup failed\n");
        return 1;
    }

    int32_t sel[N_SELECTIONS * N_EXPERT];
    float wts[N_SELECTIONS * N_EXPERT];
    for (uint32_t s = 0; s < N_SELECTIONS; s++) {
        uint32_t pool[N_TOTAL_EXPERT];
        for (uint32_t e = 0; e < N_TOTAL_EXPERT; e++) pool[e] = e;
        for (uint32_t k = 0; k < N_EXPERT; k++) {
            const uint32_t j = k + (uint32_t)((rng_next(&rng) >> 33u) % (N_TOTAL_EXPERT - k));
            const uint32_t t = pool[k]; pool[k] = pool[j]; pool[j] = t;
            sel[s * N_EXPERT + k] = (int32_t)pool[k];
            wts[s * N_EXPERT + k] = 0.05f + (float)(rng_next(&rng) >> 40u) / (float)(1u << 24u) * 0.2f;
        }
    }
    float x[IN_DIM];
    for (uint32_t i = 0; i < IN_DIM; i++) {
        x[i] = ((double)(rng_next(&rng) >> 40u) / (double)(1ull << 24u) - 0.5) * 2.0;
    }
    const uint64_t pair_floats = (uint64_t)N_EXPERT * IN_DIM;
    ds4_gpu_tensor *sel_t = ds4_gpu_tensor_alloc(sizeof(sel));
    ds4_gpu_tensor *wts_t = ds4_gpu_tensor_alloc(sizeof(wts));
    ds4_gpu_tensor *x_t = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(IN_DIM * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(pair_floats * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(pair_floats * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(pair_floats * sizeof(float));
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(pair_floats * sizeof(float));
    ds4_gpu_tensor *sel_v[N_SELECTIONS], *wts_v[N_SELECTIONS];
    if (!sel_t || !wts_t || !x_t || !out || !gate || !up || !mid || !down ||
        !ds4_gpu_tensor_write(sel_t, 0u, sel, sizeof(sel)) ||
        !ds4_gpu_tensor_write(wts_t, 0u, wts, sizeof(wts)) ||
        !ds4_gpu_tensor_write(x_t, 0u, x, sizeof(x))) {
        fprintf(stderr, "bench: tensor setup failed\n");
        return 1;
    }
    for (uint32_t s = 0; s < N_SELECTIONS; s++) {
        sel_v[s] = ds4_gpu_tensor_view(sel_t, (uint64_t)s * N_EXPERT * 4u, N_EXPERT * 4u);
        wts_v[s] = ds4_gpu_tensor_view(wts_t, (uint64_t)s * N_EXPERT * 4u, N_EXPERT * 4u);
        if (!sel_v[s] || !wts_v[s]) return 1;
    }

#define CALL(s) ds4_gpu_routed_moe_one_tensor( \
        out, gate, up, mid, down, model, model_size, gate_off, up_off, down_off, \
        TYPE_IQ2_XXS, TYPE_Q2_K, gate_expert_bytes, gate_row_bytes, \
        down_expert_bytes, down_row_bytes, IN_DIM, MID_DIM, IN_DIM, \
        sel_v[s], wts_v[s], N_TOTAL_EXPERT, N_EXPERT, CLAMP, x_t, NULL, 3u, false)

    /* Pass 1, synced: one call per selection, outputs hashed. */
    uint64_t hash = 1469598103934665603ull;
    float o[IN_DIM];
    double t0 = now_sec();
    for (uint32_t s = 0; s < N_SELECTIONS; s++) {
        if (!CALL(s) || !ds4_gpu_synchronize() || !ds4_gpu_tensor_read(out, 0u, o, sizeof(o))) {
            fprintf(stderr, "bench: call %u failed\n", s);
            return 1;
        }
        hash = fnv1a(hash, o, sizeof(o));
    }
    const double synced = (now_sec() - t0) / N_SELECTIONS;
    /* Pass 2, queued: rounds x 64 calls back to back, one sync. */
    t0 = now_sec();
    for (uint32_t r = 0; r < rounds; r++) {
        for (uint32_t s = 0; s < N_SELECTIONS; s++) {
            if (!CALL(s)) return 1;
        }
    }
    if (!ds4_gpu_synchronize()) return 1;
    const double queued = (now_sec() - t0) / ((double)rounds * N_SELECTIONS);
    printf("routed moe decode: %.1f MiB/call, queued %.1f us (%.1f GB/s), "
           "synced %.1f us, output hash %016llx\n",
           call_bytes / 1048576.0, queued * 1e6, call_bytes / queued / 1e9,
           synced * 1e6, (unsigned long long)hash);
    ds4_gpu_cleanup();
    return 0;
}
