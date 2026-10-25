/* GLM-5.3 one-token decode matvec benchmark (ROCm).
 *
 * Times ds4_gpu_matmul_quant_decode_mpp_model_view_tensor, the entry the GLM
 * resident decode uses for its dense projections, at the model's Q8_0 and
 * Q4_K shapes. Each shape gets enough synthetic copies (>= 256 MiB) that the
 * calls rotate through memory instead of hitting the 32 MiB MALL, so the
 * GB/s column is DRAM bandwidth as decode sees it.
 *
 * "queued" issues all calls back to back with one sync at the end (decode
 * without host syncs); "synced" waits after every call, which adds the launch
 * and completion latency a lone kernel pays.
 *
 * Not a correctness harness: weights are random, only kernel cost matters.
 *
 * Usage: tests/bench_glm_matvec_rocm [rounds]
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "ds4_gpu.h"

#define TYPE_Q8_0 8u
#define TYPE_Q4_K 12u
#define MIN_SHAPE_BYTES (256ull << 20)

typedef struct {
    const char *name;
    uint32_t type;
    uint32_t in_dim;
    uint32_t out_dim;
    uint64_t row_bytes;
    uint64_t bytes;
    uint32_t copies;
    uint64_t offset;
} shape;

static shape shapes[] = {
    { "kda_v",          TYPE_Q8_0,  4096u,   8192u, 0, 0, 0, 0 },
    { "kda_output",     TYPE_Q8_0,  8192u,   4096u, 0, 0, 0, 0 },
    { "kda_q/k",        TYPE_Q4_K,  4096u,   8192u, 0, 0, 0, 0 },
    { "attn_output",    TYPE_Q8_0, 16384u,   4096u, 0, 0, 0, 0 },
    { "attn_q_b",       TYPE_Q8_0,  1536u,  16384u, 0, 0, 0, 0 },
    { "attn_q_a",       TYPE_Q8_0,  4096u,   1536u, 0, 0, 0, 0 },
    { "kv_a_mqa",       TYPE_Q8_0,  4096u,    512u, 0, 0, 0, 0 },
    { "shexp gate/up",  TYPE_Q8_0,  4096u,   2048u, 0, 0, 0, 0 },
    { "shexp down",     TYPE_Q8_0,  2048u,   4096u, 0, 0, 0, 0 },
    { "dense down",     TYPE_Q8_0, 12288u,   4096u, 0, 0, 0, 0 },
    { "kda_f_a/g_a",    TYPE_Q8_0,  4096u,    128u, 0, 0, 0, 0 },
    { "kda_beta",       TYPE_Q8_0,  4096u,     64u, 0, 0, 0, 0 },
    { "output head",    TYPE_Q8_0,  4096u, 154880u, 0, 0, 0, 0 },
};
#define N_SHAPES (sizeof(shapes) / sizeof(shapes[0]))

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t rng_next(uint64_t *s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return *s;
}

/* Random quants with small, finite fp16 scales: kernel cost does not depend
 * on the values, but NaN/denormal scales could take slow paths. */
static void fill_rows(uint8_t *p, const shape *s, uint64_t *rng) {
    for (uint64_t i = 0; i < s->bytes; i += 8u) {
        const uint64_t v = rng_next(rng);
        memcpy(p + i, &v, s->bytes - i < 8u ? s->bytes - i : 8u);
    }
    const uint64_t block = s->type == TYPE_Q8_0 ? 34u : 144u;
    const uint32_t scales = s->type == TYPE_Q8_0 ? 1u : 2u;
    for (uint64_t b = 0; b < s->bytes; b += block) {
        for (uint32_t k = 0; k < scales; k++) {
            const uint16_t h = (uint16_t)(0x1400u + (rng_next(rng) >> 56u));
            memcpy(p + b + 2u * k, &h, 2u);
        }
    }
}

static int run(const shape *s, ds4_gpu_tensor *out, ds4_gpu_tensor *x,
               const void *model, uint64_t model_size, uint32_t calls, int synced) {
    for (uint32_t c = 0; c < calls; c++) {
        const uint64_t off = s->offset + (uint64_t)(c % s->copies) * s->bytes;
        if (!ds4_gpu_matmul_quant_decode_mpp_model_view_tensor(
                    out, model, model_size, off, s->type, s->in_dim, s->out_dim, x, 1)) {
            return 0;
        }
        if (synced && !ds4_gpu_synchronize()) return 0;
    }
    return ds4_gpu_synchronize();
}

int main(int argc, char **argv) {
    const uint32_t rounds = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 4u;
    uint64_t model_size = 0, max_in = 0, max_out = 0;
    for (uint32_t i = 0; i < N_SHAPES; i++) {
        shape *s = &shapes[i];
        s->row_bytes = s->type == TYPE_Q8_0 ? (uint64_t)s->in_dim / 32u * 34u
                                            : (uint64_t)s->in_dim / 256u * 144u;
        s->bytes = s->row_bytes * s->out_dim;
        s->copies = (uint32_t)((MIN_SHAPE_BYTES + s->bytes - 1u) / s->bytes);
        s->offset = model_size;
        model_size += (s->bytes * s->copies + 4095u) & ~4095ull;
        if (s->in_dim > max_in) max_in = s->in_dim;
        if (s->out_dim > max_out) max_out = s->out_dim;
    }
    fprintf(stderr, "bench: model image %.2f GiB, %u rounds\n",
            (double)model_size / 1073741824.0, rounds);

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
    uint64_t offsets[N_SHAPES], sizes[N_SHAPES], max_tensor = 0;
    for (uint32_t i = 0; i < N_SHAPES; i++) {
        const shape *s = &shapes[i];
        for (uint32_t c = 0; c < s->copies; c++) {
            fill_rows((uint8_t *)model + s->offset + (uint64_t)c * s->bytes, s, &rng);
        }
        offsets[i] = s->offset;
        sizes[i] = s->bytes * s->copies;
        if (sizes[i] > max_tensor) max_tensor = sizes[i];
    }

    if (!ds4_gpu_init()) {
        fprintf(stderr, "bench: ds4_gpu_init failed\n");
        return 1;
    }
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    if (!ds4_gpu_set_model_map(model, model_size) ||
        !ds4_gpu_set_model_fd(fileno(model_file)) ||
        !ds4_gpu_set_model_map_spans(model, model_size, offsets, sizes,
                                     (uint32_t)N_SHAPES, max_tensor)) {
        fprintf(stderr, "bench: model cache setup failed\n");
        return 1;
    }
    float *xh = (float *)malloc(max_in * sizeof(float));
    for (uint64_t i = 0; i < max_in; i++) {
        xh[i] = ((double)(rng_next(&rng) >> 40u) / (double)(1ull << 24u) - 0.5) * 0.25;
    }
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(max_in * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(max_out * sizeof(float));
    if (!xh || !x || !out || !ds4_gpu_tensor_write(x, 0u, xh, max_in * sizeof(float))) {
        fprintf(stderr, "bench: tensor setup failed\n");
        return 1;
    }

    printf("%-14s %5s %6s %7s %8s %10s %8s %10s\n",
           "shape", "type", "in", "out", "MiB", "queued us", "GB/s", "synced us");
    for (uint32_t i = 0; i < N_SHAPES; i++) {
        const shape *s = &shapes[i];
        const uint32_t calls = s->copies * rounds;
        if (!run(s, out, x, model, model_size, s->copies, 0)) { /* warm-up */
            fprintf(stderr, "bench: %s failed\n", s->name);
            return 1;
        }
        double t0 = now_sec();
        if (!run(s, out, x, model, model_size, calls, 0)) return 1;
        const double queued = (now_sec() - t0) / calls;
        t0 = now_sec();
        if (!run(s, out, x, model, model_size, calls, 1)) return 1;
        const double synced = (now_sec() - t0) / calls;
        printf("%-14s %5s %6u %7u %8.1f %10.1f %8.1f %10.1f\n",
               s->name, s->type == TYPE_Q8_0 ? "q8_0" : "q4_K",
               s->in_dim, s->out_dim, (double)s->bytes / 1048576.0,
               queued * 1e6, (double)s->bytes / queued / 1e9, synced * 1e6);
    }
    ds4_gpu_cleanup();
    return 0;
}
