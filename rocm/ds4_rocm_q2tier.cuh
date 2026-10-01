/* Q2 tier compute: the experts that ds4_gpu_q2tier_request() registered for
 * this decode step, read by the selected load into g_q2tier.buf, run through
 * the IQ2_XXS gate/up and Q2_K down decode kernels with their own gate
 * weights. The result is a separate routed partial that the caller adds. */
static struct {
    int32_t *sel;
    float *weights;
    cuda_block_q8_K *xq;
    float *gate;
    float *up;
    float *mid;
    uint32_t mid_dim;
    uint32_t xq_blocks;
} g_q2tier_scratch;

struct q2tier_args {
    float weights[DS4_ROCM_N_EXPERT_USED];
    uint32_t n;
};

/* Writes the compact ids 0..n-1 and the Q2 gate weights from kernel
 * arguments, so no host-to-device copy blocks the host. */
__global__ static void q2tier_set_args_kernel(int32_t *sel, float *weights, q2tier_args a) {
    const uint32_t i = threadIdx.x;
    if (i < a.n) {
        sel[i] = (int32_t)i;
        weights[i] = a.weights[i];
    }
}

__global__ static void q2tier_zero_weights_kernel(float *weights, uint32_t mask, uint32_t n) {
    const uint32_t i = threadIdx.x;
    if (i < n && (mask & (1u << i))) weights[i] = 0.0f;
}

/* Zeroes the router weights of the slots in mask, in stream order. */
extern "C" int ds4_gpu_q2tier_zero_weights(ds4_gpu_tensor *weights, uint32_t mask, uint32_t n) {
    if (!weights || n == 0 || n > DS4_ROCM_N_EXPERT_USED ||
        !cuda_tensor_has_elems2(weights, 1, n, sizeof(float))) {
        return 0;
    }
    q2tier_zero_weights_kernel<<<1, 32>>>((float *)weights->ptr, mask, n);
    return cuda_ok(cudaGetLastError(), "q2 tier zero weights");
}

extern "C" int ds4_gpu_q2tier_moe_one(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *x,
        const float *weights,
        uint32_t n,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        float clamp) {
    if (!out || !x || !weights || n == 0 || n > DS4_ROCM_N_EXPERT_USED ||
        n != g_q2tier.loaded_n || !g_q2tier.buf ||
        gate_expert_bytes != g_q2tier.gate_expert_bytes ||
        down_expert_bytes != g_q2tier.down_expert_bytes ||
        expert_in_dim % CUDA_QK_K != 0 || expert_mid_dim % CUDA_QK_K != 0 ||
        expert_in_dim / CUDA_QK_K > 16u ||
        !cuda_tensor_has_elems2(x, 1, expert_in_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(out, 1, out_dim, sizeof(float))) {
        return 0;
    }
    /* The regular routed launch normally finished the shared read set. */
    if (g_stream_selected_pending.active && !cuda_stream_selected_finish_pending_missing(0)) {
        return 0;
    }
    g_q2tier.loaded_n = 0;
    auto &s = g_q2tier_scratch;
    const uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    if (!s.sel || s.mid_dim < expert_mid_dim || s.xq_blocks < xq_blocks) {
        (void)cudaFree(s.sel);
        (void)cudaFree(s.weights);
        (void)cudaFree(s.xq);
        (void)cudaFree(s.gate);
        (void)cudaFree(s.up);
        (void)cudaFree(s.mid);
        memset(&s, 0, sizeof(s));
        const size_t fbytes = (size_t)DS4_ROCM_N_EXPERT_USED * expert_mid_dim * sizeof(float);
        if (cudaMalloc((void **)&s.sel, DS4_ROCM_N_EXPERT_USED * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void **)&s.weights, DS4_ROCM_N_EXPERT_USED * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void **)&s.xq, (size_t)xq_blocks * sizeof(cuda_block_q8_K)) != cudaSuccess ||
            cudaMalloc((void **)&s.gate, fbytes) != cudaSuccess ||
            cudaMalloc((void **)&s.up, fbytes) != cudaSuccess ||
            cudaMalloc((void **)&s.mid, fbytes) != cudaSuccess) {
            (void)cudaGetLastError();
            return 0;
        }
        s.mid_dim = expert_mid_dim;
        s.xq_blocks = xq_blocks;
    }
    q2tier_args args;
    memset(&args, 0, sizeof(args));
    for (uint32_t i = 0; i < n; i++) args.weights[i] = weights[i];
    args.n = n;
    q2tier_set_args_kernel<<<1, 32>>>(s.sel, s.weights, args);
    if (!cuda_ok(cudaGetLastError(), "q2 tier args")) return 0;
    const char *gate_base = g_q2tier.buf;
    const char *up_base = gate_base + (uint64_t)DS4_ROCM_N_EXPERT_USED * gate_expert_bytes;
    const char *down_base = up_base + (uint64_t)DS4_ROCM_N_EXPERT_USED * gate_expert_bytes;

    q8_K_quantize_kernel<<<dim3(xq_blocks, 1, 1), 256>>>(s.xq, (const float *)x->ptr, expert_in_dim, 1);
    if (!cuda_ok(cudaGetLastError(), "q2 tier x quantize")) return 0;
    moe_gate_up_mid_decode_lut_qwarp32_kernel<<<dim3((expert_mid_dim + 127u) / 128u, n, 1), 256>>>(
            s.gate, s.up, s.mid, gate_base, up_base, s.xq, s.sel, s.weights,
            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n,
            0u, 0xffffffffu, clamp);
    if (!cuda_ok(cudaGetLastError(), "q2 tier gate/up")) return 0;
    uint32_t rows_per_block = cuda_runtime_config()->moe_decode_down_rpb;
    if (rows_per_block == 0u) rows_per_block = 1u;
    moe_down_q2K_sum_rows_w32_kernel<<<dim3((out_dim + rows_per_block - 1u) / rows_per_block, 1, 1),
                                       rows_per_block * 32u>>>(
            (float *)out->ptr, down_base, s.mid, s.sel, 1, expert_mid_dim, out_dim,
            down_expert_bytes, down_row_bytes, n);
    return cuda_ok(cudaGetLastError(), "q2 tier down");
}
