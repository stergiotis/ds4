/* Model-backed oracle for Qwen3.8 multi-session decode batches on ROCm.
 *
 * N sessions (DS4_TEST_SESSION_COUNT, default 4) prefill different prompts
 * and then advance DS4_TEST_STEPS greedy tokens together through
 * ds4_sessions_eval_batch; every frontier's logits are archived.  The
 * batched sessions are freed and each prompt is replayed alone with the
 * same tokens.  Batching changes the order of floating-point reductions, so
 * the logits are not expected to match bit for bit; the test reports the
 * largest absolute logit difference, the mean KL of the softmax and the
 * argmax agreement per session, and fails on non-finite logits, a KL above
 * DS4_TEST_MAX_KL (default 0.02) or more than one argmax disagreement.
 *
 *   DS4_TEST_MODEL=/path/to/qwen.gguf ./tests/test_rocm_qwen_batch
 */
#include "ds4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

#define MAX_SESSIONS 16

static const char *prompts[] = {
    "Write a short paragraph about the history of the bicycle.",
    "Explain in simple terms how a heat pump works.",
    "List five practical tips for learning a new programming language.",
    "Describe a quiet morning in a mountain village.",
    "What are the main differences between TCP and UDP?",
    "Summarize the plot of a classic adventure novel of your choice.",
    "Give a recipe for a simple vegetable soup.",
    "Why is the sky blue? Answer in three sentences.",
};

static double kl(const float *p_logits, const float *q_logits, int n) {
    double mp = -INFINITY, mq = -INFINITY;
    for (int i = 0; i < n; i++) { if (p_logits[i] > mp) mp = p_logits[i]; if (q_logits[i] > mq) mq = q_logits[i]; }
    double zp = 0, zq = 0;
    for (int i = 0; i < n; i++) { zp += exp(p_logits[i] - mp); zq += exp(q_logits[i] - mq); }
    double k = 0;
    for (int i = 0; i < n; i++) {
        const double lp = p_logits[i] - mp - log(zp), lq = q_logits[i] - mq - log(zq);
        k += exp(lp) * (lp - lq);
    }
    return k;
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) { fprintf(stderr, "FAIL: DS4_TEST_MODEL is not set\n"); return 1; }
    const int n = getenv("DS4_TEST_SESSION_COUNT") ? atoi(getenv("DS4_TEST_SESSION_COUNT")) : 4;
    const int steps = getenv("DS4_TEST_STEPS") ? atoi(getenv("DS4_TEST_STEPS")) : 32;
    const double max_kl = getenv("DS4_TEST_MAX_KL") ? atof(getenv("DS4_TEST_MAX_KL")) : 0.02;
    const int ctx = 2048;
    if (n < 2 || n > MAX_SESSIONS || steps < 1) { fprintf(stderr, "FAIL: bad session count or steps\n"); return 1; }

    ds4_engine_options opt = {
        .model_path = model,
        .backend = DS4_BACKEND_CUDA,
        .context_size = ctx,
        .n_threads = 1,
        .share_session_prefill_workspace = true,
        .placement_ctx_hint = (uint32_t)ctx,
        .placement_session_count_hint = n,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) { fprintf(stderr, "FAIL: engine open\n"); return 1; }
    const int vocab = ds4_engine_vocab_size(engine);
    char err[256] = {0};

    ds4_session *s[MAX_SESSIONS] = {0};
    ds4_tokens prompt[MAX_SESSIONS] = {0};
    for (int i = 0; i < n; i++) {
        ds4_encode_chat_prompt(engine, NULL, prompts[i % 8], DS4_THINK_NONE, &prompt[i]);
        if (ds4_session_create(&s[i], engine, (uint32_t)ctx) != 0 ||
            ds4_session_sync(s[i], &prompt[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "FAIL: prefill %d: %s\n", i, err); return 1;
        }
    }
    float *arch = malloc((size_t)(steps + 1) * n * vocab * sizeof(float));
    int *tok = malloc((size_t)(steps + 1) * n * sizeof(int));
    float *cur = malloc((size_t)vocab * sizeof(float));
    if (!arch || !tok || !cur) { fprintf(stderr, "FAIL: alloc\n"); return 1; }
    double batch_ms = 0; int batch_n = 0;
    for (int step = 0; step <= steps; step++) {
        ds4_decode_item items[MAX_SESSIONS];
        for (int i = 0; i < n; i++) {
            const size_t f = (size_t)step * n + i;
            if (ds4_session_copy_logits(s[i], arch + f * vocab, vocab) != vocab) { fprintf(stderr, "FAIL: copy\n"); return 1; }
            tok[f] = ds4_session_argmax(s[i]);
            items[i] = (ds4_decode_item){ .session = s[i], .token = tok[f] };
        }
        if (step == steps) break;
        const double t0 = now_ms();
        if (ds4_sessions_eval_batch(items, n, err, sizeof(err)) != 0) { fprintf(stderr, "FAIL: batch: %s\n", err); return 1; }
        if (step >= 2) { batch_ms += now_ms() - t0; batch_n++; }
    }
    if (batch_n) printf("batch of %d: %.2f ms per step, %.1f tok/s total\n", n, batch_ms / batch_n, n * 1000.0 * batch_n / batch_ms);
    for (int i = 0; i < n; i++) ds4_session_free(s[i]);

    int failed = 0;
    for (int i = 0; i < n; i++) {
        ds4_session *c = NULL;
        if (ds4_session_create(&c, engine, (uint32_t)ctx) != 0 ||
            ds4_session_sync(c, &prompt[i], err, sizeof(err)) != 0) { fprintf(stderr, "FAIL: control %d: %s\n", i, err); return 1; }
        double sum_kl = 0, worst_kl = 0; float worst_abs = 0; int disagree = 0, nonfinite = 0;
        for (int step = 0; step <= steps; step++) {
            const size_t f = (size_t)step * n + i;
            if (ds4_session_copy_logits(c, cur, vocab) != vocab) { fprintf(stderr, "FAIL: copy\n"); return 1; }
            const float *ref = arch + f * vocab;
            for (int v = 0; v < vocab; v++) {
                const float d = fabsf(cur[v] - ref[v]);
                if (!isfinite(d)) nonfinite++; else if (d > worst_abs) worst_abs = d;
            }
            if (step > 0) {   /* step 0 is the prefill frontier, identical by construction */
                const double k = kl(cur, ref, vocab);
                sum_kl += k; if (k > worst_kl) worst_kl = k;
                if (ds4_session_argmax(c) != tok[f]) disagree++;
            }
            if (step == steps) break;
            if (ds4_session_eval(c, tok[f], err, sizeof(err)) != 0) { fprintf(stderr, "FAIL: control eval: %s\n", err); return 1; }
        }
        const double mean_kl = sum_kl / steps;
        const int bad = nonfinite || mean_kl > max_kl || disagree > 1;
        failed |= bad;
        printf("session %d: steps %d, max |dlogit| %.4g, KL mean %.3g max %.3g, argmax disagree %d%s\n",
               i, steps, worst_abs, mean_kl, worst_kl, disagree, bad ? "  FAIL" : "");
        ds4_session_free(c);
    }
    ds4_engine_close(engine);
    printf("%s\n", failed ? "FAIL" : "PASS");
    return failed;
}
