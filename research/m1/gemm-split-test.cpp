// Isolated test: mul_mat_id unsplit vs K-half split-sum (F32 + Q4_K).
// Build: g++ -O2 -I ggml/include -I ggml/src gemm-split-test.cpp -o /tmp/gemm-test \
//        build-cpu/bin/libggml-base.so build-cpu/bin/libggml-cpu.so -Wl,-rpath,... -lpthread
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

static float randf() { return (float)(rand() % 10000) / 100.0f - 50.0f; }

static void ref_split_compare(ggml_type type, int64_t K, int64_t N, int64_t E, int n_expert_used) {
    const size_t full_bytes = ggml_row_size(type, K) * N * E;
    const int64_t n_elems = K * N * E;
    std::vector<float> fdata(n_elems);
    for (int64_t i = 0; i < n_elems; i++) fdata[i] = randf();
    std::vector<uint8_t> wdata(full_bytes);
    if (type == GGML_TYPE_F32) {
        memcpy(wdata.data(), fdata.data(), full_bytes);
    } else {
        // quantize fdata into wdata (Q4_K/Q4_0 block size 32/256 chunks)
        const int blck = ggml_blck_size(type);
        ggml_quantize_chunk(type, fdata.data(), wdata.data(), 0, n_elems / blck, blck, nullptr);
    }

    ggml_init_params ip = { 64u*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_backend_t back = ggml_backend_cpu_init();

    ggml_tensor * w  = ggml_new_tensor_3d(ctx, type, K, N, E);
    ggml_tensor * s0 = ggml_new_tensor_3d(ctx, type, K/2, N, E);
    ggml_tensor * s1 = ggml_new_tensor_3d(ctx, type, K/2, N, E);

    const int M = n_expert_used;
    std::vector<float> cur(K*M);
    for (auto & v : cur) v = randf();
    std::vector<int32_t> ids(M); for (int i = 0; i < M; i++) ids[i] = i * (E / M);
    ggml_tensor * b  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, M, 1);
    ggml_tensor * id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, M, 1);
    ggml_backend_buffer_t bufs = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),
        ggml_nbytes(w) + ggml_nbytes(s0) + ggml_nbytes(s1) + ggml_nbytes(b) + ggml_nbytes(id));
    if (!bufs) { printf("alloc failed\n"); return; }
    size_t off = 0;
    for (ggml_tensor * t : {w, s0, s1, b, id}) {
        ggml_backend_tensor_alloc(bufs, t, (char*) ggml_backend_buffer_get_base(bufs) + off);
        off += ggml_nbytes(t);
    }
    // copy full data; shards get the STRIDED per-row halves (llama_numa_shard_copy layout)
    {
        const size_t row = ggml_row_size(type, K);
        const size_t row_s = ggml_row_size(type, K/2);
        const int64_t nr = N * E;
        std::vector<uint8_t> sh(wdata.size()/2);
        for (int k = 0; k < 2; k++) {
            const uint8_t * srow = wdata.data() + k*row_s;
            uint8_t * drow = sh.data();
            for (int64_t r = 0; r < nr; r++) memcpy(drow + r*row_s, srow + r*row, row_s);
            ggml_backend_tensor_set(k == 0 ? s0 : s1, sh.data(), 0, sh.size());
        }
    }
    ggml_backend_tensor_set(w, wdata.data(), 0, full_bytes);
    ggml_backend_tensor_set(b, cur.data(), 0, cur.size()*sizeof(float));
    ggml_backend_tensor_set(id, ids.data(), 0, ids.size()*sizeof(int32_t));

    ggml_tensor * full = ggml_mul_mat_id(ctx, w, b, id);
    ggml_tensor * b0 = ggml_view_3d(ctx, b, K/2, M, 1, b->nb[1], b->nb[2], 0);
    ggml_tensor * b1 = ggml_view_3d(ctx, b, K/2, M, 1, b->nb[1], b->nb[2], (K/2)*b->nb[0]);
    ggml_tensor * p0 = ggml_mul_mat_id(ctx, s0, b0, id);
    ggml_tensor * p1 = ggml_mul_mat_id(ctx, s1, b1, id);
    ggml_tensor * split = ggml_add(ctx, p0, p1);

    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, full);
    ggml_build_forward_expand(g, split);
    ggml_gallocr_t gall = ggml_gallocr_new(ggml_backend_get_default_buffer_type(back));
    ggml_gallocr_alloc_graph(gall, g);
    ggml_backend_cpu_set_n_threads(back, 1);
    ggml_backend_graph_compute(back, g);
    ggml_gallocr_free(gall);

    const int n_out = N * M;
    std::vector<float> a(n_out), c(n_out);
    if (type == GGML_TYPE_F32) {
        printf("  full[0..3] = %.3f %.3f %.3f %.3f\n", a[0], a[1], a[2], a[3]);
        printf("  split[0..3] = %.3f %.3f %.3f %.3f\n", c[0], c[1], c[2], c[3]);
    }
    ggml_backend_tensor_get(full, a.data(), 0, n_out*sizeof(float));
    ggml_backend_tensor_get(split, c.data(), 0, n_out*sizeof(float));
    if (type == GGML_TYPE_F32) {
        float wchk[4], bchk[4];
        ggml_backend_tensor_get(w, wchk, 0, sizeof wchk);
        ggml_backend_tensor_get(b, bchk, 0, sizeof bchk);
        printf("  w[0..3]=%.1f %.1f %.1f %.1f  b[0..3]=%.1f %.1f %.1f %.1f\n", wchk[0],wchk[1],wchk[2],wchk[3], bchk[0],bchk[1],bchk[2],bchk[3]);
        printf("  full[0..7]=%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]);
    }
    if (type == GGML_TYPE_Q4_0) {
        printf("  q4_0 full[0..3]=%.2f %.2f %.2f %.2f  split[0..3]=%.2f %.2f %.2f %.2f\n", a[0],a[1],a[2],a[3], c[0],c[1],c[2],c[3]);
    }
    double max_diff = 0, sum = 0;
    for (int i = 0; i < n_out; i++) {
        double d = std::fabs((double)a[i] - c[i]);
        max_diff = std::max(max_diff, d);
        sum += std::fabs((double)a[i]);
    }
    printf("%-6s K=%-5lld N=%-5lld E=%-3lld M=%d  max|a-c|=%.3e  ref|a|avg=%.3e  %s\n",
        ggml_type_name(type), (long long)K, (long long)N, (long long)E, M,
        max_diff, sum/n_out, max_diff < 1e-4 ? "MATCH" : "MISMATCH");
    ggml_backend_free(back);
    ggml_free(ctx);
}

int main() {
    srand(42);
    ref_split_compare(GGML_TYPE_Q4_0, 512, 2048, 8, 2);
    ref_split_compare(GGML_TYPE_Q4_K, 2048, 512, 8, 2);
    ref_split_compare(GGML_TYPE_F32, 2048, 512, 8, 2);
    return 0;
}
