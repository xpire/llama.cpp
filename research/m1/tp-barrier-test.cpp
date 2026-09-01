// tp-barrier-test.cpp — 2-process stress of llama_tp_context::allreduce in isolation.
// each rank writes a known value (rank+1) per element; after allreduce every element must
// equal sum(rank+1) = 3.0. skewed arrivals + varying sizes stress the sense-reversing barrier.
#include "llama-tp.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <unistd.h>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <rank> <n_iters> [shm-name]\n", argv[0]);
        return 2;
    }
    const int   rank = atoi(argv[1]);
    const int   iters = atoi(argv[2]);
    const char * name = argc > 3 ? argv[3] : "barr";

    llama_tp_context tp;
    if (!tp.init(2, rank, name)) {
        fprintf(stderr, "rank %d: init failed\n", rank);
        return 1;
    }

    std::mt19937 rng(rank + 1);
    for (int i = 0; i < iters; i++) {
        // vary the size between calls and inject skew to race the barrier
        const size_t n = 1024 + ((i % 4) * 4096); // same size on both ranks // 1k..17k floats
        std::vector<float> buf(n);
        for (size_t j = 0; j < n; j++) {
            buf[j] = (float) (rank + 1);
        }
        if (i % 7 == 0) {
            usleep(1000 + (rank * 791)); // different skew per rank
        }
        tp.allreduce(buf.data(), n);
        for (size_t j = 0; j < n; j++) {
            if (buf[j] != 3.0f) {
                fprintf(stderr, "rank %d: CORRUPT at iter %d elem %zu (%.2f)\n", rank, i, j, buf[j]);
                return 3;
            }
        }
        if (i % 2500 == 0) {
            fprintf(stderr, "rank %d: iter %d ok\n", rank, i);
        }
    }
    fprintf(stderr, "rank %d: all %d iters OK\n", rank, iters);
    tp.finalize();
    return 0;
}
