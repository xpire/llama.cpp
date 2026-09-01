#pragma once

#include <cstddef>
#include <cstdint>

// minimal CPU tensor-parallel runtime over POSIX shared memory (two sockets on one board):
// ranks rendezvous on a shm segment; every layer's MoE output is all-reduced across ranks so the
// routed-expert compute can be sharded per rank (expert-parallel). f32 only.
//
// ponytail: fixed-capacity segment, sense-reversing two-phase barrier. sliced loading (each rank
//   reads only its experts) is a later step — the load is replicated today (2x RAM, fine for
//   <= ~90 GB models); the compute split is what buys decode throughput.
struct llama_tp_context {
    int32_t size = 1; // total ranks
    int32_t rank = 0; // this rank

    // expert shard range for this rank: contiguous split of n_expert (expert-parallel)
    int32_t expert_lo(int32_t n_expert) const;
    int32_t expert_hi(int32_t n_expert) const;

    bool init(int32_t size, int32_t rank, const char * name);
    void finalize();

    // sum `data` across all ranks, in place. called with the same shape on every rank, in the same
    // order (the graphs are identical per rank). blocks until every rank arrives (per-layer barrier).
    void allreduce(float * data, size_t n);

private:
    struct shm_hdr;

    shm_hdr * hdr      = nullptr;
    float   * areas    = nullptr; // [size][cap] floats
    void    * map_base = nullptr;
    size_t    map_len  = 0;
    int       shm_fd   = -1;
    bool      sense    = true; // local sense for the barrier (starts true so the first round waits)
    size_t    cap      = 0;    // floats per rank area
};
