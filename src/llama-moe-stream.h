#pragma once

#include "llama-mmap.h"

#include "ggml-cpp.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// SSD streaming of MoE routed expert weights
//
// Streamed layers do not materialize their ffn_*_exps tensors; instead each weight gets a
// device-side cache tensor of n_slots expert slabs, filled on demand from the GGUF file by an
// id-remapping custom op that runs on the CPU right after the router top-k. The remap only
// changes which cache slot an expert id resolves to - it never changes which experts the router
// selected, so streaming affects latency, not outputs.
//
// Missing experts are loaded by a pool of I/O threads while the remap op waits; eviction is by
// decaying route hotness with an LRU tiebreak. Reads are buffered by default, or O_DIRECT with
// LLAMA_MOE_STREAM_DIRECT=1 (bypasses the page cache; recommended when the model far exceeds RAM).
//
// note: multiple contexts decoding the same streamed model concurrently are not supported -
// one context can evict slots referenced by the other's in-flight graph.

struct llama_moe_stream;

enum llama_moe_stream_slot_state : uint8_t {
    LLAMA_MOE_STREAM_SLOT_EMPTY    = 0,
    LLAMA_MOE_STREAM_SLOT_LOADING  = 1, // reserved, load queued or in flight
    LLAMA_MOE_STREAM_SLOT_RESIDENT = 2,
};

// one streamed weight tensor (gate/up/down or fused gate_up) of one layer
struct llama_moe_stream_weight {
    ggml_tensor * cache = nullptr; // cache tensor {ne0, ne1, n_slots}
    ggml_tensor * host  = nullptr; // fully materialized CPU tensor (decode phase uses this, not the cache)

    // P1 fix: NUMA tensor split — when the host is sharded, the per-node shards hold the expert
    //   data (K-halves). the I/O workers then copy the cache slabs from the shards instead of the
    //   host (the host buffer is released post-load), and the graph's shard registry (keyed on the
    //   host pointer) redirects decode to the shards.
    ggml_tensor * shards[2] = {nullptr, nullptr};

    uint16_t file_idx  = 0; // GGUF split file index
    size_t   offs      = 0; // file offset of the full exps tensor data
    size_t   nb_expert = 0; // bytes per expert slab
};

struct llama_moe_stream_layer;

// userdata of one wave's custom ops (multi-pass prefill): identifies which pass this is
struct llama_moe_stream_wave {
    llama_moe_stream_layer * sl   = nullptr;
    int32_t                  wave = -1;
};

// per-layer streaming state - also the userdata of the id-remapping custom op
struct llama_moe_stream_layer {
    llama_moe_stream * mgr = nullptr;

    int32_t  il       = -1;
    uint32_t n_expert = 0; // routed expert count (from the expert weights); the slot machinery size
    uint32_t n_slots  = 0;

    std::vector<llama_moe_stream_weight> weights;      // 2 (fused gate_up + down) or 3 entries
    std::vector<llama_moe_stream_weight> attn_weights; // window mode: attention projections, one slab = whole tensor
    bool last_is_attn = false;                         // the most recent create_cache_tensor call pushed an attention weight

    // window-mode graph build state: the attention wait-op is inserted once per layer per build
    // (the first streamed attention projection consumed by build_lora_mm); reset by begin_graph_build
    bool wait_inserted = false;

    // residency state, guarded by mgr->mtx
    std::vector<int32_t>                 slot_expert;   // [n_slots] expert id or -1
    std::vector<uint8_t>                 slot_state;    // [n_slots] llama_moe_stream_slot_state
    std::vector<uint8_t>                 slot_claimed;  // [n_slots] a worker owns the load
    std::vector<uint64_t>                slot_gen;      // [n_slots] reservation generation
    std::vector<int64_t>                 slot_last_use; // [n_slots] LRU stamps
    std::unordered_map<int32_t, int32_t> expert_slot;   // RESIDENT and LOADING entries

    std::vector<uint32_t> route_hotness; // [n_expert] decayed selection counts, for eviction
    std::vector<uint8_t>  seen;          // [n_expert] for cold-miss attribution
    int64_t use_counter = 0;

    // scratch for the remap callback
    std::vector<int32_t> uniq;
    std::vector<uint8_t> touched;
    std::vector<uint8_t> keep;         // [n_slots] slots the current call must not evict
    std::vector<int32_t> demand_slots; // slots the current call waits on

    // wave plan for multi-pass prefill (guarded by mgr->mtx): the touched experts are split into
    // plan_n_waves passes of at most plan_capacity experts each, run one pass at a time
    uint32_t plan_capacity  = 0;  // experts per wave, set at graph build
    uint32_t plan_n_waves   = 0;  // waves of the current call
    int32_t  plan_next_wave = -1; // wave expected to run next (ordering guard)
    std::vector<uint8_t> expert_wave; // [n_expert] wave each touched expert belongs to, 0xff = untouched
    std::vector<int32_t> plan_pool;   // resident slots the masked-out pairs of this wave park on
    std::vector<int32_t> pool_used;   // scratch: pool slots already used in the current token row

    std::vector<std::unique_ptr<llama_moe_stream_wave>> wave_ud; // stable per-wave op userdata

    // stable userdata for wave w (grows lazily); called at graph build time only
    llama_moe_stream_wave * wave_userdata(int32_t wave, uint32_t capacity);

    // whether the exps tensors passed to build_moe_ffn are this layer's cache tensors
    // (e.g. grovemoe evaluates a second, unstreamed expert group on the same layer index)
    bool matches(const ggml_tensor * gate, const ggml_tensor * up,
                 const ggml_tensor * down, const ggml_tensor * gate_up) const;

    // the fully materialized host tensor backing a cache tensor (decode phase), or nullptr
    const ggml_tensor * host_for(const ggml_tensor * cache) const {
        for (const auto & w : weights) {
            if (w.cache == cache) {
                return w.host;
            }
        }
        for (const auto & w : attn_weights) {
            if (w.cache == cache) {
                return w.host;
            }
        }
        return nullptr;
    }

    // whether the given tensor is one of this layer's streamed attention projections
    bool is_attn_cache(const ggml_tensor * t) const {
        for (const auto & w : attn_weights) {
            if (w.cache == t) {
                return true;
            }
        }
        return false;
    }
};

// one queued expert load
struct llama_moe_stream_work {
    llama_moe_stream_layer * sl = nullptr;

    int32_t  expert = -1;
    int32_t  slot   = -1;
    uint64_t gen    = 0; // stale unless it matches slot_gen[slot]
};

struct llama_moe_stream {
    uint32_t n_slots      = 0; // expert cache slots per streamed layer (window mode: == n_expert)
    int32_t  n_io_threads = 0;

    // layer-window mode (LvLLM-style rolling residency): n_window pool slots, each holding one
    // layer's FULL expert set; layer N's cache tensors are window_pool[N % n_window][k]. the pool
    // tensors are keyed by weight role + shape, so the weights of one layer (gate/up/down) never
    // collide while layers of equal geometry share the slot machinery.
    // 0 = expert-slot mode (the PR #25294 machinery). No slot floor in window mode.
    uint32_t n_window = 0;
    std::vector<std::vector<ggml_tensor *>> window_pool;      // [slot][weight-index]
    std::vector<std::vector<std::string>>   window_pool_role; // [slot][weight-index] "ffn_gate_exps" etc.
    std::vector<int32_t> window_pool_gen;                     // [slot] layer whose data it holds (-1 = none)

    std::vector<std::unique_ptr<llama_moe_stream_layer>> layers; // [n_layer], null = not streamed

    llama_moe_stream(uint32_t n_layer, uint32_t n_slots, int32_t n_io_threads, bool direct, uint32_t n_window = 0);
    ~llama_moe_stream();

    llama_moe_stream_layer * layer(int32_t il) const {
        return il >= 0 && (size_t) il < layers.size() ? layers[il].get() : nullptr;
    }

    // registers a streamed weight of layer il and returns its cache tensor
    ggml_tensor * create_cache_tensor(
            int32_t il, ggml_backend_buffer_type_t buft, const ggml_tensor * meta,
            uint16_t file_idx, size_t offs);

    // allocate the cache tensor buffers (after all create_cache_tensor calls)
    void alloc_bufs(bool no_alloc);

    // link the materialized host tensor to its cache tensor (decode phase uses the host tensor);
    // sh0/sh1 attach the NUMA split shards (K-halves) so the I/O workers can source cache slabs
    // from them once the host buffer is released post-load
    void set_host(int32_t il, ggml_tensor * host, ggml_tensor * sh0 = nullptr, ggml_tensor * sh1 = nullptr);

    // host tensor backing the given streamed cache tensor (decode phase), or nullptr
    const ggml_tensor * host_for(const ggml_tensor * cache) const {
        for (const auto & sl : layers) {
            if (sl) {
                const ggml_tensor * h = sl->host_for(cache);
                if (h) {
                    return h;
                }
            }
        }
        return nullptr;
    }

    // page-lock the materialized host tensors so host->device copies use DMA (M4 / #26659);
    // no-op unless a backend registers host buffers (CUDA gates on GGML_CUDA_REGISTER_HOST)
    void pin_hosts();

    // reopen the GGUF files for streaming reads
    void open_files(const std::vector<std::string> & paths);

    size_t size_bufs() const;

    void print_stats() const;

    bool use_direct_io = false; // O_DIRECT streaming reads (LLAMA_MOE_STREAM_DIRECT), no page cache

    llama_files files; // privately reopened GGUF files, same indices as the loader's

    size_t  max_nb_expert      = 0;
    int64_t hot_decay_interval = 0; // remap calls between route-hotness halvings (0 = no decay)

    std::vector<std::pair<ggml_backend_buffer_type_t, ggml_context_ptr>> ctxs; // one per buft
    std::vector<ggml_backend_buffer_ptr> bufs;

    // pinned host tensors (M4); unregistered in the destructor
    std::vector<ggml_tensor *> pinned;
    void (*pinned_unreg)(void *) = nullptr;
    // page-aligned buffer bases registered (whole-buffer registration; tensor data is not page-aligned)
    std::vector<void *> pinned_bufs;

    // load pool (queue and all layer residency state guarded by mtx)
    mutable std::mutex      mtx;
    std::condition_variable cv_work; // queued work or shutdown
    std::condition_variable cv_done; // a load committed or failed

    std::deque<llama_moe_stream_work> q_demand;

    std::vector<std::thread> workers;
    bool workers_started = false;
    bool shutting_down   = false;
    bool load_failed     = false;

    bool debug = false;

    struct {
        int64_t n_calls     = 0; // remap invocations
        int64_t n_hit       = 0; // touched experts already resident or loading
        int64_t n_miss      = 0; // demand loads issued
        int64_t n_miss_cold = 0; // first-ever touch of an expert
        int64_t t_stall_us  = 0; // wait time in miss handling

        int64_t n_wave_calls     = 0; // wave-ids invocations (>= n_calls under multi-pass prefill)
        int64_t n_waves_run      = 0; // non-empty waves
        int64_t n_preload_issued = 0; // next-wave loads started during a wave's compute
        int64_t n_preload_ready  = 0; // wave experts already resident from the previous preload
        int64_t t_stall_wave_us  = 0; // wait time in wave miss handling

        int64_t n_attn_waits     = 0; // attention wait-op invocations (window mode)
        int64_t t_stall_attn_us  = 0; // wait time in the attention wait-op
    } stats;

    // internals
    void start_workers_locked();
    void worker_loop();
    int32_t pick_victim_locked(llama_moe_stream_layer & sl, const uint8_t * keep) const;
    void reserve_slot_locked(llama_moe_stream_layer & sl, int32_t expert, int32_t slot);

    // layer-window mode: enqueue the next layer's FULL expert load (no wait); called from the
    // remap after the current layer is resident. layer order is deterministic, so the copies
    // overlap the current layer's compute (the LvLLM prefetch-window trick).
    void prefetch_layer(int32_t il);

    // layer-window mode: if the pool slot currently holds a different layer's bytes (or none),
    // reset this layer's slot state and claim the slot for this layer. called under mtx.
    void claim_slot_locked(llama_moe_stream_layer & sl, uint32_t slot);

    // layer-window mode: make the layer's pool slot resident (attention + experts), waiting for
    // its loads. used by the remap (after the prefetch of the previous layer) and by the attention
    // wait-op (the first consumer of the layer's pool slot). called under mtx, releases it on wait.
    void ensure_layer_resident_locked(std::unique_lock<std::mutex> & lk, llama_moe_stream_layer & sl);

    // build_lora_mm hook: for window mode, the first streamed attention projection of a layer
    // consumed by the graph (per build) returns the layer so the caller inserts the wait-op; later
    // projections of the same layer return null (the slot is resident by then). null for decode
    // (host tensors) and for expert-slot mode.
    llama_moe_stream_layer * attn_wait(const ggml_tensor * w);

    // called once per graph build (llama_model::build_graph): allow one attention wait-op per layer
    void begin_graph_build();

    // multi-pass prefill helpers (called by llama_moe_stream_wave_ids, all under mtx)
    void plan_waves_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n); // wave 0: build the plan
    void stage_wave_locked(std::unique_lock<std::mutex> & lk, llama_moe_stream_layer & sl, int32_t w, uint32_t n_ids); // make wave w resident + preload next
    void emit_wave_slots(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out, int32_t w, uint32_t n_ids, int64_t n_tok); // write the slot ids
};

// callback of the id-remapping custom op inserted by build_moe_ffn
void llama_moe_stream_remap(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);

// callback of the attention wait-op inserted by build_lora_mm (window mode): blocks until the
// layer's pool slot holds its attention + expert weights, then passes the activations through
void llama_moe_stream_wait(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);

// callbacks of the multi-pass prefill custom ops inserted by build_moe_ffn when a ubatch touches
// more experts than the cache holds; each src[0] is the contiguous selected ids
//   wave_ids:  makes wave w's expert slice resident and emits slot ids (masked pairs park on a pool)
//   wave_mask: emits 1.0 for pairs belonging to wave w, 0.0 otherwise
void llama_moe_stream_wave_ids (ggml_tensor * dst, int ith, int nth, void * userdata);
void llama_moe_stream_wave_mask(ggml_tensor * dst, int ith, int nth, void * userdata);
