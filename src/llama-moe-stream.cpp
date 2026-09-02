#include "llama-moe-stream.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#include <malloc.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

static const uint32_t MOE_STREAM_IO_THREADS_DEFAULT = 9;
static const uint32_t MOE_STREAM_IO_THREADS_MAX     = 18;
static const int64_t  MOE_STREAM_HOT_DECAY_TOKENS   = 64;

// O_DIRECT alignment: 4096 is a multiple of any device logical block size (512/4096), so it is
// universally valid, and reading a few extra KB of head/tail padding per slab is negligible
static const size_t MOE_STREAM_DIRECT_ALIGN = 4096;

// saturating increment - route-hotness counters accumulate over a whole run and must not wrap
static uint32_t sat_inc(uint32_t & c) {
    if (c < UINT32_MAX - 1) {
        c++;
    }
    return c;
}

// page-aligned allocation, required both for O_DIRECT reads and for Metal private-buffer uploads
static void * moe_aligned_alloc(size_t n) {
#ifdef _WIN32
    return _aligned_malloc(n, MOE_STREAM_DIRECT_ALIGN);
#else
    void * p = nullptr;
    if (posix_memalign(&p, MOE_STREAM_DIRECT_ALIGN, n) != 0) {
        p = nullptr;
    }
    return p;
#endif
}

static void moe_aligned_free(void * p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

// read len bytes at file offset offs into staging (thread-safe positional read); staging must have
// room for len (+ 2*MOE_STREAM_DIRECT_ALIGN when direct). returns a pointer to the len bytes
// within staging, or nullptr on failure
static const uint8_t * llama_moe_stream_pread(llama_file & file, uint8_t * staging, size_t len, size_t offs, bool direct) {
#ifdef _WIN32
    GGML_UNUSED(direct);
    // no positional read primitive; serialize the seek+read pairs
    static std::mutex io_mtx;
    std::lock_guard<std::mutex> lock(io_mtx);
    try {
        file.seek(offs, SEEK_SET);
        file.read_raw(staging, len);
        return staging;
    } catch (...) {
        return nullptr;
    }
#else
    const int fd = file.file_id();

    if (direct) {
        // O_DIRECT requires the offset, length, and buffer all block-aligned
        const size_t a     = MOE_STREAM_DIRECT_ALIGN;
        const size_t aoffs = offs & ~(a - 1);
        const size_t head  = offs - aoffs;
        const size_t total = ((head + len + a - 1)/a)*a;
        ssize_t r;
        do {
            r = pread(fd, staging, total, aoffs);
        } while (r < 0 && errno == EINTR);
        if (r < 0 || (size_t) r < head + len) {
            return nullptr;
        }
        return staging + head;
    }

    uint8_t * p    = staging;
    size_t    left = len;
    while (left > 0) {
        const ssize_t r = pread(fd, p, left, offs);
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return nullptr;
        }
        if (r == 0) {
            return nullptr; // unexpected EOF
        }
        p    += r;
        offs += (size_t) r;
        left -= (size_t) r;
    }
    return staging;
#endif
}

// true iff all of the given exps tensors are this layer's cache tensors - guards against a second,
// non-streamed expert group on the same layer index (e.g. grovemoe chexps)
bool llama_moe_stream_layer::matches(const ggml_tensor * gate, const ggml_tensor * up,
                                     const ggml_tensor * down, const ggml_tensor * gate_up) const {
    auto is_cache = [this](const ggml_tensor * t) {
        for (const auto & w : weights) {
            if (w.cache == t) {
                return true;
            }
        }
        return false;
    };

    size_t n = 0;
    for (const ggml_tensor * t : { gate, up, down, gate_up }) {
        if (t == nullptr) {
            continue;
        }
        if (!is_cache(t)) {
            return false;
        }
        n++;
    }

    return n > 0 && n == weights.size();
}

// weight role of a streamed expert tensor: the name between the layer prefix and the ".weight"
// suffix, e.g. "blk.12.ffn_gate_up_exps.weight" -> "ffn_gate_up_exps"
static std::string llama_moe_stream_weight_role(const ggml_tensor * meta) {
    const std::string name = meta->name;
    const size_t dot = name.rfind('.');
    const size_t dot2 = name.rfind('.', dot - 1);
    GGML_ASSERT(dot != std::string::npos && dot2 != std::string::npos);
    return name.substr(dot2 + 1, dot - dot2 - 1);
}

// sizes the per-layer table and clamps the I/O thread count; workers are spawned lazily on first use
llama_moe_stream::llama_moe_stream(uint32_t n_layer, uint32_t n_slots, int32_t n_io_threads, bool direct, uint32_t n_window) : n_slots(n_slots), n_window(n_window) {
    layers.resize(n_layer);

    this->n_io_threads = n_io_threads <= 0 ? MOE_STREAM_IO_THREADS_DEFAULT : n_io_threads;
    this->n_io_threads = std::min<int32_t>(this->n_io_threads, MOE_STREAM_IO_THREADS_MAX);

    debug         = std::getenv("LLAMA_MOE_STREAM_DEBUG") != nullptr;
    use_direct_io = direct;
}

// stop and join the I/O workers before the cache buffers and files they use are destroyed
llama_moe_stream::~llama_moe_stream() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        shutting_down = true;
        q_demand.clear();
    }
    cv_work.notify_all();
    for (auto & w : workers) {
        w.join();
    }
    if (pinned_unreg != nullptr) {
        for (void * base : pinned_bufs) {
            pinned_unreg(base);
        }
    }
}

ggml_tensor * llama_moe_stream::create_cache_tensor(
        int32_t il, ggml_backend_buffer_type_t buft, const ggml_tensor * meta,
        uint16_t file_idx, size_t offs) {
    GGML_ASSERT(il >= 0 && (size_t) il < layers.size());
    GGML_ASSERT(ggml_is_contiguous(meta));
    GGML_ASSERT(meta->ne[2] > 0 && meta->ne[3] == 1);

    const uint32_t n_expert  = meta->ne[2];
    const size_t   nb_expert = ggml_nbytes(meta) / n_expert;
    GGML_ASSERT(nb_expert * n_expert == ggml_nbytes(meta));

    // window mode streams attention projections too: they have no expert dim (ne[2] == 1), the
    // whole tensor is one slab, copied alongside the first expert of the layer load
    const bool is_attn = n_window > 0 && n_expert == 1;
    if (!is_attn) {
        GGML_ASSERT(n_slots > 0 && n_slots <= n_expert); // window mode: n_slots == n_expert
    }

    // layer-window mode holds a FULL layer per pool slot; expert mode holds n_slots slabs
    const uint32_t n_slots_eff = n_window > 0 ? n_expert : n_slots;

    ggml_context * ctx = nullptr;
    for (auto & [cur_buft, cur_ctx] : ctxs) {
        if (cur_buft == buft) {
            ctx = cur_ctx.get();
            break;
        }
    }
    if (ctx == nullptr) {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*(layers.size()*8 + 1),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create ggml context for MoE expert streaming");
        }
        ctxs.emplace_back(buft, ctx);
    }

    ggml_tensor * cache = nullptr;
    if (n_window > 0) {
        // shared pool slot: layer N uses window_pool[N % n_window][k] where k is this weight's role;
        // the slot's tensors are created on the first layer mapped to them and reused (overwritten)
        // by later ones. tensors are keyed by weight role + shape so that the gate/up/down weights
        // of one layer are distinct pool tensors (they hold different bytes of the same layer)
        // while layers with equal expert geometry share the slot machinery.
        const uint32_t slot = (uint32_t) il % n_window;
        if (window_pool.size() <= slot) {
            window_pool.resize(slot + 1);
            window_pool_role.resize(slot + 1);
            window_pool_gen.resize(slot + 1, -1);
        }
        auto & ws = window_pool[slot];
        auto & rs = window_pool_role[slot];
        const std::string role = llama_moe_stream_weight_role(meta);
        ggml_tensor * base = nullptr;
        for (size_t k = 0; k < ws.size(); k++) {
            ggml_tensor * t = ws[k];
            if (rs[k] == role && t->type == meta->type && t->ne[0] == meta->ne[0] && t->ne[1] == meta->ne[1]) {
                base = t;
                break;
            }
        }
        if (base == nullptr) {
            base = ggml_new_tensor_3d(ctx, meta->type, meta->ne[0], meta->ne[1], n_expert);
            ws.push_back(base);
            rs.push_back(role);
        }
        // attention projections are consumed by build_lora_mm, which must find the CURRENT layer's
        // host tensor in decode (the base tensor is shared across the layers of a window slot, so a
        // pointer lookup would be ambiguous). give each layer a distinct view of the shared base -
        // same memory, unambiguous identity. expert weights keep the shared base (their decode path
        // is already layer-scoped in build_moe_ffn).
        if (is_attn) {
            cache = ggml_view_3d(ctx, base, base->ne[0], base->ne[1], base->ne[2], base->nb[1], base->nb[2], 0);
        } else {
            cache = base;
        }
        ggml_format_name(cache, "%s.stream_window", meta->name);
    } else {
        cache = ggml_new_tensor_3d(ctx, meta->type, meta->ne[0], meta->ne[1], n_slots);
        ggml_format_name(cache, "%s.stream_cache", meta->name);
    }
    GGML_ASSERT(ggml_nbytes(cache) == nb_expert * n_slots_eff);

    auto & sl = layers[il];
    if (!sl) {
        sl = std::make_unique<llama_moe_stream_layer>();
        sl->mgr      = this;
        sl->il       = il;
        // n_expert / slot arrays are sized when the first EXPERT weight registers (attention
        // projections register first and are only slab lists, no slot machinery)
    }
    if (is_attn) {
        sl->attn_weights.push_back({ cache, nullptr, {nullptr, nullptr}, file_idx, offs, nb_expert });
        sl->last_is_attn = true;
    } else {
        if (sl->n_slots == 0) {
            sl->n_expert = n_expert;
            sl->n_slots  = n_slots_eff;
            sl->slot_expert  .resize(n_slots_eff, -1);
            sl->slot_state   .resize(n_slots_eff, LLAMA_MOE_STREAM_SLOT_EMPTY);
            sl->slot_claimed .resize(n_slots_eff, 0);
            sl->slot_gen     .resize(n_slots_eff, 0);
            sl->slot_last_use.resize(n_slots_eff, 0);
            sl->route_hotness.resize(n_expert, 0);
            sl->seen         .resize(n_expert, 0);
            sl->keep         .resize(n_slots, 0);
        }
        GGML_ASSERT(sl->n_expert == n_expert);
        sl->weights.push_back({ cache, nullptr, {nullptr, nullptr}, file_idx, offs, nb_expert });
        sl->last_is_attn = false;
    }

    max_nb_expert = std::max(max_nb_expert, nb_expert);

    return cache;
}

void llama_moe_stream::alloc_bufs(bool no_alloc) {
    for (auto & [buft, ctx_ptr] : ctxs) {
        ggml_context * ctx = ctx_ptr.get();
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        ggml_backend_buffer_t buf;
        if (no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        }
        if (buf == nullptr) {
            throw std::runtime_error(format("unable to allocate %s buffer for MoE expert streaming", ggml_backend_buft_name(buft)));
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        bufs.emplace_back(buf);

        LLAMA_LOG_INFO("%s: %12s expert cache size = %8.2f MiB (%u slots per layer)\n",
                __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0, n_slots);
    }
}

void llama_moe_stream::open_files(const std::vector<std::string> & paths) {
    for (const auto & path : paths) {
        if (path.empty()) {
            throw std::runtime_error("MoE expert streaming requires a file-based model (not a stream/file descriptor)");
        }
    }

    auto open_all = [&](bool direct) {
        files.clear();
        for (const auto & path : paths) {
            files.emplace_back(new llama_file(path.c_str(), "rb", direct));
        }
    };

    open_all(use_direct_io);

    // fall back to buffered when O_DIRECT is unusable: either the open did not honor it (macOS,
    // Windows, unsupported filesystems), or it opened but a probe read fails (some network/overlay
    // filesystems accept the flag then reject aligned reads). reopening is needed because O_DIRECT
    // is a property of the fd. done here, single-threaded, before any worker starts.
    if (use_direct_io) {
        bool ok = !files.empty() && files.front()->has_direct_io();
        if (ok) {
            uint8_t * probe = (uint8_t *) moe_aligned_alloc(MOE_STREAM_DIRECT_ALIGN);
            GGML_ASSERT(probe != nullptr);
            ok = llama_moe_stream_pread(*files.front(), probe, MOE_STREAM_DIRECT_ALIGN, 0, /*direct =*/ true) != nullptr;
            moe_aligned_free(probe);
        }
        if (!ok) {
            LLAMA_LOG_WARN("%s: O_DIRECT not usable, falling back to buffered streaming reads\n", __func__);
            use_direct_io = false;
            open_all(false);
        }
    }

    if (use_direct_io) {
        LLAMA_LOG_INFO("%s: MoE expert streaming uses O_DIRECT (page cache bypassed)\n", __func__);
    }

    // one token drives ~one remap per streamed layer, so decaying every 64 tokens is
    //   64 * n_streamed_layers remap calls (computed once here, off the hot path)
    int64_t n_streamed = 0;
    for (const auto & sl : layers) {
        n_streamed += sl != nullptr;
    }
    hot_decay_interval = MOE_STREAM_HOT_DECAY_TOKENS * n_streamed;
}

// spawn the I/O thread pool on first use (from the remap callback, under mtx)
void llama_moe_stream::start_workers_locked() {
    if (workers_started) {
        return;
    }
    workers_started = true;
    workers.reserve(n_io_threads);
    for (int32_t i = 0; i < n_io_threads; i++) {
        workers.emplace_back([this]() { worker_loop(); });
    }
}

// I/O worker: pops a reserved load, reads its expert slab(s) from the GGUF file into the cache
// slot, and marks the slot RESIDENT (or flags load_failed); stale/duplicate items are skipped
void llama_moe_stream::worker_loop() {
    // page-aligned staging (Metal private buffers require page-aligned source + page-multiple
    // length; O_DIRECT needs the extra head/tail slack for its aligned reads)
    uint8_t * staging = (uint8_t *) moe_aligned_alloc(max_nb_expert + 2*MOE_STREAM_DIRECT_ALIGN);
    GGML_ASSERT(staging != nullptr);

    std::unique_lock<std::mutex> lk(mtx);
    while (true) {
        cv_work.wait(lk, [&]{ return shutting_down || !q_demand.empty(); });
        if (shutting_down) {
            break;
        }

        llama_moe_stream_work w = q_demand.front();
        q_demand.pop_front();

        auto & sl = *w.sl;
        if (w.gen != sl.slot_gen[w.slot] ||
            sl.slot_state[w.slot] != LLAMA_MOE_STREAM_SLOT_LOADING ||
            sl.slot_expert[w.slot] != w.expert ||
            sl.slot_claimed[w.slot]) {
            continue; // stale or duplicate item
        }
        sl.slot_claimed[w.slot] = 1;

        lk.unlock();

        bool ok = true;
        for (const auto & wt : sl.weights) {
            if (wt.host == nullptr) {
                ok = false;
                break;
            }
            // RAM -> device: copy the expert slab from the materialized host tensor into the cache.
            //   set_tensor copies on cudaStreamPerThread (worker-local) and syncs — safe from worker
            //   threads. M4-proper (async on the scheduler stream) requires the copies to be enqueued
            //   by the scheduler thread (a graph-level copy op); worker-thread async stream enqueue
            //   races with the main thread and segfaults (measured).
            if (wt.shards[0] != nullptr && wt.shards[1] != nullptr) {
                // P1 fix: NUMA split — the host buffer is released post-load, so the expert slab is
                //   assembled from the two per-node shards (each holds half of every row)
                const size_t nb_half = wt.nb_expert / 2;
                ggml_backend_tensor_set(wt.cache,
                        (const uint8_t *) wt.shards[0]->data + (size_t) w.expert*nb_half,
                        (size_t) w.slot*wt.nb_expert, nb_half);
                ggml_backend_tensor_set(wt.cache,
                        (const uint8_t *) wt.shards[1]->data + (size_t) w.expert*nb_half,
                        (size_t) w.slot*wt.nb_expert + nb_half, nb_half);
            } else {
                const uint8_t * src = (const uint8_t *) wt.host->data + (size_t) w.expert*wt.nb_expert;
                ggml_backend_tensor_set(wt.cache, src, (size_t) w.slot*wt.nb_expert, wt.nb_expert);
            }
            if (n_window > 0 && debug) {
                fprintf(stderr, "DBG copy il=%d wt=%s expert=%d slot=%d nb=%zu\n", sl.il, ggml_get_name(wt.cache), w.expert, w.slot, wt.nb_expert);
            }
        }
        if (ok && w.expert == 0) {
            // window mode: the layer's attention projections ride along with its first expert slab
            for (const auto & wt : sl.attn_weights) {
                if (wt.host == nullptr) {
                    ok = false;
                    break;
                }
                ggml_backend_tensor_set(wt.cache, wt.host->data, 0, wt.nb_expert);
                if (debug) {
                    fprintf(stderr, "DBG copy il=%d attn=%s nb=%zu\n", sl.il, ggml_get_name(wt.cache), wt.nb_expert);
                }
            }
        }

        lk.lock();

        sl.slot_claimed[w.slot] = 0;
        if (!ok) {
            load_failed = true;
        } else {
            sl.slot_state[w.slot] = LLAMA_MOE_STREAM_SLOT_RESIDENT;
        }
        cv_done.notify_all();
    }
    lk.unlock();

    moe_aligned_free(staging);
}

// least valuable evictable slot: empty first, then coldest resident (min route hotness, oldest use
// as tiebreak); LOADING and keep slots are never candidates. returns -1 when no candidate exists
int32_t llama_moe_stream::pick_victim_locked(llama_moe_stream_layer & sl, const uint8_t * keep) const {
    int32_t v = -1;

    for (uint32_t s = 0; s < sl.n_slots; s++) {
        if ((keep && keep[s]) || sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
            continue;
        }
        if (sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_EMPTY) {
            return s;
        }
        if (v < 0) {
            v = s;
            continue;
        }
        const uint32_t hs = sl.route_hotness[sl.slot_expert[s]];
        const uint32_t hv = sl.route_hotness[sl.slot_expert[v]];
        if (hs < hv || (hs == hv && sl.slot_last_use[s] < sl.slot_last_use[v])) {
            v = s;
        }
    }

    return v;
}

// bind expert -> slot and mark it LOADING: evict the slot's prior occupant, bump slot_gen (so any
// in-flight load for the old occupant is recognized as stale), and update the expert_slot index
void llama_moe_stream::reserve_slot_locked(llama_moe_stream_layer & sl, int32_t expert, int32_t slot) {
    if (sl.slot_expert[slot] >= 0) {
        if (debug) {
            LLAMA_LOG_DEBUG("%s: layer %d: evict expert %d from slot %d\n", __func__, sl.il, sl.slot_expert[slot], slot);
        }
        sl.expert_slot.erase(sl.slot_expert[slot]);
    }

    sl.slot_expert[slot] = expert;
    sl.slot_state[slot]  = LLAMA_MOE_STREAM_SLOT_LOADING;
    sl.slot_gen[slot]++;
    sl.slot_last_use[slot] = ++sl.use_counter;
    sl.expert_slot[expert] = slot;
    sl.seen[expert] = 1;
}

// layer-window mode: if the pool slot holds a different layer's bytes (or none), reset this
// layer's slot state and claim the slot. the prior occupant's GEMMs finished before this point
// (graph order, n_window >= 2), so overwriting is safe.
void llama_moe_stream::claim_slot_locked(llama_moe_stream_layer & sl, uint32_t slot) {
    if (n_window == 0 || window_pool_gen[slot] == sl.il) {
        return;
    }
    if (debug) {
        LLAMA_LOG_DEBUG("%s: layer %d claims pool slot %u (was layer %d)\n", __func__, sl.il, slot, window_pool_gen[slot]);
    }
    sl.expert_slot.clear();
    std::fill(sl.slot_expert.begin(), sl.slot_expert.end(), -1);
    std::fill(sl.slot_state.begin(), sl.slot_state.end(), LLAMA_MOE_STREAM_SLOT_EMPTY);
    std::fill(sl.slot_claimed.begin(), sl.slot_claimed.end(), 0);
    window_pool_gen[slot] = sl.il;
}

// layer-window mode: make the layer's pool slot resident (attention + experts), waiting for its
// loads. the previous layer's remap prefetched it (attention first - its copies ride on the first
// expert slab), so the wait is short; the first touch of a layer (or the very first layer of a
// prefill) reserves and demand-loads it here. the attention wait-op and the expert remap both
// call this before the layer's GEMMs read the slot.
void llama_moe_stream::ensure_layer_resident_locked(std::unique_lock<std::mutex> & lk, llama_moe_stream_layer & sl) {
    GGML_ASSERT(n_window > 0 && sl.n_expert > 0);
    start_workers_locked(); // the attention wait-op of layer 0 is the first op of the graph - no remap started them yet
    claim_slot_locked(sl, (uint32_t) sl.il % n_window);
    if (sl.expert_slot.size() < sl.n_expert) {
        // first touch of the layer (prefetch normally reserves the whole layer already)
        for (uint32_t e = 0; e < sl.n_expert; e++) {
            if (sl.expert_slot.find(e) != sl.expert_slot.end()) {
                continue;
            }
            const int32_t v = pick_victim_locked(sl, /*keep=*/nullptr);
            if (v < 0) {
                break;
            }
            reserve_slot_locked(sl, (int32_t) e, v);
            q_demand.push_back({ &sl, (int32_t) e, v, sl.slot_gen[v] });
            cv_work.notify_one();
        }
    }
    // the prefetch reserves the layer's slots (LOADING) from the previous layer's remap, which
    // this call did not wait on - always wait for the full layer to be RESIDENT before the
    // attention GEMMs read the pool slot
    sl.demand_slots.clear();
    for (uint32_t e = 0; e < sl.n_expert; e++) {
        sl.demand_slots.push_back(sl.expert_slot.at(e));
    }
    const int64_t t0 = ggml_time_us();
    cv_done.wait(lk, [&]{
        if (load_failed) {
            return true;
        }
        for (const int32_t s : sl.demand_slots) {
            if (sl.slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                return false;
            }
        }
        return true;
    });
    if (load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }
}

// build_lora_mm hook: for window mode, the first streamed attention projection of a layer
// consumed by the graph (per build) returns the layer so the caller inserts the wait-op; later
// projections of the same layer return null (the slot is resident by then). null for decode
// (host tensors) and for expert-slot mode.
llama_moe_stream_layer * llama_moe_stream::attn_wait(const ggml_tensor * w) {
    if (n_window == 0) {
        return nullptr;
    }
    for (auto & sl : layers) {
        if (sl && sl->is_attn_cache(w)) {
            if (sl->wait_inserted) {
                return nullptr; // already gated this layer in this graph build
            }
            sl->wait_inserted = true;
            return sl.get();
        }
    }
    return nullptr;
}

// called once per graph build (llama_model::build_graph): allow one attention wait-op per layer
void llama_moe_stream::begin_graph_build() {
    for (auto & sl : layers) {
        if (sl) {
            sl->wait_inserted = false;
        }
    }
}

// layer-window mode: enqueue the next layer's FULL expert load without waiting (LvLLM's
// prefetch-window trick — layer order is deterministic, so the copies overlap the current
// layer's compute). the pool slot's prior occupant finished its GEMMs before this point.
void llama_moe_stream::prefetch_layer(int32_t il) {
    if (n_window == 0 || il < 0 || (size_t) il >= layers.size() || layers[il] == nullptr) {
        return;
    }
    auto & sl = *layers[il];
    std::unique_lock<std::mutex> lk(mtx);
    start_workers_locked();
    const uint32_t slot = (uint32_t) il % n_window;
    claim_slot_locked(sl, slot);
    if (!sl.expert_slot.empty()) {
        return; // already loaded or loading for this layer
    }
    for (uint32_t e = 0; e < sl.n_expert; e++) {
        const int32_t v = pick_victim_locked(sl, /*keep=*/nullptr);
        if (v < 0) {
            continue;
        }
        reserve_slot_locked(sl, (int32_t) e, v);
        q_demand.push_back({ &sl, (int32_t) e, v, sl.slot_gen[v] });
        cv_work.notify_one();
    }
}

// link the materialized host tensor to its cache tensor (decode phase computes from the host tensor)
void llama_moe_stream::set_host(int32_t il, ggml_tensor * host, ggml_tensor * sh0, ggml_tensor * sh1) {
    // pool tensors are shared across layers in window mode, so the host link must target the
    // weight entry just pushed by create_cache_tensor for this layer
    auto & sl = layers[il];
    GGML_ASSERT(sl && (!sl->weights.empty() || !sl->attn_weights.empty()));
    llama_moe_stream_weight & w = sl->last_is_attn ? sl->attn_weights.back() : sl->weights.back();
    w.host  = host;
    w.shards[0] = sh0;
    w.shards[1] = sh1;
}

// page-lock the materialized host tensors so host->device copies use DMA instead of the driver's
// bounce buffer. resolved via ggml_backend_reg_get_proc_address so src/ gains no backend-specific
// dependency; backends that do not provide it fall through unchanged (CUDA gates on
// GGML_CUDA_REGISTER_HOST). registration makes pages non-pageable, so failures are ignored.
void llama_moe_stream::pin_hosts() {
    using reg_fn   = bool (*)(void *, size_t);
    using unreg_fn = void (*)(void *);

    reg_fn reg = nullptr;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t r = ggml_backend_reg_get(i);
        if (r != nullptr) {
            reg = (reg_fn) ggml_backend_reg_get_proc_address(r, "ggml_backend_register_host_buffer");
            if (reg != nullptr) {
                break;
            }
        }
    }
    if (reg == nullptr) {
        return;
    }
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t r = ggml_backend_reg_get(i);
        if (r != nullptr) {
            pinned_unreg = (unreg_fn) ggml_backend_reg_get_proc_address(r, "ggml_backend_unregister_host_buffer");
            if (pinned_unreg != nullptr) {
                break;
            }
        }
    }

    size_t n_pinned = 0;
    std::vector<ggml_backend_buffer_t> seen;
    for (auto & sl : layers) {
        if (!sl) {
            continue;
        }
        for (auto & w : sl->weights) {
            if (w.host == nullptr || w.host->buffer == nullptr) {
                continue;
            }
            ggml_backend_buffer_t buf = w.host->buffer;
            if (std::find(seen.begin(), seen.end(), buf) != seen.end()) {
                continue; // register each host buffer once (tensor data is not page-aligned)
            }
            seen.push_back(buf);
            void * base = ggml_backend_buffer_get_base(buf);
            size_t size = ggml_backend_buffer_get_size(buf);
            if (reg(base, size)) {
                pinned_bufs.push_back(base);
                n_pinned += size;
            }
        }
    }

    if (n_pinned > 0) {
        LLAMA_LOG_INFO("%s: page-locked %.2f GiB of host expert buffers for DMA\n",
                __func__, n_pinned / 1024.0 / 1024.0 / 1024.0);
    }
}

size_t llama_moe_stream::size_bufs() const {    size_t size = 0;
    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }
    return size;
}

void llama_moe_stream::print_stats() const {
    std::lock_guard<std::mutex> lock(mtx);

    const int64_t n_touched = stats.n_hit + stats.n_miss;
    // direct dump: common_log is buffered and lost on exit
    fprintf(stderr, "%s: moe stream: remap calls = %" PRId64 ", expert hits = %" PRId64 ", misses = %" PRId64 " (%" PRId64 " cold), hit rate = %.2f%%\n",
            __func__, stats.n_calls, stats.n_hit, stats.n_miss, stats.n_miss_cold,
            n_touched > 0 ? 100.0*stats.n_hit/n_touched : 0.0);
    fprintf(stderr, "%s: moe stream: load stall = %.2f ms total (%.3f ms per remap call)\n",
            __func__, stats.t_stall_us/1000.0, stats.n_calls > 0 ? stats.t_stall_us/1000.0/stats.n_calls : 0.0);
    if (stats.n_attn_waits > 0) {
        fprintf(stderr, "%s: moe stream: attention waits = %" PRId64 ", attn stall = %.2f ms total (%.3f ms per wait)\n",
                __func__, stats.n_attn_waits, stats.t_stall_attn_us/1000.0, stats.n_attn_waits > 0 ? stats.t_stall_attn_us/1000.0/stats.n_attn_waits : 0.0);
    }

    LLAMA_LOG_INFO("%s: moe stream: remap calls = %" PRId64 ", expert hits = %" PRId64 ", misses = %" PRId64 " (%" PRId64 " cold), hit rate = %.2f%%\n",
            __func__, stats.n_calls, stats.n_hit, stats.n_miss, stats.n_miss_cold,
            n_touched > 0 ? 100.0*stats.n_hit/n_touched : 0.0);
    LLAMA_LOG_INFO("%s: moe stream: load stall = %.2f ms total (%.3f ms per remap call)\n",
            __func__, stats.t_stall_us/1000.0, stats.n_calls > 0 ? stats.t_stall_us/1000.0/stats.n_calls : 0.0);
    if (stats.n_wave_calls > 0) {
        LLAMA_LOG_INFO("%s: moe stream: waves = %" PRId64 " (%" PRId64 " non-empty), preloads issued = %" PRId64 " (ready on arrival = %" PRId64 "), wave stall = %.2f ms\n",
                __func__, stats.n_wave_calls, stats.n_waves_run, stats.n_preload_issued, stats.n_preload_ready, stats.t_stall_wave_us/1000.0);
    }
    if (stats.n_attn_waits > 0) {
        LLAMA_LOG_INFO("%s: moe stream: attention waits = %" PRId64 ", attn stall = %.2f ms total (%.3f ms per wait)\n",
                __func__, stats.n_attn_waits, stats.t_stall_attn_us/1000.0, stats.n_attn_waits > 0 ? stats.t_stall_attn_us/1000.0/stats.n_attn_waits : 0.0);
    }
}

// custom-op callback (single-threaded on ith 0): given the router's expert ids, ensure every touched
// expert is resident - reserving cache slots and demand-loading misses, stalling until they commit -
// then rewrite each id to its cache slot. this only relabels ids, so the same experts are computed
// in the same order; the result matches a non-streamed run (bit-exact when both paths use the same
// kernels, as on CUDA; a CPU build that repacks the non-streamed weights can differ in the last bits).
void llama_moe_stream_remap(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * sl  = (llama_moe_stream_layer *) userdata;
    auto * mgr = sl->mgr;

    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_are_same_shape(a, dst));

    const int64_t n = ggml_nelements(a);

    const int32_t * ids = (const int32_t *) a->data;
          int32_t * out = (int32_t *) dst->data;

    std::unique_lock<std::mutex> lk(mgr->mtx);

    if (mgr->load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }

    mgr->stats.n_calls++;
    mgr->start_workers_locked();

    if (mgr->n_window > 0) {
        // layer-window mode: the pool slot holds the FULL layer at identity slots (expert e ->
        // slot e), so the load is routing-independent and valid across ubatches/warmup. wait for
        // the load (prefetched from the previous layer, or demand-loaded by the attention wait-op
        // that runs before this layer's attention GEMMs), then pass the ids through unchanged.
        const int64_t t0 = ggml_time_us();
        mgr->ensure_layer_resident_locked(lk, *sl);
        mgr->stats.t_stall_us += ggml_time_us() - t0;

        for (int64_t i = 0; i < n; i++) {
            out[i] = ids[i];
        }
        lk.unlock();
        mgr->prefetch_layer(sl->il + 1);
        return;
    }

    // distinct experts touched by this ubatch, in first-use order
    sl->touched.assign(sl->n_expert, 0);
    sl->uniq.clear();
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && (uint32_t) e < sl->n_expert);
        if (!sl->touched[e]) {
            sl->touched[e] = 1;
            sl->uniq.push_back(e);
        }
    }

    if (sl->uniq.size() > sl->n_slots) {
        GGML_ABORT("MoE expert streaming: layer %d needs %zu distinct experts but the cache has only %u slots; "
                   "increase --moe-stream-cache or reduce the ubatch size (-ub)",
                sl->il, sl->uniq.size(), sl->n_slots);
    }

    // route hotness for eviction; halved periodically so a formerly-hot expert ages out
    for (const int32_t e : sl->uniq) {
        sat_inc(sl->route_hotness[e]);
    }
    if (mgr->hot_decay_interval > 0 && mgr->stats.n_calls % mgr->hot_decay_interval == 0) {
        for (auto & sl2 : mgr->layers) {
            if (sl2) {
                for (auto & h : sl2->route_hotness) {
                    h >>= 1;
                }
            }
        }
    }

    // classify the touched experts; reserve and enqueue demand loads in deterministic order
    std::fill(sl->keep.begin(), sl->keep.end(), 0);
    sl->demand_slots.clear();

    bool waited = false;
    for (const int32_t e : sl->uniq) {
        const auto it = sl->expert_slot.find(e);
        if (it != sl->expert_slot.end()) {
            const int32_t s = it->second;
            if (sl->slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
                mgr->q_demand.push_back({ sl, e, s, sl->slot_gen[s] });
                mgr->cv_work.notify_one();
                waited = true;
            }
            mgr->stats.n_hit++;
            sl->keep[s] = 1;
            sl->demand_slots.push_back(s);
        } else {
            int32_t v;
            while ((v = mgr->pick_victim_locked(*sl, sl->keep.data())) < 0) {
                // every allowed slot is loading; wait for a commit and retry
                mgr->cv_done.wait(lk);
                if (mgr->load_failed) {
                    GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
                }
            }
            if (!sl->seen[e]) {
                mgr->stats.n_miss_cold++;
            }
            mgr->reserve_slot_locked(*sl, e, v);
            mgr->q_demand.push_back({ sl, e, v, sl->slot_gen[v] });
            mgr->cv_work.notify_one();
            mgr->stats.n_miss++;
            waited = true;
            sl->keep[v] = 1;
            sl->demand_slots.push_back(v);
        }
    }

    if (waited) {
        const int64_t t0 = ggml_time_us();
        mgr->cv_done.wait(lk, [&]{
            if (mgr->load_failed) {
                return true;
            }
            for (const int32_t s : sl->demand_slots) {
                if (sl->slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                    return false;
                }
            }
            return true;
        });
        if (mgr->load_failed) {
            GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
        }
        mgr->stats.t_stall_us += ggml_time_us() - t0;
    }

    for (int64_t i = 0; i < n; i++) {
        const int32_t s = sl->expert_slot.at(ids[i]);
        sl->slot_last_use[s] = ++sl->use_counter;
        out[i] = s;
    }
    if (mgr->n_window > 0 && sl->il < 2) {
        fprintf(stderr, "DBG remapout il=%d n=%lld in[0..7]=%d,%d,%d,%d,%d,%d,%d,%d out[0..7]=%d,%d,%d,%d,%d,%d,%d,%d\n",
                sl->il, (long long) n, ids[0],ids[1],ids[2],ids[3],ids[4],ids[5],ids[6],ids[7],
                out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
    }

    if (mgr->n_window > 0 && std::getenv("LLAMA_MOE_STREAM_NO_PREFETCH") == nullptr) {
        // layer-window mode: prefetch the next layer's full expert set during this layer's compute
        lk.unlock();
        mgr->prefetch_layer(sl->il + 1);
    }
}

// Custom-op callback of the attention wait-op (window mode): a no-op map_custom1 on the attention
// input, inserted by build_lora_mm before the first streamed attention GEMM of each layer. it makes
// the layer's pool slot resident - the previous layer's remap prefetched it, so the wait is short -
// then passes the activations through unchanged. the graph orders it before the GEMM (data
// dependency), so the GEMM cannot read unwritten pool memory.
void llama_moe_stream_wait(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * sl  = (llama_moe_stream_layer *) userdata;
    auto * mgr = sl->mgr;

    GGML_ASSERT(a->type == dst->type);
    GGML_ASSERT(ggml_are_same_shape(a, dst));

    std::unique_lock<std::mutex> lk(mgr->mtx);

    if (mgr->load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }

    mgr->stats.n_attn_waits++;

    const int64_t t0 = ggml_time_us();
    mgr->ensure_layer_resident_locked(lk, *sl);
    mgr->stats.t_stall_attn_us += ggml_time_us() - t0;

    // no-op: pass the activations through unchanged
    memcpy(dst->data, a->data, ggml_nbytes(a));
    lk.unlock();

    // prefetch the next layer HERE (start of this layer) so its load overlaps this layer's FULL
    // compute (attention + experts); the remap also prefetches it (idempotent no-op). only safe
    // for W >= 2: the next layer's slot differs from this layer's, and this layer's stream was
    // drained by the wait-op's input copy, so overwriting the next slot is safe. at W=1 the remap
    // prefetch (after this layer's attention GEMMs) is the correct trigger.
    if (mgr->n_window >= 2) {
        mgr->prefetch_layer(sl->il + 1);
    }
}

// stable per-wave userdata; grows lazily and records the per-wave expert capacity (set at build)
llama_moe_stream_wave * llama_moe_stream_layer::wave_userdata(int32_t wave, uint32_t capacity) {
    GGML_ASSERT(capacity >= 1 && capacity <= n_slots);
    plan_capacity = capacity;
    while ((size_t) wave >= wave_ud.size()) {
        auto ud = std::make_unique<llama_moe_stream_wave>();
        ud->sl   = this;
        ud->wave = (int32_t) wave_ud.size();
        wave_ud.push_back(std::move(ud));
    }
    return wave_ud[wave].get();
}

// wave 0 of a ubatch: record the distinct touched experts (sl.uniq, first-use order) and split them
// into consecutive groups of plan_capacity, one group per wave (sl.expert_wave[e] = e's wave)
void llama_moe_stream::plan_waves_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n) {
    stats.n_calls++;
    start_workers_locked();

    sl.touched.assign(sl.n_expert, 0);
    sl.uniq.clear();
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && (uint32_t) e < sl.n_expert);
        if (!sl.touched[e]) {
            sl.touched[e] = 1;
            sl.uniq.push_back(e);
        }
    }

    GGML_ASSERT(sl.plan_capacity > 0);
    sl.expert_wave.assign(sl.n_expert, 0xff);
    for (size_t i = 0; i < sl.uniq.size(); i++) {
        GGML_ASSERT(i/sl.plan_capacity < 0xff);
        sl.expert_wave[sl.uniq[i]] = (uint8_t) (i/sl.plan_capacity);
    }
    sl.plan_n_waves   = (uint32_t) ((sl.uniq.size() + sl.plan_capacity - 1)/sl.plan_capacity);
    sl.plan_next_wave = 0;
}

// make wave w's expert slice (uniq[w*cap .. +count)) resident, waiting for its loads, and best-effort
// preload the next wave so its loads overlap this wave's compute. leaves sl.demand_slots = this wave's
// slots and sl.plan_pool = the resident parking pool (>= n_ids slots) the emit draws masked pairs from
void llama_moe_stream::stage_wave_locked(std::unique_lock<std::mutex> & lk, llama_moe_stream_layer & sl, int32_t w, uint32_t n_ids) {
    const size_t first = (size_t) w*sl.plan_capacity;
    const size_t count = first < sl.uniq.size() ? std::min<size_t>(sl.plan_capacity, sl.uniq.size() - first) : 0;

    std::fill(sl.keep.begin(), sl.keep.end(), 0);
    sl.demand_slots.clear();

    // a small final wave has fewer than n_ids own slots; borrow the rest from the previous wave's
    //   pool so every token row has n_ids distinct resident parking slots for its masked pairs
    std::vector<int32_t> borrowed;
    if (count < n_ids) {
        GGML_ASSERT(sl.plan_pool.size() >= n_ids - count);
        for (size_t i = 0; i < n_ids - count; i++) {
            borrowed.push_back(sl.plan_pool[i]);
            sl.keep[sl.plan_pool[i]] = 1; // parking slots must survive this wave's loads
        }
    }

    // protect the next wave's already-resident experts so this wave's victims do not evict them
    const size_t nfirst = first + sl.plan_capacity;
    const size_t ncount = nfirst < sl.uniq.size() ? std::min<size_t>(sl.plan_capacity, sl.uniq.size() - nfirst) : 0;
    for (size_t i = nfirst; i < nfirst + ncount; i++) {
        const auto it = sl.expert_slot.find(sl.uniq[i]);
        if (it != sl.expert_slot.end()) {
            sl.keep[it->second] = 1;
        }
    }

    // reserve and demand-load this wave's experts (per-expert, same path as the decode remap)
    bool waited = false;
    if (count > 0) {
        stats.n_waves_run++;
        for (size_t i = first; i < first + count; i++) {
            const int32_t e  = sl.uniq[i];
            const auto    it = sl.expert_slot.find(e);
            if (it != sl.expert_slot.end()) {
                // already in the cache (resident, or still loading from the previous wave's preload)
                const int32_t s = it->second;
                if (sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
                    q_demand.push_back({ &sl, e, s, sl.slot_gen[s] }); // promote to demand, wait for it
                    cv_work.notify_one();
                    waited = true;
                } else {
                    stats.n_preload_ready++; // resident from the previous wave's preload
                }
                stats.n_hit++;
                sl.keep[s] = 1;
                sl.demand_slots.push_back(s);
            } else {
                // miss: evict a non-kept slot and queue the load
                int32_t v;
                while ((v = pick_victim_locked(sl, sl.keep.data())) < 0) {
                    cv_done.wait(lk);
                    if (load_failed) {
                        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
                    }
                }
                if (!sl.seen[e]) {
                    stats.n_miss_cold++;
                }
                reserve_slot_locked(sl, e, v);
                q_demand.push_back({ &sl, e, v, sl.slot_gen[v] });
                cv_work.notify_one();
                stats.n_miss++;
                waited = true;
                sl.keep[v] = 1;
                sl.demand_slots.push_back(v);
            }
        }
    }

    // best-effort preload of the next wave so its loads overlap this wave's compute; never waits,
    //   whatever cannot be reserved now simply becomes the next wave's demand load
    if (std::getenv("LLAMA_MOE_STREAM_NO_PRELOAD") == nullptr) {
        for (size_t i = nfirst; i < nfirst + ncount; i++) {
            const int32_t e = sl.uniq[i];
            if (sl.expert_slot.find(e) != sl.expert_slot.end()) {
                continue;
            }
            const int32_t v = pick_victim_locked(sl, sl.keep.data());
            if (v < 0) {
                continue;
            }
            if (!sl.seen[e]) {
                stats.n_miss_cold++;
            }
            reserve_slot_locked(sl, e, v);
            sl.keep[v] = 1;
            q_demand.push_back({ &sl, e, v, sl.slot_gen[v] });
            cv_work.notify_one();
            stats.n_preload_issued++;
        }
    }

    if (waited) {
        const int64_t t0 = ggml_time_us();
        cv_done.wait(lk, [&]{
            if (load_failed) {
                return true;
            }
            for (const int32_t s : sl.demand_slots) {
                if (sl.slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                    return false;
                }
            }
            return true;
        });
        if (load_failed) {
            GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
        }
        stats.t_stall_wave_us += ggml_time_us() - t0;
    }

    // parking pool: this wave's own resident slots plus the borrowed ones (all keep-protected;
    //   the next same-layer reservation is ordered after this wave's GEMMs by the graph)
    sl.plan_pool = sl.demand_slots;
    sl.plan_pool.insert(sl.plan_pool.end(), borrowed.begin(), borrowed.end());
    GGML_ASSERT(sl.plan_pool.size() >= n_ids);
}

// write out[i] = the cache slot the GEMM should index for each (token, expert) pair of wave w, one
// token row at a time: pairs whose expert is in this wave get its real slot; the rest park on distinct
// resident pool slots (pool_used prevents a repeat within the row, required by the Metal kernel)
void llama_moe_stream::emit_wave_slots(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out,
        int32_t w, uint32_t n_ids, int64_t n_tok) {
    for (int64_t t = 0; t < n_tok; t++) {
        sl.pool_used.clear();

        // pass 1: pairs whose expert belongs to this wave -> that expert's real (resident) slot
        for (uint32_t kk = 0; kk < n_ids; kk++) {
            const int64_t i = t*n_ids + kk;
            const int32_t e = ids[i];
            GGML_ASSERT(sl.expert_wave[e] != 0xff);
            if (sl.expert_wave[e] == (uint8_t) w) {
                const int32_t s = sl.expert_slot.at(e);
                GGML_ASSERT(sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
                sl.slot_last_use[s] = ++sl.use_counter;
                out[i] = s;
                sl.pool_used.push_back(s);
            }
        }

        // pass 2: the remaining (masked) pairs -> the next pool slot not yet used in this row
        size_t pi = 0;
        for (uint32_t kk = 0; kk < n_ids; kk++) {
            const int64_t i = t*n_ids + kk;
            if (sl.expert_wave[ids[i]] == (uint8_t) w) {
                continue;
            }
            while (std::find(sl.pool_used.begin(), sl.pool_used.end(), sl.plan_pool[pi]) != sl.pool_used.end()) {
                pi++;
                GGML_ASSERT(pi < sl.plan_pool.size());
            }
            GGML_ASSERT(sl.slot_state[sl.plan_pool[pi]] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
            out[i] = sl.plan_pool[pi];
            sl.pool_used.push_back(sl.plan_pool[pi]);
            pi++;
        }
    }
}

// Custom-op callback for one pass of multi-pass prefill. When a ubatch touches more experts than the
// cache holds, build_moe_ffn runs the expert GEMMs in several waves; this runs once per wave (single-
// threaded on ith 0), in wave order. For wave w it makes that wave's expert slice resident (preloading
// the next wave), then writes the slot ids the GEMM indexes - see plan_waves_locked / stage_wave_locked
// / emit_wave_slots. The router's expert choice is untouched, so the output matches a non-streamed run.
void llama_moe_stream_wave_ids(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * ud  = (llama_moe_stream_wave *) userdata;
    auto * sl  = ud->sl;
    auto * mgr = sl->mgr;

    const int32_t w = ud->wave;

    const ggml_tensor * a = dst->src[0]; // contiguous selected ids
    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(a));
    GGML_ASSERT(dst->data != a->data); // the emit must not clobber the ids other waves read

    const int64_t   n   = ggml_nelements(a);
    const int32_t * ids = (const int32_t *) a->data;
          int32_t * out = (int32_t *) dst->data;

    std::unique_lock<std::mutex> lk(mgr->mtx);

    if (mgr->load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }

    mgr->stats.n_wave_calls++;

    if (w == 0) {
        mgr->plan_waves_locked(*sl, ids, n);
    }
    GGML_ASSERT(sl->plan_next_wave == w); // waves must run in order (enforced by the graph ordering token)

    const uint32_t n_ids = (uint32_t) a->ne[0]; // experts per token (n_expert_used)

    mgr->stage_wave_locked(lk, *sl, w, n_ids); // make this wave resident, preload the next, build the pool
    sl->plan_next_wave = w + 1;

    mgr->emit_wave_slots(*sl, ids, out, w, n_ids, a->ne[1]);
}

// multi-pass prefill: 1.0 for pairs whose expert belongs to wave w, 0.0 otherwise; multiplied into
// this wave's expert GEMM output so the masked-out (parked) pairs contribute nothing to the sum
void llama_moe_stream_wave_mask(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * ud  = (llama_moe_stream_wave *) userdata;
    auto * sl  = ud->sl;
    auto * mgr = sl->mgr;

    const int32_t w = ud->wave;

    const ggml_tensor * a = dst->src[0]; // contiguous selected ids
    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(a));

    const int64_t   n   = ggml_nelements(a);
    const int32_t * ids = (const int32_t *) a->data;
          float   * out = (float *) dst->data;

    std::lock_guard<std::mutex> lock(mgr->mtx);

    GGML_ASSERT(sl->plan_next_wave > w); // this wave's ids op has already run

    for (int64_t i = 0; i < n; i++) {
        out[i] = sl->expert_wave[ids[i]] == (uint8_t) w ? 1.0f : 0.0f;
    }
}
