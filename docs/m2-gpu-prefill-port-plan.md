# M2 — Port plan: GPU prefill for MoE experts (feat/prefill-gpu-ada)

> Status: **assessment complete 2026-08-30**. This is the M2 deliverable of the roadmap in
> `research/llm-homelab-to-sort-out.md` §3. It covers: what PR #25294 gives us, what transfers,
> what gets discarded, the RAM→GPU adaptation, where #26659 plugs in, and the concrete starting
> diff already applied on this branch. M1 baseline numbers: see the bench section at the bottom.

## 0. TL;DR

- PR #25294 (`feat/moe-streaming-core`, 2 feature commits, **+1450/-16, 17 files**) is the streaming
  core we port: **cache slots + id-remap + wave-partitioned prefill** — all of it transferable.
- Its **disk path (pread pool, O_DIRECT, mmap-disable) is discarded** — our weights are RAM-resident.
- It is **807 commits behind current master** (16/17 touched files changed since; llama-graph.cpp
  drifted 14 commits, the loader/model/context/mmap set 61). A cherry-pick alone will not apply —
  the rebase is the real work (roadmap's 2–3 week estimate holds).
- The machinery compiles against current mainline **as-is** (verified: `llama_file` already exposes
  `file_id()`/`has_direct_io()`, `select_weight_buft`/`weight_buft_supported` still exist,
  `cparams.op_offload` exists, the CUDA backend still exposes the #26659 host-buffer hook).
- **This branch already carries the skeleton commit** (module + params + loader routing + context
  guards; graph injection deferred). Next: graph injection → RAM→GPU copy-pool adaptation.

## 1. What PR #25294 actually is

Two commits (branch `feat/moe-streaming-core`, author Junchao Lyu):

| Commit | Content |
|---|---|
| `4260e4608` | "llama: stream MoE routed experts from disk (core + async pool + O_DIRECT)" — 17 files, +1043. |
| `db5268ffa` | "support waved prefill" — 4 files, +418. |

### 1.1 The core mechanism (all transferable)

**Loader routing (`llama-model.cpp`/`llama-model-loader.cpp`)**
- New `TENSOR_STREAMED` flag (1<<4). `create_tensor()` routes `ffn_{gate,up,down,gate_up}_exps.weight`
  tensors to the stream engine instead of materializing them: accounting kept consistent via a
  `TENSOR_STREAMED`-flagged skip call, and a cache tensor is returned in place of the real one.
- `llama_moe_stream_resolve_slots()`: per-layer cache slot count from `--moe-stream-cache`
  (GiB budget or `Ns` slots), with an auto default of `clamp(2*n_expert_used, 16, n_expert)`.
- `llama_moe_stream_select_buft()`: like `select_weight_buft` but only device **default** bufts
  (extra bufts like CPU repacking can't do partial slot writes).

**Cache slots (`src/llama-moe-stream.{h,cpp}`, +717)**
- Per streamed layer: a cache tensor `{ne0, ne1, n_slots}` (n_slots expert slabs of `nb_expert`
  bytes), allocated on the layer's chosen buft. Slot state machine `EMPTY → LOADING → RESIDENT`.
- `expert_slot` map (expert id → slot), reservation generations (`slot_gen`) so in-flight loads
  for evicted occupants are recognized stale, LRU stamps + decaying **route hotness** eviction
  (halved every `64 * n_streamed_layers` remap calls).
- Async load pool: workers pop demand work, read the expert slab(s) (pread), commit via
  `ggml_backend_tensor_set()`, notify `cv_done`.

**ID-remap op (`llama-graph.cpp`)**
- In `build_moe_ffn()`, after top-k: `ggml_map_custom1(cont(selected_experts),
  llama_moe_stream_remap, 1, msl)`. The callback (single-threaded, ith 0) makes every touched
  expert resident (stalling on misses), then rewrites each expert id to its cache slot.
- `build_lora_mm_id()` gains an `ids_scale` param: per-expert scale/biases (`w_s`) are gathered
  with the **original** ids, the GEMMs (`ggml_mul_mat_id`) index the **remapped** slots.
- `llama_moe_stream_layer::matches()` guards against a second non-streamed expert group on the
  same layer index (e.g. grovemoe chexps).

**Wave-partitioned prefill (`db5268ffa`)**
- When a ubatch touches more distinct experts than the cache holds, the touched set is split into
  waves of `cap = (n_slots - n_expert_used)/2` (cache holds: this wave + next-wave preload +
  n_expert_used parking slots). Per wave: `llama_moe_stream_wave_ids` (make wave resident, preload
  next wave best-effort, emit slot ids with masked pairs parked on distinct resident pool slots)
  + `llama_moe_stream_wave_mask` (1.0/0.0 per pair) + the expert GEMMs + accumulate.
- **Each expert is loaded at most once per ubatch regardless of batch size** — this is the
  "wave-partitioned, each expert loaded once" insight the roadmap calls out.
- `graph_max_nodes()` adds bounded extra node budget for waves.

**Context guards (`llama-context.cpp`)**
- Single-pass mode: cap `n_ubatch ≤ n_slots / n_expert_used` (a mul_mat_id needs every expert a
  ubatch touches resident at once). Wave mode removes the cap.
- Disable `op_offload` when the stream cache lives in host memory (op offload snapshots host
  weights per split and assumes they don't change mid-graph).

## 2. What to DISCARD (disk path — our weights are RAM-resident)

| Piece | Where | Verdict |
|---|---|---|
| `pread` pool + `llama_moe_stream_pread` (O_DIRECT aligned reads, EINTR loops) | llama-moe-stream.cpp | **Discard** → replaced by a copy pool (RAM → device) |
| `open_files()` / `file_paths` / `llama_file` reopen + `file_idx`/`offs` slab offsets | llama-moe-stream, model-loader | **Discard** (tensor data is already in host buffers) |
| `--moe-stream-direct`, O_DIRECT probe + fallback | arg.cpp, llama-moe-stream | **Discard** |
| `params.use_mmap = false` when streaming | llama.cpp | **Discard** — mmap is exactly what we keep (weights in RAM) |
| `--moe-stream-io-threads` | arg.cpp | Repurpose → copy-pool threads |
| `LLAMA_MOE_STREAM_DEBUG` / `NO_PRELOAD` env | llama-moe-stream.cpp | **Keep** (useful) |

## 3. What to ADAPT (disk → RAM → GPU)

The insight that makes this a *port* and not a rewrite: **the streaming engine is source-agnostic**.
It does "make this expert resident in the cache tensor, tell me its slot, evict cold experts".
Today the source is a file pread; for us the source is the **host-RAM expert slab** (the model
loads fully into RAM; only attention + KV go to GPU via `-ngl`). The cache tensor moves from a
host/CPU buft to a **CUDA buft**.

1. **Cache buffer type** — `llama_moe_stream_select_buft()` already prefers the device default
   buft; the change is in *where the layer's expert group lands*: streamed layers route their
   expert cache to the CUDA backend instead of CPU. `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` stays.
2. **Load path** — worker body becomes:
   `ggml_backend_tensor_set_async(cuda_backend, cache, host_expert_ptr + slot*nb, 0, nb)` (or the
   synchronous `ggml_backend_tensor_set` first — async is M4). The source pointer comes from a
   per-tensor registration of the expert tensor's **host buffer address**, not a file offset.
3. **`cache_on_host` guard** (op_offload disable) — becomes moot when the cache is device-resident;
   drop or invert. Keep the wave-mode n_ubatch logic.
4. **Eviction policy** — route hotness + LRU: **keep as-is**. The 12 GB VRAM budget is the cache
   constraint (vs RAM before); `--moe-stream-cache` maps to a VRAM GiB budget now.
5. **Bit-exactness caveat** — PR docs: output is bit-identical to unstreamed when both paths use
   the same kernels (true on CUDA). Our split (experts on GPU for prefill, CPU for decode with
   `--n-cpu-moe`) deliberately uses *different* kernels per phase — fine, phases don't mix.

### Batch gating (constraint from the roadmap, M3 knob but M2 design)
Engage the GPU-prefill expert path only above a token threshold (LvLLM's 4096 lesson) — short
prompts stay on the CPU path where expert GEMMs are tiny and H2D staging would dominate. Concretely:
in `llama_decode()`, branch on `n_tokens >= prefill_gpu_min_batch` when choosing the graph path;
decode (`n_tokens == 1` or small) is **untouched** (`--n-cpu-moe` as today).

## 4. Where #26659 plugs in (host-buffer / pinned staging)

- PR #26659 (47 lines): `llama_mmap` ctor/dtor call `ggml_backend_register_host_buffer()` /
  `unregister` resolved via `ggml_backend_reg_get_proc_address()`. The CUDA backend implements it
  (ggml-cuda.cu:4641) gated on env `GGML_CUDA_REGISTER_HOST`; SYCL exposes the same names. No
  backend-specific dependency in src/.
- **Why we need it**: with the cache tensor on the GPU, every expert copy is a H2D DMA. Page-locked
  source pages (the mmap'd weights registered as host buffers) make those copies DMA instead of
  bounce-buffered — this is the LvLLM pinned-staging pattern (their PR #36) made llama.cpp-native.
- **Where it plugs in**: before/alongside the copy pool in step M2-3. The mapping is created at
  load (`llama_mmap` ctor already matches the mapping lifetime); M4 double-buffering builds on the
  same pinned staging.
- **Rebase note**: current mainline `llama_mmap` ctor has a 4th parameter vs the PR's era — the
  patch's ctor-body insertion still applies, just against the newer signature.

## 5. File-by-file transfer map

| File | PR change | Transfer | Drift risk |
|---|---|---|---|
| `src/llama-moe-stream.{h,cpp}` | new module (+717) | ✅ keep (rewrite load path) | none — self-contained |
| `src/llama-graph.cpp/h` | remap op, ids_scale, wave ops, graph_max_nodes | ✅ keep | **high** (14 commits) |
| `src/llama-model.cpp/h` | TENSOR_STREAMED routing, resolve_slots, accessor, defaults | ✅ keep | medium |
| `src/llama-model-loader.cpp/h` | TENSOR_STREAMED skip, export buft helpers | ✅ keep | medium |
| `src/llama-context.cpp` | n_ubatch cap, op_offload guard, wave stats | ✅ keep | low |
| `common/arg.cpp`, `common/common.h/cpp` | `--moe-stream*` flags | ✅ keep (drop direct/io-threads rename) | low |
| `include/llama.h` | llama_model_params fields | ✅ keep | low |
| `src/llama-adapter.cpp` | LoRA guard for streamed tensors | ✅ keep | low |
| `src/llama.cpp` | mmap disable | ❌ discard | — |
| `src/llama-mmap.cpp/h` | #26659 host registration | ✅ keep (separate commit) | low |

## 6. Suggested implementation order (M2)

1. **[done on this branch] Skeleton** — module verbatim + CMake entry + params plumbing
   (arg/common/llama.h/model) + loader routing + context guards, graph injection deferred.
   Proof: compiles against current master (verified `llama_file`, `select_weight_buft`,
   `op_offload` all present).
2. **Graph injection** — cherry-pick the `llama-graph.cpp/h` hunks onto the current
   `build_moe_ffn` (same shape as the PR's era — verified by eye); `ids_scale` on
   `build_lora_mm_id`; wave ops; `graph_max_nodes` budget.
3. **RAM→GPU adaptation** — replace the pread pool with a copy pool from registered host
   pointers; cache buft → CUDA; delete O_DIRECT/mmap-disable; wire `--moe-stream-cache` as VRAM
   budget. (Start of the `#26659` host-registration commit.)
4. **Batch gate + knob** — `--prefill-min-batch-size` (default 4096); decode path untouched.
5. **Verify** — `llama-bench -p 512,2048,8192` before/after; expect prefill to move from CPU
   (bandwidth-bound) toward the 4070 Ti's FP16 ceiling; watch the stream stats line
   (`llama_moe_stream_print_stats` hit rate + stall). M3 placement policy / M4 overlap follow.

## 7. Risks / open items (honest)

- **Wave-mode + CUDA MUL_MAT_ID**: the PR ran on CUDA + Metal; the "parking slots must be
  distinct per token row" constraint is a Metal kernel requirement — CUDA doesn't need it but
  tolerates it. Expect no functional issue; verify by eye during step 2.
- **No multi-context** support in the PR (one context can evict another's in-flight slots) —
  documented limitation; fine for llama-bench/llama-cli, note for llama-server (parallel > 1).
- **MXFP4 experts (GLM-5.3)**: cache tensor type is preserved (`meta->type`), cuBLAS/MMQ handles
  mxfp4 via the #22378 dequant path; `nb_expert` math is type-agnostic. Verify on the P720 later.
- **op_offload** currently defaults on in mainline; the PR's guard only trips when the cache is
  host-resident — with a CUDA cache it stays on. Keep the guard for the skeleton's correctness.
- The 807-commit drift means **this branch's skeleton must be rebased again** before upstream
  discussion; treat as a fork-first effort (roadmap already assumes fork).

## 8. M1 baseline (this box: Ryzen 7 7800X3D, 96 GB RAM, RTX 4070 Ti 12 GB)

Measured 2026-08-30 with the CUDA 13.3 build on `feat/prefill-gpu-ada` (commit `3ef6aa0dc`).
Model: Qwen3.6-35B-A3B-UD-Q4_K_XL (22 GB, HF-cached; also Q8_K_XL for the user-config check).
Raw JSON in `research/m1/`.

| Test | Pure CPU (`-ngl 0`) | Attn GPU, experts CPU (`-ngl 999 -ncmoe 40`) | Q8, user config (`-ngl 999 -ncmoe 35`, b1024) |
|---|---|---|---|
| PP 128 | 179 t/s | 220 t/s | _pending_ |
| PP 512 | 551 t/s | 619 t/s | — |
| PP 1024 | — | — | _pending_ |
| PP 2048 | 555 t/s | 612 t/s | — |
| TG 32/64/128 | 16.7 t/s | 51.3 t/s | _pending_ |

**Conclusion**: on this box the CPU (AVX-512 VNNI, ~3+ TOPS int8) prefills a 3B-active model
at 550-620 t/s — 2-4× above the 4070 Ti's mmq prefill ceiling (120-250 t/s). GPU prefill has no
payoff for A3B-class models here; the P720 + GLM-5.3 (Skylake, no VNNI, 31-47 t/s CPU) remains the
valid GPU-prefill target. Decode: attention-on-GPU lifts TG from 16.7 to 51.3 t/s — that offload
is already in production config.
