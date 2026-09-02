# Implementation prompt: fix the two P1 NUMA bugs (memory doubling + numa-init wiring)

**Branch:** `feat/prefill-gpu-ada` (working tree on top of `9b2214642`). Build with CPU-only
(`cmake -B build -DCMAKE_BUILD_TYPE=Release`), testbed is a single-NUMA box (must stay inert there),
**functional target is the P720** (dual-socket, 2 NUMA nodes, Debian).

## Goal

Single-process CPU tensor-parallel decode for llama.cpp (spec: `docs/cpu-tp-single-process-numa-spec.md`).
Three increments are implemented and committed; **two bugs remain** that must be fixed before the
feature works on the P720. Implement both fixes, build, verify 1-node inertness, commit, push
(`git push xpire feat/prefill-gpu-ada`).

---

## Background (what exists)

- **Increment 1 (committed):** `ggml_tensor.numa_node` (int32_t, default −1, in the trailing padding,
  ABI-stable). `ggml_compute_forward_mul_mat_id` (ggml/src/ggml-cpu/ggml-cpu.c) honors it: after the
  pool barrier, threads not on the op's node return (barrier-synced, no work). Node mapping =
  `ith % ggml_numa_nodes()` (matches `set_numa_thread_affinity` under `--numa distribute`). Also
  falls back to `src0->numa_node` (the weight) when `dst->numa_node < 0`.
- **Increment 2 (committed):** `llama_numa_mbind_tensor()` (raw `mbind` syscall, `MPOL_BIND` +
  `MPOL_MF_MOVE` — note MOVE is `(1<<1)`), applied in a post-load pass in `load_all_data`
  (src/llama-model-loader.cpp). `LLAMA_NUMA_PLACE_LAYER=1` env tags tensors by `tn.bid % 2`.
- **Increment 3 (committed):** `llama_numa_shard_tensor(src, ctx)` (src/llama-model-loader.cpp) splits
  a loaded expert tensor's n_ff (ne[0]) into two per-node shards (strided per-row halves, page-aligned
  heap wrapped in `ggml_backend_cpu_buffer_from_ptr`, mbind'd, `numa_node` 0/1). The model pimpl owns
  the shard context + data. `build_moe_ffn` (src/llama-graph.cpp) runs the expert-GEMM lambda twice
  (once per shard, same ids) and combines with `ggml_add` when the shard registry matches. Registry
  wired through `llm_graph_params.tp_shards`. Enabled by `LLAMA_NUMA_TENSOR_SPLIT=1` **and**
  `ggml_numa_nodes() > 1`.

## BUG 1 — `llama_numa_init` is never called by llama-cli/llama-server

**Symptom:** on the P720, `ggml_numa_nodes()` returns 0 even with `--numa distribute`, so the shard
gate (`is_exps && ggml_numa_nodes() > 1 && getenv("LLAMA_NUMA_TENSOR_SPLIT")`) never passes — the
split never engages (numastat shows the whole model on one node, total ≈ model size).

**Root cause:** `llama_numa_init(params.numa)` (defined src/llama.cpp:136, wraps the CPU backend's
`ggml_backend_cpu_numa_init`) is only called by some tools — grep shows:
`tools/llama-bench/llama-bench.cpp:2391`, `tools/completion/completion.cpp:129`,
`tools/imatrix/imatrix.cpp:1150`, `tools/batched-bench/batched-bench.cpp:42`,
`tools/fit-params/fit-params.cpp:28`, `tools/cvector-generator/cvector-generator.cpp:426`.
**It is NOT called by `tools/llama-cli` or `tools/server` (llama-server).** Without it,
`ggml_numa_init` never populates `g_state.numa.n_nodes` (ggml/src/ggml-cpu/ggml-cpu.c ~636), so
`ggml_numa_nodes()` stays 0 and `set_numa_thread_affinity` never binds threads.

**Fix:** call `llama_numa_init(params.numa)` in the llama-cli and llama-server init paths, at the
same point the other tools do (after `common_params_parse`, before model load — copy the pattern from
`tools/llama-bench/llama-bench.cpp:2391`). Verify it's called for both, guarded as the other tools
are. (Do NOT rely on llama.cpp's internal `--numa` flag doing it.)

## BUG 2 — the split doubles the expert memory (fatal for the big models)

**Symptom (design flaw):** in the moe-stream expert branch of `llama_model_base::create_tensor`
(src/llama-model.cpp), the full host tensor is created and loaded (`ml.create_tensor(...)`, resident
for the unsharded decode path) and the two shards are created **as copies on top**. Result: the
expert weights exist twice. A3B ≈ 40 GB resident; **V4-Flash 156 GB → ~312 GB, GLM 157 GB → ~314 GB
— neither fits the P720's 192 GB.** The split's whole purpose is the big models, so this is fatal.

**Design intent:** the shards *replace* the full tensor — each socket holds half of each expert's
weights, total memory = model size (the non-expert tensors stay single-copy).

**Fix:**
1. Add an all-or-nothing guard: track whether every expert tensor of the model sharded successfully
   (the pimpl already has `tp_shards`; add e.g. `bool tp_split_partial` set when any
   `llama_numa_shard_tensor` call returns null shards). If partial, the split is disabled entirely
   (clear `tp_shards`, keep all full host tensors) — the graph must never see a freed host.
2. When the split is fully engaged, **free the full host tensor's buffer after the shard copies
   exist** (`ggml_backend_buffer_free(host->buffer); host->buffer = nullptr;`), so the resident
   expert memory is the shards only. Do this once, after the model's tensors are loaded (after the
   `load_all_data` call in the model load path) — the shard copies read the host data during load,
   so the free must happen after both.
3. Verify: on a 2-node machine the split decode still produces byte-identical output to the
   unsplit path (the moe-stream decode swap feeds the host tensors into the graph; the shard registry
   keys on the host *pointer*, so the graph's shard path uses the shards — the freed host data is
   never read as long as the registry always matches, which the all-or-nothing guard guarantees).

## Files to touch (expected)

- `tools/llama-cli/llama-cli.cpp` and `tools/server/server.cpp` (or their common init): add
  `llama_numa_init(params.numa)`.
- `src/llama-model.cpp`: the shard gate + failure tracking + the post-load buffer free. The impl
  struct (pimpl) already has `tp_shards` and `tp_shard_ctx` — add the partial flag + the free loop.
- Possibly `src/llama-model.h` if a flag accessor is needed.

## Constraints

- Do not modify mainline behavior: everything stays gated behind `LLAMA_NUMA_TENSOR_SPLIT=1` +
  multi-node. On the 1-node testbed the feature must be fully inert (no behavior change, no crash).
- No new dependencies (the mbind/affinity use raw syscalls already in the code).
- Keep the existing moe-stream / attention-streaming machinery intact.
- Build CPU-only; also confirm the CUDA build (`-DGGML_CUDA=ON`) still compiles (the ggml.h struct
  change is ABI-stable — verify).
- The P720's `--numa distribute` may interact with moe-stream thread binding (research F1) — if the
  combination crashes/hangs, that's a known risk to report, not to paper over.

## Verification checklist

1. Testbed (1 node): build clean; bench without env == bench with env == previous numbers.
2. P720: `LLAMA_NUMA_TENSOR_SPLIT=1 llama-cli ... --numa distribute --moe-stream-window 1 ...` runs.
3. numastat (P720): total ≈ model size (NOT 2×), expert pages split across both nodes (~50/50).
4. Oracle (P720): split output byte-identical to unsplit (same seed/prompt), both with
   `--numa distribute`.
5. Commit with a clear message; push to `xpire/feat/prefill-gpu-ada`.
