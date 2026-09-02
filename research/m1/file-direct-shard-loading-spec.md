# File-direct shard loading — kill the NUMA split's 2× load transient (2026-09-02)

Branch `feat/prefill-gpu-ada` · open item #2 from `session-context-2026-09-02.md` · status: implemented, testbed-validated

## 1. The problem

The NUMA tensor split (`LLAMA_NUMA_TENSOR_SPLIT=1`) materializes a full-size **host copy** of every
expert tensor and a second full-size **shard** allocation. Both live from create-time through the
finalize pass in `load_tensors`, so:

```
peak anon = host (1×) + shards (1×) = 2× model size
```

V4-Flash MXFP4 (156 GB) → ~312 GB on a 192 GB P720 → OOM before the split can even engage. Anything
> ~96 GB cannot split-load today. (Session context §4.5 / §6, item #2.)

## 2. Why the host exists at all — and why it shouldn't

`create_tensor` (moe-stream expert path, `src/llama-model.cpp` ~2110) creates the host in its buft
ctx, then **re-homes** it into a dedicated `ggml_backend_buft_alloc_buffer(CPU)` heap buffer so it
can be freed independently after the shard copy. The re-home is what forces `load_all_data` to
memcpy the file bytes into anon RAM (`ggml_backend_tensor_set`, the `cur->data != nullptr` branch).

Post-finalize the host is **pure staging**: it is freed and never read again — the graph runs the
two shard GEMMs, and the moe-stream I/O workers assemble cache slabs from the shards
(`set_host(..., sh0, sh1, axis)`). The bytes it holds are the GGUF's own bytes.

## 3. The fix

Under the plain-CPU mmap load path (the CPU-only default — `ml.use_mmap` + host buft == CPU default
buft), the loader already supports **aliasing tensors into the GGUF mapping**:
`load_tensors` creates a `buffer_from_host_ptr` view over the file range and `load_all_data`
`ggml_backend_tensor_alloc`s the tensor into it — zero anon, no memcpy. Non-split streamed hosts
already take this path today.

So: **when the split would alias the host to the mmap, skip the re-home.** The host stays in its
buft ctx; `load_all_data` points `host->data` at the mapping; the finalize
`llama_numa_shard_copy(shards, host, axis)` then reads the shard bytes **straight from the file
mapping** (faulting clean, reclaimable page cache) and writes the shards.

```
peak anon (CPU-only, mmap) = shards (1×) only   ← the 2× never exists
```

The finalize's `tp_host_bufs` free loop is empty in this mode; no "assign then free" remains — and
load is faster too (the host write + host read round trip disappears; the file is read once).

### 3.0 Why this needs a load-mode change (found during bring-up)

`llama_model_load` (`src/llama.cpp`) **forces `load_mode AUTO → NONE` whenever `params.moe_stream`** —
its I/O workers copy per-expert slabs from the materialized host tensors, and mmap-backed (page-cache)
hosts measured ~40% slower as the copy source (561 vs 788 t/s at ub512, GPU-hybrid). But under the
**split**, the workers copy from the *shards*, never the hosts — so the heap-forcing buys nothing and
would keep the 2× transient forever (and the mmap gate would never fire). Fix: skip the downgrade
when the split is active (same gate as `create_tensor`: env + `ggml_numa_nodes() > 1`):

```cpp
// src/llama.cpp, llama_model_load
const bool numa_split_active =
    params.moe_stream && getenv("LLAMA_NUMA_TENSOR_SPLIT") != nullptr && ggml_numa_nodes() > 1;
if (params.moe_stream && params.load_mode == LLAMA_LOAD_MODE_AUTO && !numa_split_active) {
    params.load_mode = LLAMA_LOAD_MODE_NONE;  // heap hosts (original behavior)
}
```

Explicit `--load-mode mmap` was never downgraded — that is how the file-direct path is validated on
the single-node testbed. On the P720 the auto path keeps mmap once the split engages.

### 3.1 Predicate (must not change the GPU-hybrid path)

The GPU-hybrid build picks the pinned `CUDA_Host` buft for streamed hosts (DMA-optimal for the
host→device decode swap). A pinned-host ctx is **not** mmap-viewable (not the device-default buft),
so there the host must stay materialized + re-homed exactly as today — otherwise its (shared) ctx
buffer would never be freed and the resident footprint would go from 1× to 2× permanently.

File-direct engages only when the loader's own buft selection for the host is the **plain CPU
default** AND mmap is on:

```cpp
const bool host_is_mmap_alias =
    ml.use_mmap &&
    select_weight_buft(hparams, w->tensor, GGML_OP_MUL_MAT_ID, host_buft_list) ==
        ggml_backend_cpu_buffer_type();
```

This replicates `ml.create_tensor`'s internal selection (layer-repeating MUL_MAT_ID expert tensor,
no overrides) with the same helper + same inputs, so it cannot diverge from the actual choice.
CPU-only → plain CPU buft → file-direct. GPU-hybrid → pinned buft → old materialize+re-home path.
`!ml.use_mmap` (load-mode none / direct-io) → old path.

### 3.2 Finalize

Unchanged logic; `tp_host_bufs` is empty under file-direct so the free loop no-ops. The
`host->buffer = nullptr` detach is kept (the graph must never execute a host weight under the
split; today it also prevents use-after-free of the freed dedicated buffer). Under file-direct the
data pointer stays valid (it aliases the live mapping), so even a stray read would see correct
bytes rather than garbage — strictly safer than before.

## 4. Exact changes

1. `src/llama-model.cpp`, `create_tensor`, `if (tp_split)` block (~2115): compute `host_is_mmap_alias`
   (3.1); run the existing re-home only when `!host_is_mmap_alias`.
2. `src/llama-model.cpp`, finalize comment (~1972) + engaged log line (~2031): note the mmap-alias
   mode.
3. `src/llama.cpp`, `llama_model_load`: keep mmap (skip the moe-stream AUTO→NONE downgrade) when the
   split is active (3.0); `#include "ggml-cpu.h"` for `ggml_numa_nodes()`.

## 5. Verification (testbed, single-node, A3B Q4_K_XL 22.8 GB)

The 1-node testbed cannot engage the split (`ggml_numa_nodes() > 1` gate). Engaged-path tests used
a **temporary gate bypass** (not committed) + explicit `--load-mode mmap` (single-node AUTO keeps
NONE — the auto-keep needs 2 nodes, P720). Forced tests cover **load only** (shard byte-exactness +
memory), which is exactly what this change touches; decode coherence stays a P720 item.

| # | test | result |
|---|---|---|
| T1 | build-cpu + CUDA build | clean |
| T2 | inert: env off vs `LLAMA_NUMA_TENSOR_SPLIT=1`, no force | pp512 110.26 vs 110.01 t/s; tg128 17.01 vs 17.05 — identical |
| T3 | forced load, `LLAMA_NUMA_VERIFY=1` | 120/120 shard pairs byte-exact (source = mmap), 0 mismatches |
| T4 | forced load peak RssAnon: old code vs file-direct | **40,309 → 20,072 MiB** (2× → 1×); load wall 30 → 23 s |
| T5 | normal (unforced) oracle, W=1 | "Here's a thinking process:" coherent, rc=0 |

(T4 note: the loader runs two passes — a `no_alloc` metadata pass and the real one; file-direct
engages on the real pass: `lm=mmap use_mmap=1 alias=1`.)

## 6. P720 acceptance (user runs)

```bash
# V4-Flash Q4_K (~165 GB), once open item #1 (download) is done:
LLAMA_NUMA_TENSOR_SPLIT=1 ./build/bin/llama-cli -m <v4flash-q4k> --moe-stream-window 1 \
  --numa distribute --spec-type none -p "The capital of France is" -n 32 -s 42 -t 16 \
  --single-turn --no-display-prompt --temp 0 -c 2048 < /dev/null
# 1. loads without OOM (was: 2× transient → ~330 GB)
# 2. peak RSS ≈ 165 GB, not 312
# 3. coherent output (semantic oracle — split logits diverge ~1e-6)
LLAMA_NUMA_TENSOR_SPLIT=1 LLAMA_NUMA_VERIFY=1 ...   # all shard pairs OK (source = mmap)
```

## 7. Notes / non-goals

- The no-mmap fallback (`--load-mode none`, or the moe-stream AUTO forcing when the split is not
  active) keeps the old materialize→copy→release behavior (still 2× transient there, unchanged
  from HEAD) — correct, just not file-direct. On the P720 the split keeps AUTO (3.0) → file-direct.
- GPU-hybrid + split: unchanged (pinned hosts still materialized + re-homed + freed). Not the P720
  case; a future GPU-hybrid file-direct would need the host in a plain-CPU (mmap-able) buft, which
  conflicts with the pinned decode-swap path — out of scope.
- The engaged expert file ranges stay mapped for model life (clean, reclaimable page cache) — no
  unmap pass added; page cache is not an OOM risk.
- `LLAMA_NUMA_PLACE_LAYER` (test-only mbind of host tensors) now operates on mmap pages; mbind
  failures on shared file mappings are non-fatal warnings and the env is off in real runs.

## 8. P720 acceptance note

The auto-keep-mmap path (3.0) requires `--numa distribute` (llama_numa_init populates
`ggml_numa_nodes()`; without it the split is inert anyway). Explicit `--load-mode mmap` also works
and bypasses the moe-stream downgrade on any box.
