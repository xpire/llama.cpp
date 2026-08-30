# Layer-window streaming — spec (rolling residency, LvLLM-style)

Branch `feat/prefill-gpu-ada` · 2026-08-30 · status: **proposed → implementing**

## Problem it solves

The expert-slot streaming (PR #25294 port) has a **slot floor**: wave mode needs ≥3×n_expert_used
slots/layer. Big-expert models (Qwen3.5-122B-A10B: ~470 MB/expert → 24 slots = ~16 GB) **abort on
12 GB VRAM** — the 12 GB card cannot run GPU prefill for them, even though the CPU baseline
(212-1117 t/s) is beatable in principle (transfer ceiling ~3,300 t/s at ub 8192).

LvLLM avoids this with **layer-granular residency + prefetch**: layer order is deterministic, so
the next layer's weights are prefetchable without routing knowledge. Their
`LVLLM_ENABLE_MOE_LAYERWISE_LOAD` + `PREFETCH_WINDOW=1-2` streams whole layers through a small
in-flight window — VRAM = window + attention, no slot floor.

## Design

**Rolling layer window**: W pool slots, each holding one layer's *full* expert set
(cache tensor `[ne0, ne1, n_expert]`). Layer N's expert GEMMs read pool slot `N % W`. A prefetch
controller (hooked into the existing remap op) enqueues layer N+1's full load during layer N's
compute — deterministic (layer order), so the copies overlap the GEMMs via the existing safe
worker pool (per-thread stream + cv).

**Reuse of the existing machinery** (why this is small):
- The per-layer cache machinery already does: worker pool, cv ordering, host materialization,
  demand loads, per-layer slot state. Window mode = `n_slots = n_expert` (full layer per cache
  tensor) + pool-shared cache tensors + a next-layer prefetch hook.
- With `n_slots = n_expert`, the remap degenerates to: wait-for-layer-loaded + identity ids
  (slot e = expert e) — no id rewrite needed, no wave abort (touched = 256 ≤ slots).
- The per-layer slot state is naturally per-layer, so pool reuse needs no cross-layer eviction
  bookkeeping: layer N's state starts fresh for its slot; the physical bytes of layer N-W are
  simply overwritten after N-W's GEMMs completed (guaranteed by graph order, W ≥ 2).

**VRAM**: attention + W × layer_expert_bytes + KV + buffers. On 12 GB: A3B → W ≈ 18
(~0.45 GB/layer), A10B → W ≈ 5 (~1.4 GB/layer). Transfer per ubatch is unchanged (each layer's
experts stream once) — the win is (a) no slot floor → big-expert models run, (b) deterministic
prefetch → copies hide behind compute.

## Implementation

1. `llama-moe-stream.{h,cpp}`: `n_window` mode — `create_cache_tensor` returns pool tensors
   (`window_pool[slot][k]`, W×3 of them); `resolve` forces `n_slots = n_expert`; new
   `prefetch_layer(N+1)` enqueues the next layer's full expert load (no wait), called from the
   remap after the current layer is resident.
2. `llama-graph.cpp`: no structural change — the remap op already sits before the expert GEMMs;
   window mode just makes it identity + prefetch hook.
3. `llama-model.cpp` / params: `--moe-stream-window <W>` (0 = expert mode).
4. `llama-bench`: flag.

## Test matrix (this run)

| Model | CPU (mlock) | Expert-stream (12 GB) | **Window-stream (12 GB)** |
|---|---|---|---|
| A3B (22 GB) | 520-2560 | 790-1230 (works) | expect ≈ parity or better (deterministic prefetch) |
| A10B (71 GB) | 212-1117 | **ABORT** (slot floor) | expect 800-2500 (the new capability) |

Success criteria: A10B window-streaming at ub ≥ 2048 **beats the CPU baseline (573-1117)**;
A3B window-streaming stays within noise of the expert cache.
