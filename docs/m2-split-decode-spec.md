# Split-mode decode — spec (window prefill + op-offload decode)

Branch `feat/prefill-gpu-ada` · 2026-09-03 · status: **implement** · testbed-validated evidence

## 1. The measured problem

On the 7800X3D + 4070 Ti testbed, A3B (Q4_K_XL), `-t 8 -b/ub 3000 -fa on`:

| config | prefill (pp3000) | decode (tg128) |
|---|---|---|
| window mode (W=2) | **~3226** | 29.7 |
| `-ncmoe 40` (normal offload, no window) | 611 | **52.0** |
| window + `-ncmoe 40` | 621 | 29.7 (window still caps decode) |

The two modes have complementary strengths: window streaming is a **5× prefill** win (experts streamed to the GPU cache, loads overlapped with compute); normal offload is a **1.75× decode** win. Currently the mode is fixed at context init, so you must pick one. Goal: **window-streamed prefill + op-offloaded decode in one context** → ~3200 prefill + ~52 decode.

## 2. Root cause (code-verified)

1. `llama-context.cpp:283-289`: when MoE expert streaming is enabled, **op offload is disabled globally**:
   > "op offload snapshots host weights to the device per graph split, which assumes they do not change during the graph - streamed caches are rewritten between waves, and the decode phase deliberately computes experts from CPU-resident host tensors (per-token snapshots to the GPU would defeat that)"

   The disable is blanket — it also kills decode-side offload of the **attention** projections, which are static host tensors that could safely snapshot once per (reused) decode graph.

2. `llama-graph.cpp:1525-1539` (`build_lora_mm`): in window mode at decode (`n_tokens == 1`), attention projections are swapped to their **CPU host tensors** (`mstream->host_for(w)`), so decode attention runs on CPU. In `-ncmoe` mode (no window), attention weights are GPU-resident and decode attention runs on the GPU → the 52 vs 29.7 gap.

3. `-ncmoe` proves op offload + CPU-resident expert MUL_MAT_ID coexist correctly (decode 52 t/s with experts on CPU — the scheduler does not per-token offload expert GEMMs), so the blanket disable is not required for the expert path either.

## 3. The change (two steps, each independently testable)

**Step 1 — re-enable op offload under moe-stream (guarded, then default):**
- `llama-context.cpp`: skip the disable when an env/flag allows it; test decode/prefill/correctness. If prefill regresses (pool-rewrite hazard), scope op offload to the decode graph only (see step 1b).
- 1b (if needed): per-graph op offload — the scheduler takes `cparams.op_offload` at creation; use a decode-side scheduler flag or split the graphs so op offload applies to decode passes only.

**Step 2 — decode attention from GPU-resident weights:**
- If step 1 does not lift decode to ~-ncmoe level (op offload not engaging on the host attention tensors, or snapshot-per-graph still paying per token), add a GPU-resident attention copy used at decode: load attention into the streaming cache (prefill path unchanged) **and** into a permanent GPU buffer whose tensor the decode graph consumes (`build_lora_mm` host-swap skipped when a decode-side GPU tensor exists). VRAM cost ≈ attention size (A3B ~2-3 GB; fits 12 GB alongside KV+pool).

## 4. Acceptance criteria — RESULTS (2026-09-03, testbed 4070 Ti)

All three implemented + measured:

| criterion | target | measured | status |
|---|---|---|---|
| decode (window) | ≥ ~45 | **45.7 t/s @ ub2500** (was 29.7) — `LLAMA_MOE_STREAM_ATTN_RESIDENT=1` | ✅ +54% |
| prefill (window) | ≥ ~3000 | **3057 t/s @ ub2500** (2531 @ ub2048 vs 2321 streamed — resident attention helps prefill) | ✅ |
| correctness | Paris coherent | **both paths coherent** (streamed + resident) | ✅ |

Step 1 alone (op offload re-enabled, `LLAMA_MOE_STREAM_OP_OFFLOAD=1`): decode 27.2 → 29.45 (+8%) — not the main gap; kept as an option.
Step 2 (attention resident, `LLAMA_MOE_STREAM_ATTN_RESIDENT=1`): the real win. Decode approaches the `-ncmoe` reference (52). VRAM: resident attention costs ~2-3 GB → ub ceiling drops 3000→2500 with the desktop holding ~5 GB; headless (11.9 GB free) restores ub3000+.

## 5. Implementation

- `src/llama-model.cpp` create_tensor: `attn_stream` gate — attention projections skip the streaming pool (load to the normal layer GPU buft, decode computes on-device from resident weights; experts stream unchanged) when `LLAMA_MOE_STREAM_ATTN_RESIDENT=1`. Default (big-model, attention streamed) unchanged.
- `src/llama-context.cpp`: op-offload disable under moe-stream is skipped when `LLAMA_MOE_STREAM_OP_OFFLOAD=1`.

## 6. Test matrix (testbed, nvidiabox, 4070 Ti)

```
llama-bench -m A3B -ngl 999 --moe-stream-window 2 -fa on -t 8 -b 3000 -ub 3000 -p 3000 -n 128
  → expect pp ≥3000, tg ≥45
llama-server oracle (W=2) → "Paris" coherence
llama-server W=2 + --spec-type draft-mtp → decode vs the 3.3 t/s regression baseline
```
