# Attention-streaming window — spike spec (rev 2)

Branch `feat/prefill-gpu-ada` · 2026-08-30 · status: **assessment updated — implementation still
gated** (window machinery is now scale-correct; see §0). Replaces rev 1.

## 0. What changed since rev 1 (all measured)

- The A10B pp2048 crash was **VRAM capacity, not a window bug** — W=1 works (run-log §15b).
- Window mode is now scale-correct through 64K (run-log §15c):
  - A3B window: 610 / 2201 / **1998** t/s at pp 512/8K/64K (ub 2048) — 1.58x CPU at 64K
  - A10B window W=1: 190 / 737 / **704** t/s — beats CPU +29% at pp8K/ub2048
- The A10B's ub-8192 wall is **compute-buffer VRAM** ("failed to create context"), not the pool and
  not attention residency.
- The blocking dependency (rev 1 §6) is gone: the window machinery is scale-correct.

## 1. The wall this targets

Window mode removes the expert-slot floor, but **attention residency remains the binding wall**:
GLM-5.3-class attention (~10 GB at Q4) + W>=2 window (4.4 GB) + KV does not fit 12 GB. LvLLM has
the same wall (attention/KV resident; only experts managed — verified from their docs). Result:
GLM-5.3 GPU prefill is impossible on 12 GB, in every engine.

## 2. The idea

Stream attention through the same window, deterministically: the pool slot holds a layer's
**attention + experts**; layer N+1's full slot is prefetched during layer N's compute (layer order
fixed → exact prefetch, same machinery as the expert window). Nothing dense stays resident except
the tiny fixed parts (norms ~0.2 GB, embeddings ~0.5-1 GB, router). Decode: attention uploaded
once after prefill (~0.4 s for 10 GB) — the existing phase-switch.

## 3. Does it tank? (the spike's core question — math + measured context)

The stall would come from the attention GEMMs waiting on the attention load. Transfer vs compute:

| Model | Attention transfer (per ubatch, 26.9 GB/s) | Attention compute (per ubatch) | Stream bytes added |
|---|---|---|---|
| A3B (2.3 GB attn) | 0.085 s @ b2048 | ~1.7 s @ b2048 | +13% (17→19.3 GB) |
| A10B (0.9 GB attn) | 0.03 s @ b2048 | small | +1.3% (67→68 GB) |
| GLM-5.3 (10 GB attn) | 0.37 s @ b2048 | ~3.5 s @ b2048 | +7% (140→150 GB) |

At batch ≥ ~256 the attention transfer is ≤10% of the attention compute → the load hides under the
GEMMs with W>=2 prefetch. The tank only appears at tiny batches (the P2 batch gate disengages GPU
prefill there anyway). **Feasible — no structural tank.** The A3B spike measures it directly
(stream-attn vs resident-attn on the same model).

## 4. Payoff — re-scoped with the measured matrix

| Target | Payoff | Verdict |
|---|---|---|
| **GLM-5.3-class (attn ~10 GB)** | the ONLY way to fit 12 GB; ~120-600 t/s vs P720 CPU 31-47 = **3-19x** | **the headline — unchanged** |
| A3B (attn 2.3 GB) | frees ~2.3 GB VRAM → bigger W or batch; direct mechanism gain ≈ 0 (stream is already hidden) | marginal; the freed VRAM is the lever |
| A10B (attn 0.9 GB) | frees ~0.9 GB — **not enough to clear the ub-8192 compute-buffer wall** | marginal |

So the spike's promise is NOT "attention streaming itself speeds prefill" — it's (a) the only path
to GLM-5.3-class on 12 GB, and (b) freed VRAM for a larger window/batch on the small-attention
models. The "increase prefill" framing applies mostly to the freed-VRAM side.

## 5. Implementation scope (refined — cleaner than rev 1)

1. **Loader routing** (`llama-model.cpp` create_tensor): in window mode, ALSO route attention
   projections (attn_q/k/v/o, fused qkv) to the window pool (host-materialize + pool-slot pattern,
   same as exps). Norms/embeddings stay resident.
2. **Wait-op placement — the refinement**: attention projections consume their weights via
   `build_lora_mm` (dense GEMM). Inserting the wait-op **inside `build_lora_mm`** for streamed
   attention weights (a no-op map_custom1 on `cur` keyed to the weight's layer slot) gives the gate
   for free at every projection — no per-arch surgery in the attention build paths. The expert
   wait-op in `build_moe_ffn` already triggers the next layer's prefetch; the prefetch now loads
   attention + experts of layer N+1 (attention first — it runs first in the layer).
3. **Window slot composition**: slot tensor list grows to q/k/v/o + gate/up/down; role-keyed pool
   already supports this (the agent's fix).
4. **KV**: unchanged (`-nkvo`, RAM).
5. **Decode**: one-time attention upload after prefill (existing host-tensor path is the fallback).

## 6. Risks / dependencies

- The A10B ub-8192 compute-buffer wall is unchanged by attention streaming (frees only ~1 GB) —
  the big-batch regime on small cards stays bounded by graph compute buffers, not weights.
- The wait-op inside `build_lora_mm` must not reorder the graph badly (the no-reorder expand
  cluster around attention; verify the split count doesn't explode).
- GLM-5.3 is not loadable on this box (156 GB > 96 GB RAM) — the spike measures on the A3B
  (stream-attn vs resident-attn) and validates the mechanism; GLM's number is projected, verified
  on the P720.
- Correctness oracle: byte-identical vs the resident-attention window run (same seed).

## 7. Spike exit criteria

1. A3B window+attention-streaming: byte-identical vs resident-attention window (correctness).
2. Overhead ≤10% at ub ≥ 512 (the "no tank" claim) AND the freed VRAM buys a measurable bigger
   window or batch on the A3B (the actual prefill lever).
3. Projected GLM-5.3 prefill documented (stream ceiling + compute estimate) for the P720 decision.

## 8. Decision gate

Implement the spike only if: the paseo agent's window work is committed and stable (done — W=1
through 64K), AND the user approves the re-scoped payoff (headline = GLM-5.3-class on 12 GB;
A3B/A10B gain = freed VRAM, not the mechanism).
