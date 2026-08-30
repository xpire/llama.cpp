# Attention-streaming window — spike spec

Branch `feat/prefill-gpu-ada` · 2026-08-30 · status: **assessment only — DO NOT implement until the
A10B window crash fix lands** (paseo agent `dae0c238`, run-log §15). Dependent on window mode
(`--moe-stream-window`, commit `4de44bae4` + the correctness fix `ec4628af4`).

## 1. The wall this targets

Window mode removes the expert-slot floor, but **attention residency remains the binding wall**:
GLM-5.3-class attention (~10 GB at Q4) + W≥2 window (4.4 GB) + KV does not fit 12 GB. LvLLM has the
same wall (their attention/KV is resident; only experts are managed — verified from their docs).
Result: GLM-5.3 GPU prefill is impossible on 12 GB, in every engine.

## 2. The idea

Stream attention through the same window, deterministically: the pool slot holds a layer's
**attention + experts**; layer N+1's full slot is prefetched during layer N's compute (layer order
is fixed, so the prefetch is exact — same machinery as the expert window). Nothing dense needs to
be *resident* except the tiny fixed parts (norms ~0.2 GB, embeddings ~0.5-1 GB, router).

Per-ubatch stream becomes attention (10 GB) + experts (140 GB) for GLM-5.3-class. Decode stays
resident: after prefill, attention is uploaded once (~0.4 s for 10 GB) — the phase-switch from the
existing design.

## 3. Does it tank? (the spike's core question — the math)

The stall would come from the attention GEMMs waiting on the attention load. Whether it hides
depends on transfer-per-token vs compute-per-token:

| Model | Attention transfer (per ubatch, 26.9 GB/s) | Attention compute (per ubatch) |
|---|---|---|
| A3B (2.3 GB attn) | 0.085 s @ b2048 | ~1.7 s @ b2048 (8.4 GFLOP/token) |
| GLM-5.3 (10 GB attn) | 0.37 s @ b2048 | ~3.5 s @ b2048 (34 GFLOP/token) |

At batch ≥ ~256 the attention **transfer is ≤10% of the attention compute** — the load hides under
the GEMMs with W≥2 prefetch. The tank only appears at tiny batches, where the batch gate (P2,
min-batch) disengages GPU prefill anyway. **Feasible — no structural tank at prefill batch sizes.**

## 4. Expected payoff (GLM-5.3 on 12 GB, single 4070 Ti)

Stream per ubatch = 150 GB → at batch 8192: 18 MB/token → ~1,470 t/s ceiling. Compute-bound
realistically ~120-600 t/s (the roadmap's Ada mmq range) — **vs the P720's 31-47 t/s CPU = 3-19x**,
on a card that today cannot touch the model at all. The spike measures the real number.

## 5. Implementation scope (for the spike)

1. **Loader routing** (`llama-model.cpp` create_tensor): in window mode, ALSO route the attention
   projection tensors (attn_q/k/v/o, and fused qkv if present) to the window pool (same
   host-materialize + pool-slot pattern as the exps). Norms/embeddings stay resident.
2. **Graph hook** (`llama-graph.cpp`): a wait-op (identity/no-op, like the expert remap's window
   branch) before the attention projection GEMMs — the q/k/v build path feeding `build_attn`
   (line ~2799); the expert wait-op is already in `build_moe_ffn`. Both gate on the same pool slot
   generation; both trigger the next layer's prefetch (attention + experts in slot order).
3. **Window slot composition**: per-slot tensor list grows to q/k/v/o (+ gate/up/down); the
   shape-keyed pool already supports this (the agent's fix made it role-keyed).
4. **KV**: stays where it is today (RAM via `-nkvo`, or GPU if it fits) — prefill writes KV, it
   doesn't stream weights for it.
5. **Decode**: phase-switch uploads attention once after prefill (the existing host-tensor decode
   path is the fallback; the one-time upload is the migration piece already specced).

## 6. Risks / dependencies

- **BLOCKING: the A10B pp2048 CUDA crash** (mmq batch-boundary, paseo agent in progress) — the
  window machinery must be scale-correct first. This spike builds on it.
- The attention wait-op must not reorder the graph badly (build_attn comments note the
  no-reorder expand pattern — the wait-op must join that cluster).
- GLM-5.3 is not loadable on this box (156 GB > 96 GB RAM) — the spike measures on the A3B
  (attention-streaming on/off, same model) and validates the mechanism; the GLM number is
  projected, verified on the P720.
- Correctness oracle: byte-identical vs the resident-attention window run (same seed).

## 7. Spike exit criteria

1. A3B window+attention-streaming runs byte-identical (correctness oracle).
2. Measured prefill: attention-streaming adds ≤10% overhead vs attention-resident at ub ≥ 512
   (the "no tank" claim), and the A3B's 2537 t/s at ub 2048 holds or improves.
3. Projected GLM-5.3 prefill documented (stream ceiling + compute estimate) for the P720 decision.
