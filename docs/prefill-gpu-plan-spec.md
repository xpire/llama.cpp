# GPU Prefill for MoE — Plan Spec (phase-split architecture)

Branch `feat/prefill-gpu-ada` · 2026-08-30 · companion: `docs/m2-gpu-prefill-port-plan.md`, `research/m1/run-log-2026-08-30.md`

## 1. Findings (all measured on the 7800X3D / 4070 Ti box)

| # | Finding | Value |
|---|---|---|
| F1 | CPU prefill (A3B Q4) scales with ubatch via weight amortization | 520 → 2,560 t/s (ub 512 → 8192) |
| F2 | GPU expert-streaming prefill also scales, but caps below CPU at scale | 620 → 962 t/s (ub 512 → 2048); **OOM at ub ≥ 4096** (12 GB VRAM) |
| F3 | Streaming's only edge is small ubatch (+19% at ub 512); CPU wins +43% at ub 2048 | crossover ~ub 1024 |
| F4 | PCIe H2D ceiling | 19.1 GB/s pageable, 26.9 GB/s pinned (+41% via #26659) |
| F5 | GPU wins 3.7× when model is resident (dense 14B) | 3,192 vs 871 t/s |
| F6 | Streaming decode is worse than CPU-resident experts | 36 vs 51 t/s |
| F7 | 12 GB VRAM is structurally too small for big-active streaming: attention residency + wave-slot floor (≥3×n_expert_used) | GLM-4.5-Air-class: attention ~10 GB + 24-slot floor ~10 GB → impossible |
| F8 | Streaming machinery works (slots/remap/waves live, 68/256 slots, 9 I/O threads) | commit `34b806a3b` |

## 2. Architecture decision: phase-split placement

Prefill and decode want opposite expert placement. One process, two phases:

- **Prefill** (large ubatch): experts **stream to GPU cache** on demand (amortized: each expert read once per ubatch); attention on GPU.
- **Decode** (1 token): experts **CPU-resident** (RAM, bandwidth-bound — the existing `--n-cpu-moe` config); attention on GPU.

Mechanism: experts are **materialized in RAM** (host tensors) AND mirrored in the **VRAM stream cache**. Prefill GEMMs index the cache via the id-remap op; decode GEMMs index the host tensors directly (no remap). No mid-run buffer migration needed for experts — the phase switch is a graph-level tensor selection.

Attention: stays GPU-resident both phases (fits for A3B; for GLM-5.3-class needs ≥24 GB GPU or per-ubatch attention preload — Phase 3 below).

## 3. Implementation phases

| Phase | Work | Status |
|---|---|---|
| P0 | Streaming core port (skeleton + graph injection + fixes + bench flags) | **DONE** (`3ef6aa0dc`…`8c2201c5e`) |
| P1 | **RAM→GPU copy pool + host materialization** (this run): experts materialized in RAM; stream cache filled by copies from host tensors, not disk pread; decode phase selects host tensors (no remap) | IN PROGRESS |
| P2 | Parameterized engagement gate: `--moe-stream-min-ubatch <N>` (streaming only for ubatch ≥ N; encode F3 — the LvLLM 4096 default is wrong for bandwidth-bound A3B) | next |
| P3 | GLM-5.3-class (P720): needs ≥24 GB GPU; optional attention per-ubatch preload for 12 GB cards; batch 8192 to amortize | P720-gated |
| P4 | Overlap: prefetch next ubatch's experts during current compute (CUDA event chains, #26659 pinned staging) | after P1-P2 |

## 4. Per-model targets

| Model | Box | Prefill target | Blockers |
|---|---|---|---|
| Qwen3.6-35B-A3B Q4 (22 GB) | this box | P1 phase-split; PP via stream (parity+), TG via host experts (~51 t/s) | none (implementation in flight) |
| GLM-5.3-Flash IQ4_XS (156 GB) | P720 (192 GB) | ≥120 t/s streaming vs 31-47 CPU | **NAS shard 1 incomplete (9.4 MB stub, needs re-download); 156 GB > 96 GB RAM (this box cannot run it at all); NAS 1 GbE too slow to stream from — must be local NVMe; ≥24 GB GPU** |
| Qwen3.8-27B Q8 (28 GB) | this box | n/a — dense, no experts to stream (bench only, queued) | — |

## 5. Key risks / decisions

- **Decode correctness**: P1 must prove TG-through-host equals TG-through-cache equals baseline (compare outputs).
- **op_offload**: PR disables it when cache is host-resident; with the cache on GPU it stays enabled — but the decode host-tensor path must not trigger the snapshot path (verify).
- **RAM cost of host materialization**: +17 GB (A3B) / +140 GB (GLM-5.3, still fits 192 GB P720).
- **Graph reuse**: phase switch must not poison the reuse cache (decode vs prefill graphs already differ by ubatch shape → `can_reuse` naturally separates them).
- **Engagement gate**: parameterized (P2), default derived per model from F1/F3, not hardcoded 4096.
