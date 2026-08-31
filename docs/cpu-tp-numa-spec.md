# CPU Tensor Parallelism (rank = socket) — Plan Spec

Branch: `feat/cpu-tp-numa` · 2026-08-30 · source: upstream RFC PR #25209 (withdrawn, diff reviewed) · companion: `research/llm-homelab-performance-estimate.md`, `research/llm-homelab-inference-performance.md`, `research/llm-homelab-fleet-plan.md`, `research/llm-homelab-vs-sglang-benchmark.md`

## 1. Goal

Give the P720 (2× Xeon 6138, 12ch DDR4-2133, 192 GB, 4070 Ti) a **CPU-only inference path that operates near the 2133 bandwidth wall with zero weight duplication and interconnect traffic limited to activation combines** (~3% class, LvLLM-style):

| Metric | Baseline (single-process, v2 estimates) | TP target (rank=socket) | 2133 wall |
|---|---|---|---|
| V4-Flash-0731 decode | ~10 t/s (hybrid ~11) | **~13-15** | ~16 |
| Qwen3.8-Flash-Next decode | ~22 (hybrid ~25) | **~27-31** | ~34 |
| GLM-5.3-Flash decode | ~7 (hybrid ~8) | **~9-10.5** | ~11 |
| Prefill | 1x | **~1.8-2x** (PR: Phi-3 1.84x, 235B 2.1x) | — |

Fleet goal is >10 t/s for all three; GLM-5.3 hybrid (7-8) currently misses. TP at ~85% of wall gets all three to/over the line (GLM marginal).

## 2. Findings

| # | Finding | Source |
|---|---|---|
| F1 | Current NUMA plan = `numactl --interleave=all` + `--numa numactl`; **`--numa distribute` is explicitly avoided with `--n-cpu-moe`** (thread-binding collisions) | inference-performance.md |
| F2 | Interleave stripes every tensor across both sockets → each access ~50% local / ~50% UPI; TP rank=socket makes weight reads **100% local by construction** | analysis |
| F3 | PR #25209 (withdrawn RFC): multi-process CPU TP — per-rank weight sharding at load + one all-reduce per layer as a ggml custom op; author-reported 2× cross-NUMA, near-linear over IB; measured anchors: 235B Q8 2-socket **+36% decode**, Phi-3 **1.84× prefill** | PR body/diff |
| F4 | PR gaps for our use: UCX-only transport (need shm default), synchronous `n_tasks=1` all-reduce (needs overlap), requires `GGML_CPU_REPACK=OFF` + `--no-mmap` (needs repack-aware sharding), MoE dense-FFN all-reduce suppression is model-family-assumptive | diff review |
| F5 | 6138 decode sits at ~60-65% of the 2133 wall (10/22/7 vs 16/34/11) — locality fix can reclaim most of the gap; the wall itself is physics (`GLM-5.3 hard-capped ~14 — no flag set changes that`) | performance-estimate.md v2 |
| F6 | 4070 Ti hybrid adds only 10-40% (expert offload; PCIe 3.0 16 GB/s ceiling) — **TP-CPU at the wall may match or exceed hybrid decode** for these small-active models (hypothesis → P2 benchmark) | performance-estimate.md |
| F7 | Multi-process gives **hard per-socket ownership** (threads cannot cross nodes) vs soft pinning in one process — eliminates the `--n-cpu-moe` × distribute hazard entirely | analysis |

## 3. Architecture decision

**Multi-process TP, rank = NUMA socket (2 ranks on the P720).** Weights sharded per rank at load — no duplication, no mirror, no interleave — each rank computes only its local shards; one all-reduce per sharded layer moves activations only. This is the LvLLM "3% cross-node" scheme expressed as processes instead of thread teams.

- CPU-only path. The 4070 Ti hybrid (single-process `--cpu-moe` / moe-stream) remains the interactive path — orthogonal, not composable in this PR.
- GPU-powered-down mode (S3/power budget): TP-CPU at 2×100W RAPL is the low-power inference path.

## 4. Design (PR #25209 core + our deltas, Δ = change)

**Loader sharding** (`llama-model-loader.cpp`): per-rank `tp_shard_plan` per tensor — COLUMN / ROW / EXPERT / per-head-segment, quant-block-aligned, fail-loud on unsplittable shapes. Δ: **repack-aware** — materialize each rank's shard already in repacked layout so the fast Q4_K kernels are not lost (PR forces `GGML_CPU_REPACK=OFF`, which forfeits decode speed the bandwidth win claims to buy).

**All-reduce op** (`llama_tp_allreduce_op` as `ggml_map_custom1_inplace`, no core ggml changes). Δ1: **transport interface** — POSIX shm + futex default (zero deps, the 2-socket case needs nothing else), plain TCP fallback, UCX/IB behind a build flag (future multi-node). Δ2: **overlapped** — dedicated comm thread + double-buffered partials so layer L's all-reduce runs under layer L+1's compute (the PR's synchronous single-thread op serializes prefill).

**Graph insertion** (`build_ffn` down-proj, `build_attn` wo, `build_moe_ffn` combine, mamba2 `ssm_out`). Δ: **structural suppression** — key the dense-FFN all-reduce off whether the tensor was actually sharded, not the "pure-MoE models" assumption (breaks for shared-expert hybrids).

**MoE modes**: EP (shard expert set) for prefill/capacity; tensor (split each expert's n_ff) for decode balance. Δ: **data-dependent default** (decode → tensor; prefill ubatch ≥ 2048 → EP) to dodge EP's routing straggler problem.

**Replicated per rank**: router, embeddings, lm_head, norms, shared experts, KV cache (2× KV RAM — fine at 192 GB).

**Surface**: `--tp-size N --tp-rank N --tp-moe <ep|tensor> [--tp-attn] [--tp-ssm] [--tp-peer ADDR]`; llama-bench flags; per-rank `numactl --cpunodebind --membind` wrappers (their documented recipe).

## 5. Integration with current branch

- **Orthogonal to moe-stream**: TP is multi-process CPU; moe-stream is single-process GPU hybrid. Inert unless `--tp-size > 1`. No shared state.
- Merge surfaces: `llama-graph.cpp` `build_moe_ffn` (conflict with stream-injection ops — both gated by env/flags, coexist), `common/arg.cpp` (new `--tp-*` flags), `tools/llama-bench` (bench flags + 2-rank orchestrator), `llama-model-loader.cpp` (clean, untouched by stream work).
- Build: our Δ keeps repack ON — **one binary serves both paths**. `--no-mmap` only for TP runs (matches the existing server macro).
- Do NOT combine TP with `--cpu-moe`/`--n-cpu-moe` in the same run — TP is the CPU-only mode; the hybrid is the other process topology.

## 6. P720 config

```bash
# rank 0 (socket 0)  ·  rank 1 (socket 1)  — 2× 6138, 192 GB
numactl --cpunodebind=0 --membind=0 ./llama-server -m <model>.gguf \
  --tp-size 2 --tp-rank 0 --tp-moe tensor -t 20 --no-mmap -b 2048 -ub 2048 -c 32768 -ctk q8_0 -ctv q8_0
numactl --cpunodebind=1 --membind=1 ./llama-server -m <model>.gguf \
  --tp-size 2 --tp-rank 1 --tp-moe tensor -t 20 --no-mmap -b 2048 -ub 2048 -c 32768 -ctk q8_0 -ctv q8_0
```

## 7. Feasibility verdict — does this make llama on the P720 more feasible?

**Yes, materially — as the CPU-only path.** It is the missing *hard-locality* layer: CPU-only decode moves from ~60-65% toward ~80-90% of the 2133 wall (V4-Flash 10→13-15, Flash-Next 22→27-31, GLM-5.3 7→9-10.5), prefill ~2× (agentic-heavy workload), the `--n-cpu-moe` × distribute hazard disappears, and it enables GPU-off inference at 2×100W. It may even match/exceed the 4070 Ti hybrid on decode for these small-active models (F6 — benchmark before assuming).

**What it does NOT change:** the 205 GB/s physics (GLM-5.3 ~14 t/s cap), the unified-memory economics gap (Spark 27.7 t/s, Macs), the 6240/6248R+2933 upgrade ladder, or the hybrid's role for interactive use. TP is the CPU-side counterpart to the GPU streaming roadmap, not a replacement.

## 8. Phases

| Phase | Work | Status |
|---|---|---|
| P0 | shm transport + all-reduce op + dense sharding (FFN/attn) + repack-aware loader | proposed |
| P1 | MoE (tensor + EP) + structural suppression + llama-bench flags + 2-rank bench harness | next |
| P2 | P720 benchmark matrix (3 models × moe mode × ubatch) vs hybrid + interleave baselines | gates verdict |
| P3 | overlapped all-reduce; optional UCX multi-node | optional |

## 9. Risks / open questions

- Load-time shard read for 112-165 GB GGUFs (strided vs span; repack-aware may need a double pass) — measure load time on the 2 TB NVMe.
- All-reduce latency at batch-1 decode over shm (expect µs-class; verify — this is what F6's hypothesis hinges on).
- Cross-rank determinism: replicated router/logits must stay bit-identical (same graph, same seed).
- llama-bench multi-process: needs a 2-rank orchestrator (wrap the two commands).
- Whether TP-CPU beats the hybrid on decode at all — P2 decides; if not, TP is still the prefill + GPU-off path.

## 10. References

- PR #25209 (body, comments, full diff) + its `docs/cpu-tensor-parallel.md`
- `research/llm-homelab-performance-estimate.md` (v2 walls/efficiency), `research/llm-homelab-inference-performance.md` (NUMA/engine), `research/llm-homelab-fleet-plan.md` (roles/power), `research/llm-homelab-vs-sglang-benchmark.md` (lk_moe benchmark)
- LvLLM / LSGLang READMEs (`LVLLM_MOE_NUMA_ENABLED`, `LK_THREAD_BINDING`, `LVLLM_ENABLE_NUMA_INTERLEAVE`)
