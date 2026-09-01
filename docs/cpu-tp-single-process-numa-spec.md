# Single-Process NUMA — llama.cpp-native thread-team TP

Branch `feat/prefill-gpu-ada` · 2026-09-02 · status: **P1 in progress** (P720 measurements 2026-09-01)
Companion: [cpu-tp-numa-spec.md](cpu-tp-numa-spec.md) (multi-process TP — **superseded by this spec**), `research/llm-homelab-p520-flash-next-report.md`, `research/llm-homelab-to-sort-out.md` §5 (ktransformers NUMA port sizing)

## 12. P1 implementation status (2026-09-02)

**Findings (what already exists):** llama.cpp's `--numa distribute` implements thread affinity only
(`set_numa_thread_affinity`: pool thread `ith` → node `ith % n_nodes`) — there is NO per-tensor memory
placement in the loader, and no multi-CPU-device support. So the split's locality requires a new
per-op thread-team mechanism.

**Increment 1 — DONE (compiles, inert on single-node):**
- `ggml_tensor.numa_node` (int32_t, default -1, absorbed into the trailing padding — ABI-stable)
- `ggml_compute_forward_mul_mat_id`: after the pool barrier, threads not on `dst->numa_node`
  return (barrier-synced, no work) — the chunk loop is atomic-counter coordinated, so a subset of
  threads can do the work. `ggml_numa_nodes()` accessor added.
- Inert until tensors are tagged + shard ops exist; the 1-node testbed cannot exercise it.

**Increment 2 — DONE (2026-09-02, validated mechanically):**
- `llama_model_loader`: `llama_numa_mbind_tensor()` — raw `mbind` syscall (no libnuma), `MPOL_BIND`
  + `MPOL_MF_MOVE`, applied in a post-load pass over every `numa_node`-tagged tensor.
  Bug fixed during bring-up: `MPOL_MF_MOVE` is `(1<<1)`, not `(1<<4)` (EINVAL).
- `llama_model_base::create_tensor`: env-gated tagging (`LLAMA_NUMA_PLACE_LAYER=1` → `numa_node =
  tn.bid % 2`) on both the moe-stream host tensors and the normal load path — a mechanism test; the
  real per-shard assignment arrives with the sharded loader.
- Verified: tagging fires (blk.N tensors → node N%2), the pass runs, mbind applies. NOTE: mbind on
  unfaulted mmap pages sets the policy silently (no node validation — pages fault later onto the
  policy'd node), which IS the desired placement. The 1-node testbed cannot show migration; the P720
  `numastat` check is the proof: run with the env, then
  `numastat -p <pid>` / `grep "N1=" /proc/<pid>/numa_maps` — odd-layer expert pages must land on node 1.
**Increment 3 — DONE (2026-09-02, build + 1-node-inertness verified; P720 is the functional test):**
- `llama_numa_shard_tensor()`: per-node n_ff shards of a loaded expert tensor (strided per-row
  halves, page-aligned heap wrapped in `ggml_backend_cpu_buffer_from_ptr`, mbind'd, `numa_node`
  0/1). model pimpl owns the shard ctx + data (freed on destruction).
- `build_moe_ffn`: the expert-GEMM lambda is parameterized by its weight tensors; when the shard
  registry matches (decode path — the host tensors are swapped in), the pipeline runs twice (one
  per node's shard) and the expert outputs are added (the combine). `mul_mat_id`'s node check
  falls back to the weight's `numa_node` (the GEMM outputs are untagged).
- Gated: `LLAMA_NUMA_TENSOR_SPLIT=1` AND `ggml_numa_nodes() > 1` (inert on the single-node
  testbed — verified no regression). Prefill/waves use the unsharded cache tensors (EP/link-bound
  path unchanged); only decode (host-swap) takes the split path — per the spec.
**Increment 4:** `--tp-mode <tensor|ep>` wiring; P720 validation (cross the 40 GB/s interleave
ceiling).

## 1. TL;DR

Give the P720 **100%-local weight reads** (both sockets' RAM bandwidth, no UPI tax) **inside one llama.cpp process** — the LvLLM/lk_moe "3% cross-node" scheme expressed as **two thread teams instead of two processes**. This banks the dual-Xeon bandwidth that interleave leaves on the table (measured ceiling ~2×UPI ≈ 40 GB/s), without the multi-process TP's desync flaw (one scheduler = nothing to diverge).

**Target on the P720** (vs the interleave cap measured/derived on 2026-09-01):

| model | interleave cap (≈2×UPI) | this spec's target | wall |
|---|---|---|---|
| V4-Flash (7.15 GB/token) | ~5.6 t/s | **~20-28 t/s** | 28 (102 GB/s × 2 sockets) |
| Flash-Next (5.9 GB/token) | ~6.8 | **~25-34** | 34 |
| A3B (2.7 GB/token) | ~15 (not the binder) | **~30-38** | 38 |

## 2. Why — the evidence

1. **The interleave UPI tax is real and severe.** `numactl --interleave=all` stripes every tensor across both sockets → ~50% of weight reads cross UPI (~20 GB/s on 6138s) → **effective decode bandwidth ≤ ~40 GB/s**, i.e. V4-Flash ~5.6 t/s — *before* the 205 GB/s wall is ever reached. The P520 report's "interleave may cost some" was optimistic.
2. **The multi-process TP (rank=socket) solved locality but introduced desync.** Two independent processes take different internal graph paths (warmup/batch-shape/rebuild decisions are per-context) → all-reduce rendezvous diverges → hang or silent corruption (proven: P720 server deadlock at round 161; standalone barrier CORRUPT case). The locality win is real but the architecture is flawed.
3. **LvLLM/LSGLang has the locality but at the ecosystem price** — no GGUF, CUDA-lock, SNC4 BIOS, fork maintenance. We want GGUF + mainline.

The A3B's 9.67 t/s decode hid the UPI problem (compute-bound at 26 GB/s < 40) — the big-active models are where it binds.

## 3. Design overview

**One process, two socket thread-teams.** The router, KV cache, graph, scheduler, and context are shared (single instance — the thing that makes desync impossible). The CPU expert work is split between two thread teams, each bound to one socket and reading only **its own socket's RAM** (100% local by construction). The per-layer combine moves activations only (~3% class — the LvLLM number).

```
            one process (llama-server / llama-bench)
┌───────────────────────────────────────────────┐
│  router ──(runs ONCE)──▶ expert ids            │
│                                               │
│  ┌────────── socket-0 team ──────────┐        │
│  │  threads 0-19 (affinity 0-19)     │        │
│  │  weights: node-0 RAM (mbind)      │        │
│  │  computes its expert shard        │        │
│  └──────────────┬────────────────────┘        │
│                 │ partial                     │
│  ┌────────── socket-1 team ──────────┐        │
│  │  threads 20-39 (affinity 20-39)   │        │
│  │  weights: node-1 RAM (mbind)      │        │
│  │  computes its expert shard        │        │
│  └──────────────┬────────────────────┘        │
│                 │ partial                     │
│  combine (intra-process sum, ~3% traffic) ──▶ next layer
└───────────────────────────────────────────────┘
```

## 4. The three mechanisms

### 4a. Weight placement (the memory half)

Allocate each socket-team's weight shard **in that socket's RAM**, deterministically — not via kernel migration:

- **Primary: `mbind(2)`** per tensor range with `MPOL_BIND` to the owning node, after allocation (heap path) or on the mmap range (`MPOL_MF_MOVE` for the mmap path). Explicit, kernel-guaranteed.
- Fallback: **load-time first-touch** — pin the loading threads per socket while faulting in their shards.
- Explicitly NOT: relying on kernel NUMA balancing (our recipes disable it) or interleave.

### 4b. Thread teams (the compute half)

Two per-socket affinity groups in the ggml threadpool:

- `pthread_setaffinity_np` per pool thread — team 0 → CPUs 0-19, team 1 → 20-39 (the ktransformers port item #1; llama.cpp's `--numa distribute` conflicts with `--n-cpu-moe` — F1 — so this is an in-engine binding, not the flag).
- Cleanest: **two ggml threadpools** (one per socket), expert-GEMM ops dispatched to the owning pool; attention/dense on either/both. The scheduler is unchanged — one graph, nodes dispatched per team.

### 4c. The split + combine (the parallel half)

The spec's data-dependent default (cpu-tp-numa-spec P1):

- **Decode (n_tokens == 1): tensor-split** — split each expert's `n_ff` (gate/up COLUMN, down ROW) across the teams → both sockets' cores work on **every** expert → balanced (avoids the EP routing straggler at batch 1). Each team reads its shard → local.
- **Prefill (ubatch ≥ 2048): expert-parallel (EP)** — each team owns half the experts; the batch's tokens route to both; combine per layer (the straggler averages out at big batch).

**The combine** is an **intra-process** reduction: the graph runs the expert GEMMs per team, then a custom op sums the partials with a thread barrier. **No shm, no rendezvous protocol, no process boundary** — this is the entire point.

## 5. Why it cannot desync

The multi-process flaw was *two state machines*. Here there is **one** scheduler, one graph, one context: every layer's combine op is dispatched once to the pool; both teams participate in the same op execution — they are structurally always at the same layer. Threads in one process cannot disagree on the work list. The failure spectrum from the TP experiment (hang / silent corruption / divergence) does not exist here.

## 6. Reuse from the multi-process TP work (b817a5e72)

The process layer (shm transport, futex barrier, `--tp-peer`) is **discarded**; the graph machinery is **kept**:

| piece | status |
|---|---|
| `llama_tp_mask_ids_op` (non-local ids → −1) | keep — team filter instead of rank filter |
| `llama_tp_allreduce_op` + the ggml negative-id patches (cpu/repack/backend) | keep — becomes the intra-process combine; negative-id support still needed for the mask |
| `--tp-size / --tp-rank` flags | keep — now mean intra-process thread teams (`--tp-mode <tensor|ep>` for the split; `--tp-peer` dropped) |
| shm transport (`src/llama-tp.cpp` barrier) | **drop** — replaced by the pool's own thread sync |
| `llama_tp_context` | reduced to team/affinity state (no shm fd, no futex) |

## 7. Integration points

| file | change |
|---|---|
| `src/llama-model-loader.cpp` | per-tensor shard plan (COLUMN/ROW/EXPERT, quant-block-aligned) + `mbind` placement to the owning node |
| `src/llama-graph.cpp` `build_moe_ffn` / `build_ffn` | per-team expert GEMMs on sharded weights + combine op; tensor-split (decode) vs EP (prefill) selected by `n_tokens` |
| `ggml` threadpool (`ggml/src/ggml-cpu`) | per-team affinity (or two pools); the riskiest change |
| `common/arg.cpp` | `--tp-mode`, reuse `--tp-size` |
| `src/llama-tp.cpp` | trim to team state; delete shm/futex/barrier |

## 8. Phases + sizing (from to-sort-out §5: items #1-2 ≈ 1-2 wks, #1-4 ≈ 6-8 wks)

| phase | work | size | exit |
|---|---|---|---|
| **P1 — decode side** | thread teams + mbind placement + tensor-split decode + intra-process combine | **1-2 wks** | V4-Flash decode > 15 t/s on the P720 (vs ~5.6 interleave cap); byte-identical output |
| **P2 — prefill** | EP path + batch gate (≥2048 → EP, else tensor) | +2-3 wks | prefill ≥ interleave baseline; no regression |
| **P3 — optional** | GPU layer residency (compose with moe-stream hybrid) | optional | the full LvLLM shape: GPU attention + NUMA CPU teams |

## 9. Verification (P720)

1. **Correctness oracle**: same prompt/seed, output byte-identical to a single-team run (the mask+combine must be exact — the TP mask validated this).
2. **The bandwidth proof**: decode t/s × GB/token must **cross the 40 GB/s interleave ceiling** — V4-Flash > ~6.8 t/s is the falsifiable claim; the target band is 20-28.
3. **Thread-balance check**: tensor-split decode should show both teams' CPU busy (EP at batch 1 would not — the straggler).
4. Matrix: A3B + V4-Flash pp/tg ladders (GLM after — 156 GB, needs the 2 TB NVMe local).

## 10. Risks / open questions

- **ggml threadpool surgery** is the hard part — two pools / per-team affinity interacts with every op, not just MoE. Mitigate: bind at the pool level, not per-op; the attention/dense path is single-team-safe.
- **mbind on the mmap path** (GGUF is one big mapping; per-tensor sub-range mbind needs `MPOL_MF_MOVE` and a faulted-in region) — verify page placement with `numastat` / `/sys/devices/system/node/node*/numastat` after load.
- **Decode balance at small active counts**: tensor-split makes both teams work every expert, but the combine adds a per-layer barrier — must hide under the bandwidth win (P1's 15 t/s exit proves it).
- Does the kernel's `kernel.numa_balancing=0` (our recipe) interfere with `MPOL_MF_MOVE`? It shouldn't (mbind is explicit) — verify.
- **Interaction with moe-stream (GPU hybrid)**: compose = GPU attention/dense + NUMA CPU teams; the moe-stream pool machinery targets the GPU, the teams target the CPU experts — orthogonal buffers, but the graph has both injection points; test together in P3.

## 11. References

- `cpu-tp-numa-spec.md` (multi-process design — superseded; its F2/F4/F7 analysis and loader shard plan carry over)
- `research/llm-homelab-to-sort-out.md` §5 (ktransformers NUMA port: thread binding, interleave, GPU residency sizing)
- `research/llm-homelab-p520-flash-next-report.md` (§5: interleave UPI caveat, bandwidth-wall targets)
- P720 measurements 2026-09-01: A3B decode 9.67 t/s (compute-bound), TP 2-rank 7.73 t/s (EP straggler at batch 1), server deadlock at round 161 (process desync)
