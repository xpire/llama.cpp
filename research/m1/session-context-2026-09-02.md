# Session context — P720 prefill/NUMA work (2026-09-02, full record)

This is the complete working-context record of the session: hardware, what was built, what was
measured, every bug found and fixed, and the current open state. For the implementation agents:
read this + `tooling-and-debugging-guide.md` + the relevant spec/prompt docs. Branch:
`feat/prefill-gpu-ada` (fork remote: `xpire`).

## 1. Hardware context

| Box | Specs | Role |
|---|---|---|
| testbed | 7800X3D, 96 GB, RTX 4070 Ti 12 GB, Bazzite (atomic) | dev; all runs inside the `nvidiabox` podman container. **Single-NUMA** — NUMA machinery inert here |
| P720 | 2× Xeon 6138 (Skylake-SP, **no VNNI**, 2.4 GHz all-core), 192 GB (12×16 GB, 12ch @ 2133), Debian at 192.168.0.251 | the target. **No SSH from the testbed** — the user runs commands + pastes output |

Models: A3B = `Qwen3.6-35B-A3B-UD-Q4_K_XL` (22.8 GB, 3B active — the -MTP variant);
V4-Flash = `DeepSeek-V4-Flash-0731` (284B, 13B active); NAS at `/mnt/prox_share` (TrueNAS
192.168.0.20). The P720's 2 TB drive = `/dev/sda` (whole-disk ext4, `~/models` symlinks to it).

## 2. Key measured facts (ground truth)

- P720 A3B CPU-only decode: **9.7 t/s peak at t=8-16** (the small-GEMM/latency wall — flat from
  8→16 threads, *declines* past 16: 6.7 at t=40). Prefill ~122 t/s at t=40.
- 7800X3D (VNNI) A3B decode: 16.2 t/s CPU-only, 29.2 t/s GPU-hybrid. Prefill 512 t/s CPU-only.
- The 6138s are healthy (2.3-2.7 GHz under load, 12 DIMMs confirmed) — the no-VNNI ISA gap is the
  real per-core story (~21× per-core vs the 7800X3D on int8 Q4).
- MTP spec decode on the A3B: **5× regression** (1.1 vs 5.7 t/s) — no compute headroom on the 6138s.
- **V4-Flash MXFP4 on the P720 CPU is unusable**: 5.4 t/s prefill, **1.9 t/s decode** — MXFP4 has no
  fast CPU kernel on no-VNNI Skylake (A3B Q4_K does 122/9.7 on the same build → the 23× gap is the
  quant, not the CPU). A K-quant GGUF is required for CPU work.

## 3. What was built on the branch (in order)

1. **moe-stream machinery** (pre-existing, this branch's GPU-hybrid work): layer-window streaming
   (`--moe-stream-window N`), attention-streaming, expert-slot mode. Validated on GPU.
2. **Multi-process CPU TP** (b817a5e72): shm transport + EP id-mask + per-layer all-reduce +
   `--tp-size/--tp-rank/--tp-peer`. **Rejected**: P720 measured 7.7 t/s vs 9.7 single-process, and
   the server path deadlocks (two independent processes take different graph paths — desync).
3. **Single-process NUMA spec** (`docs/cpu-tp-single-process-numa-spec.md`): the locality goal kept,
   threads instead of processes. Increments:
   - inc-1: `ggml_tensor.numa_node` + mmid node-skip (per-op thread-team affinity)
   - inc-2: mbind placement + env tagging
   - inc-3: per-node n_ff shards + two-GEMM combine

## 4. The bug chain (each found by a specific method — see the debugging guide)

1. **Futex PRIVATE bug**: `FUTEX_WAIT_PRIVATE` is per-mm — a wake in process A never reaches a
   waiter in process B. Found via the standalone 2-process barrier test + `/proc/<pid>/syscall`.
2. **Timed futex never returns** on this host → barrier switched to spin+yield.
3. **`MPOL_MF_MOVE` = (1<<1), not (1<<4)** → mbind EINVAL. Found via an isolated syscall test.
4. **`llama_numa_init` never wired into llama-cli/llama-server** → `ggml_numa_nodes()` stays 0 →
   the split gate never passed. Found by grepping call sites.
5. **Memory doubling**: the split kept the full host tensor + shard copies = 2× expert memory
   (fatal: V4-Flash 156 GB → ~312 GB). The host is freed post-shard-copy; **but the load-time
   transient is still 2×** (150 GB host + 150 GB shards > 192 GB RAM — big models still can't
   split-load; needs file-direct shard loading).
6. **Shard split the wrong axis**: mul_mat_id weights are contraction-first — gate/up split ne[1]
   (output → concat), down split ne[0] (input → add). Verified from the real GGUF shapes.
7. **Prefill slab assembly axis-dependent**: gate/up (contiguous blocks) concat; down (per-row
   halves) interleave.
8. **mmid node-skip dropped chunks**: the ith-indexed fast path broke with a thread subset; the
   one-chunk-per-thread break is disabled for node-tagged ops.
9. **The moe-stream-CPU corruption** (the big one): `--moe-stream-window N` on the CPU-only build
   produced degenerate looping output ("WBK/Württembergische…"). The model is fine (answers Paris
   on plain CPU and GPU+moe-stream). **Fixed by the implementation agent** (88b056be8) — output
   now correct on both paths.

## 5. Current state (validated)

- The moe-stream-CPU path: correct output after 88b056be8.
- The NUMA split (LLAMA_NUMA_TENSOR_SPLIT=1): shards byte-exact (verified via LLAMA_NUMA_VERIFY,
   120+ pairs OK), output coherent and consistent with the unsplit until a ~1e-6 fp-rounding token
   flip at ~token 18 (inherent to the split-sum — compare *semantically*, not diff-empty).
- Memory: final footprint 1×, split ~50/50 across nodes (numastat confirmed).
- **V4-Flash split test pending**: the MXFP4 file is unusable on CPU (quant issue); a K-quant GGUF
   is required. Also: the split's load-time 2× transient OOMs at 150 GB — file-direct shard
   loading is the missing piece for big models.

## 6. Open items / next steps (ranked)

1. **V4-Flash Q4_K GGUF download** (~165 GB, unsloth/DeepSeek-V4-Flash-0731-GGUF) → re-run the
   plain CPU baseline → the real bandwidth-bound number the split targets.
2. **File-direct shard loading** (shards read their halves from the GGUF; never materialize the
   full host) — removes the 2× load transient, making the split viable for >96 GB models.
3. The fp-rounding token flip: decide whether it's acceptable (semantic equivalence) or needs the
   int32-exact accumulation path.
4. Bisect/confirm the moe-stream-CPU corruption fix didn't regress the GPU hybrid.

## 7. Key commands (see the tooling guide for the full set)

```bash
# bench (testbed, inside nvidiabox):
podman exec nvidiabox bash -lc 'cd /var/home/xpirep/dev/llama.cpp && ./build-cpu/bin/llama-bench -m $A3B -t 8 -b 2048 -ub 2048 -p 512 -n 128 -r 3'
# A3B (testbed): ~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf

# P720 oracle (user runs): llama-cli -m <A3B> --moe-stream-window 1 --numa distribute --spec-type none \
#   -p "..." -n 32 -s 42 -t 16 --single-turn --no-display-prompt --temp 0 -c 2048 < /dev/null
# split: prefix LLAMA_NUMA_TENSOR_SPLIT=1; verify: + LLAMA_NUMA_VERIFY=1

# the honest CPU baseline (no moe-stream window):
./build/bin/llama-bench -m <model> -t 40 -b 2048 -ub 2048 -p 512 -n 128 -r 1
```

## 8. Reference docs in the repo

- `docs/cpu-tp-single-process-numa-spec.md` — the single-process NUMA design + P1 status
- `docs/cpu-tp-numa-spec.md` — the (superseded) multi-process design
- `research/m1/tooling-and-debugging-guide.md` — environment + run patterns + debugging playbook
- `research/m1/p1-fix-implementation-prompt.md` — the agent task for the two P1 fixes
- `research/m1/moe-stream-cpu-fix-prompt.md` — the agent task for the moe-stream-CPU corruption
- `research/m1/run-log-2026-08-30.md` — the experiment log
- `research/m1/run-experiments.sh`, `oracle.sh`, `tp-barrier-test.cpp` — test helpers
