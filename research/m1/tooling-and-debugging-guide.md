# Tooling & Debugging Guide — P720 prefill/NUMA work

Branch `feat/prefill-gpu-ada`. This is the on-ramp: what the environment is, how to run tests on
the testbed (gaming PC), and how past debugging was actually done. Written 2026-09-02 from a full
working session. Companion: `p1-fix-implementation-prompt.md` (the active task).

## 1. The machines

| Box | Hardware | Role | Access |
|---|---|---|---|
| **testbed** ("gaming PC") | 7800X3D, 96 GB, RTX 4070 Ti 12 GB, Bazzite (atomic/ostree) | dev + all mechanism tests | this box; everything runs **inside the `nvidiabox` podman container** |
| **P720** | 2× Xeon 6138 (no VNNI), 192 GB, Debian, 192.168.0.251 | the target; NUMA behavior only valid here | **no SSH from here** — the user runs commands and pastes output |

Critical constraint: the testbed is **single-NUMA** (`ggml_numa_nodes() == 1`), so every NUMA
mechanism is inert there. The testbed proves *build correctness + no-regression + non-crash*; the
P720 is the only place the NUMA behavior can be functionally verified.

## 2. The build environment (nvidiabox)

All builds/executions happen inside the `nvidiabox` podman toolbox container. `toolbox` is not on
PATH — use podman directly:

```bash
podman exec nvidiabox bash -lc 'cd /var/home/xpirep/dev/llama.cpp && ninja -C build-cpu llama-bench 2>&1 | grep -E "error:|FAILED" | head'
```

Three build dirs (all inside the repo):
- `build/` — CUDA build (`-DGGML_CUDA=ON`, default generator). Used for GPU-hybrid tests.
- `build-cpu/` — CPU-only (`cmake -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release`). The workhorse
  for CPU/NUMA tests.
- `build-dbg/` — RelWithDebInfo, for gdb backtraces.

Running CUDA binaries needs the toolkit's libs in `LD_LIBRARY_PATH` (conda packages —
`miniconda3/pkgs/cuda-*/targets/x86_64-linux/lib` plus `libcuda*/lib` and `libcu*/lib`, joined with
`build/bin`). If `libcuda.so.1` is missing inside the container, copy it from the host's
`/usr/lib64/` (`podman cp`) — but prefer starting the container via the nvidia hooks.

`gdb` was installed into the container (`dnf install -y gdb`).

**Gotcha:** `pkill -f llama-bench` inside a `bash -lc '... llama-bench ...'` wrapper **kills the
wrapper shell itself** (its own cmdline contains the pattern). Kill by anchored pattern
(`pkill -f "^/tmp/tp-bt"`) or by PID. This wasted an hour once.

## 3. Models & data

- **A3B** (the main test model): `~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (also Q8_K_XL). The `-MTP` variant includes the MTP draft head.
- **NAS** (TrueNAS 192.168.0.20, `/mnt/prox_share` — systemd automount, 10 s idle-unmount): A10B, V4-Flash-0731, GLM-5.3-Flash, Hunyuan-A13B, Flash-Next, Qwen3-14B, Qwen3.8-27B.
- **P720**: `~/models/` symlinks to the 2 TB drive (`/mnt/models`, whole-disk ext4 on `/dev/sda`).
- Rule: copy models to NVMe before benchmarking — NFS streaming crippled the Hunyuan numbers.

## 4. Git workflow

- `origin` = upstream ggml-org/llama.cpp; **`xpire` = the fork** (https://github.com/xpire/llama.cpp).
- Work on `feat/prefill-gpu-ada`, commit, `git push xpire feat/prefill-gpu-ada`. The P720 pulls
  (`git pull xpire feat/prefill-gpu-ada`) and rebuilds.
- `.pi/loops/` and `research/m1/runs/` are local junk — never commit them.

## 5. Running tests on the testbed

Standard bench (inside nvidiabox):

```bash
podman exec nvidiabox bash -lc 'cd /var/home/xpirep/dev/llama.cpp && \
  ./build-cpu/bin/llama-bench -m $A3B -t 8 -b 2048 -ub 2048 -p 512 -n 128 -r 3'
```

Two-process TP test (the multi-process experiment — both ranks, same `--tp-peer`):

```bash
( ./build-cpu/bin/llama-bench -m $A3B -t 4 --tp-size 2 --tp-rank 0 --tp-peer smoke -b 2048 -ub 2048 -p 64 -n 32 -r 1 > /tmp/r0.out 2>&1 &
  ./build-cpu/bin/llama-bench -m $A3B -t 4 --tp-size 2 --tp-rank 1 --tp-peer smoke -b 2048 -ub 2048 -p 64 -n 32 -r 1 > /tmp/r1.out 2>&1 &
  wait )
```

Correctness oracle (token comparison): `llama-cli` with `--single-turn --no-display-prompt
--temp 0 -s <seed>`, env on/off, `diff` the outputs. (llama-cli in this build is the server
wrapper, but `--single-turn` gives one-shot generation. `curl` may be intercepted by the
context-mode hook — use `python3 -c urllib.request` for server endpoints.)

Helper scripts that already exist: `research/m1/run-experiments.sh` (phase runner, tees to files),
`research/m1/oracle.sh`, `research/m1/tp-barrier-test.cpp` (standalone 2-process barrier stress —
the repro that found the futex bug).

## 6. The P720 (validation)

The user runs commands there and pastes output. Test commands must be written for them to run,
with the exact expected result spelled out. Key P720 facts: `--numa distribute` is required for the
NUMA state (`llama_numa_init`) — **but it is NOT wired into llama-cli/llama-server** (that's the
active fix). The A3B decode sweet spot is `-t 8-16` (9.7 t/s); `-t 40` gives 6.7. The A3B is
latency-bound — the split/NUMA work is judged on V4-Flash (bandwidth-bound).

## 7. Debugging playbook (how the bugs were actually found)

General method: **isolate the mechanism before integrating.** Standalone repros found every bug:

1. **Futex PRIVATE bug (multi-process TP barrier deadlocked):** wrote `tp-barrier-test.cpp` (2
   processes, plain allreduce loop) — it deadlocked with zero llama.cpp involved. Then
   `/proc/<pid>/syscall` showed both processes futex-waiting at addresses **inside the shm segment**
   (`/proc/<pid>/maps` confirmed), and `od -A d -t d4 -N 24 /dev/shm/llama_tp_*` showed the barrier
   counters interleaved across rounds. Root cause: `FUTEX_WAIT_PRIVATE` is **per-mm** — a wake in
   process A can never reach a waiter in process B. Fix: non-private futexes. Lesson: for
   cross-process sync, private futexes are for threads only.

2. **Timed futex never returning:** a 5-line test (`futest.cpp`: `FUTEX_WAIT` with a 700 ms
   timespec) hung on this host — the timeout didn't fire even with a valid timespec. Rather than
   fight it, the barrier was switched to **spin + `sched_yield()`** (correct for a 2-rank local
   barrier: the partner is always computing on other cores).

3. **`MPOL_MF_MOVE` wrong constant:** every `mbind` returned EINVAL. An isolated syscall test
   (`/tmp/mbtest.cpp`, mmap + memset + mbind per node) showed the constant was wrong: MOVE is
   `(1<<1)`, not `(1<<4)`. Lesson: verify syscall constants against a minimal test before blaming
   the caller.

4. **`--numa distribute` does nothing for the split:** numastat showed the model on one node with
   no shards. Root cause found by grepping call sites: `llama_numa_init` is called by llama-bench/
   completion/imatrix but **not llama-cli/llama-server**, so `ggml_numa_nodes()` stays 0 and the
   shard gate never passes. Lesson: verify the init chain, not just the flag.

5. **Memory-doubling design flaw:** the user's `numastat` + arithmetic (V4-Flash 156 GB → 312 GB)
   exposed that the split kept the full host tensor *and* added shard copies. Lesson: always check
   the memory multiple against the real target models, not the test model.

Diagnostic toolkit (all via `/proc` or small tools — no strace in the container):
- `/proc/<pid>/syscall` — what the process is blocked on (futex addr, args, timeout ptr).
- `/proc/<pid>/wchan`, `/proc/<pid>/stat` (state + utime over time → busy vs blocked).
- `/proc/<pid>/maps` — is the futex address inside the shm mapping?
- `numastat -p <pid>` — memory placement per node.
- `gdb -p <pid> -batch -ex "thread apply all bt"` — needs the debug build.
- `strings build/bin/libllama.so.0.3.0 | grep <marker>` — is my change actually in the binary?
- Timed-wait diagnostics must use a **timed futex** (or a bounded spin with an abort), not a bare
  loop around futex_wait — a blocking futex never returns, so the check after it never runs.

Other lessons:
- **Logs buffer when redirected** — an "empty output" process can be mid-inference. Check utime.
- **The oracle must actually exercise the path** — the "byte-identical" A3B test was vacuous
  because the split never engaged (bug 4). Verify with memory/numastat that the mechanism ran.
- **The build dirs can vanish** (container/session churn) — `git status` confirms the source is
  fine; rebuild.
- **`ggml_tensor` struct changes** must keep `sizeof` a multiple of `GGML_MEM_ALIGN` — the field
  went into the trailing padding (int32_t + padding[4] replacing padding[8]) and the designated
  initializer in `ggml.c` must stay in declaration order.

## 8. Key flags & env

| Flag / env | What it does |
|---|---|
| `--moe-stream-window N` | layer-window streaming (GPU hybrid machinery) |
| `--tp-size/--tp-rank/--tp-peer` | multi-process TP (the rejected experiment) |
| `LLAMA_NUMA_TENSOR_SPLIT=1` | enable the single-process split (needs `ggml_numa_nodes() > 1`) |
| `LLAMA_NUMA_PLACE_LAYER=1` | tag tensors by layer % 2 (placement mechanism test) |
| `--numa distribute` | thread affinity + numa init — **required but not yet wired into cli/server** |
