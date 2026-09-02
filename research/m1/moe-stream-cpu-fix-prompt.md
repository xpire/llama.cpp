# Implementation prompt: fix the moe-stream CPU-only corruption (A3B output)

**Branch:** `feat/prefill-gpu-ada` (working tree near `923756cf8` + uncommitted NUMA verify work).
Build CPU-only (`build-cpu`, `-DGGML_CUDA=ON` build is `build`). Testbed = single-NUMA 7800X3D box,
all commands run inside the `nvidiabox` podman container. Environment details: see
`research/m1/tooling-and-debugging-guide.md`. Model paths: see below.

## The bug (measured 2026-09-02)

The A3B model produces **degenerate, looping output** ("WBK steht für: * Württembergische Bank...",
repeating "* Württembergische Bergwerksgesellschaft" forever) **when run with `--moe-stream-window N`
on the CPU-only build**. The same model answers correctly everywhere else:

| config | prompt "What is the capital of France? Answer briefly." | result |
|---|---|---|
| CPU-only, NO moe-stream | `llama-cli -m $A3B --spec-type none -p ... -n 200 -t 16` | ✅ "Paris" (4×), proper thinking |
| GPU (`-ngl 999`) + `--moe-stream-window 2` | same | ✅ "Paris" |
| **CPU-only + `--moe-stream-window 1`** | same | ❌ WBK loop |

So: the model file is fine, the common CPU kernels are fine (a non-MoE model, stories15M, generates
coherent text on the same build), the moe-stream machinery works on GPU. **The corruption is
specifically the moe-stream streaming machinery on the CPU-only path.**

Repro (testbed, inside nvidiabox):

```bash
M=/var/home/xpirep/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
# broken:
./build-cpu/bin/llama-cli -m $M --moe-stream-window 1 --spec-type none \
  -p "What is the capital of France? Answer briefly." -n 64 -s 42 -t 16 \
  --single-turn --no-display-prompt --temp 0 -c 2048 < /dev/null
# good reference:
./build-cpu/bin/llama-cli -m $M --spec-type none \
  -p "What is the capital of France? Answer briefly." -n 64 -s 42 -t 16 \
  --single-turn --no-display-prompt --temp 0 -c 2048 < /dev/null
```

## Why this matters

All the recent NUMA/single-process-split validation on the P720 was done on the broken
moe-stream-CPU path (the split hooks into the moe-stream host tensors), so its text-output
correctness was never actually validated. The moe-stream machinery was built and validated on GPU
(the testbed's 29.2 t/s A3B matrix); the CPU-only path may never have been text-validated.

## Task 1 — bisect: regression vs pre-existing

1. Build the A3B + `--moe-stream-window 1` CPU-only at commit **`134a8e884`** (the attention-streaming
   spike, before the NUMA work and before the implementation agent's changes). Procedure:
   `git stash` (uncommitted), `git checkout 134a8e884`, `cmake --build build-cpu -j`, run the repro.
   - Answers "Paris" → the corruption is a **regression** introduced between 134a8e884 and HEAD.
     Bisect the introducing commit (`git bisect` between 134a8e884 and HEAD, using the repro as the
     gate) and fix/revert it.
   - Also loops → **pre-existing** in the moe-stream CPU path. Then Task 2.
2. Restore: `git checkout feat/prefill-gpu-ada` + `git stash pop` + rebuild.

## Task 2 (if pre-existing) — locate and fix the moe-stream CPU-only corruption

Suspects, in the moe-stream machinery (src/llama-moe-stream.cpp/.h and the graph ops in
src/llama-graph.cpp) when running without GPU offload:
- the layer-window **cache assembly** (the worker copies expert slabs into the window pool; verify a
  loaded slab byte-matches the source host tensor on CPU)
- the **attention-streaming wait-op / remap ops** (map_custom1 callbacks that gate GPU residency —
  on CPU-only the residency logic may misbehave)
- the **decode host-swap** (`host_for` path)
- quant/repack interaction on the CPU-only path

Approach: instrument the worker's slab copy (compare cache bytes vs the source tensor for one
expert/layer) and the decode swap; find where the output first diverges from the good reference.
Note whether the moe-stream CPU-only path is *supposed* to be supported (the feature targets
GPU-hybrid inference; if CPU-only moe-stream is simply out of scope, the fix may be to gate it off
with a clear error instead of silently corrupting).

## Constraints

- Do not modify mainline. Work on `feat/prefill-gpu-ada`.
- Keep the GPU-hybrid path (which works) untouched — the fix must not regress `-ngl 999`.
- The NUMA single-process split work (LLAMA_NUMA_TENSOR_SPLIT etc.) is entangled with moe-stream;
  it must keep compiling, but its correctness validation is downstream of this fix.
- No new dependencies. Build CPU-only + confirm the CUDA build still compiles.
- Commit with a clear message (state whether it was a regression or pre-existing); push to
  `xpire/feat/prefill-gpu-ada`.

## Deliverable

A short writeup (append to `research/m1/run-log-2026-08-30.md` or a new doc): the bisect result
(which commit introduced it, or "pre-existing"), the root cause, the fix, and the verification
(CPU-only moe-stream now answers "Paris" for the repro prompt; GPU path still answers correctly).
