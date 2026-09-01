#!/usr/bin/env bash
# P720 prefill spike experiments — synchronous runner.
# usage: run-experiments.sh <a3b|a10b|hunyuan|all>
# every phase tees stdout+stderr to files under research/m1/runs/<ts>/
set -u

cd /var/home/xpirep/dev/llama.cpp || exit 1

# CUDA runtime libs (conda pkgs) + build/bin
PKGS=/var/home/xpirep/miniconda3/pkgs
export LD_LIBRARY_PATH="/var/home/xpirep/dev/llama.cpp/build/bin:$(for d in $PKGS/cuda-*/targets/x86_64-linux/lib $PKGS/cuda-*/lib $PKGS/libcuda*/targets/x86_64-linux/lib $PKGS/libcuda*/lib $PKGS/libcu*/targets/x86_64-linux/lib $PKGS/libcu*/lib; do [ -d "$d" ] && echo -n "$d:"; done | sed 's/:$//')"

BENCH=./build/bin/llama-bench
CLI=./build/bin/llama-cli

A3B=/var/home/xpirep/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
A10B=/var/home/xpirep/models/Qwen3.5-122B-A10B/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf
HUNYUAN=/mnt/prox_share/models/Hunyuan-A13B-Instruct/Hunyuan-A13B-Instruct-Q4_K_M.gguf

TS=$(date +%Y%m%d-%H%M%S)
OUT=/var/home/xpirep/dev/llama.cpp/research/m1/runs/$TS
mkdir -p "$OUT"

run() { # run <label> <cmd...>  — tee both streams to $OUT/<label>.out
    local label=$1; shift
    echo "== $(date +%H:%M:%S) [$label] $*" | tee -a "$OUT/runner.log"
    "$@" >"$OUT/$label.out" 2>&1
    local rc=$?
    echo "== $(date +%H:%M:%S) [$label] rc=$rc" | tee -a "$OUT/runner.log"
    [ $rc -ne 0 ] && echo "  !! FAILED (see $OUT/$label.out)" | tee -a "$OUT/runner.log"
}

phase_a3b() {
    echo "### PHASE A3B ($A3B)"
    # oracle (run-log 16.3): expect "Here's a thinking process:" — llama-cli is the server wrapper
    # in this build; -c 512 -nkvo 1 keeps the KV alloc tiny (recorded run was the interactive CLI)
    run a3b-oracle $CLI -m "$A3B" -ngl 999 --moe-stream-window 8 -c 512 -nkvo 1 \
        -p "The capital of France is" -n 32 -s 42 -t 8 --temp 0
    # stress matrix (16.4): W in {1,2,4,8}, pp {1K,2K,8K,32K,64K} x2 + tg1000, ub2048.
    # KV OFFLOADED + flash attn: A3B KV is tiny (40 layers, 2 kv heads -> 5.1 GB f16 @ 64K, fits 12 GB);
    # the recorded flat ~1600 t/s matrix requires GPU attention (nkvo makes it KV-copy-bound)
    for W in 1 2 4 8; do
        run a3b-W${W}-pp $BENCH -m "$A3B" -ngl 999 --moe-stream-window $W -b 2048 -ub 2048 -fa on -t 8 \
            -p 1000 -p 2000 -p 8000 -p 32000 -p 64000 -n 0 -r 2
        run a3b-W${W}-tg $BENCH -m "$A3B" -ngl 999 --moe-stream-window $W -b 2048 -ub 2048 -fa on -t 8 \
            -p 0 -n 1000 -r 1
    done
    # expert-slot regression (16.3): --moe-stream-cache 32s (correctness, small prompt)
    run a3b-cache32s $BENCH -m "$A3B" -ngl 999 --moe-stream-cache 32s -b 2048 -ub 2048 -fa on -t 8 \
        -p 16 -n 0 -r 1
}

phase_a10b() {
    echo "### PHASE A10B ($A10B)"
    # 16.4 table: streamed W=1 / W=2 at pp2048 + pp8192, ub2048, KV in RAM
    for W in 1 2; do
        run a10b-W${W}-pp $BENCH -m "$A10B" -ngl 999 --moe-stream-window $W -b 2048 -ub 2048 -nkvo 1 -t 8 \
            -p 2048 -p 8192 -n 0 -r 2
    done
    # 16.8 pending: A10B W=2 ubatch scaling (16.9: ub1024-8192 pins the compute-buffer constant).
    # ub4096 skipped: W=2 + ub4096 compute buffer OOMs the 12 GB card (user: stay at W<=2)
    run a10b-W2-ub1024 $BENCH -m "$A10B" -ngl 999 --moe-stream-window 2 -b 2048 -ub 1024 -nkvo 1 -t 8 \
        -p 2048 -n 0 -r 2
    # decode sanity at W=2
    run a10b-W2-tg $BENCH -m "$A10B" -ngl 999 --moe-stream-window 2 -b 2048 -ub 2048 -nkvo 1 -t 8 \
        -p 0 -n 128 -r 1
}

phase_hunyuan() {
    echo "### PHASE HUNYUAN ($HUNYUAN)"
    # oracle via llama-server + urllib (llama-cli hangs; see oracle.sh)
    # 16.10: W=2 ub4096 = 8.9 GB VRAM fits the 12 GB card; pp sweep + tg, KV in RAM
    for W in 1 2; do
        run hy-W${W}-pp $BENCH -m "$HUNYUAN" -ngl 999 --moe-stream-window $W -b 4096 -ub 4096 -nkvo 1 -t 8 \
            -p 4096 -p 8192 -n 0 -r 2
        run hy-W${W}-tg $BENCH -m "$HUNYUAN" -ngl 999 --moe-stream-window $W -b 4096 -ub 4096 -nkvo 1 -t 8 \
            -p 0 -n 256 -r 1
    done
}

case "${1:-all}" in
    a3b)     phase_a3b ;;
    a10b)    phase_a10b ;;
    hunyuan) phase_hunyuan ;;
    all)     phase_a3b; phase_a10b; phase_hunyuan ;;
    *) echo "usage: $0 <a3b|a10b|hunyuan|all>"; exit 2 ;;
esac

echo "### DONE — outputs in $OUT" | tee -a "$OUT/runner.log"
