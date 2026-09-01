#!/usr/bin/env bash
# A3B correctness oracle via llama-server + curl (llama-cli is a server wrapper that hangs).
# usage: oracle.sh <model> <label>
set -u
cd /var/home/xpirep/dev/llama.cpp || exit 1
PKGS=/var/home/xpirep/miniconda3/pkgs
export LD_LIBRARY_PATH="/var/home/xpirep/dev/llama.cpp/build/bin:$(for d in $PKGS/cuda-*/targets/x86_64-linux/lib $PKGS/cuda-*/lib $PKGS/libcuda*/targets/x86_64-linux/lib $PKGS/libcuda*/lib $PKGS/libcu*/targets/x86_64-linux/lib $PKGS/libcu*/lib; do [ -d "$d" ] && echo -n "$d:"; done | sed 's/:$//')"

MODEL=$1; LABEL=${2:-oracle}; PORT=8123
OUT=/tmp/${LABEL}-oracle.out
rm -f "$OUT"

timeout 240 ./build/bin/llama-server -m "$MODEL" -ngl 999 --moe-stream-window 8 \
    -c 512 -nkvo -t 8 --port $PORT --host 127.0.0.1 >/tmp/${LABEL}-server.log 2>&1 &
SRV=$!

# wait for ready (max 180s)
ready=0
for i in $(seq 1 90); do
    if curl -s -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1; then ready=1; break; fi
    sleep 2
done
if [ $ready -ne 1 ]; then
    echo "server not ready"; tail -5 /tmp/${LABEL}-server.log; kill $SRV 2>/dev/null; exit 1
fi

curl -s -m 120 http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
    -d '{"prompt":"The capital of France is","n_predict":32,"seed":42,"temperature":0,"cache_prompt":false}' \
    > "$OUT" 2>&1
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
echo "rc=$? — output:"
python3 -c "import json,sys; d=json.load(open('$OUT')); print(d.get('content','<none>')[:200])" 2>/dev/null || cat "$OUT" | head -5
