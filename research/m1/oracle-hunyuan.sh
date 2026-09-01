#!/usr/bin/env bash
# Hunyuan oracle: llama-server W=2 + urllib, robust poll, result to file.
set -u
cd /var/home/xpirep/dev/llama.cpp || exit 1
PKGS=/var/home/xpirep/miniconda3/pkgs
export LD_LIBRARY_PATH="/var/home/xpirep/dev/llama.cpp/build/bin:$(for d in $PKGS/cuda-*/targets/x86_64-linux/lib $PKGS/cuda-*/lib $PKGS/libcuda*/targets/x86_64-linux/lib $PKGS/libcuda*/lib $PKGS/libcu*/targets/x86_64-linux/lib $PKGS/libcu*/lib; do [ -d "$d" ] && echo -n "$d:"; done | sed 's/:$//')"
MODEL=/mnt/prox_share/models/Hunyuan-A13B-Instruct/Hunyuan-A13B-Instruct-Q4_K_M.gguf
OUT=/tmp/hy-oracle-result.json
LOG=/tmp/hy-oracle-server.log
rm -f "$OUT" "$LOG"
./build/bin/llama-server -m "$MODEL" -ngl 999 --moe-stream-window 2 -c 512 -nkvo -t 8 \
    --port 8123 --host 127.0.0.1 >"$LOG" 2>&1 &
SRV=$!
python3 - "$OUT" <<'EOF'
import json, sys, time, urllib.request, urllib.error
out = sys.argv[1]
# poll health up to 12 min (NFS heap load of 48.8 GB)
for i in range(360):
    try:
        urllib.request.urlopen('http://127.0.0.1:8123/health', timeout=2)
        break
    except Exception:
        if i % 30 == 0:
            print(f'poll {i*2}s', flush=True)
        time.sleep(2)
else:
    open(out, 'w').write(json.dumps({'error': 'never ready'}))
    raise SystemExit(1)
print('READY', flush=True)
body = json.dumps({'prompt': 'The capital of France is', 'n_predict': 32, 'seed': 42,
                   'temperature': 0, 'cache_prompt': False}).encode()
req = urllib.request.Request('http://127.0.0.1:8123/completion', data=body,
                             headers={'Content-Type': 'application/json'})
d = json.load(urllib.request.urlopen(req, timeout=600))
open(out, 'w').write(json.dumps(d))
print('content:', d.get('content', '<none>')[:150], flush=True)
EOF
RC=$?
kill $SRV 2>/dev/null
echo "rc=$RC"
