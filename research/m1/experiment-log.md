# m1 experiment log (prefill-gpu-ada)

Branch `feat/prefill-gpu-ada` · started 2026-08-30 · box: 4070 Ti 12 GB (PCIe4 x16,
26.9 GB/s), 7800X3D, 96 GB RAM, Bazzite; runs inside `nvidiabox` (podman).

## Hardware context (2026-08-31)

**The P720 (2x 6138, 192 GB, 12-channel) is still in transit — this 7800X3D box is the TESTBED
until it lands.** Consequences for the log:

- **Runnable here**: everything on the A3B (22 GB) and A10B (71 GB) — the mechanism testbeds.
  These validate what the P720 will later configure/run (batch gate, ubatch, W, attention
  streaming correctness).
- **P720-bound (not runnable here)**: all "Flash"-family models — GLM-5.3-Flash (157 GB),
  DeepSeek-V4-Flash (156 GB), Qwen3.8-Flash-Next (~160 GB) — every one exceeds this box's
  96 GB RAM. Their experiments are projections/planning (B7, B11) or wait for the P720.
- **Classification tag in the backlog**: [here] = runs on this box now; [projection] = planning
  for the P720's arrival; [P720] = needs the hardware.
- The buy/no-buy questions (12 vs 24 vs 32 GB VRAM; single-card Flash streaming) are decided
  from [here]-measurements + [projection] analysis, BEFORE the P720 config is fixed.

## Convention

Every experiment = one entry:

1. **Command** - the exact invocation (no scripting needed for benches; llama-bench
   is a single command with flags).
2. **Outputs** - the timestamped files under /tmp (run-<ts>-<label>.{json,err,out}).
   Runs are detached (podman exec -d + nohup) and polled via the files.
3. **Overview** - the numbers + what they mean.

Models:
- A3B = Qwen3.6-35B-A3B-UD-Q4_K_XL (22.8 GB), 40 layers, 256 experts/8 used
- A10B = Qwen3.5-122B-A10B-Q4_K_M (76.5 GB, 3-part), 43 layers, 256 experts/8 used
- Hunyuan-A13B Q4_K_M (45.4 GB, NAS) - 32 layers, 64 experts/~11 used
- Qwen3.8-Flash-Next UD-IQ1_M (NAS) - 48 layers, 512 experts/10 used

Flags common to every bench:
`-b 2048 -ub 2048 -t 8 -ctk q8_0 -ctv q8_0 -nkvo 1 -sm layer -mg 0 -o json`

## Completed

### E1. Correctness oracle, window W=8 (attention streamed) vs resident - PASS
- command: `llama-cli -m <A3B> -ngl 999 --moe-stream --moe-stream-window 8 --spec-type none -p "The capital of France is" -n 32 -s 42 -t 8 --single-turn --no-display-prompt --temp 0 -c 2048 < /dev/null`
- outputs: /tmp/run-20260830-210727-attnstream.out (streamed), /tmp/run-20260830-211426-resident.out (resident, stashed build)
- overview: both produce "Here's a thinking process:"; model output BYTE-IDENTICAL
  (only the loading-spinner length and the timing line differ). W=1 (run-20260830-223801-w1.out)
  and expert-slot 32s (run-20260830-224034-cache32s.out) also pass.

### E2. A10B W=2 fit test - the spike's key result
- command: `llama-bench -m <A10B> -ngl 999 --moe-stream --moe-stream-window 2 -b 2048 -ub 2048 -t 8 -p 2048 -n 0 -ctk q8_0 -ctv q8_0 -nkvo 1 -sm layer -mg 0 -o json`
- outputs: /tmp/run-20260830-224048-a10b-w2-pp2048.json (first, desktop-contended: 411.8 t/s),
  /tmp/run-20260830-225159-a10b-w2.json (clean same-session: pp2048 679.8, pp8192 675.6)
- overview: W=2 at pp2048/ub2048 previously OOM'd at ~10363 MiB with resident
  attention; attention streaming (~0.86 GB freed) makes it FIT. Clean numbers:
  W=2 pp2048 679.8 / pp8192 675.6 vs W=1 570.3/569.0 (run-20260830-225159-a10b-w1b.json) -
  W=2 beats W=1 by ~19% (load hiding: 14.4 ms vs 44.9 ms stall per wait).

### E3. A3B stress matrix (pp 1K..64K x2 + tg1000, ub2048)
- command: `llama-bench -m <A3B> -ngl 999 --moe-stream --moe-stream-window <1|2|4|8> -b 2048 -ub 2048 -t 8 -p 1000-64000*2 -n 1000 -ctk q8_0 -ctv q8_0 -nkvo 1 -sm layer -mg 0 -o json` (per W; expert-slot variant: `--moe-stream-cache 32s`)
- outputs: /tmp/run-20260830-230409-a3b-w{1,2,4,8}.json, -a3b-cache32s.json, -a10b-w2.json
- overview (t/s):

| A3B ub2048 | W=1 | W=2 | W=4 | W=8 | expert-slot 32s |
|------------|-----|-----|-----|-----|-----------------|
| pp1000     | 969 | 1042| 1047| 1061| 606 |
| pp2000     | 1630| 1825| 1840| 1888| 605 |
| pp8000     | 1626| 1878| 1836| 1883| 605 |
| pp32000    | 1604| 1848| 1829| 1839| 596 |
| pp64000    | 1512| 1702| 1721| 1716| 575 |
| tg1000     | 21.2| 21.2| 21.3| 21.1| 30 |

- overview: W>=2 gives the same prefetch behavior (W only buys VRAM); W=1 ~15%
  slower (serial loads); window mode is 3x expert-slot at large prompts (the
  wave machinery costs ~605 t/s); tg is W-independent (decode uses host tensors).

### E4. Attention-streaming overhead vs resident (A3B, ub2048)
- command: `llama-bench -m <A3B> -ngl 999 --moe-stream --moe-stream-window 8 -b 2048 -ub 2048 -t 8 -p 512 -p 8192 -n 0 -ctk q8_0 -ctv q8_0 -nkvo 1 -sm layer -mg 0 -o json`
- outputs: /tmp/run-20260830-212941-a3b-attnstream.json (563.5/1725.4),
  /tmp/run-20260830-213643-a3b-resident.json (637.0/2225.3, stashed build)
- overview: attention streaming costs ~11% (pp512) to ~22% (pp8192); the
  prefetch-at-wait-op fix (E5) recovered most of the gap.

### E5. Prefetch trigger: remap -> wait-op (A3B, ub2048)
- command: same as E4, window 8, pp512+pp8192
- outputs: /tmp/run-20260830-215325-a3b-prefetch.json (584.9/1932.7)
- overview: moving the next-layer prefetch from the remap (mid-layer) to the
  wait-op (start of layer) gives the load the FULL layer compute to hide under:
  stall per wait dropped 17.96 -> 7.87 ms (pp512) and 15.36 -> 3.82 ms (pp8192).

### E6. Hunyuan-A13B bring-up (ctx overflow fix)
- command: `llama-cli -m /var/mnt/prox_share/models/Hunyuan-A13B-Instruct/Hunyuan-A13B-Instruct-Q4_K_M.gguf -ngl 999 --moe-stream --moe-stream-window 2 --spec-type none -p "x" -n 1 -s 42 -t 8 --single-turn --no-display-prompt --temp 0 -c 512 < /dev/null`
- outputs: /tmp/hy2.log (rc=0)
- overview: exposed a real bug - the moe_stream ggml context was sized
  layers*4+1 tensor slots, but all-dense-attention layers (Hunyuan: 4 attn
  views/layer x 32 = 142 > 129) overflow it (GGML_ASSERT(obj_new)). Fixed to
  layers*8+1 (src/llama-moe-stream.cpp). After the fix + a container restart
  (page-cache/swap pressure was OOM-killing the 45 GB load), the model loads
  and runs at W=2 (VRAM 7.0 GB). Prompt 5.0 t/s at 1 token (batch-gate regime).

### E7. External reference: A10B on an RTX A5500 Laptop (16 GB)
- command (user run): `./llama-bench -hf unsloth/Qwen3.5-122B-A10B-MTP-GGUF:UD-Q4_K_M -ncmoe 49 -fa on`
- outputs: user-reported (no local file)
- overview: resident/CPU-MoE config (49 MoE layers on CPU): pp512 96.2 t/s,
  tg128 13.89 t/s. Reference point for the 16 GB-class analysis.

## Backlog (to run, in order - A3B/A10B first, Hunyuan last)

Tagging per the hardware context: **[here]** runs on this box; **[projection]** = planning;
**[P720]** = needs the P720.

Models: A3B + A10B stay LOCAL (NVMe) for the benches; Hunyuan is deferred until the
A3B/A10B experiments are proven, then copied from the NAS to the NVMe.

- **[here] B1. A10B ub-scaling (the flat-regime proof): `llama-bench -m <A10B> --moe-stream-window 2 -b 8192 -ub 1024,2048,4096 -t 8 -p 8192 -n 0 ...` + W=1 ub 2048,4096; and A3B W=2 ub 2048,4096,8192 + W=8 ub 2048,4096. Prediction: t/s flat (~680 A10B, ~1900 A3B) - compute-bound; VRAM linear (pins the compute-buffer constant). Script: /tmp/ub-scaling.sh (locked). The earlier attempts died to monitor-infra reaping; run detached.
- **[here] B4. A10B W=2 @ub4096 (compute-buffer pin): included in B1 (a10b-w2-ub ub4096); fits ~11.2 GB on 12 GB, expected ~680 (flat).
- **[here] B5. Wave-removal verification (per docs/m2-wave-removal-spec.md): byte-identical window + single-pass expert-slot; multi-pass expert-slot now aborts.
- **[here?] B3. Qwen3.8-Flash-Next UD-IQ1_M (NAS): verify the download (the 3 parts total 74 GB on disk vs ~28 GB expected for IQ1_M - part1 at 10.9 MB looks wrong; check HF file sizes), then the guarded oracle + bench (10 active experts).
- **[here] B2. Hunyuan oracle + prefill bench - AFTER the above are proven (user gate: copy Hunyuan from NAS to NVMe only once the A3B/A10B experiments are run and reported). `llama-bench -m <Hunyuan> --moe-stream-window 2 -p 512 -p 2048 -n 0 ...`; expected ~700-900 t/s (compute-bound, 11 active experts).
- **[here] B6. Hunyuan on freed RAM or a smaller quant (Q3_K_XL / IQ2_M ~25-30 GB) if the box stays memory-pressured.
- **[projection] B7. P720 projections (recorded in run-log 16.5/16.6/16.9, not runnable here): GLM-5.3-Flash / DSV4-Flash at 16 GB x ub3072-4096 = 270-390 t/s on PCIe3.

### Suggested additions (from the main agent, 2026-08-31)

- **[here] B8. Prompt-length crossover (the P2 batch-gate number)**: pp 128/256/512/1024/2048 at ub
  2048, window W=8 vs expert-slot vs CPU on the A3B. Pins where GPU prefill crosses the CPU
  (window wins big-batch, expert wins small-batch) - the production engagement threshold.
  Prediction: window crosses CPU around pp 512-1024; expert crosses ~pp 1024-2048.
- **[here] B9. Attention-streamed correctness at scale**: the E1 oracle was a 7-token prompt. Run the
  4K-token byte-identity oracle (same seed) with W=8 attention-streamed vs resident - the
  attention slots get rewritten across ubatches (2 ubatches @ ub 2048) - must stay byte-identical.
- **[here] B10. Decode after attention-streamed prefill**: TG speed (n 1000, c 64K) with a streamed-
  attention prefill - confirms the decode phase-switch (host attention path) is unaffected at
  long context (vs expert-mode TG ~51 t/s A3B).
- **[projection] B11. P720 projection: Flash family on a SINGLE 12 GB with attention streaming**: V4-Flash /
  GLM-5.3-Flash attention (~1.6/1.7 GB) streams -> W=3-4 fits on 12 GB. Project prefill
  (compute-bound estimate + stream ceiling) and decide whether the single-card Flash case beats
  the CPU on the P720 before buying anything.
- **[here] B12. Attention-streamed overhead re-measure (post-fix)**: E4's 11-22% was measured BEFORE the
  E5 wait-op prefetch fix. Re-run W=8-streamed vs resident on the A3B (pp512+pp8192) with the
  current code - expect the overhead to shrink toward the spec's <=10% bar.
