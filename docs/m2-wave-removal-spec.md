# Wave-machinery removal — spec

Branch `feat/prefill-gpu-ada` · 2026-08-31 · status: **proposal** (spike validated)

## 1. Purpose

Remove the multi-pass prefill wave machinery (the `llama_moe_stream_wave_ids`
/ `llama_moe_stream_wave_mask` custom ops and their helpers) from MoE expert
streaming. Keep the layer-window mode (`--moe-stream-window`) and the
expert-slot mode's single-pass path (`--moe-stream-cache`, distinct experts
<= cache slots) byte-identical.

## 2. Why the waves exist, and why they are now obsolete

The waves were the answer to: "a ubatch touches more distinct experts than the
expert cache holds." In expert-slot mode the cache holds n_slots slabs; when
`n_touch_max = min(n_expert, n_tokens * n_expert_used) > n_slots`, the expert
GEMMs run in waves of at most `(n_slots - n_expert_used)/2` experts, the other
waves' pairs masked to zero and summed.

The window mode removed the problem: a window slot holds a FULL layer, so
`n_slots == n_expert` and `n_touch_max <= n_slots` always. The wave trigger in
build_moe_ffn (`if (n_touch_max > msl->n_slots)`) can never fire in window
mode; `n_stream_waves` stays 1 and the wave block is unreachable dead code.

Measured (A3B, ub2048, stress matrix 16.4): the wave regime is the slow path.
Expert-slot 32s (256 distinct experts vs 32 slots = 8 waves/layer) prefills at
~605 t/s vs the window mode's ~1878 t/s at the same ubatch — the window's
3x win is largely *because* it avoids the waves.

## 3. Scope

Delete:

- `llama_moe_stream_wave_ids` and `llama_moe_stream_wave_mask` callbacks.
- `plan_waves_locked`, `stage_wave_locked`, `emit_wave_slots`,
  `wave_userdata`, the `llama_moe_stream_wave` struct.
- The per-layer wave state (`plan_capacity`, `plan_n_waves`,
  `plan_next_wave`, `expert_wave`, `plan_pool`, `pool_used`, `wave_ud`).
- The wave stats (`n_wave_calls`, `n_waves_run`, `n_preload_issued`,
  `n_preload_ready`, `t_stall_wave_us`) and their print_stats lines.
- The wave node-budget term in `llama_context::graph_max_nodes`
  (`24 * n_waves * layers`), which currently over-reserves ~2880 phantom nodes
  for every window-mode build (n_waves computes as 3 there even though waves
  never fire).

Keep: the window mode (residency, prefetch, wait-op), the remap op, the
single-pass expert-slot branch in build_moe_ffn (`n_stream_waves == 1`), the
multi-pass abort ("increase --moe-stream-cache or reduce -ub").

## 4. Behavior matrix

| path | before | after |
|------|--------|-------|
| window mode (any W) | waves never fire | unchanged, byte-identical |
| expert-slot, distinct <= slots | single-pass remap | unchanged, byte-identical |
| expert-slot, distinct > slots | wave-split, ~605 t/s (A3B) | abort with the existing "increase --moe-stream-cache or reduce -ub" message |

Perf impact: none on the window and single-pass paths (verified structurally:
the wave block is a separate `if`; removing it does not touch the remap
branch). The multi-pass regime stops running instead of crawling — a
functional boundary, not a regression of a fast path. The `graph_max_nodes`
budget becomes exact (smaller node-array allocation; base budget still has
~30x headroom over actual usage, so no allocation risk).

## 5. Verification

1. Rebuild llama-cli/llama-bench (nvidiabox, ninja).
2. Correctness oracle, byte-identical vs the pre-removal outputs:
   - window W=8 (attention streamed): "Here's a thinking process:" identical
     to run-20260830-210727-attnstream.out.
   - window W=1: identical to run-20260830-223801-w1.out.
   - expert-slot 32s single-pass (small prompt, distinct <= 32): identical to
     run-20260830-224034-cache32s.out.
3. The multi-pass abort: expert-slot 32s at ub2048 (pp >= 1K) must now abort
   with the clear message (was 605 t/s). Documented, expected.
4. Perf: window W=8 pp512/pp8192 ub2048 within noise of the pre-removal
   numbers (584.9/1932.7 t/s).

## 6. Constraints

- Do not modify mainline. Branch `feat/prefill-gpu-ada` only.
- The window + attention-streaming machinery (this spike) must stay intact.
- ASCII-only comments; 1-2 line comments where needed; no new dependencies.
- Keep the multi-pass abort message so misconfiguration fails loudly instead
  of silently running CPU prefill.
