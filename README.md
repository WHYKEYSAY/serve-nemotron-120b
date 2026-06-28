# serve-nemotron-120b — how to serve **Nemotron-3-Super-120B-A12B** efficiently (any GPU)

*A GPU-agnostic operations manual for one model. The reference tok/s is from one rig; the **recipe applies to any GPU**
— figure out your VRAM, then follow it.*

- **Architecture:** MoE (120B total / ~12B active)
- **Fits in VRAM?** NO — UD-IQ4_NL ≈61 GB; offload-bound

## Recipe
**Let `-fit on` solve the split:** this model twice OOM'd on hand-picked `--n-cpu-moe` (28→37 GB on the big card; 44/40,6→32.8 GB) — **`-fit on` auto-computes the partition and loads first try.** + `-fa on` + q8 KV. **The generalizable lesson: stop hand-tuning `--n-cpu-moe` for 120B; `-fit on` fills VRAM without OOM.**

**Serving flags (llama.cpp):**
```
-fit on -fa on --cache-type-k q8_0 --cache-type-v q8_0 -c 4096
```

## Reference throughput
~16.3 tok/s decode, 7/7 clean. Beats GLM-Air (12) at the same class — smaller resident footprint (61 GB vs 73 GB) → fewer GB read per token from RAM. Still ~2× under 122B-MTP (no spec-decode).

## Failures → fixes
- Manual `--n-cpu-moe 28` / `44` → **OOM ×2**. Fix = `-fit on`.
- `invalid split file name` → keep the `-00001-of-00003` shard suffix.
- Quant: unsloth ships no plain Q4_K_M; **UD-IQ4_NL (4.5 bpw)** is the highest 4-bit → honors the Q4 floor.

## Verdict
Fastest no-spec >100B offload model tested (16 > GLM's 12). The headline is the `-fit on` rescue of a model that twice OOM'd on manual tuning.

---
## The one decision: does it FIT in your VRAM?
Estimate size ≈ params × bytes/weight (Q4≈0.5, Q8≈1, FP16≈2 B/param) + KV + ~2–3 GB overhead.
- **Fits** → full GPU residency, **no offload**, single card if it fits on one → *bandwidth-bound, fast.*
- **Doesn't fit** → offload experts to RAM (use `-fit on`), keep the active path on GPU → *RAM-bandwidth-bound, slower.*

## Measure honestly
Use the server's **`/completion` decode timings** (`predicted_per_second`), greedy, cache off, multiple workloads —
NOT short OpenAI wall-time (it understates decode). See `bench_decode.py`.

## Files
- `REPORT.md` — the detailed benchmark (throughput · config · tuning-research+sources · analysis · failures), if present.
- `bench_decode.py` — honest decode-tok/s measurement (`/completion` timings).
- `launcher-entry.json` — a ready-to-paste config (name + serving flags) for whatever model server you run.

*Part of a per-model serving-playbook set.*
