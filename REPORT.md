# nemotron-120b-a12b — NVIDIA Nemotron-3-Super (MoE) — `-fit on` rescued it

NVIDIA `Nemotron-3-Super-120B-A12B` MoE (120B total / ~12B active) at **UD-IQ4_NL (61 GB, 3 shards)**. Doesn't fit
48 GB → offload. **Decode ~16.3 tok/s, 7/7 clean.** Previously OOM'd twice on hand-tuned `--n-cpu-moe` splits; **`-fit
on` loaded it first try** — the same lever that rescued Leanstral.

## Decode throughput (7 workloads, `-fit on`)
| Workload | decode tok/s |
|---|---|
| Translation (multilingual) | **16.9** |
| Math / reasoning | 16.8 |
| Summarization | 16.8 |
| Chat / dialogue | 16.4 |
| JSON / structured | 16.3 |
| Code generation | 16.2 |
| Free-form prose | 14.6 |
| **Average** | **~16.3** (prefill ~28.3) |

## Serving configuration
| Param | Value |
|---|---|
| Backend | llama.cpp build-master, :8018 |
| Quant | **UD-IQ4_NL, 61 GB, 3 shards** (unsloth dynamic; ≈Q4_K_M quality at 4.5 bpw — chosen to honor the Q4 floor) |
| Placement | **`-fit on`** — no manual `--n-cpu-moe`/`--tensor-split` |
| KV / ctx / fa | `--cache-type-k/v q8_0` · 4096 · `-fa on` |
| VRAM | 5090 ~31 GB + 5080 ~14–15.6 GB on GPU; ~15 GB experts in CPU RAM |

## Tuning-research + the rescue
- **History:** earlier attempts at `--n-cpu-moe 28` (→ 37 GB on the 5090, OOM) and `44 / 40,6` (→ 32.8 GB, OOM) both
  failed — hand-picking the offload count for a 120B is fragile.
- **Fix = `-fit on`:** auto-computes the GPU/CPU partition to exactly fill VRAM without OOM. First-try clean load,
  7/7 workloads, no crashes. **The generalizable lesson: stop hand-tuning `--n-cpu-moe`; let `-fit on` solve it.**
- **Quant choice:** unsloth ships no plain Q4_K_M for this model; **UD-IQ4_NL (4.5 bpw dynamic)** is the highest 4-bit
  and keeps important layers high-precision → honors the "no worse than Q4" quality floor.

## Analysis — why it beats GLM-Air (16 vs 12) despite both being offload-bound
Both are ~12B-active 100B-class MoE on the offload side of the cliff, so both are CPU-RAM-bandwidth-bound. Nemotron's
edge comes from a **smaller resident footprint** (61 GB IQ4_NL vs GLM's 73 GB Q4_K_M → fewer GB read per token from
RAM) — less data movement per token = higher tok/s. The dead-flat curve (16.2–16.9, prose aside) is the textbook
bandwidth-bound signature. Neither has spec-decode, so both sit well below the MTP-equipped 122B (~30).

## Failures → fixes
- **Earlier OOM ×2** on manual `--n-cpu-moe` — resolved by `-fit on` (see above).
- **`invalid split file name`** on first batch — shards needed the canonical `-00001-of-00003` suffix (don't rename on
  download). Fixed.

## Verdict
✅ **Works, 7/7 clean, ~16.3 tok/s** — fastest of the no-spec >100B offload models tested (beats GLM-Air's 12). Still
~2× slower than 122B-MTP because it lacks speculative decoding. A solid NVIDIA-stack 120B option. **Disk: candidate for
cleanup** (not a designated keeper). The headline is the **`-fit on` rescue** of a model that twice OOM'd on manual tuning.

_2026-06-27 · `-fit on`, 7-workload clean run, UD-IQ4_NL (Q4 floor) · the test rig · >120B offload-class survey._
