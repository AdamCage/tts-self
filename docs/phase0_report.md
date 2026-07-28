# Phase 0 — Codec feasibility report

**Decision: GO** (exit 0)

## Cache gate note (bf16 vs structural)

Prior NO-GO used a single bf16 `max_abs` threshold of `1e-2`. On the seed-42 first utterance
that peak was `0.01416`, but it was **global numerical noise**, not a streaming seam:

- bf16: `interior_max_abs == max_abs` (`0.01416`), `boundary_max_abs ≈ 7.3e-4` (ratio `0.75`)
- fp32 (same latents path / fresh encode): `max_abs ≈ 6.3e-5`, `boundary_ratio ≈ 0.90`
- lengths match (`len_delta=0`); chunked concat aligns with full decode

Fix: structural cache equivalence runs in **float32** (`cache_max_abs_diff=1e-3`) with a
boundary-spike detector (`boundary_max > 10× interior_p95` and ≥ `1e-3`). bf16 max-abs is
still reported with a soft ceiling `2e-2`. Operating decode remains bf16; codec is not trained.

```json
{
  "decision": "GO",
  "go": true,
  "deterministic_sample_false": true,
  "cache_check_dtype": "float32",
  "cache_max_abs_diff": 6.310641765594482e-05,
  "cache_max_abs_diff_bf16": 0.01416015625,
  "cache_stats": {
    "n_samples": 67200,
    "len_full": 67200,
    "len_cached": 67200,
    "len_delta": 0,
    "max_abs": 6.310641765594482e-05,
    "mean_abs": 2.5803410608205013e-06,
    "p95_abs": 1.0498151823412627e-05,
    "p99_abs": 2.056036828435026e-05,
    "rms": 4.949340109305922e-06,
    "interior_max_abs": 6.310641765594482e-05,
    "interior_p95_abs": 1.0581497917883098e-05,
    "boundary_max_abs": 9.51811671257019e-06,
    "boundary_ratio": 0.8995056074702092
  },
  "cache_stats_bf16": {
    "n_samples": 67200,
    "len_full": 67200,
    "len_cached": 67200,
    "len_delta": 0,
    "max_abs": 0.01416015625,
    "mean_abs": 0.0002688615641091019,
    "p95_abs": 0.0009765625,
    "p99_abs": 0.001953125,
    "rms": 0.0005345905665308237,
    "interior_max_abs": 0.01416015625,
    "interior_p95_abs": 0.0009765625,
    "boundary_max_abs": 0.000732421875,
    "boundary_ratio": 0.75
  },
  "cache_ok": true,
  "cache_reason": "ok",
  "decoder_rtf": 0.0042902349999986355,
  "rtf_ok": true,
  "wer_status": "ok",
  "asr_error": null,
  "wer_orig": 0.036818851251840944,
  "wer_codec": 0.048600883652430045,
  "delta_wer_pp": 1.1782032400589102,
  "cer_orig": 0.02304544475554598,
  "cer_codec": 0.03532199009261253,
  "delta_cer_pp": 1.2276545337066551,
  "bootstrap_wer": {
    "ci95_low": -0.1383463918959644,
    "ci95_high": 2.5450458144114454,
    "p_delta_gt_4pp": 0.0
  },
  "bootstrap_cer": {
    "ci95_low": 0.08571374826022078,
    "ci95_high": 2.797012385147429,
    "p_delta_gt_4pp": 0.0
  },
  "n_samples": 120,
  "n_with_text": 120,
  "latent_layout": "btd",
  "note": "vae_std=0.625 is NOT a guaranteed noise budget; see sensitivity curves.",
  "gate": {
    "GO": "\u0394WER \u2264 2 pp and CI95 high \u2264 4 pp",
    "REVIEW": "\u0394WER in 2\u20134 pp or CI crosses 4 pp; or ASR unavailable",
    "NO-GO": "\u0394WER > 4 pp and CI95 low > 4 pp"
  }
}
```

## Decoder sensitivity sweep

Sweep ASR on n=10 utterances. Treat as empirical sensitivity — do **not** assume σ=0.625 is safe a priori.

### white
| sigma | rms | wer | cer |
|------:|----:|----:|----:|
| 0.0 | 0.0593 | 0.022222222222222223 | 0.01079136690647482 |
| 0.05 | 0.0593 | 0.022222222222222223 | 0.01079136690647482 |
| 0.1 | 0.0593 | 0.022222222222222223 | 0.01079136690647482 |
| 0.2 | 0.0594 | 0.022222222222222223 | 0.01079136690647482 |
| 0.3 | 0.0592 | 0.022222222222222223 | 0.01079136690647482 |
| 0.45 | 0.0598 | 0.022222222222222223 | 0.01079136690647482 |
| 0.625 | 0.0592 | 0.022222222222222223 | 0.01079136690647482 |
| 0.9 | 0.0597 | 0.022222222222222223 | 0.01079136690647482 |
| 1.25 | 0.0594 | 0.022222222222222223 | 0.01079136690647482 |

### ar1
| sigma | rms | wer | cer |
|------:|----:|----:|----:|
| 0.0 | 0.0593 | 0.022222222222222223 | 0.01079136690647482 |
| 0.05 | 0.0593 | 0.022222222222222223 | 0.01079136690647482 |
| 0.1 | 0.0595 | 0.022222222222222223 | 0.01079136690647482 |
| 0.2 | 0.0594 | 0.022222222222222223 | 0.01079136690647482 |
| 0.3 | 0.0597 | 0.022222222222222223 | 0.01079136690647482 |
| 0.45 | 0.0607 | 0.022222222222222223 | 0.01079136690647482 |
| 0.625 | 0.0593 | 0.022222222222222223 | 0.01079136690647482 |
| 0.9 | 0.0596 | 0.022222222222222223 | 0.01079136690647482 |
| 1.25 | 0.0597 | 0.022222222222222223 | 0.01079136690647482 |

Per-utterance records: `/home/adam/projects/tts-self/docs/phase0_asr_records.jsonl`