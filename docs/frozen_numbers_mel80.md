# Frozen numbers — mel80 (`fmax = sr/2`)

Tag: **`paper2-frozen-mel80`** @ commit `56c8182`.

These numbers were measured in the original Track A mel domain (`MEL80`, `fmax = sample_rate/2 = 11025 Hz`). After any new mel target (e.g. `mel80_hifigan`) absolute CER values are **not** comparable to this table without re-running under the frozen tag/domain.

Do **not** push the tag without an explicit user request.

## Production checkpoint

| Field | Value | Artifact |
|-------|-------|----------|
| Ckpt | `runs/mel_baseline_v01_s4/best.pt` | WSL run dir |
| Train size | 12490 utts / 15.41 h | `runs/mel_baseline_v01_s4/run_manifest.json`, `s4_budget_post.json` |
| Gap mode | `expand_neighbor` | same |

## Free synthesis (S4)

| Metric | Value | Artifact |
|--------|------:|----------|
| val100 Whisper CER (mean) | **0.048** | `out/intelligibility_mel_s4_val100.json` |
| val100 CER (median) | 0.026 | same |
| val100 WER (mean) | 0.153 | same (also cited as ≈0.138 in summary tables; use JSON) |
| val100 NEAR_SILENT | 0 | same |
| demo5 CER (griffin-lim) | **0.048** | `out/intelligibility_mel_s4_demo.json` / `out/intelligibility_mel_s4.json` |
| demo5 CER (HiFi-GAN) | **0.042** | `out/intelligibility_mel_s4_demo_neural.json` |

## Live eval suite (S4 + HiFi-GAN)

| Set | mean CER | Artifact |
|-----|---------:|----------|
| demo5 | 0.099 | `out/eval_mel_suite_s4_live.json` |
| homograph | 0.034 | same |
| bank_entity (after TN/ASR-score fix) | 0.093 | `out/eval_mel_suite_s4_bankfix.json` |

## Copy-synthesis / oracle toplines

| Condition | CER | Artifact |
|-----------|----:|----------|
| GT mel → Griffin-Lim | **0.027** | `out/intelligibility_gt_mel_griffin.json` |
| S1e oracle-dur | 0.031 | `out/intelligibility_mel_s1e_oracle_dur.json` |
| S1d oracle-dur | 0.052 | `out/intelligibility_mel_s1d_oracle_dur.json` |
| S1e free train (text-only) | 0.211 | `out/intelligibility_mel_s1e_train_free.json` |

## Timeline coverage (CTC blank gaps)

| Condition | cover_ratio | Artifact |
|-----------|------------:|----------|
| Raw CTC (pre-fix, S1c era) | **0.318** | `out/cover_ratio_s1d.json` (pre) / diagnosis in `docs/next_intelligibility.md` |
| After `expand_neighbor` (S1e) | **1.000** | `out/cover_ratio_s1e.json` |
| After `sil_fill` (S1d) | **1.000** | `out/cover_ratio_s1d.json` |

## CFM (parked)

| Condition | CER | Artifact |
|-----------|----:|----------|
| Mel CFM S2 free demo | **0.699** | docs / CFM run artifacts under `out/intelligibility_mel_cfm*` |
| Mel regression S2 free demo | **0.079** | `out/intelligibility_mel_s2.json` |

## Notes

- Primary narrative: `docs/next_intelligibility.md`.
- Naturalness (UTMOS / NISQA) was **not** measured at freeze time — see roadmap tasks T02/T04.
- JSON paths above live primarily under WSL `~/projects/tts-self/out/`; many were mirrored to Windows `C:\projects\tts-self\out\`.
