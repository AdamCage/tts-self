# Corpus QC

Source: `/home/adam/datasets/tts/ruslan/corpus_qc.jsonl`  
Backend: `faster_whisper` large-v3 (greedy, temperature=0)  
Git: `/home/adam/projects/tts-self` @ see run log `runs/corpus_qc/console.log`

## Decision

Applied plan defaults **clean≤0.15 / review≤0.25 / else exclude** after looking at percentiles.

- CER p50=0.10 is healthy; mean is inflated by a long right tail (CER>1 from insertion-heavy ASR / text mismatches).
- Exclude rate is high (**7811 / 21642 = 36.1%**), but non-exclude still leaves **13831 utts / 17.78 h** for training (train non-exclude: 12490).
- Thresholds frozen with `test_set_sha256=d015f46d39b6aa81b1a67a6b00fc063af81bba6bf09a1147fc46813e33e2a77b` — do not re-split.
- Listen set: 80 examples in `out/listen_qc/` (Windows) from CER band ~0.15–0.35 (fallback: worst).

`max_dur_s` raised to **15.0** (config commit) to keep 10–15s phrases for length-degradation metrics; dropped_dur=262.

## Manifest summary

```json
{
  "n": 21642,
  "hours": 29.610154814814894,
  "hours_by_split": {
    "train": 25.714562847222254,
    "val": 2.0108271064814836,
    "test": 1.8847648611111099
  },
  "hours_by_qc_tag": {
    "clean": 15.030273564814829,
    "exclude": 11.8333608912037,
    "review": 2.746520358796293
  },
  "qc_tags": {
    "clean": 12035,
    "exclude": 7811,
    "review": 1796
  },
  "qc_thresholds": {
    "clean_max": 0.15,
    "review_max": 0.25
  },
  "filter_counters": {
    "pairs_found": 22200,
    "dropped_dur": 262,
    "dropped_unreadable": 0,
    "dropped_dup_text": 296,
    "dropped_utt_id_collision": 0
  },
  "utt_id_scheme": "ruslan_{stem}",
  "test_set_sha256": "d015f46d39b6aa81b1a67a6b00fc063af81bba6bf09a1147fc46813e33e2a77b"
}
```

## Tag counts

```json
{
  "clean": 12035,
  "exclude": 7811,
  "review": 1796
}
```

## CER percentiles

```json
{
  "p50": 0.1,
  "p75": 0.49230769230769234,
  "p90": 1.5135135135135136,
  "p95": 2.7083333333333335,
  "p99": 6.534751131221721,
  "mean": 0.572399948075319
}
```

## Top-50 worst CER

- `ruslan_008684_RUSLAN` CER=80.000 `/home/adam/datasets/tts/ruslan/wavs/008684_RUSLAN.wav`
- `ruslan_022143_RUSLAN` CER=76.000 `/home/adam/datasets/tts/ruslan/wavs/022143_RUSLAN.wav`
- `ruslan_014628_RUSLAN` CER=52.000 `/home/adam/datasets/tts/ruslan/wavs/014628_RUSLAN.wav`
- `ruslan_021619_RUSLAN` CER=50.000 `/home/adam/datasets/tts/ruslan/wavs/021619_RUSLAN.wav`
- `ruslan_013797_RUSLAN` CER=48.071 `/home/adam/datasets/tts/ruslan/wavs/013797_RUSLAN.wav`
- `ruslan_021479_RUSLAN` CER=47.000 `/home/adam/datasets/tts/ruslan/wavs/021479_RUSLAN.wav`
- `ruslan_005777_RUSLAN` CER=29.333 `/home/adam/datasets/tts/ruslan/wavs/005777_RUSLAN.wav`
- `ruslan_009341_RUSLAN` CER=28.333 `/home/adam/datasets/tts/ruslan/wavs/009341_RUSLAN.wav`
- `ruslan_020510_RUSLAN` CER=27.250 `/home/adam/datasets/tts/ruslan/wavs/020510_RUSLAN.wav`
- `ruslan_016829_RUSLAN` CER=25.333 `/home/adam/datasets/tts/ruslan/wavs/016829_RUSLAN.wav`
- `ruslan_000799_RUSLAN` CER=22.000 `/home/adam/datasets/tts/ruslan/wavs/000799_RUSLAN.wav`
- `ruslan_006729_RUSLAN` CER=21.250 `/home/adam/datasets/tts/ruslan/wavs/006729_RUSLAN.wav`
- `ruslan_016473_RUSLAN` CER=20.250 `/home/adam/datasets/tts/ruslan/wavs/016473_RUSLAN.wav`
- `ruslan_020020_RUSLAN` CER=19.800 `/home/adam/datasets/tts/ruslan/wavs/020020_RUSLAN.wav`
- `ruslan_017176_RUSLAN` CER=19.600 `/home/adam/datasets/tts/ruslan/wavs/017176_RUSLAN.wav`
- `ruslan_021302_RUSLAN` CER=19.500 `/home/adam/datasets/tts/ruslan/wavs/021302_RUSLAN.wav`
- `ruslan_013244_RUSLAN` CER=18.200 `/home/adam/datasets/tts/ruslan/wavs/013244_RUSLAN.wav`
- `ruslan_013738_RUSLAN` CER=18.200 `/home/adam/datasets/tts/ruslan/wavs/013738_RUSLAN.wav`
- `ruslan_011790_RUSLAN` CER=18.000 `/home/adam/datasets/tts/ruslan/wavs/011790_RUSLAN.wav`
- `ruslan_018181_RUSLAN` CER=17.800 `/home/adam/datasets/tts/ruslan/wavs/018181_RUSLAN.wav`
- `ruslan_005999_RUSLAN` CER=17.600 `/home/adam/datasets/tts/ruslan/wavs/005999_RUSLAN.wav`
- `ruslan_001805_RUSLAN` CER=17.000 `/home/adam/datasets/tts/ruslan/wavs/001805_RUSLAN.wav`
- `ruslan_010262_RUSLAN` CER=17.000 `/home/adam/datasets/tts/ruslan/wavs/010262_RUSLAN.wav`
- `ruslan_007024_RUSLAN` CER=16.800 `/home/adam/datasets/tts/ruslan/wavs/007024_RUSLAN.wav`
- `ruslan_014176_RUSLAN` CER=16.333 `/home/adam/datasets/tts/ruslan/wavs/014176_RUSLAN.wav`
- `ruslan_020838_RUSLAN` CER=16.333 `/home/adam/datasets/tts/ruslan/wavs/020838_RUSLAN.wav`
- `ruslan_017686_RUSLAN` CER=16.000 `/home/adam/datasets/tts/ruslan/wavs/017686_RUSLAN.wav`
- `ruslan_016993_RUSLAN` CER=15.500 `/home/adam/datasets/tts/ruslan/wavs/016993_RUSLAN.wav`
- `ruslan_006993_RUSLAN` CER=15.200 `/home/adam/datasets/tts/ruslan/wavs/006993_RUSLAN.wav`
- `ruslan_014660_RUSLAN` CER=15.167 `/home/adam/datasets/tts/ruslan/wavs/014660_RUSLAN.wav`
- `ruslan_018268_RUSLAN` CER=15.000 `/home/adam/datasets/tts/ruslan/wavs/018268_RUSLAN.wav`
- `ruslan_009360_RUSLAN` CER=14.667 `/home/adam/datasets/tts/ruslan/wavs/009360_RUSLAN.wav`
- `ruslan_012907_RUSLAN` CER=14.500 `/home/adam/datasets/tts/ruslan/wavs/012907_RUSLAN.wav`
- `ruslan_005811_RUSLAN` CER=14.333 `/home/adam/datasets/tts/ruslan/wavs/005811_RUSLAN.wav`
- `ruslan_012299_RUSLAN` CER=13.500 `/home/adam/datasets/tts/ruslan/wavs/012299_RUSLAN.wav`
- `ruslan_016134_RUSLAN` CER=13.500 `/home/adam/datasets/tts/ruslan/wavs/016134_RUSLAN.wav`
- `ruslan_011774_RUSLAN` CER=13.200 `/home/adam/datasets/tts/ruslan/wavs/011774_RUSLAN.wav`
- `ruslan_013391_RUSLAN` CER=13.143 `/home/adam/datasets/tts/ruslan/wavs/013391_RUSLAN.wav`
- `ruslan_008565_RUSLAN` CER=13.000 `/home/adam/datasets/tts/ruslan/wavs/008565_RUSLAN.wav`
- `ruslan_011347_RUSLAN` CER=12.857 `/home/adam/datasets/tts/ruslan/wavs/011347_RUSLAN.wav`
- `ruslan_009585_RUSLAN` CER=12.667 `/home/adam/datasets/tts/ruslan/wavs/009585_RUSLAN.wav`
- `ruslan_022070_RUSLAN` CER=12.500 `/home/adam/datasets/tts/ruslan/wavs/022070_RUSLAN.wav`
- `ruslan_012675_RUSLAN` CER=12.375 `/home/adam/datasets/tts/ruslan/wavs/012675_RUSLAN.wav`
- `ruslan_018018_RUSLAN` CER=12.250 `/home/adam/datasets/tts/ruslan/wavs/018018_RUSLAN.wav`
- `ruslan_021585_RUSLAN` CER=12.143 `/home/adam/datasets/tts/ruslan/wavs/021585_RUSLAN.wav`
- `ruslan_021336_RUSLAN` CER=12.125 `/home/adam/datasets/tts/ruslan/wavs/021336_RUSLAN.wav`
- `ruslan_016080_RUSLAN` CER=11.857 `/home/adam/datasets/tts/ruslan/wavs/016080_RUSLAN.wav`
- `ruslan_017217_RUSLAN` CER=11.857 `/home/adam/datasets/tts/ruslan/wavs/017217_RUSLAN.wav`
- `ruslan_021513_RUSLAN` CER=11.556 `/home/adam/datasets/tts/ruslan/wavs/021513_RUSLAN.wav`
- `ruslan_021299_RUSLAN` CER=11.400 `/home/adam/datasets/tts/ruslan/wavs/021299_RUSLAN.wav`
## Listen set (Windows)

Copied 80 examples to `out/listen_qc/`.


## Gate B (offline caches)

```json
{
  "latent_index": 21642,
  "text_hidden_index": 21642,
  "align_index": 21642,
  "non_exclude": 13831,
  "non_exclude_lat_ok": 13831,
  "non_exclude_th_ok": 13831,
  "non_exclude_align_ok": 13831,
  "miss_lat": 0,
  "miss_th": 0,
  "miss_align": 0
}
```

- `stats.safetensors`: True
- encode `ac1_mean`: -0.02351510524749756
- text_hidden layers: [11, 17, 23]
- align meta: `{"requested": "C", "mean_defect_rate": null, "n": 21642}`
- **GATE_B**: PASS

