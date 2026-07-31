# M03 — Диагноз провала variance (CER 0.430)

Обучение не запускалось.

- index_n: **12490**
- train_hit_rate: **0.6413** (12490/19477)
- val_hit_rate: **0.0000**
- sim batch=8 missing rate: **0.9670**
- sample pitch/energy stats: {"min": 0.0, "max": 125.67105102539062, "mean": 42.93393325805664, "std": 36.617637634277344, "col0_mean": 39.0228271484375, "col1_mean": 46.84504318237305, "col0_std": 51.32132339477539, "col1_std": 7.323028087615967}

## Первичный механизм

- **partial_f0_cache_plus_collate** [high]: train_hit_rate=0.6413, sim_batch8_missing_rate=0.9670. Mixed batches drop pitch/energy loss entirely; infer still predicts.
- **pitch_not_zscored** [medium]: pitch col0 mean=39.02 looks like raw Hz/log not unit-Gaussian; Linear(1,d) adaptor may be poorly scaled vs trained mel path

