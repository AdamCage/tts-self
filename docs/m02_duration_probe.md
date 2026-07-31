# M02 — Длительности в production-синтезе

**Ответ: production использует `mean_stats`, не `dur_head`.**

Eval suite и `28_infer_mel.py` (default) подают табличные длительности в `model.infer(..., durations=...)`, поэтому `use_pred_duration=False`.

Demo5 синтезирован обоими режимами:
- `C:/projects/tts-self/out/m02_dur_demo/mean_stats/`
- `C:/projects/tts-self/out/m02_dur_demo/pred/`

Pred короче по числу кадров (см. JSON `frame_comparison`).

Рекомендация: слепой A/B в M08/M09 — кандидат на бесплатный слышимый выигрыш.
