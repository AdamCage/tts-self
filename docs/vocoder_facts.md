# Vocoder environment facts (Track A)

Collected for roadmap task **T01**. Read-only probe of `~/models/hifigan` and mel extraction backends.

## Local HiFi-GAN checkpoint

| Field | Value |
|-------|-------|
| Path | `~/models/hifigan/generator.pt` (+ `generator.ckpt`, `hyperparams.yaml`, `source.json`) |
| Source repo | `speechbrain/tts-hifigan-libritts-22050Hz` ([HF card](https://huggingface.co/speechbrain/tts-hifigan-libritts-22050Hz)) |
| Downloader | `scripts/33_download_hifigan.py` |

### `hyperparams.yaml` (generator architecture only)

Local file lists generator topology (`in_channels: 80`, upsample factors `[8,8,2,2]`, etc.) and **does not** store mel STFT hyperparameters. Mel geometry must come from the model card / SpeechBrain recipe.

### Published mel geometry (authoritative for `mel_fmax`)

From the Hugging Face README “Spectrogram to Waveform Conversion” block for this exact repo — parameters required for “optimal results”:

| Parameter | Value | Source |
|-----------|------:|--------|
| `sample_rate` | **22050** | HF README |
| `n_fft` | **1024** | HF README |
| `hop_length` | **256** | HF README |
| `win_length` | **1024** | HF README |
| `n_mels` | **80** | HF README |
| `f_min` / mel_fmin | **0.0** | HF README |
| `f_max` / **mel_fmax** | **8000.0** | HF README |
| `power` | **1** | HF README |
| `norm` | **`"slaney"`** | HF README |
| `mel_scale` | **`"slaney"`** | HF README |
| `normalized` | `False` | HF README |
| `min_max_energy_norm` | `True` | HF README (SpeechBrain energy norm; **not** used in MiniSpeech `mel.py`) |
| `compression` | `True` | HF README (SpeechBrain dynamic-range log; MiniSpeech uses `log(max(mel, 1e-5))`) |

**Answer:** vocoder `mel_fmax = 8000.0` (Hz), documented on the SpeechBrain model card for `tts-hifigan-libritts-22050Hz`. Local `source.json` note also records `fmax train=8000`.

## Project mel80 (current)

From `src/minispeech/data/mel.py` at freeze:

| Backend | fmin | fmax | norm | mel_scale |
|---------|-----:|-----:|------|-----------|
| librosa | 0.0 | `sr/2` = **11025** | default **slaney** | default **slaney** (`htk=False`) |
| torchaudio | 0.0 | `sr/2` = **11025** | **None** (implicit) | **`"htk"`** (default) |

So:

1. Project `fmax=11025` ≠ vocoder train `fmax=8000` — confirmed domain mismatch hypothesis for Gate V0.
2. librosa vs torchaudio branches are **not** filter-identical today (slaney+slaney vs htk+None).

## WSL runtime probe (2026-07-30)

```bash
cd ~/projects/tts-self && source .venv/bin/activate
python scripts/_probe_vocoder_facts.py
```

Measured:

| Field | Value |
|-------|-------|
| `mel_backend_name()` | **librosa** |
| librosa | 0.11.0 |
| torchaudio | 2.11.0+cu130 |
| `$HOME` free | **895.8 GB** |

## Verdict for T03 / T06

- Use **fmax=8000**, **norm=slaney**, **mel_scale=slaney** for the HiFi-GAN-matched target.
- Keep legacy `MEL80` behavior unchanged (`fmax=11025`, slaney/slaney) for paper2 freeze reproducibility.
- Residual mismatches (`min_max_energy_norm`, SpeechBrain `compression`) are out of scope for V0; note them if copy-synth UTMOS still lags real audio after domain align.
