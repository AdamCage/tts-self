#!/usr/bin/env python3
"""Minimal FastAPI listen service for discrete Speech LM (no Gradio)."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse

SERVICE_ROOT = Path(__file__).resolve().parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))
REPO_ROOT = SERVICE_ROOT.parent
SRC = REPO_ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv(SERVICE_ROOT / ".env")

from runtime.bundle import ListenBundle, load_bundle, resolve_device  # noqa: E402
from runtime.synthesize import synthesize  # noqa: E402

_BUNDLE: ListenBundle | None = None
_STATUS: dict[str, Any] = {"loaded": False}


def _default_ckpt() -> str:
    env = os.environ.get("CKPT_PATH", "").strip()
    if env:
        return env
    local = SERVICE_ROOT / "artifacts" / "best.pt"
    if local.is_file():
        return str(local)
    return str(
        REPO_ROOT / "runs" / "discrete_speech_lm_d1b_confirm_v2_seed43" / "best.pt"
    )


def _default_tokenizer() -> str:
    return os.environ.get(
        "TOKENIZER_PATH",
        str(Path.home() / "datasets" / "tts" / "models" / "Qwen3.5-0.8B-Base"),
    ).strip()


def _default_device() -> str:
    env = os.environ.get("DEVICE", "").strip().lower()
    if env in {"cuda", "cpu"}:
        resolved, _ = resolve_device(env, warn=False)
        return resolved.type
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def preload() -> None:
    global _BUNDLE, _STATUS
    device = _default_device()
    t0 = time.perf_counter()
    _BUNDLE = load_bundle(
        _default_ckpt(),
        device=device,
        tokenizer_id=_default_tokenizer() or None,
        codec_model_id=os.environ.get("CODEC_MODEL_ID") or None,
    )
    ms = (time.perf_counter() - t0) * 1000.0
    _STATUS = {
        "loaded": True,
        "device": str(_BUNDLE.device),
        "ckpt": str(_BUNDLE.ckpt_path),
        "tokenizer_id": _BUNDLE.tokenizer_id,
        "preload_ms": round(ms, 1),
        "cuda_available": bool(__import__("torch").cuda.is_available()),
    }
    print(f"model loaded on {_BUNDLE.device}", flush=True)
    print(json.dumps(_STATUS, ensure_ascii=False), flush=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    preload()
    yield


app = FastAPI(title="Discrete listen", lifespan=lifespan)


INDEX_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <title>Discrete listen</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
    textarea, select, input, button { width: 100%; margin: 0.4rem 0; font-size: 1rem; }
    textarea { min-height: 5rem; }
    button { padding: 0.6rem; cursor: pointer; }
    #status { color: #444; font-size: 0.9rem; white-space: pre-wrap; }
    #meta { background: #f4f4f4; padding: 0.75rem; overflow: auto; font-size: 0.8rem; }
    audio { width: 100%; margin-top: 1rem; }
    .err { color: #b00020; }
  </style>
</head>
<body>
  <h1>Discrete Speech LM — listen</h1>
  <p id="boot">loading status…</p>
  <label>Текст</label>
  <textarea id="text">Привет, это тест озвучки.</textarea>
  <label>Device</label>
  <select id="device">
    <option value="cuda">cuda</option>
    <option value="cpu">cpu</option>
  </select>
  <label>Speaker id</label>
  <input id="speaker" type="number" value="1088"/>
  <button id="go" type="button">Generate</button>
  <p id="status"></p>
  <audio id="player" controls></audio>
  <pre id="meta"></pre>
  <script>
    async function refreshStatus() {
      const r = await fetch('/health');
      const j = await r.json();
      document.getElementById('boot').textContent =
        'loaded=' + j.loaded + ' device=' + (j.device || '?') +
        ' cuda=' + j.cuda_available;
      if (j.device) document.getElementById('device').value =
        j.device.startsWith('cuda') ? 'cuda' : 'cpu';
    }
    refreshStatus().catch(e => {
      document.getElementById('boot').textContent = 'health failed: ' + e;
    });
    document.getElementById('go').onclick = async () => {
      const status = document.getElementById('status');
      const meta = document.getElementById('meta');
      const player = document.getElementById('player');
      status.textContent = 'running…';
      status.className = '';
      meta.textContent = '';
      player.removeAttribute('src');
      const body = new FormData();
      body.append('text', document.getElementById('text').value);
      body.append('device', document.getElementById('device').value);
      body.append('speaker_id', document.getElementById('speaker').value);
      const t0 = performance.now();
      try {
        const r = await fetch('/synthesize', { method: 'POST', body });
        const j = await r.json();
        const ms = Math.round(performance.now() - t0);
        if (!r.ok || j.error) {
          status.className = 'err';
          status.textContent = 'error (' + ms + ' ms): ' + (j.error || r.status);
          meta.textContent = JSON.stringify(j, null, 2);
          return;
        }
        status.textContent = 'ok in ' + ms + ' ms; audio_s=' + j.meta.audio_s +
          ' rtf=' + j.meta.rtf + ' frames=' + j.meta.n_frames;
        meta.textContent = JSON.stringify(j.meta, null, 2);
        player.src = 'data:audio/wav;base64,' + j.audio_base64;
        player.play().catch(() => {});
      } catch (e) {
        status.className = 'err';
        status.textContent = String(e);
      }
    };
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/health")
def health() -> dict[str, Any]:
    return dict(_STATUS)


def _wav_base64(wav: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    try:
        import soundfile as sf

        sf.write(buf, wav, sr, format="WAV")
    except Exception:
        import wave

        pcm = np.clip(wav, -1.0, 1.0)
        pcm_i16 = (pcm * 32767.0).astype(np.int16)
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm_i16.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/synthesize")
def synthesize_endpoint(
    text: str = Form(...),
    device: str = Form("cuda"),
    speaker_id: int = Form(1088),
) -> JSONResponse:
    global _BUNDLE
    text = (text or "").strip()
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)
    if _BUNDLE is None:
        return JSONResponse({"error": "model not loaded"}, status_code=503)

    requested = (device or "").strip().lower() or _default_device()
    resolved, warn = resolve_device(requested, warn=False)
    if resolved != _BUNDLE.device:
        _BUNDLE = load_bundle(
            _default_ckpt(),
            device=resolved.type,
            tokenizer_id=_default_tokenizer() or None,
            codec_model_id=os.environ.get("CODEC_MODEL_ID") or None,
        )
        _STATUS.update({"device": str(_BUNDLE.device), "loaded": True})

    try:
        wav, sr, meta = synthesize(_BUNDLE, text, speaker_id=int(speaker_id))
        if warn:
            meta = dict(meta)
            meta["device_warning"] = warn
            meta["device_requested"] = requested
        b64 = _wav_base64(wav, sr)
        return JSONResponse({"meta": meta, "audio_base64": b64, "sample_rate": sr})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
