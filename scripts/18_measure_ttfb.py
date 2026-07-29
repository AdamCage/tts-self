#!/usr/bin/env python3
"""Workstream E: cold/warm streaming TTFB with Qwen/Talker/sampler/codec breakdown."""

from __future__ import annotations

import argparse
import json
import time

import torch
from transformers import AutoTokenizer

from minispeech.codec.stats import LatentNormalizer
from minispeech.codec.wrapper import VibeVoiceCodec
from minispeech.constants import QWEN_MODEL_ID, SR
from minispeech.infer.generate import load_checkpoint, synthesize_streaming
from minispeech.infer.stream import measure_ttfb
from minispeech.utils import expand, load_yaml


DEFAULT_TEXTS = [
    "Привет, это тест задержки первого аудио.",
    "Сегодня хорошая погода, можно пройтись пешком до парка.",
    "Пожалуйста, повторите ещё раз чуть медленнее и громче.",
]


def _resolve_qwen_id(infer_cfg: dict, qwen_config: str | None) -> str:
    if qwen_config:
        qcfg = load_yaml(qwen_config).get("qwen", {})
        mid = qcfg.get("model_id")
        if mid:
            return str(expand(mid))
    return str(infer_cfg.get("qwen", QWEN_MODEL_ID))


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure streaming TTFB (cold + warm)")
    ap.add_argument("--ckpt", default="runs/v01_regression_direct/best.pt")
    ap.add_argument("--config", default="configs/infer.yaml")
    ap.add_argument("--qwen-config", default="configs/qwen_local.yaml")
    ap.add_argument("--out", default="runs/v01_regression_direct/ttfb_phase4.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--warm-runs", type=int, default=3)
    ap.add_argument("--long-text", default="", help="Optional long-form stability probe")
    args = ap.parse_args()

    cfg = load_yaml(args.config)["infer"]
    stream_cfg = cfg.get("stream", {})
    chunk = int(stream_cfg.get("chunk_latents", 4))
    min_pcm_ms = float(stream_cfg.get("min_pcm_ms", 40.0))
    stop = cfg.get("stop", {})
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"device={device} ckpt={args.ckpt} chunk={chunk} min_pcm_ms={min_pcm_ms}")
    model, ckpt = load_checkpoint(args.ckpt, device)
    codec = VibeVoiceCodec(model_id=cfg.get("codec", "microsoft/VibeVoice-AcousticTokenizer"))
    normalizer = LatentNormalizer.from_file(cfg["stats"])
    qwen_id = _resolve_qwen_id(cfg, args.qwen_config)
    print(f"qwen={qwen_id}")
    tok = AutoTokenizer.from_pretrained(qwen_id, trust_remote_code=True)
    try:
        from transformers import Qwen3_5TextModel

        text_model = Qwen3_5TextModel.from_pretrained(
            qwen_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    except Exception:
        from transformers import AutoModel

        text_model = AutoModel.from_pretrained(
            qwen_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    text_model.eval()

    def run_one(text: str):
        return synthesize_streaming(
            text,
            model,
            codec,
            normalizer,
            tok,
            text_model,
            cfg_scale=float(cfg.get("cfg_scale", 1.3)),
            flow_steps=int(cfg.get("flow_steps", 4)),
            use_stress=bool(cfg.get("stress", {}).get("use_accentuator", True)),
            stop=stop,
            chunk_latents=chunk,
            min_pcm_ms=min_pcm_ms,
        )

    results: dict = {
        "ckpt": str(args.ckpt),
        "git_sha": (ckpt.get("git_sha") if isinstance(ckpt, dict) else None),
        "chunk_latents": chunk,
        "min_pcm_ms": min_pcm_ms,
        "stop": stop,
        "cold": [],
        "warm": [],
        "long_form": None,
    }

    # Cold: first utterance after load (models already loaded — true cold would include load;
    # we report post-load cold as first synthesize, which matches Phase-4 protocol).
    print("--- cold ---")
    for i, text in enumerate(DEFAULT_TEXTS[:1]):
        if device.type == "cuda":
            torch.cuda.synchronize()
        bd = measure_ttfb(lambda t=text: run_one(t), min_pcm_ms=min_pcm_ms)
        row = bd.to_dict()
        row["text"] = text
        results["cold"].append(row)
        print(
            f"cold[{i}] ttfb={bd.ttfb_ms:.1f}ms estimated={bd.estimated} "
            f"qwen={bd.qwen_ms:.1f} talker={bd.talker_ms:.1f} sampler={bd.sampler_ms:.1f} "
            f"codec={bd.codec_first_ms:.1f} rtf={bd.rtf:.3f} audio_s={bd.audio_s:.2f}"
        )

    print("--- warm ---")
    for i, text in enumerate(DEFAULT_TEXTS):
        for k in range(args.warm_runs if i == 0 else 1):
            if device.type == "cuda":
                torch.cuda.synchronize()
            bd = measure_ttfb(lambda t=text: run_one(t), min_pcm_ms=min_pcm_ms)
            row = bd.to_dict()
            row["text"] = text
            row["rep"] = k
            results["warm"].append(row)
            print(
                f"warm[{i}.{k}] ttfb={bd.ttfb_ms:.1f}ms estimated={bd.estimated} "
                f"qwen={bd.qwen_ms:.1f} talker={bd.talker_ms:.1f} sampler={bd.sampler_ms:.1f} "
                f"codec={bd.codec_first_ms:.1f} rtf={bd.rtf:.3f} audio_s={bd.audio_s:.2f}"
            )

    long_text = args.long_text.strip() or (
        "Это проверка длинной фразы для устойчивости потокового декодера. "
        "Мы говорим несколько предложений подряд, чтобы убедиться, что кэш кодека "
        "не ломает швы и что отношение реального времени остаётся приемлемым "
        "на протяжении всей генерации без обрыва или тишины."
    )
    print("--- long-form ---")
    t0 = time.perf_counter()
    audio, info = run_one(long_text)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0
    audio_s = len(audio) / float(SR)
    peak = float(abs(audio).max()) if len(audio) else 0.0
    results["long_form"] = {
        "text": long_text,
        "audio_s": audio_s,
        "total_ms": total_ms,
        "rtf": total_ms / 1000.0 / max(audio_s, 1e-6),
        "n_frames": info.get("n_frames"),
        "peak_abs": peak,
        "silence_like": peak < 1e-4,
        "ttfb_ms": info.get("first_pcm_ms"),
        "estimated": info.get("estimated", False),
        "qwen_ms": info.get("qwen_ms"),
        "talker_ms": info.get("talker_ms"),
        "sampler_ms": info.get("sampler_ms"),
        "codec_first_ms": info.get("codec_first_ms"),
        "stop_triggered": info.get("stop_triggered"),
    }
    print(
        f"long audio_s={audio_s:.2f} rtf={results['long_form']['rtf']:.3f} "
        f"peak={peak:.4f} ttfb={results['long_form']['ttfb_ms']}"
    )

    # Warm aggregate
    warm_ttfb = [r["ttfb_ms"] for r in results["warm"]]
    warm_rtf = [r["rtf"] for r in results["warm"]]
    results["summary"] = {
        "cold_ttfb_ms": results["cold"][0]["ttfb_ms"] if results["cold"] else None,
        "warm_ttfb_ms_mean": sum(warm_ttfb) / len(warm_ttfb) if warm_ttfb else None,
        "warm_ttfb_ms_min": min(warm_ttfb) if warm_ttfb else None,
        "warm_rtf_mean": sum(warm_rtf) / len(warm_rtf) if warm_rtf else None,
        "any_estimated": any(r["estimated"] for r in results["cold"] + results["warm"]),
        "mode": "regression" if float(results["warm"][0]["sampler_ms"]) == 0.0 else "flow",
    }

    out = expand(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print("summary:", json.dumps(results["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
