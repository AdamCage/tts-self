#!/usr/bin/env python3
"""Train Track N acoustic σ-VAE tokenizer (stage T0) + copy-synth CER.

T0 criterion: reconstruction loss plateaus on 1 h RUSLAN; copy-synth on the
same hour is intelligible (overfit OK). Held-out quality is out of scope.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from minispeech.constants import get_target
from minispeech.data.manifest import Utterance, filter_split, read_manifest
from minispeech.data.mel import load_wav_mono
from minispeech.eval.asr import AsrEngine, load_asr_engine
from minispeech.eval.asr_wer import compute_cer, compute_wer
from minispeech.frontend.normalize import normalize_for_asr
from minispeech.models.tokenizer import AcousticTokenizer, TokenizerLossBundle
from minispeech.utils import expand, load_yaml, set_seed, write_run_manifest


def _select_hours(rows: list[Utterance], max_hours: float) -> list[Utterance]:
    budget = float(max_hours) * 3600.0
    acc, kept = 0.0, []
    for r in rows:
        d = float(r.dur_s or 0.0)
        if d <= 0:
            continue
        if acc + d > budget:
            break
        kept.append(r)
        acc += d
    return kept


class WaveformDataset(Dataset):
    """Unlabeled waveform rows for acoustic tokenizer training."""

    def __init__(
        self,
        rows: list[Utterance],
        *,
        sample_rate: int,
        hop: int,
        max_samples: int,
        min_samples: int,
    ):
        self.rows = rows
        self.sample_rate = int(sample_rate)
        self.hop = int(hop)
        self.max_samples = int(max_samples)
        self.min_samples = int(min_samples)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        u = self.rows[idx]
        wav_np, sr = load_wav_mono(u.wav, target_sr=self.sample_rate)
        wav = torch.from_numpy(np.asarray(wav_np, dtype=np.float32))
        if wav.numel() > self.max_samples:
            # Random crop in train; deterministic start-0 for eval via set_seed.
            start = int(random.randint(0, wav.numel() - self.max_samples))
            wav = wav[start : start + self.max_samples]
        if wav.numel() < self.min_samples:
            wav = F.pad(wav, (0, self.min_samples - wav.numel()))
        # Pad to hop multiple so latent frames align.
        rem = wav.numel() % self.hop
        if rem:
            wav = F.pad(wav, (0, self.hop - rem))
        return {
            "utt_id": u.utt_id,
            "wav": wav,
            "text": u.text_stressed or u.text_raw or "",
            "n_samples": int(wav.numel()),
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [int(b["n_samples"]) for b in batch]
    t_max = max(lengths)
    wavs = torch.zeros(len(batch), 1, t_max, dtype=torch.float32)
    for i, b in enumerate(batch):
        n = lengths[i]
        wavs[i, 0, :n] = b["wav"][:n]
    return {
        "utt_id": [b["utt_id"] for b in batch],
        "wav": wavs,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "text": [b["text"] for b in batch],
    }


def _lr_at(step: int, *, lr: float, warmup: int, total: int, min_ratio: float) -> float:
    if step < warmup:
        return lr * float(step + 1) / float(max(1, warmup))
    progress = (step - warmup) / float(max(1, total - warmup))
    progress = min(1.0, max(0.0, progress))
    return lr * (min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress)))


def _crop_to_valid(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop batch to min valid length so STFT never sees zero-pad edges."""
    n = int(lengths.min().item())
    n = max(n, 1)
    return pred[..., :n], target[..., :n]


def train(cfg: dict[str, Any], *, data_cfg: dict[str, Any]) -> Path:
    tcfg = cfg["train"]
    dcfg = cfg.get("data", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(tcfg.get("seed", 42)))

    target = get_target(cfg.get("target", "sigma_vae64_t0"))
    hop = int(target.samples_per_frame)
    sr = int(target.sample_rate)

    rows = filter_split(read_manifest(expand(data_cfg["manifest"])), str(dcfg.get("split", "train")))
    excl = set(dcfg.get("exclude_tags", ["exclude"]))
    rows = [r for r in rows if (r.qc_tag or "clean") not in excl]
    rows = _select_hours(rows, float(dcfg.get("max_hours", 1.0)))
    if not rows:
        raise SystemExit("no training rows after hour/tag filter")

    hours = sum(float(r.dur_s or 0.0) for r in rows) / 3600.0
    print(f"[data] utts={len(rows)} hours={hours:.3f} sr={sr} hop={hop}")

    ds = WaveformDataset(
        rows,
        sample_rate=sr,
        hop=hop,
        max_samples=int(dcfg.get("max_samples", 240_000)),
        min_samples=int(dcfg.get("min_samples", 16_000)),
    )
    loader = DataLoader(
        ds,
        batch_size=int(tcfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(tcfg.get("num_workers", 2)),
        collate_fn=_collate,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )

    model = AcousticTokenizer.from_config_dict(cfg.get("model", {})).to(device)
    nparams = model.count_parameters()
    print(f"[model] params={nparams}")
    init_ckpt = cfg.get("train", {}).get("init_ckpt")
    if init_ckpt:
        payload = torch.load(expand(init_ckpt), map_location=device, weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        print(f"[model] warm-start from {init_ckpt} step={payload.get('step')}")

    lcfg = cfg.get("loss", {})
    loss_fn = TokenizerLossBundle(
        wave_l1_weight=float(lcfg.get("wave_l1_weight", 0.1)),
        stft_sc_weight=float(lcfg.get("stft_sc_weight", 1.0)),
        stft_mag_weight=float(lcfg.get("stft_mag_weight", 1.0)),
        mel_l1_weight=float(lcfg.get("mel_l1_weight", 0.0)),
        kl_weight=float(lcfg.get("kl_weight", 1e-4)),
        latent_l2_weight=float(lcfg.get("latent_l2_weight", 0.0)),
        adv_weight=float(lcfg.get("adv_weight", 0.0)),
        feat_weight=float(lcfg.get("feat_weight", 0.0)),
        discriminator=None,  # T0 recon-only
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg.get("lr", 2e-4)),
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
        betas=(0.9, 0.99),
    )

    run_dir = expand(tcfg.get("run_dir", "runs/tokenizer_t0"))
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        run_dir,
        cfg,
        extra={
            "stage": "T0",
            "target": target.name,
            "n_utts": len(rows),
            "hours": hours,
            "params": nparams,
            "loss_mode": "reconstruction_only",
            "discriminator": None,
            "note": (
                "T0 acoustic σ-VAE; adv_weight=0; DiscriminatorProtocol reserved for T1+. "
                "P0 frame rate 7.5 Hz."
            ),
        },
    )

    steps = int(tcfg.get("steps", 8000))
    warmup = int(tcfg.get("warmup_steps", 200))
    base_lr = float(tcfg.get("lr", 2e-4))
    min_ratio = float(tcfg.get("min_lr_ratio", 0.05))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    log_every = int(tcfg.get("log_every", 50))
    save_every = int(tcfg.get("save_every", 500))
    use_bf16 = device.type == "cuda" and str(tcfg.get("precision", "bf16")).lower() in (
        "bf16",
        "bfloat16",
    )

    history: list[dict[str, float]] = []
    last_best_save = -10**9
    model.train()
    data_iter = iter(loader)
    t0 = time.time()
    best_loss = float("inf")

    for step in range(1, steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        lr = _lr_at(step, lr=base_lr, warmup=warmup, total=steps, min_ratio=min_ratio)
        for pg in opt.param_groups:
            pg["lr"] = lr

        wav = batch["wav"].to(device, non_blocking=True)
        lengths = batch["lengths"]

        opt.zero_grad(set_to_none=True)
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16
            else nullcontext()
        )
        with ctx:
            out = model(wav, sample=False)
            pred, tgt = _crop_to_valid(out.recon, wav, lengths)
            losses = loss_fn.generator_loss(pred, tgt, mean=out.mean, vae_std=model.vae_std)
            loss = losses["loss"]

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at step {step}: { {k: float(v) for k, v in losses.items()} }"
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        loss_f = float(loss.detach().cpu())
        if step % log_every == 0 or step == 1:
            row = {
                "step": float(step),
                "loss": loss_f,
                "wave_l1": float(losses["wave_l1"].cpu()),
                "stft_sc": float(losses["stft_sc"].cpu()),
                "stft_mag": float(losses["stft_mag"].cpu()),
                "kl": float(losses["kl"].cpu()),
                "latent_l2": float(losses.get("latent_l2", losses["kl"]*0).cpu()) if "latent_l2" in losses else 0.0,
                "lr": lr,
            }
            history.append(row)
            elapsed = time.time() - t0
            print(
                f"[step {step}/{steps}] loss={loss_f:.4f} "
                f"wave={row['wave_l1']:.4f} sc={row['stft_sc']:.4f} "
                f"mag={row['stft_mag']:.4f} kl={row['kl']:.5f} "
                f"lr={lr:.2e} ({elapsed:.0f}s)"
            )

        if loss_f < best_loss:
            best_loss = loss_f
            # Throttle disk writes: frequent best saves were hanging the run.
            if step == 1 or (step - last_best_save) >= max(50, save_every // 2):
                tmp = run_dir / "best.tmp.pt"
                final = run_dir / "best.pt"
                print(f"[ckpt] writing best step={step} loss={best_loss:.4f}", flush=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg,
                        "step": step,
                        "loss": best_loss,
                        "params": nparams,
                    },
                    tmp,
                )
                tmp.replace(final)
                last_best_save = step
                print("[ckpt] best saved", flush=True)

        if save_every > 0 and step % save_every == 0:
            # Skip optimizer in mid-ckpts (was ~2GB and a hang risk on slow FS).
            print(f"[ckpt] writing step_{step:06d}", flush=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "step": step,
                    "history": history,
                    "params": nparams,
                },
                run_dir / f"step_{step:06d}.pt",
            )
            print("[ckpt] step saved", flush=True)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "cfg": cfg,
            "step": steps,
            "history": history,
            "params": nparams,
        },
        run_dir / "last.pt",
    )
    (run_dir / "loss_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    # Convergence summary for acceptance.
    if len(history) >= 4:
        early = sum(h["loss"] for h in history[: max(1, len(history) // 5)]) / max(
            1, len(history) // 5
        )
        late = sum(h["loss"] for h in history[-max(1, len(history) // 5) :]) / max(
            1, len(history) // 5
        )
        plateau = abs(history[-1]["loss"] - history[-2]["loss"]) if len(history) >= 2 else None
        summary = {
            "early_mean_loss": early,
            "late_mean_loss": late,
            "final_loss": history[-1]["loss"],
            "best_loss": best_loss,
            "monotone_drop": late < early,
            "n_log_points": len(history),
            "last_delta": plateau,
            "hours": hours,
            "n_utts": len(rows),
            "params": nparams,
        }
    else:
        summary = {"error": "insufficient history", "n_log_points": len(history)}
    (run_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[train] done summary={summary}")
    # Free train-time GPU memory before copy-synth ASR.
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return run_dir


@torch.no_grad()
def copy_synth_eval(
    cfg: dict[str, Any],
    *,
    data_cfg: dict[str, Any],
    run_dir: Path,
    ckpt_path: Path | None = None,
) -> dict[str, Any]:
    """Copy-synthesis CER on the same 1 h train subset (overfit OK on T0)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ecfg = cfg.get("eval", {})
    dcfg = cfg.get("data", {})
    target = get_target(cfg.get("target", "sigma_vae64_t0"))
    hop = int(target.samples_per_frame)
    sr = int(target.sample_rate)

    ckpt_path = ckpt_path or (run_dir / "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = AcousticTokenizer.from_config_dict(ckpt.get("cfg", cfg).get("model", cfg.get("model", {})))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    rows = filter_split(read_manifest(expand(data_cfg["manifest"])), str(dcfg.get("split", "train")))
    excl = set(dcfg.get("exclude_tags", ["exclude"]))
    rows = [r for r in rows if (r.qc_tag or "clean") not in excl]
    rows = _select_hours(rows, float(dcfg.get("max_hours", 1.0)))
    n = int(ecfg.get("copy_synth_n", 32))
    rows = rows[:n]

    asr_path = ecfg.get("asr_config", "configs/asr.yaml")
    asr: AsrEngine = load_asr_engine(asr_path, device=device)

    wav_dir = run_dir / "copy_synth_wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    refs: list[str] = []
    hyps: list[str] = []
    per_utt: list[dict[str, Any]] = []

    import soundfile as sf

    for u in rows:
        wav_np, _ = load_wav_mono(u.wav, target_sr=sr)
        wav = torch.from_numpy(np.asarray(wav_np, dtype=np.float32))
        rem = wav.numel() % hop
        if rem:
            wav = F.pad(wav, (0, hop - rem))
        x = wav.unsqueeze(0).unsqueeze(0).to(device)
        sample = bool(ecfg.get("sample", False))
        out = model(x, sample=sample)
        recon = out.recon.squeeze().float().cpu().numpy()
        # Trim to original (pre-pad) length.
        recon = recon[: len(wav_np)]
        out_path = wav_dir / f"{u.utt_id}.wav"
        sf.write(str(out_path), recon, sr)

        ref = normalize_for_asr(u.text_stressed or u.text_raw or "")
        hyp = asr.transcribe_normalized(recon, sr)
        refs.append(ref)
        hyps.append(hyp)
        cer_i = compute_cer([ref], [hyp])
        per_utt.append(
            {
                "utt_id": u.utt_id,
                "ref": ref,
                "hyp": hyp,
                "cer": cer_i,
                "wav": str(out_path),
            }
        )
        print(f"[copy-synth] {u.utt_id} cer={cer_i:.4f}")

    result = {
        "n": len(rows),
        "cer": compute_cer(refs, hyps),
        "wer": compute_wer(refs, hyps),
        "ckpt": str(ckpt_path),
        "sample": bool(ecfg.get("sample", False)),
        "target": target.name,
        "subset": "train_1h",
        "note": "T0 overfit copy-synth on training hour; held-out not required.",
        "per_utt": per_utt,
    }
    out_json = run_dir / "copy_synth_train1h.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[copy-synth] CER={result['cer']:.4f} WER={result['wer']:.4f} → {out_json}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/tokenizer_t0.yaml")
    ap.add_argument("--data", default=None, help="Override data yaml (default from config)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--init-ckpt", default=None, help="Warm-start model weights")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_path = args.data or cfg.get("data", {}).get("config", "configs/data_ruslan_local.yaml")
    data_cfg = load_yaml(data_path)["data"]
    if args.steps is not None:
        cfg.setdefault("train", {})["steps"] = args.steps
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = args.batch_size
    if args.run_dir:
        cfg.setdefault("train", {})["run_dir"] = args.run_dir
    if getattr(args, "init_ckpt", None):
        cfg.setdefault("train", {})["init_ckpt"] = args.init_ckpt

    run_dir = expand(cfg["train"]["run_dir"])
    if not args.skip_train:
        run_dir = train(cfg, data_cfg=data_cfg)
    if not args.skip_eval:
        ckpt = Path(args.ckpt) if args.ckpt else None
        copy_synth_eval(cfg, data_cfg=data_cfg, run_dir=run_dir, ckpt_path=ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
