#!/usr/bin/env python3
"""Train N2 (Qwen latent backbone + DDPM head). Plan A4 S1 overfit path.

Example (WSL, RUSLAN 32 utts)::

  export PYTHONPATH=/mnt/c/projects/tts-self/src
  export HF_HOME=$HOME/hf_cache
  export MINISPEECH_ALLOW_DIRTY=1
  python /mnt/c/projects/tts-self/scripts/65_train_n2.py \\
    --config configs/n2_qwen_latent_v01.yaml \\
    --data configs/data_ruslan_local.yaml \\
    --stage S1 --max-utts 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from minispeech.codec.stats import LatentNormalizer
from minispeech.train.n2_trainer import (
    LISTEN_DEMO_RECIPE,
    N2Trainer,
    build_n2_model_from_cfg,
    evaluate_n2_free_running,
    evaluate_n2_teacher_forced_diagnostics,
    make_n2_loaders,
    resolve_data_cfg,
)
from minispeech.utils import auto_batch_size, expand, load_yaml, set_seed


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/n2_qwen_latent_v01.yaml")
    ap.add_argument(
        "--data",
        default=None,
        help="Override data YAML (default: data.config from N2 yaml)",
    )
    ap.add_argument("--stage", default="S1", help="S1/S2/S3 or all")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max-utts", type=int, default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument(
        "--source",
        choices=("ruslan", "multicorpus"),
        default=None,
        help="Override data.source in N2 yaml",
    )
    ap.add_argument(
        "--skip-final-eval",
        action="store_true",
        help="Skip post-train FR/diagnostics JSON write",
    )
    ap.add_argument(
        "--text-usage",
        action="store_true",
        help="After train, run scripts/67_eval_n2_text_usage.py on best.pt",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_yaml(args.config)
    if args.source:
        cfg.setdefault("data", {})["source"] = args.source
    if args.run_dir:
        cfg.setdefault("train", {})["run_dir"] = args.run_dir
    data_cfg = resolve_data_cfg(cfg, args.data)
    set_seed(cfg.get("train", {}).get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    qcfg = cfg.get("qwen", {})
    tok = AutoTokenizer.from_pretrained(
        str(qcfg.get("model_id")),
        trust_remote_code=True,
        local_files_only=bool(qcfg.get("local_files_only", False)),
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = build_n2_model_from_cfg(cfg, device=device, load_pretrained=True)
    try:
        normalizer = LatentNormalizer.from_file(data_cfg["stats_path"])
    except Exception:
        normalizer = None

    stages = list(cfg.get("train", {}).get("stages", []))
    if args.stage and args.stage != "all":
        stages = [s for s in stages if s["name"] == args.stage]
    if not stages:
        stages = [{"name": "S1", "max_utterances": 32, "steps": 3000}]
    if args.max_utts is not None:
        for s in stages:
            s["max_utterances"] = args.max_utts

    bcfg = cfg.get("train", {}).get("batch", {})
    if args.batch_size:
        bs = args.batch_size
    elif bcfg.get("mode") == "auto" and device.type == "cuda":

        def probe(b: int) -> None:
            tl, _, _ = make_n2_loaders(
                cfg,
                data_cfg,
                stages[0],
                b,
                tokenizer=tok,
                bos_latent=model.backbone.bos_latent.detach().cpu(),
                normalizer=normalizer,
            )
            batch = next(iter(tl))
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            model.train()
            out = model(
                batch["input_ids"],
                batch["z_input"],
                batch["z_target"],
                text_mask=batch["text_mask"],
                frame_mask=batch["frame_mask"],
            )
            out.loss.backward()
            model.zero_grad(set_to_none=True)

        bs = auto_batch_size(
            probe,
            list(bcfg.get("candidates", [1, 2, 4, 8])),
            float(bcfg.get("target_vram_gb", 14.5)),
        )
        print(f"auto batch_size={bs}")
    else:
        bs = int(args.batch_size or 2)

    metrics_out: dict = {"stages": {}}
    for stage in stages:
        # A4 default: only S1. Explicit --stage S2/S3 allowed later after overfit.
        if args.stage == "all" and stage["name"] != "S1":
            print(
                f"Skipping {stage['name']}: refuse S2/S3 until S1 overfits "
                "(re-run with --stage S2 after gate)"
            )
            continue
        train_loader, val_loader, train_ds = make_n2_loaders(
            cfg,
            data_cfg,
            stage,
            bs,
            tokenizer=tok,
            bos_latent=model.backbone.bos_latent.detach().cpu(),
            normalizer=normalizer,
        )
        print(f"stage={stage['name']} n_train={len(train_ds)} batch={bs}")
        # Per-stage run dir for S1
        tcfg = cfg.setdefault("train", {})
        if stage["name"] == "S1" and "n2_s1" not in str(tcfg.get("run_dir", "")):
            tcfg["run_dir"] = "runs/n2_s1_ruslan"
        trainer = N2Trainer(
            model,
            train_loader,
            val_loader or train_loader,
            cfg,
            device,
            train_dataset=train_ds,
        )
        steps = int(args.steps or stage.get("steps", 1000))
        last = trainer.train_steps(steps, stage=stage["name"])
        print(stage["name"], last)

        if not args.skip_final_eval:
            try:
                fr = evaluate_n2_free_running(
                    trainer.ema.shadow,
                    val_loader or train_loader,
                    device=device,
                    max_items=min(16, len(train_ds)),
                    sample_steps=4,
                    max_frames_cap=64,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[final-fr] failed: {exc}")
                fr = {"error": str(exc), "n": 0, "stability": {}, "head_diagnostics": {}}
            tf = evaluate_n2_teacher_forced_diagnostics(
                trainer.ema.shadow,
                val_loader or train_loader,
                device=device,
                max_items=min(32, len(train_ds)),
            )
            report = {
                "stage": stage["name"],
                "steps": steps,
                "batch_size": bs,
                "n_train": len(train_ds),
                "run_dir": str(trainer.run_dir),
                "final_train": last,
                "free_running": fr,
                "teacher_forced": tf,
                "copy_synth_cer_topline": 0.073,
                "listen_demo_recipe": LISTEN_DEMO_RECIPE.strip(),
                "overfit_heuristic": {
                    "loss_curve": last.get("loss_curve_points"),
                    "tf_loss": tf.get("tf_loss_mean"),
                    "fr_r2": fr.get("head_diagnostics", {}).get("r2_overall"),
                    "fr_var_ratio": fr.get("head_diagnostics", {}).get("var_ratio"),
                    "fr_mean_dtw_l1": fr.get("mean_dtw_l1"),
                },
            }
            out_name = f"n2_{stage['name'].lower()}_ruslan.json"
            out_path = expand(f"out/{out_name}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            (trainer.run_dir / "final_metrics.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            print(f"Wrote {out_path}")
            metrics_out["stages"][stage["name"]] = report

    if args.text_usage:
        best = expand(cfg["train"]["run_dir"]) / "best.pt"
        if best.is_file():
            from subprocess import call

            rc = call(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "67_eval_n2_text_usage.py"),
                    "--ckpt",
                    str(best),
                    "--config",
                    args.config,
                    "--data",
                    args.data or cfg.get("data", {}).get("config", "configs/data_ruslan_local.yaml"),
                ]
            )
            if rc != 0:
                return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
