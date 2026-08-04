#!/usr/bin/env python3
"""Round-3 live probe: production aggregation → emitter (FAIL #4)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util

from minispeech.data.collate_codec_seq import collate_codec_seq
from minispeech.models.discrete_speech_lm import DiscreteSpeechLM, DiscreteSpeechLMConfig


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    mod = _load_script("74_eval_discrete_lm.py")
    cfg = DiscreteSpeechLMConfig.tiny()
    cfg.termination_mode = "semantic_eos_v1"
    cfg.eos_id = cfg.codebook_size
    cfg.codec_vocab_size = cfg.codebook_size
    cfg.semantic_vocab_size = cfg.codebook_size + 1
    m = DiscreteSpeechLM(cfg)
    batch = collate_codec_seq(
        [
            {
                "utt_id": "a",
                "input_ids": torch.tensor([3, 4]),
                "codes": torch.randint(0, m.codebook_size, (2, m.n_codebooks)),
                "speaker_id": 0,
            },
            {
                "utt_id": "b",
                "input_ids": torch.tensor([5, 6, 7]),
                "codes": torch.randint(0, m.codebook_size, (4, m.n_codebooks)),
                "speaker_id": 1,
            },
        ],
        pad_token_id=0,
        speech_eos_id=m.eos_id,
        termination_mode="semantic_eos_v1",
    )
    measured = [
        mod.teacher_forced_ce(m, batch, torch.device("cpu")),
        mod.teacher_forced_ce(m, batch, torch.device("cpu")),
    ]
    print("H1 per-batch keys", sorted(measured[0].keys()))
    # Legacy discard path
    totals = {"loss": 0.0, "ce_cb0": 0.0, "ce_residual": 0.0}
    for mm in measured:
        for k in totals:
            totals[k] += mm[k]
    legacy = {k: v / 2 for k, v in totals.items()}
    bare = legacy.get("loss")
    print("H2 legacy bare float", bare, type(bare).__name__)

    summary = mod.aggregate_teacher_forced_batches(measured)
    tf_ce = mod.teacher_forced_metrics_for_emitter(summary)
    print("H3 aggregated", {k: tf_ce.get(k) for k in (
        "teacher_forced_ce_source",
        "tf_eos_final_top1_accuracy",
        "tf_eos_early_top1_false_positive_count",
        "ce_semantic_eos",
        "total_ce",
    )})
    assert tf_ce["teacher_forced_ce_source"] == "eval_computed"
    assert "tf_eos_final_top1_accuracy" in tf_ce

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        wav = td_path / "u0.wav"
        sf.write(str(wav), np.linspace(-0.2, 0.2, 2400, dtype=np.float32), 24000)
        ckpt = td_path / "ckpt.pt"
        torch.save({"model": {}}, ckpt)

        def row(cer=0.1):
            return {
                "utt_id": "u0",
                "cer": cer,
                "ref": "привет",
                "hyp": "привет",
                "n_frames": 10,
                "gt_frames": 10,
                "wav": str(wav),
                "stop_reason": "learned_eos",
                "generation": {
                    "termination_reason": "learned_eos",
                    "n_decoded_frames": 10,
                    "early_eos_suppressed_count": 0,
                    "max_cb0_run": 2,
                },
                "generated_codes_sha256": "d" * 64,
            }

        contract = {
            "termination": {"mode": "semantic_eos_v1"},
            "sha256": "e" * 64,
            "recipe_sha256": "f" * 64,
            "asr": {"file_sha256": "1" * 64},
            "subsets": {
                "select8": {
                    "path": "configs/train_subsets/d1c_dev32.json",
                    "file_sha256": "3" * 64,
                }
            },
        }
        reports = {
            name: {"rows": [row(0.05 if name == "copy_synth" else 0.1)]}
            for name in (
                "model",
                "copy_synth",
                "wrong_utt",
                "random_codes",
                "shuffle_text",
                "mask_text",
            )
        }
        gate = {
            "verdict": "PASS",
            "criteria": {"macro_cer": 0.1, "n_catastrophic": 0},
            "text_usage_acceptance": {"pass": True},
            "stability_acceptance": {
                "phrase_loop_repetition": False,
                "runaway_or_repetition": False,
                "no_runaway": True,
            },
        }
        prov = {
            "subset_label": "select8",
            "subset_path": "x",
            "subset_file_sha256": "3" * 64,
        }
        ok = mod._emit_d1c_v2_from_gate(
            out_dir=td_path / "ok",
            contract=contract,
            ckpt=str(ckpt),
            provenance=dict(prov),
            infer={"max_frames": 750},
            teacher_forced_ce=tf_ce,
            ordered_utt_ids=["u0"],
            condition_reports=reports,
            gate=gate,
        )
        print("H4 MAIN measured path", "OK" if ok else "FAIL", ok)
        bare_out = mod._emit_d1c_v2_from_gate(
            out_dir=td_path / "bare",
            contract=contract,
            ckpt=str(ckpt),
            provenance=dict(prov),
            infer={"max_frames": 750},
            teacher_forced_ce=bare,
            ordered_utt_ids=["u0"],
            condition_reports=reports,
            gate=gate,
        )
        print("H5 bare-float soft-refuse", bare_out is None)
        score_skip = mod._emit_d1c_v2_from_gate(
            out_dir=td_path / "score_skip",
            contract=contract,
            ckpt=str(ckpt),
            provenance={**prov, "score_only": True},
            infer={"max_frames": 750},
            teacher_forced_ce=bare,
            ordered_utt_ids=["u0"],
            condition_reports=reports,
            gate=gate,
        )
        print("H6 score-only bare skip", score_skip is None)
        score_ok = mod._emit_d1c_v2_from_gate(
            out_dir=td_path / "score_ok",
            contract=contract,
            ckpt=str(ckpt),
            provenance={**prov, "score_only": True},
            infer={"max_frames": 750},
            teacher_forced_ce=tf_ce,
            ordered_utt_ids=["u0"],
            condition_reports=reports,
            gate=gate,
        )
        print("H7 score-only measured", "OK" if score_ok else "FAIL")
        report = json.loads(ok.read_text(encoding="utf-8"))
        print(
            "H8 §11 fields",
            "text_usage" in report["gates"],
            "phrase_loop_repetition" in report["summaries"],
            "runaway_or_repetition" in report["summaries"],
        )

    from minispeech.data.d1b_oracles import SUBSET_ALIASES, verify_recipe_source_manifest, load_eval_contract

    print("H9 d1c aliases", [k for k in SUBSET_ALIASES if k.startswith("d1c_")])
    contract = load_eval_contract(ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
    mismatches = verify_recipe_source_manifest(contract["recipe_source_manifest"], root=ROOT)
    print("H10 manifest", "mismatches", mismatches, "sha", contract["sha256"][:16])
    print("ROUND3_PROBE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
