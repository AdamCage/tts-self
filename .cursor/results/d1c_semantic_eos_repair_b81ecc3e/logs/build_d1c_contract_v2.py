#!/usr/bin/env python3
"""Build D1c eval contract 2.0.0 from frozen DEV32 + measured provenance."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/mnt/c/projects/tts-self-d1c")
sys.path.insert(0, str(ROOT / "src"))

from minispeech.eval.asr import asr_provenance


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def content_sha(contract: dict) -> str:
    body = {k: v for k, v in contract.items() if k not in {"sha256", "recipe_sha256"} and not str(k).startswith("_")}
    # recipe projection excludes confirm reveal fields
    return hashlib.sha256(canonical(body)).hexdigest()


def recipe_projection(contract: dict) -> dict:
    """Exclude confirmation seed reveal / confirm IDs / timestamps / full SHA."""
    exclude = {
        "sha256",
        "recipe_sha256",
        "confirmation_seed_reveal",
        "created_at",
        "timestamps",
    }
    out = {}
    for k, v in contract.items():
        if k in exclude or str(k).startswith("_"):
            continue
        if k == "subsets" and isinstance(v, dict):
            # Keep DEV; exclude confirm-only keys from recipe projection if present.
            subsets = {
                sk: sv
                for sk, sv in v.items()
                if not str(sk).startswith("confirm") and "confirm" not in str(sk)
            }
            out[k] = subsets
            continue
        out[k] = v
    return out


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    dev = json.loads((ROOT / "configs/train_subsets/d1c_dev32.json").read_text(encoding="utf-8"))
    burn_path = ROOT / "configs/train_subsets/d1_oracle_burn_registry.json"
    asr = asr_provenance(ROOT / "configs/asr.yaml")
    # Source manifest of executable/config files for recipe lock.
    source_files = [
        "configs/discrete_speech_lm_d1c_semantic_eos_w1.yaml",
        "configs/discrete_speech_lm_d1c_semantic_eos_w2.yaml",
        "src/minispeech/models/discrete_speech_lm.py",
        "src/minispeech/models/multi_codebook_head.py",
        "src/minispeech/data/collate_codec_seq.py",
        "src/minispeech/train/discrete_lm_trainer.py",
        "src/minispeech/infer/discrete_config.py",
        "src/minispeech/eval/discrete_n2.py",
        "src/minispeech/eval/discrete_report_schema.py",
        "scripts/72_train_discrete_lm.py",
        "scripts/73_infer_discrete_lm.py",
        "scripts/74_eval_discrete_lm.py",
        "scripts/76_select_discrete_checkpoint.py",
        "scripts/77_freeze_d1c_oracles.py",
    ]
    source_manifest = {
        rel: file_sha(ROOT / rel) for rel in source_files if (ROOT / rel).is_file()
    }
    recipe_source_manifest_sha256 = hashlib.sha256(canonical(source_manifest)).hexdigest()

    contract = {
        "schema": "minispeech.discrete_eval_contract",
        "id": "d1c_semantic_eos_v2_0_0",
        "version": "2.0.0",
        "plan_slug": "d1c_semantic_eos_repair_b81ecc3e",
        "description": (
            "D1c DEV-bound formal contract for semantic_eos_v1. "
            "Confirmation subset/seed reveal fields are intentionally absent until recipe lock."
        ),
        "gate_id": "N2_D1c_semantic_eos_v1",
        "termination": {
            "mode": "semantic_eos_v1",
            "eos_id": 2048,
            "semantic_vocab_size": 2049,
            "codec_vocab_size": 2048,
        },
        "loss": {
            "cb0_weight": 1.0,
            "eos_loss_weight_primary": 1.0,
            "eos_loss_weight_fallback": 2.0,
            "allowed_recipes": ["w1", "w2_conditional"],
        },
        "train": {
            "optimizer_steps": 750,
            "grad_accum": 4,
            "candidate_steps": [250, 375, 500, 625, 750],
            "dev_train_seeds": [101, 102],
            "confirm_train_seeds": [201, 202],
            "batch_size": 4,
            "lr": 0.0002,
            "precision": "bf16",
            "resume": False,
        },
        "asr": {
            "config_path": "configs/asr.yaml",
            "file_sha256": asr["asr_config_raw_sha256"],
            "parsed_sha256": asr["asr_config_parsed_sha256"],
            "backend": asr["backend"],
            "model": asr["model"],
            "language": asr["language"],
            "decoding": asr["decoding"],
            "package": asr["package"],
            "expected_weight_sha256": asr.get("expected_weight_sha256"),
            "normalizer": asr["normalizer"],
            "scoring_packages": asr["scoring_packages"],
        },
        "codec": {
            "id": "qwen3tts_12hz",
            "model_id": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            "fingerprint": {
                "status": "ok",
                "partial_sha256_16": "7030d98a18ae5968",
                "license": "Apache-2.0",
                "target": "qwen3tts_12hz",
            },
            "codec_hash": "7030d98a18ae5968",
            "sample_rate": 24000,
            "frame_rate": 12.5,
            "n_codebooks": 16,
            "codebook_size": 2048,
        },
        "text_tokenizer": {
            "local_path": "/home/adam/datasets/tts/models/Qwen3.5-0.8B-Base",
            "fingerprint_note": "local path recorded; weight digest measured at train time",
        },
        "speaker_map": {
            "path": "configs/speaker_maps/trackn_v2.json",
            "version": "2.0.0",
            "n_speakers": 1088,
            "max_speakers": 1089,
            "sha256": "64e7cdc5c3447a43d7484b7538f9fd5df88551650d7b95ddb23110f9c36dca6e",
        },
        "infer": {
            "termination_mode": "semantic_eos_v1",
            "temperature": 0.0,
            "top_k": 0,
            "min_frames": 4,
            "max_cb0_repeat": 12,
            "repetition_penalty": 2.0,
            "max_frames": 750,
            "sampler": "greedy",
            "forbidden_success_paths": [
                "text_length_hard_cap",
                "frames_per_text_token",
                "stop_threshold",
                "stop_consecutive",
                "stop_earliest_frac",
                "stop_overrun_bias",
            ],
            "shared_resolver_module": "minispeech.infer.discrete_config",
        },
        "cer_normalization": {
            "version": "normalize_for_asr_v1",
            "module": "minispeech.frontend.normalize.normalize_for_asr",
        },
        "bootstrap": {"seed": 42, "n_resamples": 1000},
        "random_code_baseline": {"base_seed": 42001},
        "catastrophic": {
            "definition_version": "d1b_v1",
            "cer_ge": 0.5,
            "cer_and_jaccard_cer": 0.35,
            "jaccard_lt": 0.2,
        },
        "runaway": {
            "definition_version": "d1c_v1",
            "cb0_max_run": 12,
            "mean_duration_ratio_range": [0.9, 1.1],
            "utt_duration_ratio_range": [0.5, 2.0],
            "require_learned_eos": True,
            "allow_safety_as_pass": False,
        },
        "text_usage": {
            "shuffle_mask_margin": 0.05,
            "prefix_change_frac": 0.25,
            "paired_model_minus_shuffle_ci95_high_lt": 0.0,
            "pairing": "utt_id",
        },
        "candidate_selection": {
            "subset": "select8",
            "steps": [250, 375, 500, 625, 750],
            "require_all_learned_eos": True,
            "no_eligible_policy": "error",
            "lexicographic": [
                "macro_cer",
                "catastrophic_count",
                "max_abs_log_dur_error",
                "mean_abs_log_dur_error",
                "tf_total_ce",
                "optimizer_step",
                "checkpoint_sha256",
            ],
        },
        "formal_mode_bypass_prohibitions": [
            "skip_eval_contract",
            "legacy_n2",
            "gt_derived_max_frames",
            "manual_tf_ce",
            "hand_authored_flat_candidates",
            "dirty_tree",
        ],
        "burn_registry": {
            "path": "configs/train_subsets/d1_oracle_burn_registry.json",
            "file_sha256": file_sha(burn_path),
        },
        "confirm_seed_commitment": {
            "path": "configs/train_subsets/d1c_confirm_seed_commitment.json",
            "status": "awaiting_external_commitment",
        },
        "subsets": {
            "dev32": {
                "path": "configs/train_subsets/d1c_dev32.json",
                "role": "development_oracle",
                "split": "full32",
                "n": 32,
                "file_sha256": dev["file_sha256"],
                "items_sha256": dev["items_sha256"],
                "split_hashes": dev["split_hashes"],
            },
            "select8": {
                "path": "configs/train_subsets/d1c_dev32.json",
                "role": "checkpoint_selection",
                "split": "select8",
                "n": 8,
                "parent": "dev32",
                "items_sha256": dev["split_hashes"]["select8"],
            },
            "gate24": {
                "path": "configs/train_subsets/d1c_dev32.json",
                "role": "heldout_gate",
                "split": "gate24",
                "n": 24,
                "parent": "dev32",
                "items_sha256": dev["split_hashes"]["gate24"],
            },
            "full32": {
                "path": "configs/train_subsets/d1c_dev32.json",
                "role": "full_oracle",
                "split": "full32",
                "n": 32,
                "parent": "dev32",
                "items_sha256": dev["split_hashes"]["full32"],
            },
        },
        "repro": {
            "dev_subset_selection_seed": 20260805,
            "bootstrap_seed": 42,
            "random_baseline_base_seed": 42001,
        },
        "recipe_source_manifest": source_manifest,
        "recipe_source_manifest_sha256": recipe_source_manifest_sha256,
    }
    contract["recipe_sha256"] = hashlib.sha256(canonical(recipe_projection(contract))).hexdigest()
    contract["sha256"] = content_sha(contract)

    out = ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json"
    out.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Also write a placeholder path note for 2.0.1 (created after reveal).
    note = ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_1.json"
    if not note.exists():
        note.write_text(
            json.dumps(
                {
                    "schema": "minispeech.discrete_eval_contract",
                    "id": "d1c_semantic_eos_v2_0_1",
                    "version": "2.0.1",
                    "status": "not_materialized",
                    "note": "Created after confirmation seed reveal + CONFIRM32 freeze; must preserve recipe_sha256 from 2.0.0.",
                    "parent_contract": "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json",
                    "parent_sha256": contract["sha256"],
                    "parent_recipe_sha256": contract["recipe_sha256"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "path": str(out),
                "sha256": contract["sha256"],
                "recipe_sha256": contract["recipe_sha256"],
                "recipe_source_manifest_sha256": recipe_source_manifest_sha256,
                "dev_items_sha256": dev["items_sha256"],
                "asr_raw_sha256": asr["asr_config_raw_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
