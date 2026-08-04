#!/usr/bin/env python3
"""Train own discrete Speech LM (D1b / Wave 2).

Canonical D1b DEV32 (requires clean git SHA; never MINISPEECH_ALLOW_DIRTY):

  cd ~/projects/tts-self && source .venv/bin/activate
  export HF_HOME=$HOME/hf_cache
  python scripts/72_train_discrete_lm.py \\
    --config configs/discrete_speech_lm_d1b_443m.yaml \\
    --subset dev32 \\
    --stage S1 \\
    --codec-tokens-dir ~/datasets/tts/ruslan/codec_tokens,~/datasets/tts/common_voice_ru17/codec_tokens

CONFIRM32:

  python scripts/72_train_discrete_lm.py \\
    --config configs/discrete_speech_lm_d1b_443m.yaml \\
    --subset confirm32 \\
    --run-name discrete_speech_lm_d1b_confirm_seed42 \\
    --seed 42 \\
    --stage S1 \\
    --codec-tokens-dir ~/datasets/tts/ruslan/codec_tokens,~/datasets/tts/common_voice_ru17/codec_tokens

Step budget is **optimizer updates** (D1b primary: 750).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from minispeech.data.codec_seq_protocol import resolve_collate_codec_seq
from minispeech.data.d1b_oracles import (
    DEFAULT_EVAL_CONTRACT,
    enforce_eval_contract,
    load_frozen_utt_ids,
)
from minispeech.data.speaker_registry import SpeakerMode, SpeakerRegistry
from minispeech.train.discrete_lm_trainer import (
    DiscreteLMTrainer,
    build_model_from_cfg,
    maybe_apply_text_init,
    resolve_optimizer_step_budget,
)
from minispeech.train.distributed import (
    cleanup_distributed,
    device_for_rank,
    get_dist_info,
    init_distributed,
    is_main_process,
)
from minispeech.utils import deep_merge, expand, load_yaml, set_seed


def _parse_codec_roots(spec: str | Path) -> list[Path]:
    """Accept one path, comma-separated paths, or a JSON list of roots/indexes."""
    raw = str(spec).strip()
    if raw.endswith(".json") and not raw.endswith("codec_tokens_index.json"):
        p = expand(raw)
        if p.is_file():
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                return [expand(x) for x in obj]
            if isinstance(obj, dict):
                # encode_meta style: index_paths → parent dirs
                idx_paths = obj.get("index_paths") or obj.get("codec_tokens_dirs")
                if isinstance(idx_paths, dict):
                    roots = []
                    for v in idx_paths.values():
                        vp = expand(v)
                        roots.append(vp.parent if vp.name.endswith(".json") else vp)
                    return roots
                if isinstance(idx_paths, list):
                    return [expand(x) for x in idx_paths]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [expand(p) for p in parts]


class CodecTokenShardDataset(Dataset):
    """Wave-1 codec-token reader (safetensors index **or** legacy ``.pt`` / jsonl).

    Preferred layout (Wave 1):
    - ``codec_tokens_XXXXX.safetensors`` + ``codec_tokens_index.json``
    - Multiple corpus roots via comma-separated ``--codec-tokens-dir``
    """

    def __init__(
        self,
        root: str | Path,
        *,
        speaker_registry: SpeakerRegistry,
        speaker_mode: SpeakerMode = "train",
        allow_unknown_speaker: bool | None = None,
        max_utterances: int | None = None,
        max_hours: float | None = None,
        frame_rate: float = 12.5,
        utt_ids: Sequence[str] | None = None,
        require_items_sha256: str | None = None,
    ):
        from minispeech.data.codec_tokens import CodecTokenShardReader
        from minispeech.data.d1b_oracles import items_sha256

        if speaker_registry is None:
            raise ValueError(
                "speaker_registry is required (order-dependent _spk2id removed)"
            )
        self.speaker_registry = speaker_registry
        self.speaker_mode: SpeakerMode = speaker_mode
        self.allow_unknown_speaker = allow_unknown_speaker
        self.roots = _parse_codec_roots(root)
        self._mode: str = "pt"
        self._readers: list[Any] = []
        self._sf_items: list[tuple[int, str]] = []  # (reader_idx, utt_id)
        self._pt_items: list[Path] = []
        self.frame_rate = float(frame_rate)
        self.utt_ids: list[str] | None = (
            [str(u) for u in utt_ids] if utt_ids is not None else None
        )
        self.items_sha256: str | None = None

        for rpath in self.roots:
            if not rpath.is_dir():
                raise FileNotFoundError(
                    f"codec tokens dir not found: {rpath}. "
                    "Run Wave-1 scripts/71_encode_codec_tokens.py first."
                )
            idx_sf = rpath / "codec_tokens_index.json"
            if idx_sf.is_file():
                reader = CodecTokenShardReader(rpath)
                ri = len(self._readers)
                self._readers.append(reader)
                for uid in sorted(reader.index.keys()):
                    self._sf_items.append((ri, uid))
                self._mode = "safetensors"
            else:
                index = rpath / "index.jsonl"
                if index.is_file():
                    for line in index.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        p = Path(row["path"])
                        if not p.is_file():
                            p = rpath / p
                        self._pt_items.append(p)
                else:
                    self._pt_items.extend(sorted(rpath.glob("**/*.pt")))

        if self._mode == "safetensors" and self._sf_items:
            items_n = len(self._sf_items)
        else:
            self._mode = "pt"
            items_n = len(self._pt_items)
        if items_n == 0:
            raise FileNotFoundError(f"no codec-token shards under {self.roots}")

        # Frozen oracle path: exact ordered utt_id list (fail-closed).
        if self.utt_ids is not None:
            if self._mode != "safetensors":
                raise RuntimeError(
                    "utt_ids subset requires safetensors codec_tokens_index.json caches"
                )
            by_uid: dict[str, tuple[int, str]] = {}
            for ri, uid in self._sf_items:
                # First hit wins (stable if duplicates across corpora — should not happen).
                by_uid.setdefault(uid, (ri, uid))
            missing = [u for u in self.utt_ids if u not in by_uid]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} utt_id(s) missing from codec cache under "
                    f"{self.roots}; first missing={missing[:5]}"
                )
            self._sf_items = [by_uid[u] for u in self.utt_ids]
            self.items_sha256 = items_sha256(self.utt_ids)
            if require_items_sha256 is not None and self.items_sha256 != require_items_sha256:
                raise ValueError(
                    f"subset items_sha256 mismatch: got={self.items_sha256} "
                    f"expected={require_items_sha256}"
                )
            return

        if max_hours is not None:
            budget = float(max_hours) * 3600.0
            acc = 0.0
            if self._mode == "safetensors":
                kept_sf: list[tuple[int, str]] = []
                for ri, uid in self._sf_items:
                    info = self._readers[ri].index[uid]
                    t = int(info.get("n_frames") or 0)
                    if t <= 0:
                        t = int(self._readers[ri].get(uid).codes.shape[0])
                    acc += t / self.frame_rate
                    kept_sf.append((ri, uid))
                    if acc >= budget:
                        break
                self._sf_items = kept_sf
            else:
                kept_pt: list[Path] = []
                for p in self._pt_items:
                    obj = torch.load(p, map_location="cpu", weights_only=False)
                    codes = obj["codes"]
                    t = int(codes.shape[0]) if torch.is_tensor(codes) else len(codes)
                    acc += t / self.frame_rate
                    kept_pt.append(p)
                    if acc >= budget:
                        break
                self._pt_items = kept_pt

        if max_utterances is not None:
            n = int(max_utterances)
            if self._mode == "safetensors":
                self._sf_items = self._sf_items[:n]
            else:
                self._pt_items = self._pt_items[:n]

    def __len__(self) -> int:
        return len(self._sf_items) if self._mode == "safetensors" else len(self._pt_items)

    def _speaker_int(self, spk: Any) -> int:
        return self.speaker_registry.to_id(
            spk,
            mode=self.speaker_mode,
            allow_unknown=self.allow_unknown_speaker,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._mode == "safetensors":
            ri, uid = self._sf_items[idx]
            ex = self._readers[ri].get(uid)
            return {
                "utt_id": uid,
                "input_ids": ex.text_ids.long(),
                "text_ids": ex.text_ids.long(),
                "codes": ex.codes.long(),
                "speaker_id": self._speaker_int(ex.speaker_id),
                "corpus": ex.corpus,
                "dur_s": float(ex.dur_s),
                "sample_weight": 1.0,
                "stress": ex.features.stress,
                "phoneme": ex.features.phoneme,
                "span_type": ex.features.span_type,
                "word_bound": ex.features.word_bound,
                "phrase_bound": ex.features.phrase_bound,
            }

        obj = torch.load(self._pt_items[idx], map_location="cpu", weights_only=False)
        if "codes" not in obj:
            raise KeyError(f"{self._pt_items[idx]} missing codes")
        ids = obj.get("input_ids", obj.get("text_ids"))
        if ids is None:
            raise KeyError(f"{self._pt_items[idx]} missing input_ids/text_ids")
        out = dict(obj)
        out["input_ids"] = ids
        out.setdefault("utt_id", self._pt_items[idx].stem)
        out["speaker_id"] = self._speaker_int(obj.get("speaker_id", 0))
        return out


def _stage_subset(stage: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_utterances": stage.get("max_utterances"),
        "max_hours": stage.get("max_hours"),
    }


def resolve_train_utt_ids(
    cfg: dict[str, Any],
    *,
    subset: str | None,
    subset_split: str | None,
) -> tuple[list[str] | None, str | None, str | None]:
    """Return (utt_ids, subset_path_label, items_sha256) for frozen oracles."""
    data = cfg.get("data", {}) or {}
    train = cfg.get("train", {}) or {}
    raw = (
        subset
        or train.get("subset")
        or train.get("subset_path")
        or data.get("subset")
        or data.get("subset_path")
    )
    if not raw:
        return None, None, None
    split = (
        subset_split
        or train.get("subset_split")
        or data.get("subset_split")
    )
    ids, _payload, digest = load_frozen_utt_ids(raw, split=split, verify_hash=True)
    return ids, str(raw), digest


def resolve_speaker_registry(cfg: dict[str, Any]) -> SpeakerRegistry:
    """Load canonical speaker map (fail-closed if missing)."""
    data = cfg.get("data", {}) or {}
    train = cfg.get("train", {}) or {}
    model_spk = (cfg.get("model") or {}).get("speaker") or {}
    path = (
        data.get("speaker_map")
        or data.get("speaker_registry")
        or train.get("speaker_map_path")
        or train.get("speaker_map")
        or (model_spk.get("map_path") if isinstance(model_spk, dict) else None)
        or cfg.get("speaker_map")
        or "configs/speaker_maps/trackn_v2.json"
    )
    reg = SpeakerRegistry.load(path)
    if reg.n_speakers <= 0:
        raise RuntimeError(
            f"speaker map {path} is empty/scaffold; rebuild with "
            "`python -m minispeech.data.speaker_registry build-trackn-v2`"
        )
    return reg


def make_loaders(
    cfg: dict[str, Any],
    stage: dict[str, Any],
    *,
    codec_tokens_dir: str | Path,
    batch_size: int,
    speaker_registry: SpeakerRegistry | None = None,
    utt_ids: Sequence[str] | None = None,
    require_items_sha256: str | None = None,
) -> tuple[DataLoader, DataLoader | None]:
    collate = resolve_collate_codec_seq()
    pad_id = int(
        cfg.get("data", {}).get("pad_token_id", cfg.get("model", {}).get("pad_token_id", 0))
    )
    subset = _stage_subset(stage)
    registry = speaker_registry or resolve_speaker_registry(cfg)
    ds_kwargs: dict[str, Any] = {
        "speaker_registry": registry,
        "speaker_mode": "train",
    }
    if utt_ids is not None:
        ds_kwargs["utt_ids"] = list(utt_ids)
        ds_kwargs["require_items_sha256"] = require_items_sha256
    else:
        # Legacy first-N / hours path — forbidden for canonical D1b YAML.
        ds_kwargs.update({k: v for k, v in subset.items() if v is not None})
    ds = CodecTokenShardDataset(codec_tokens_dir, **ds_kwargs)
    info = get_dist_info()
    sampler = (
        DistributedSampler(ds, shuffle=True, drop_last=False)
        if info.distributed
        else None
    )
    train_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        collate_fn=lambda b: collate(b, pad_token_id=pad_id),
    )
    return train_loader, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/discrete_speech_lm_d1b_443m.yaml")
    ap.add_argument("--data", default=None, help="Optional data yaml (merged under data:)")
    ap.add_argument("--stage", default="S1")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optimizer updates (not microbatches). Default: stage/config budget.",
    )
    ap.add_argument("--codec-tokens-dir", default=None)
    ap.add_argument(
        "--subset",
        default=None,
        help="Frozen oracle alias/path: dev32, confirm32, or JSON path",
    )
    ap.add_argument(
        "--subset-split",
        default=None,
        help="Split inside subset JSON: full32|select8|gate24 (CONFIRM)",
    )
    ap.add_argument("--run-name", default=None, help="Expands train.run_dir ${run_name}")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--eval-contract",
        default=None,
        help=f"Frozen eval_contract.json (default: {DEFAULT_EVAL_CONTRACT})",
    )
    ap.add_argument(
        "--skip-eval-contract",
        action="store_true",
        help="Skip fail-closed eval-contract checks (non-canonical only)",
    )
    ap.add_argument("--resume", default=None, help="ckpt path | true | false")
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Skip clean-git enforcement (FORBIDDEN for canonical D1b gates)",
    )
    ap.add_argument(
        "--text-init",
        action="store_true",
        help="Force-enable Qwen text-init for this run",
    )
    args = ap.parse_args()

    dist = init_distributed()
    cfg = load_yaml(args.config)
    if args.data:
        data_yaml = load_yaml(args.data)
        cfg = deep_merge(cfg, {"data": data_yaml.get("data", data_yaml)})
    if args.text_init:
        cfg.setdefault("model", {}).setdefault("text_init", {})["enabled"] = True

    tcfg = cfg.setdefault("train", {})
    if args.run_name:
        tcfg["run_name"] = args.run_name
    if args.seed is not None:
        tcfg["seed"] = int(args.seed)
    if args.eval_contract:
        tcfg["eval_contract_path"] = args.eval_contract
    tcfg.setdefault("eval_contract_path", DEFAULT_EVAL_CONTRACT)

    utt_ids, subset_label, subset_digest = resolve_train_utt_ids(
        cfg, subset=args.subset, subset_split=args.subset_split
    )
    cfg_name = str(args.config).replace("\\", "/")
    is_d1b_cfg = "d1b_443m" in cfg_name or "d1b_n2" in cfg_name
    is_d1c_cfg = (
        "d1c_semantic_eos" in cfg_name
        or "d1c_eos" in cfg_name
        or str((cfg.get("model") or {}).get("termination", {}) or {}).get("mode")
        == "semantic_eos_v1"
    )
    is_formal_cfg = is_d1b_cfg or is_d1c_cfg
    if is_formal_cfg and utt_ids is None:
        label = "D1c" if is_d1c_cfg else "D1b"
        raise SystemExit(
            f"{label} config requires --subset (dev32|confirm32|path); "
            "silent first-N is forbidden"
        )
    if is_d1c_cfg and args.skip_eval_contract:
        raise SystemExit(
            "D1c formal path forbids --skip-eval-contract; "
            "eval-contract enforcement is mandatory"
        )
    if utt_ids is not None:
        tcfg["subset"] = subset_label
        tcfg["subset_path"] = subset_label
        tcfg["subset_split"] = args.subset_split or tcfg.get("subset_split") or "full32"
        tcfg["subset_hash"] = subset_digest

    if not args.skip_eval_contract and (is_formal_cfg or utt_ids is not None):
        declared_hash = tcfg.get("eval_contract_hash")
        contract = enforce_eval_contract(
            cfg,
            contract_path=tcfg.get("eval_contract_path"),
            subset_items_sha256=subset_digest,
            subset_label=subset_label,
            require_clean_sha=bool(is_d1c_cfg and not args.allow_dirty),
            expected_contract_sha256=str(declared_hash) if declared_hash else None,
        )
        live_sha = str(contract.get("sha256") or "")
        if is_d1c_cfg and not declared_hash:
            raise SystemExit(
                "D1c recipe must declare train.eval_contract_hash; "
                "refusing silent overwrite"
            )
        if declared_hash and str(declared_hash) != live_sha:
            raise SystemExit(
                "eval_contract_hash mismatch (fail-closed): "
                f"declared={declared_hash} live={live_sha}"
            )
        # Stamp provenance from contract; never overwrite a mismatched declared hash.
        tcfg["eval_contract_hash"] = live_sha or str(declared_hash)
        tcfg["recipe_hash"] = str(contract.get("recipe_sha256") or "") or tcfg.get(
            "recipe_hash"
        )
        tcfg["recipe_source_manifest_hash"] = str(
            contract.get("recipe_source_manifest_sha256") or ""
        ) or tcfg.get("recipe_source_manifest_hash")
        # Prefer content hash recorded in contract.
        sp = contract.get("speaker_map") or {}
        if sp.get("sha256"):
            tcfg["speaker_map_hash"] = str(sp["sha256"])

    set_seed(int(cfg.get("train", {}).get("seed", 42)) + dist.rank)
    device = device_for_rank(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Default horizon is optimizer updates (D1b primary budget = 750).
    stages = cfg.get("train", {}).get("stages", [{"name": "S1", "steps": 750}])
    if args.stage and args.stage != "all":
        stages = [s for s in stages if s["name"] == args.stage]
    if not stages:
        raise SystemExit(f"no stages matched --stage {args.stage}")

    codec_dir = (
        args.codec_tokens_dir
        or cfg.get("data", {}).get("codec_tokens_dir")
        or cfg.get("data", {}).get("codec_tokens_path")
    )
    if not codec_dir:
        raise SystemExit(
            "Pass --codec-tokens-dir or set data.codec_tokens_dir once Wave-1 "
            "caches exist."
        )

    bcfg = cfg.get("train", {}).get("batch", {})
    bs = int(args.batch_size or bcfg.get("size") or 4)

    model = build_model_from_cfg(cfg)
    text_report = maybe_apply_text_init(model, cfg)
    if is_main_process():
        counts = model.count_parameters()
        print(f"param_count={counts}")
        if text_report is not None:
            print(f"text_init coverage={text_report.coverage:.3f} source={text_report.source}")

    resume = args.resume
    if resume is not None:
        if str(resume).lower() in {"true", "1", "yes"}:
            resume = True
        elif str(resume).lower() in {"false", "0", "no"}:
            resume = False

    enforce_clean = not args.allow_dirty
    if args.allow_dirty:
        import os

        os.environ.setdefault("MINISPEECH_ALLOW_DIRTY", "1")
        if is_d1b_cfg or is_d1c_cfg:
            label = "D1c" if is_d1c_cfg else "D1b"
            print(
                f"WARNING: --allow-dirty set with {label} config; "
                "canonical gates forbid MINISPEECH_ALLOW_DIRTY=1"
            )

    speaker_registry = resolve_speaker_registry(cfg)
    if is_main_process():
        print(
            f"speaker_registry n_speakers={speaker_registry.n_speakers} "
            f"max_speakers={speaker_registry.max_speakers} "
            f"sha256={speaker_registry.sha256[:16]}..."
        )
        if utt_ids is not None:
            print(
                f"frozen_subset={subset_label} split={tcfg.get('subset_split')} "
                f"n={len(utt_ids)} items_sha256={subset_digest}"
            )

    try:
        for stage in stages:
            train_loader, val_loader = make_loaders(
                cfg,
                stage,
                codec_tokens_dir=codec_dir,
                batch_size=bs,
                speaker_registry=speaker_registry,
                utt_ids=utt_ids,
                require_items_sha256=subset_digest,
            )
            # Ensure DDP block exists so trainer can read find_unused_parameters.
            cfg.setdefault("train", {}).setdefault("ddp", {})
            cfg["train"]["ddp"].setdefault("find_unused_parameters", False)
            trainer = DiscreteLMTrainer(
                model,
                train_loader,
                val_loader,
                cfg,
                device,
                resume=resume,
                text_init_report=text_report,
                enforce_clean_git=enforce_clean,
            )
            steps = int(
                args.steps
                if args.steps is not None
                else resolve_optimizer_step_budget(cfg, stage)
            )
            if is_main_process():
                print(
                    f"stage={stage['name']} optimizer_steps={steps} batch={bs} "
                    f"grad_accum={trainer.grad_accum} "
                    f"find_unused_parameters={trainer.find_unused_parameters} "
                    f"n_train={len(train_loader.dataset)} "  # type: ignore[arg-type]
                    f"world_size={dist.world_size}"
                )
            last = trainer.train_steps(steps, stage=stage["name"])
            if is_main_process():
                print(f"stage={stage['name']} done last={last}")
            # Only resume from disk on the first constructed trainer
            resume = False
            text_report = None
            model = trainer._bare()
    finally:
        cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
