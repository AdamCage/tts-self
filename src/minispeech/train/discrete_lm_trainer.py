"""Trainer for discrete Speech LM (bf16, grad accum, resume, DDP-ready).

Step semantics (D1b normative):
- ``optimizer_step`` / ``global_step`` = optimizer updates (after ``opt.step()``)
- ``microbatch_step`` = forward/backward microbatches (including accumulation)
- scheduler, warmup, save/eval intervals, and train budgets use optimizer updates
- checkpoints only on optimizer boundaries (``accum_counter == 0``)
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from minispeech.models.discrete_speech_lm import (
    DiscreteSpeechLM,
    DiscreteSpeechLMConfig,
    build_discrete_speech_lm,
)
from minispeech.models.text_init import (
    TextInitReport,
    apply_qwen_text_init,
    text_init_manifest_extra,
)
from minispeech.train.distributed import (
    barrier,
    device_for_rank,
    get_dist_info,
    init_distributed,
    is_main_process,
    unwrap_model,
    wrap_ddp,
)
from minispeech.train.schedule import lr_multiplier, set_optimizer_lr
from minispeech.data.d1b_oracles import expand_path_template
from minispeech.utils import config_hash, expand, write_run_manifest

STEP_UNIT_OPTIMIZER = "optimizer"
STEP_UNIT_MICROBATCH = "microbatch"


class ResumeMismatchError(ValueError):
    """Fail-closed resume incompatibility (step unit, hashes, accum, …)."""


def fr_selection_key(metrics: dict[str, Any]) -> tuple:
    """Higher tuple wins. Prefer lower free-running CER when present."""
    if metrics.get("cer") is not None:
        return (
            -float(metrics["cer"]),
            -float(metrics.get("ce_cb0", metrics.get("loss", 0.0))),
        )
    # Teacher-forced CE as diagnostic fallback only
    return (0.0, -float(metrics.get("loss", metrics.get("ce_cb0", 1e9))))


def _capture_rng() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _as_cpu_byte_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().to(dtype=torch.uint8)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return torch.frombuffer(bytearray(value), dtype=torch.uint8).clone()
    return torch.as_tensor(np.asarray(value), dtype=torch.uint8).cpu().contiguous()


def _restore_rng(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_as_cpu_byte_tensor(state["torch"]))
    if "cuda" in state and torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all([_as_cpu_byte_tensor(s) for s in state["cuda"]])


def _file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = expand(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_speaker_map_path(cfg: dict[str, Any]) -> Path | None:
    """Locate the canonical speaker-map JSON for hashing / run-dir copy."""
    tcfg = cfg.get("train", {}) or {}
    data = cfg.get("data", {}) or {}
    model_spk = (cfg.get("model") or {}).get("speaker") or {}
    candidates = [
        tcfg.get("speaker_map_path"),
        tcfg.get("speaker_map"),
        data.get("speaker_map"),
        data.get("speaker_registry"),
        model_spk.get("map_path") if isinstance(model_spk, dict) else None,
        cfg.get("speaker_map"),
        "configs/speaker_maps/trackn_v2.json",
    ]
    for raw in candidates:
        if not raw or not isinstance(raw, str):
            continue
        p = expand(raw)
        if p.is_file():
            return p
    return None


def resolve_repro_hashes(cfg: dict[str, Any]) -> dict[str, str | None]:
    """Collect hashes stored in checkpoints / checked on resume."""
    tcfg = cfg.get("train", {}) or {}
    data = cfg.get("data", {}) or {}
    repro = cfg.get("repro", {}) or tcfg.get("repro", {}) or {}
    model_cfg = {"model": cfg.get("model", {})}

    def _pick(*keys: str, path_keys: tuple[str, ...] = ()) -> str | None:
        for k in keys:
            v = repro.get(k)
            if v is None:
                v = tcfg.get(k)
            if v is not None and str(v).strip():
                return str(v)
        for pk in path_keys:
            raw = repro.get(pk)
            if raw is None:
                raw = tcfg.get(pk)
            if raw is None:
                raw = data.get(pk)
            digest = _file_sha256(raw) if raw else None
            if digest is not None:
                return digest
        return None

    speaker_hash = _pick(
        "speaker_map_hash",
        path_keys=("speaker_map_path", "speaker_map"),
    )
    if speaker_hash is None:
        sp_path = resolve_speaker_map_path(cfg)
        if sp_path is not None:
            try:
                from minispeech.data.speaker_registry import SpeakerRegistry

                speaker_hash = SpeakerRegistry.load(sp_path).sha256
            except Exception:
                speaker_hash = _file_sha256(sp_path)

    subset_hash = _pick(
        "subset_hash",
        path_keys=(),
    )
    if subset_hash is None:
        # Prefer explicit items sha when subset_path is a frozen oracle JSON.
        for pk in ("subset_path", "train_subset", "subset"):
            raw = repro.get(pk) or tcfg.get(pk) or data.get(pk)
            if not raw:
                continue
            try:
                from minispeech.data.d1b_oracles import load_frozen_utt_ids

                split = (
                    repro.get("subset_split")
                    or tcfg.get("subset_split")
                    or data.get("subset_split")
                )
                _, _, digest = load_frozen_utt_ids(
                    raw, split=split, verify_hash=True
                )
                subset_hash = digest
                break
            except Exception:
                subset_hash = _file_sha256(raw)
                if subset_hash is not None:
                    break

    out: dict[str, str | None] = {
        "config": config_hash(cfg),
        "model_config": config_hash(model_cfg),
        "speaker_map": speaker_hash,
        "subset": subset_hash,
        "eval_contract": _pick("eval_contract_hash", path_keys=()),
    }
    if out["eval_contract"] is None:
        for pk in ("eval_contract_path", "eval_contract"):
            raw = repro.get(pk) or tcfg.get(pk) or data.get(pk)
            if not raw:
                continue
            try:
                from minispeech.data.d1b_oracles import load_eval_contract

                out["eval_contract"] = str(load_eval_contract(raw).get("sha256") or "")
                if out["eval_contract"]:
                    break
            except Exception:
                out["eval_contract"] = _file_sha256(raw)
                if out["eval_contract"] is not None:
                    break
    return out


def resolve_optimizer_step_budget(
    cfg: dict[str, Any],
    stage: dict[str, Any] | None = None,
) -> int:
    """Resolve train horizon in **optimizer updates**.

    D1b primary budget is 750 optimizer updates (legacy ``steps: 3000`` with
    ``grad_accum: 4``). Optional DEV-only extension to 1250 is a config
    capability (``train.dev_extension_optimizer_steps`` + opt-in flag), not an
    automatic promotion.
    """
    tcfg = cfg.get("train", {}) or {}
    stage = stage or {}
    if stage.get("optimizer_steps") is not None:
        return int(stage["optimizer_steps"])
    if stage.get("steps") is not None:
        return int(stage["steps"])

    use_ext = bool(stage.get("dev_extension", False)) or bool(
        tcfg.get("allow_dev_extension", False)
    )
    if use_ext:
        if tcfg.get("dev_extension_optimizer_steps") is not None:
            return int(tcfg["dev_extension_optimizer_steps"])
        if tcfg.get("optimizer_steps_dev_extension") is not None:
            return int(tcfg["optimizer_steps_dev_extension"])
        return 1250
    if tcfg.get("optimizer_steps") is not None:
        return int(tcfg["optimizer_steps"])
    if tcfg.get("max_optimizer_steps") is not None:
        return int(tcfg["max_optimizer_steps"])
    return 750


def resolve_candidate_steps(cfg: dict[str, Any]) -> tuple[int, ...]:
    """Immutable FR candidate optimizer-step boundaries for D1b offline selection."""
    tcfg = cfg.get("train", {}) or {}
    use_ext = bool(tcfg.get("allow_dev_extension", False))
    raw = tcfg.get("candidate_steps_dev_extension") if use_ext else tcfg.get(
        "candidate_steps"
    )
    if raw is None and use_ext:
        raw = tcfg.get("candidate_steps")
    if raw is not None:
        return tuple(int(x) for x in raw)
    from minispeech.eval.discrete_n2 import fr_candidate_steps

    return fr_candidate_steps(include_dev_extension=use_ext)


def freeze_disabled_ru_conditioning(model: torch.nn.Module) -> int:
    """Mark RU-conditioning params non-trainable when the feature is off.

    Avoids DDP unused-parameter errors without requiring ``find_unused_parameters``.
    Duck-typed: needs ``.cfg.russian_conditioning`` and ``.ru_cond``.
    """
    bare = unwrap_model(model)
    cfg = getattr(bare, "cfg", None)
    ru = getattr(bare, "ru_cond", None)
    if cfg is None or ru is None:
        return 0
    if bool(getattr(cfg, "russian_conditioning", False)):
        return 0
    n = 0
    for p in ru.parameters():
        if p.requires_grad:
            p.requires_grad = False
            n += 1
    return n


def resolve_resume_path(cfg: dict[str, Any], resume: str | Path | bool | None) -> Path | None:
    tcfg = cfg.get("train", {})
    raw = resume if resume is not None else tcfg.get("resume")
    if raw is None or raw is False:
        return None
    if raw is True:
        run_name = str(tcfg.get("run_name") or "unnamed")
        run_dir = expand_path_template(
            tcfg.get("run_dir", "runs/discrete_speech_lm_d1b_443m"),
            {"run_name": run_name},
        )
        for name in ("S1_last.pt", "S2_last.pt", "S3_last.pt", "S4_last.pt", "last.pt"):
            cand = run_dir / name
            if cand.is_file():
                return cand
        best = run_dir / "best.pt"
        return best if best.is_file() else None
    path = expand(raw)
    return path if path.is_file() else None


def residual_weights_from_cfg(cfg: dict[str, Any], n_residual: int) -> torch.Tensor:
    lcfg = cfg.get("loss", {})
    raw = lcfg.get("residual_weights")
    if raw is None:
        return torch.ones(n_residual, dtype=torch.float32)
    w = torch.tensor(list(raw), dtype=torch.float32)
    if w.numel() == 1:
        return w.repeat(n_residual)
    if w.numel() != n_residual:
        raise ValueError(
            f"loss.residual_weights length {w.numel()} != n_residual {n_residual}"
        )
    return w


class DiscreteLMTrainer:
    """DDP-ready trainer for :class:`DiscreteSpeechLM`."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        cfg: dict[str, Any],
        device: torch.device,
        *,
        fr_eval_fn: Callable[[torch.nn.Module], dict[str, Any]] | None = None,
        resume: str | Path | bool | None = None,
        text_init_report: TextInitReport | None = None,
        enforce_clean_git: bool = True,
    ):
        self.dist = init_distributed()
        self.device = device_for_rank(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.fr_eval_fn = fr_eval_fn
        self.text_init_report = text_init_report
        self.repro_hashes = resolve_repro_hashes(cfg)

        bare = model.to(self.device)
        n_frozen_ru = freeze_disabled_ru_conditioning(bare)
        tcfg = cfg.get("train", {})
        self.base_lr = float(tcfg.get("lr", 2e-4))
        trainable = [p for p in bare.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("DiscreteLMTrainer: no trainable parameters")
        self.opt = torch.optim.AdamW(
            trainable,
            lr=self.base_lr,
            weight_decay=float(tcfg.get("weight_decay", 0.01)),
        )
        ddp_cfg = tcfg.get("ddp") or {}
        self.find_unused_parameters = bool(ddp_cfg.get("find_unused_parameters", False))
        self.model = wrap_ddp(
            bare, find_unused_parameters=self.find_unused_parameters
        )

        run_name = str(tcfg.get("run_name") or "unnamed")
        self.run_dir = expand_path_template(
            tcfg.get("run_dir", "runs/discrete_speech_lm_d1b_443m"),
            {"run_name": run_name},
        )
        # Persist expanded path back so checkpoints / resume see a concrete dir.
        tcfg["run_dir"] = str(self.run_dir)
        tcfg["run_name"] = run_name
        if is_main_process():
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._materialize_run_artifacts()
        barrier()

        self.grad_clip = float(tcfg.get("grad_clip", 1.0))
        self.grad_accum = max(int(tcfg.get("grad_accum", tcfg.get("gradient_accumulation_steps", 1))), 1)
        self.warmup_steps = int(tcfg.get("warmup_steps", 200) or 0)
        self.lr_schedule = str(tcfg.get("lr_schedule", "cosine"))
        self.min_lr_ratio = float(tcfg.get("min_lr_ratio", 0.05))
        self.log_every = int(tcfg.get("log_every", 50) or 50)
        self.save_every = int(tcfg.get("save_every", 1000) or 0)
        self.eval_every = int(tcfg.get("eval_every", 500) or 0)
        prec = str(tcfg.get("precision", "bf16")).lower()
        self.use_bf16 = self.device.type == "cuda" and prec in ("bf16", "bfloat16")

        fr_cfg = tcfg.get("fr_checkpoint", {})
        self.fr_every = int(fr_cfg.get("every", tcfg.get("fr_eval_every", 0)) or 0)
        self.fr_select = bool(fr_cfg.get("enabled", self.fr_every > 0))
        self.candidate_steps = frozenset(resolve_candidate_steps(cfg))
        self._best_key: tuple | None = None

        self.cb0_weight = float(cfg.get("loss", {}).get("cb0_weight", 1.0))
        self.stop_weight = float(cfg.get("loss", {}).get("stop_weight", 1.0))
        self.stop_pos_weight = float(cfg.get("loss", {}).get("stop_pos_weight", 20.0))
        bare_m = unwrap_model(self.model)
        n_residual = int(getattr(getattr(bare_m, "heads", None), "n_residual", 15))
        self._residual_w = residual_weights_from_cfg(cfg, n_residual).to(self.device)

        # Normative counters: global_step == optimizer_step
        self.optimizer_step = 0
        self.microbatch_step = 0
        self._total_steps: int | None = None
        self._epoch = 0
        self._batches_in_epoch = 0
        self._rng_at_epoch_start: dict[str, Any] | None = None
        self._data_iter = None
        self._accum_counter = 0
        self.allow_legacy_microbatch_resume = bool(
            tcfg.get("allow_legacy_microbatch_resume", False)
        )

        counts = (
            bare_m.count_parameters()
            if hasattr(bare_m, "count_parameters")
            else {"total": sum(p.numel() for p in bare_m.parameters())}
        )
        if is_main_process():
            extra = {
                "param_count": counts,
                "distributed": self.dist.distributed,
                "world_size": self.dist.world_size,
                "rank": self.dist.rank,
                "use_bf16": self.use_bf16,
                "grad_accum": self.grad_accum,
                "step_unit": STEP_UNIT_OPTIMIZER,
                "find_unused_parameters": self.find_unused_parameters,
                "frozen_ru_conditioning_params": n_frozen_ru,
                "repro_hashes": self.repro_hashes,
                "trainable_modules": [
                    "text_emb",
                    "blocks",
                    "speech_emb",
                    "speech_start",
                    "speaker_emb",
                    "heads",
                    "ru_cond",
                ],
                "frozen_modules": ["qwen3_tts_tokenizer (external codec)"],
                **text_init_manifest_extra(text_init_report),
            }
            write_run_manifest(
                self.run_dir,
                {**cfg, "_resolved": extra},
                extra=extra,
                enforce_clean=enforce_clean_git,
            )
        barrier()

        resume_path = resolve_resume_path(cfg, resume)
        if resume_path is not None:
            self.load_checkpoint(resume_path)

    @property
    def global_step(self) -> int:
        """Alias for ``optimizer_step`` (normative D1b step unit)."""
        return int(self.optimizer_step)

    @global_step.setter
    def global_step(self, value: int) -> None:
        self.optimizer_step = int(value)

    def _autocast(self):
        if self.use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _bare(self) -> torch.nn.Module:
        return unwrap_model(self.model)

    def _materialize_run_artifacts(self) -> None:
        """Copy speaker map + write resolved config / hashes into the run dir."""
        import yaml

        resolved_path = self.run_dir / "resolved_config.yaml"
        resolved_path.write_text(
            yaml.safe_dump(self.cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        sp_src = resolve_speaker_map_path(self.cfg)
        if sp_src is not None:
            sp_dst = self.run_dir / "speaker_map.json"
            if sp_src.resolve() != sp_dst.resolve():
                shutil.copy2(sp_src, sp_dst)
            digest = self.repro_hashes.get("speaker_map") or _file_sha256(sp_dst)
            (self.run_dir / "speaker_map.sha256").write_text(
                f"{digest}  speaker_map.json\n", encoding="utf-8"
            )
        (self.run_dir / "hashes.json").write_text(
            json.dumps(self.repro_hashes, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _set_sampler_epoch(self, epoch: int) -> None:
        sampler = getattr(self.train_loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

    def _sync_context(self, *, is_final_microbatch: bool):
        """DDP ``no_sync`` on non-final accumulation microbatches."""
        if (
            not is_final_microbatch
            and isinstance(self.model, DDP)
        ):
            return self.model.no_sync()
        return nullcontext()

    def _ckpt_payload(
        self,
        stage: str,
        step: int,
        *,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sampler = getattr(self.train_loader, "sampler", None)
        sampler_state: dict[str, Any] = {
            "epoch": self._epoch,
            "batches_in_epoch": self._batches_in_epoch,
            "rng_at_epoch_start": self._rng_at_epoch_start,
        }
        if isinstance(sampler, DistributedSampler):
            sampler_state["distributed_sampler_epoch"] = int(getattr(sampler, "epoch", self._epoch))
        return {
            "model": self._bare().state_dict(),
            "optimizer": self.opt.state_dict(),
            "cfg": self.cfg,
            "stage": stage,
            "step": int(step),
            "global_step": int(step),
            "optimizer_step": int(self.optimizer_step),
            "microbatch_step": int(self.microbatch_step),
            "step_unit": STEP_UNIT_OPTIMIZER,
            "grad_accum": int(self.grad_accum),
            "hashes": dict(self.repro_hashes),
            "metrics": metrics,
            "best_key": list(self._best_key) if self._best_key is not None else None,
            "rng": _capture_rng(),
            "dataloader": sampler_state,
            "schedule": {
                "warmup_steps": self.warmup_steps,
                "lr_schedule": self.lr_schedule,
                "min_lr_ratio": self.min_lr_ratio,
                "base_lr": self.base_lr,
                "total_steps": self._total_steps,
                "grad_accum": self.grad_accum,
                "step_unit": STEP_UNIT_OPTIMIZER,
            },
        }

    def save_checkpoint(
        self,
        stage: str,
        step: int,
        *,
        metrics: dict[str, Any] | None = None,
        promote_best: bool = False,
        as_candidate: bool = False,
    ) -> None:
        if self._accum_counter != 0:
            raise RuntimeError(
                "checkpoint refused mid-accumulation "
                f"(accum_counter={self._accum_counter}); "
                "save only on optimizer boundary"
            )
        if not is_main_process():
            return
        payload = self._ckpt_payload(stage, step, metrics=metrics)
        if as_candidate:
            # Immutable FR candidate — never overwrite best.pt here.
            torch.save(
                payload,
                self.run_dir / f"candidate_step_{int(step):04d}.pt",
            )
            return
        torch.save(payload, self.run_dir / f"{stage}_last.pt")
        torch.save(payload, self.run_dir / "last.pt")
        if promote_best:
            torch.save(payload, self.run_dir / "best.pt")
            if metrics is not None:
                (self.run_dir / "best_metrics.json").write_text(
                    json.dumps(metrics, indent=2), encoding="utf-8"
                )

    def _validate_resume(self, ckpt: dict[str, Any]) -> tuple[str, int, int]:
        """Return ``(step_unit_resolved, optimizer_step, microbatch_step)``."""
        tcfg = self.cfg.get("train", {}) or {}
        step_unit = ckpt.get("step_unit")
        legacy = ckpt.get("legacy_step_unit")
        ckpt_accum = ckpt.get("grad_accum")
        if ckpt_accum is None:
            ckpt_accum = (ckpt.get("schedule") or {}).get("grad_accum")
        if ckpt_accum is not None and int(ckpt_accum) != int(self.grad_accum):
            raise ResumeMismatchError(
                f"grad_accum mismatch: ckpt={ckpt_accum} current={self.grad_accum}"
            )

        raw_step = int(ckpt.get("global_step", ckpt.get("step") or 0))
        opt_step_ck = ckpt.get("optimizer_step")
        micro_ck = ckpt.get("microbatch_step")

        migrate_legacy = False
        if step_unit == STEP_UNIT_OPTIMIZER:
            optimizer_step = int(opt_step_ck if opt_step_ck is not None else raw_step)
            microbatch_step = int(
                micro_ck
                if micro_ck is not None
                else optimizer_step * int(self.grad_accum)
            )
        elif (
            step_unit == STEP_UNIT_MICROBATCH
            or legacy == STEP_UNIT_MICROBATCH
            or (
                step_unit is None
                and opt_step_ck is None
                and (
                    self.allow_legacy_microbatch_resume
                    or bool(tcfg.get("allow_legacy_microbatch_resume", False))
                )
            )
        ):
            migrate_legacy = True
            microbatch_step = int(micro_ck if micro_ck is not None else raw_step)
            optimizer_step = microbatch_step // int(self.grad_accum)
            step_unit = STEP_UNIT_MICROBATCH
        elif step_unit is None and opt_step_ck is None:
            raise ResumeMismatchError(
                "legacy checkpoint lacks step_unit=optimizer; refuse resume. "
                "Either retrain under D1b trainer, or set "
                "train.allow_legacy_microbatch_resume=true to migrate with "
                "legacy_step_unit=microbatch "
                f"(raw_step={raw_step}, grad_accum={self.grad_accum})."
            )
        else:
            raise ResumeMismatchError(
                f"unsupported step_unit={step_unit!r} (expected '{STEP_UNIT_OPTIMIZER}')"
            )

        ckpt_hashes = ckpt.get("hashes") or {}
        current = self.repro_hashes
        for key in ("model_config", "speaker_map", "subset", "eval_contract"):
            cur_v = current.get(key)
            if cur_v is None:
                continue
            ck_v = ckpt_hashes.get(key)
            if ck_v is None:
                raise ResumeMismatchError(
                    f"resume fail-closed: current {key}={cur_v} but checkpoint "
                    "hashes lack this key"
                )
            if str(ck_v) != str(cur_v):
                raise ResumeMismatchError(
                    f"resume fail-closed: {key} mismatch ckpt={ck_v} current={cur_v}"
                )

        if migrate_legacy and is_main_process():
            print(
                f"[resume] migrated legacy microbatch step={microbatch_step} "
                f"→ optimizer_step={optimizer_step} (grad_accum={self.grad_accum})"
            )
        return step_unit, optimizer_step, microbatch_step

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        path = expand(path)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(ckpt, dict):
            raise ResumeMismatchError(f"checkpoint is not a dict: {path}")
        _, optimizer_step, microbatch_step = self._validate_resume(ckpt)

        sd = ckpt.get("model") or ckpt
        if isinstance(sd, dict) and any(k.startswith("module.") for k in sd):
            sd = {k.removeprefix("module."): v for k, v in sd.items()}
        self._bare().load_state_dict(sd, strict=True)
        if ckpt.get("optimizer") is not None:
            self.opt.load_state_dict(ckpt["optimizer"])

        self.optimizer_step = int(optimizer_step)
        self.microbatch_step = int(microbatch_step)
        # Boundary invariant: never resume mid-accumulation
        self._accum_counter = 0

        best = ckpt.get("best_key")
        if isinstance(best, (list, tuple)) and best:
            self._best_key = tuple(best)
        dl = ckpt.get("dataloader") or {}
        self._epoch = int(dl.get("epoch", 0))
        self._batches_in_epoch = int(dl.get("batches_in_epoch", 0))
        self._rng_at_epoch_start = dl.get("rng_at_epoch_start")
        sched = ckpt.get("schedule") or {}
        if sched.get("total_steps") is not None:
            self._total_steps = int(sched["total_steps"])

        if self._rng_at_epoch_start is not None:
            _restore_rng(self._rng_at_epoch_start)
        else:
            _restore_rng(ckpt.get("rng"))
        self._set_sampler_epoch(self._epoch)
        self._data_iter = iter(self.train_loader)
        for _ in range(self._batches_in_epoch):
            try:
                next(self._data_iter)
            except StopIteration:
                break
        if self._rng_at_epoch_start is not None and ckpt.get("rng") is not None:
            _restore_rng(ckpt["rng"])

        if is_main_process():
            print(
                f"[resume] loaded={path} optimizer_step={self.optimizer_step} "
                f"microbatch_step={self.microbatch_step} "
                f"epoch={self._epoch} batches_in_epoch={self._batches_in_epoch}"
            )
        barrier()
        return ckpt

    def _reset_data_iter(self) -> None:
        self._rng_at_epoch_start = _capture_rng()
        self._set_sampler_epoch(self._epoch)
        self._data_iter = iter(self.train_loader)
        self._batches_in_epoch = 0

    def _next_batch(self) -> dict[str, Any]:
        if self._data_iter is None:
            self._reset_data_iter()
        assert self._data_iter is not None
        while True:
            try:
                batch = next(self._data_iter)
                self._batches_in_epoch += 1
                return batch
            except StopIteration:
                self._epoch += 1
                self._reset_data_iter()

    def _forward_loss(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        out = self.model(
            batch["input_ids"],
            batch["codes"],
            text_mask=batch.get("text_mask"),
            frame_mask=batch.get("frame_mask"),
            speaker_id=batch.get("speaker_id"),
            phoneme_ids=batch.get("phoneme_ids"),
            stress=batch.get("stress"),
            word_bound=batch.get("word_bound"),
            phrase_bound=batch.get("phrase_bound"),
            span_type=batch.get("span_type"),
            compute_loss=True,
            cb0_weight=self.cb0_weight,
            residual_weights=self._residual_w,
            stop_weight=self.stop_weight,
            stop_pos_weight=self.stop_pos_weight,
        )
        assert out.loss is not None
        loss = out.loss["total"]
        sw = batch.get("sample_weight")
        if sw is not None and torch.is_tensor(sw):
            loss = loss * sw.mean()
        out_losses = {
            "total": loss,
            "ce_cb0": out.loss["ce_cb0"],
            "ce_residual": out.loss["ce_residual"],
        }
        if "stop" in out.loss:
            out_losses["stop"] = out.loss["stop"]
        return out_losses

    def _val_ce(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        self._bare().eval()
        totals = {"loss": 0.0, "ce_cb0": 0.0, "ce_residual": 0.0}
        n = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                with self._autocast():
                    losses = self._forward_loss(batch)
                for k in totals:
                    src = "total" if k == "loss" else k
                    totals[k] += float(losses[src].detach().float().cpu())
                n += 1
                if n >= 16:
                    break
        self.model.train()
        if n == 0:
            return {}
        return {k: v / n for k, v in totals.items()}

    def _maybe_fr_select(self, stage: str, step: int) -> None:
        """Free-running checkpoint selection hook (CER when eval fn wired)."""
        if not is_main_process():
            barrier()
            return
        rng_snap = _capture_rng()
        try:
            metrics: dict[str, Any] = dict(self._val_ce())
            # Oracle / S1 often has no val_loader. Fall back to a short teacher-forced
            # CE on the train loader so best.pt tracks overfit instead of freezing at
            # the first FR tick with an empty metric dict.
            if not metrics:
                self._bare().eval()
                totals = {"loss": 0.0, "ce_cb0": 0.0, "ce_residual": 0.0}
                n = 0
                with torch.no_grad():
                    for batch in self.train_loader:
                        batch = {
                            k: v.to(self.device) if torch.is_tensor(v) else v
                            for k, v in batch.items()
                        }
                        with self._autocast():
                            losses = self._forward_loss(batch)
                        for k in totals:
                            src = "total" if k == "loss" else k
                            totals[k] += float(losses[src].detach().float().cpu())
                        n += 1
                        if n >= 4:
                            break
                if n:
                    metrics = {k: v / n for k, v in totals.items()}
                    metrics["tf_source"] = "train_loader_probe"
            if self.fr_eval_fn is not None:
                try:
                    metrics.update(self.fr_eval_fn(self._bare()))
                except Exception as exc:  # noqa: BLE001
                    metrics["fr_error"] = str(exc)
            metrics["step"] = step
            metrics["optimizer_step"] = step
            metrics["stage"] = stage
        finally:
            _restore_rng(rng_snap)
            self.model.train()

        key = fr_selection_key(metrics)
        improved = self._best_key is None or key > self._best_key
        metrics["improved"] = improved
        metrics["selection_key"] = list(key)
        with open(self.run_dir / "val_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        if improved or not self.fr_select:
            self._best_key = key
            self.save_checkpoint(stage, step, metrics=metrics, promote_best=True)
            if metrics.get("cer") is not None:
                (self.run_dir / "best_fr_metrics.json").write_text(
                    json.dumps(metrics, indent=2), encoding="utf-8"
                )
            elif metrics.get("loss") is not None:
                (self.run_dir / "best_metrics.json").write_text(
                    json.dumps(metrics, indent=2), encoding="utf-8"
                )
        else:
            self.save_checkpoint(stage, step, metrics=metrics, promote_best=False)
        barrier()

    def train_steps(
        self,
        steps: int,
        stage: str = "S1",
        *,
        total_steps: int | None = None,
    ) -> dict[str, float]:
        """Train until ``optimizer_step`` reaches ``steps`` (optimizer updates)."""
        self.model.train()
        if total_steps is not None:
            horizon = int(total_steps)
        elif self._total_steps is not None and self.optimizer_step > 0:
            horizon = max(int(self._total_steps), int(steps))
        else:
            horizon = int(steps)
        self._total_steps = horizon
        start = int(self.optimizer_step)
        target = int(steps)
        if start >= target:
            if is_main_process():
                print(
                    f"[train] already at optimizer_step={start} >= target={target}; "
                    "nothing to do"
                )
            return {}

        pbar = tqdm(
            total=target,
            desc=f"discrete-lm-{stage}",
            disable=not is_main_process(),
            initial=start,
        )
        last: dict[str, float] = {}
        self.opt.zero_grad(set_to_none=True)
        assert self._accum_counter == 0

        while self.optimizer_step < target:
            # LR / warmup keyed by the upcoming optimizer update index
            mult = lr_multiplier(
                self.optimizer_step,
                horizon,
                warmup_steps=self.warmup_steps,
                schedule=self.lr_schedule,  # type: ignore[arg-type]
                min_lr_ratio=self.min_lr_ratio,
            )
            set_optimizer_lr(self.opt, self.base_lr, mult)

            losses: dict[str, torch.Tensor] | None = None
            for accum_i in range(self.grad_accum):
                batch = self._next_batch()
                batch = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                is_final = accum_i + 1 >= self.grad_accum
                with self._sync_context(is_final_microbatch=is_final):
                    with self._autocast():
                        losses = self._forward_loss(batch)
                        loss = losses["total"] / self.grad_accum
                    loss.backward()
                self.microbatch_step += 1
                self._accum_counter += 1

            assert losses is not None
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self._bare().parameters(), self.grad_clip
                )
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self._accum_counter = 0
            self.optimizer_step += 1
            pbar.update(1)

            last = {
                "loss": float(losses["total"].detach().float().cpu()),
                "ce_cb0": float(losses["ce_cb0"].detach().float().cpu()),
                "ce_residual": float(losses["ce_residual"].detach().float().cpu()),
                "lr": float(self.opt.param_groups[0]["lr"]),
                "optimizer_step": float(self.optimizer_step),
                "microbatch_step": float(self.microbatch_step),
            }
            if "stop" in losses:
                last["stop"] = float(losses["stop"].detach().float().cpu())
            if is_main_process() and (
                self.optimizer_step % self.log_every == 0
                or self.optimizer_step == start + 1
            ):
                pbar.set_postfix(
                    {
                        "loss": last["loss"],
                        "ce_cb0": last["ce_cb0"],
                        "lr": last["lr"],
                        "mb": int(self.microbatch_step),
                    }
                )

            do_fr = self.fr_every > 0 and self.optimizer_step % self.fr_every == 0
            do_eval = self.eval_every > 0 and self.optimizer_step % self.eval_every == 0
            do_save = self.save_every > 0 and self.optimizer_step % self.save_every == 0
            do_candidate = self.optimizer_step in self.candidate_steps
            if do_fr:
                self._maybe_fr_select(stage, self.optimizer_step)
            elif do_eval or do_save:
                if is_main_process():
                    metrics = dict(self._val_ce()) if do_eval else None
                    self.save_checkpoint(
                        stage,
                        self.optimizer_step,
                        metrics=metrics,
                        promote_best=False,
                    )
                barrier()
            if do_candidate:
                if is_main_process():
                    self.save_checkpoint(
                        stage,
                        self.optimizer_step,
                        metrics=last,
                        promote_best=False,
                        as_candidate=True,
                    )
                barrier()

        pbar.close()
        # Completed whole optimizer updates only — never flush a partial accum.
        assert self._accum_counter == 0

        if is_main_process():
            self.save_checkpoint(
                stage, self.optimizer_step, metrics=last, promote_best=False
            )
        barrier()
        return last


def maybe_apply_text_init(
    model: DiscreteSpeechLM, cfg: dict[str, Any]
) -> TextInitReport | None:
    """Apply Qwen text-init when ``model.text_init.enabled`` is true."""
    tic = cfg.get("model", {}).get("text_init") or cfg.get("text_init") or {}
    if not bool(tic.get("enabled", False)):
        return None
    model_id = str(tic.get("model_id", "Qwen/Qwen3-0.6B"))
    return apply_qwen_text_init(model, model_id=model_id)


def build_model_from_cfg(cfg: dict[str, Any]) -> DiscreteSpeechLM:
    return build_discrete_speech_lm(cfg)


__all__ = [
    "DiscreteLMTrainer",
    "DiscreteSpeechLMConfig",
    "ResumeMismatchError",
    "STEP_UNIT_MICROBATCH",
    "STEP_UNIT_OPTIMIZER",
    "build_model_from_cfg",
    "freeze_disabled_ru_conditioning",
    "fr_selection_key",
    "maybe_apply_text_init",
    "resolve_candidate_steps",
    "resolve_optimizer_step_budget",
    "resolve_repro_hashes",
    "resolve_resume_path",
    "resolve_speaker_map_path",
]
