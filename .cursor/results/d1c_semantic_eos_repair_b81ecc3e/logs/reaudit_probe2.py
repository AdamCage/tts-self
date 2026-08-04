"""Independent adversarial probe, part 2: B5 batched, B7 dynamic resume, B8 manifest.

Run from the worktree root:
    python .cursor/results/d1c_semantic_eos_repair_b81ecc3e/logs/reaudit_probe2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

RESULTS: list[tuple[str, str, str]] = []


def rec(blocker: str, status: str, detail: str) -> None:
    RESULTS.append((blocker, status, detail))
    print(f"REAUDIT {blocker} {status} {detail}", flush=True)


def _semantic_model():
    from minispeech.models.discrete_speech_lm import DiscreteSpeechLM, DiscreteSpeechLMConfig

    cfg = DiscreteSpeechLMConfig.tiny()
    cfg.termination_mode = "semantic_eos_v1"
    cfg.eos_id = cfg.codebook_size
    cfg.codec_vocab_size = cfg.codebook_size
    cfg.semantic_vocab_size = cfg.codebook_size + 1
    return DiscreteSpeechLM(cfg)


# --------------------------------------------------------- B5 batched EOS rows
def b5_batched() -> None:
    m = _semantic_model()
    eos_id = m.eos_id

    class RowAwareHead(torch.nn.Module):
        """EOS for the row flagged in ``self.eos_rows`` (matched on hidden id)."""

        def __init__(self) -> None:
            super().__init__()
            self.call = 0

        def forward(self, h):  # noqa: ANN001, ANN201
            lg = h.new_zeros((*h.shape[:-1], m.semantic_vocab_size))
            if h.shape[0] == 2:
                # full-batch semantic step: row 0 stops immediately, row 1 emits id 7
                lg[0, ..., eos_id] = 1e9
                lg[1, ..., 7] = 1e9
            else:
                # per-row residual call (B=1): never EOS
                lg[..., 7] = 1e9
            return lg

    m.heads.semantic = RowAwareHead()
    out = m.generate(
        torch.tensor([[3, 4, 5], [3, 4, 5]]),
        max_frames=3,
        min_frames=0,
        max_cb0_repeat=100,
        temperature=0.0,
        top_k=0,
    )
    lens = out.lengths.tolist()
    row0_mask = bool(out.frame_mask[0].any()) if out.frame_mask.numel() else False
    reasons = [r["termination_reason"] for r in out.rows]
    ok = lens[0] == 0 and lens[1] > 0 and not row0_mask and reasons[0] == "learned_eos"
    rec(
        "B5",
        "PASS" if ok else "FAIL",
        f"mixed batch: lengths={lens} codes.shape={tuple(out.codes.shape)} "
        f"row0_mask_any={row0_mask} reasons={reasons}",
    )
    # Row-0 codes region inside the padded tensor must be masked, not phantom-real.
    if out.codes.shape[1] > 0:
        row0_padded = out.codes[0].tolist()
        rec(
            "B5",
            "INFO",
            f"row0 padded codes (masked out) first frame={row0_padded[0] if row0_padded else None}",
        )
    # min_frames suppression
    class AlwaysEOS(torch.nn.Module):
        def forward(self, h):  # noqa: ANN001, ANN201
            lg = h.new_full((*h.shape[:-1], m.semantic_vocab_size), -1e9)
            lg[..., eos_id] = 1e9
            return lg

    m2 = _semantic_model()
    m2.heads.semantic = AlwaysEOS()
    out3 = m2.generate(
        torch.tensor([[3, 4, 5]]),
        max_frames=6,
        min_frames=4,
        max_cb0_repeat=100,
        temperature=0.0,
        top_k=0,
    )
    sup = int(out3.rows[0]["early_eos_suppressed_count"])
    rec(
        "B5",
        "PASS" if sup >= 4 else "FAIL",
        f"min_frames=4: early_eos_suppressed_count={sup} len={int(out3.lengths[0])} "
        f"reason={out3.rows[0]['termination_reason']}",
    )


# ---------------------------------------------------------- B7 dynamic resume
def b7_dynamic(tmp: Path) -> None:
    import torch as _t
    from _discrete_lm_trainer_fakes import FakeDiscreteSpeechLM
    from torch.utils.data import DataLoader, Dataset

    from minispeech.data.codec_seq_protocol import collate_codec_seq
    from minispeech.train.discrete_lm_trainer import DiscreteLMTrainer, ResumeMismatchError
    from minispeech.train.distributed import cleanup_distributed

    class DS(Dataset):
        def __init__(self) -> None:
            g = _t.Generator().manual_seed(0)
            self.items = [
                {
                    "utt_id": f"u{i}",
                    "input_ids": _t.randint(1, 60, (4 + i % 3,), generator=g),
                    "codes": _t.randint(0, 32, (5 + i % 4, 16), generator=g),
                    "speaker_id": i % 3,
                }
                for i in range(32)
            ]

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, i: int) -> dict:
            return self.items[i]

    def loader() -> DataLoader:
        return DataLoader(
            DS(), batch_size=2, shuffle=False,
            collate_fn=lambda b: collate_codec_seq(b, pad_token_id=0),
        )

    def cfg_for(run: Path, **extra) -> dict:
        train = {
            "run_dir": str(run), "lr": 1e-3, "weight_decay": 0.0, "grad_accum": 4,
            "grad_clip": 0.0, "warmup_steps": 0, "lr_schedule": "constant",
            "min_lr_ratio": 1.0, "precision": "fp32", "log_every": 100,
            "save_every": 0, "eval_every": 0,
            "fr_checkpoint": {"enabled": False, "every": 0}, "resume": False,
            "ddp": {"find_unused_parameters": False},
            "speaker_map_hash": "spk_v1", "subset_hash": "sub_v1",
            "eval_contract_hash": "ec_v1", "seed": 0,
        }
        train.update(extra)
        return {
            "model": {"d_model": 16, "n_layers": 1, "name": "fake_tiny"},
            "loss": {"cb0_weight": 1.0, "stop_weight": 1.0, "stop_pos_weight": 1.0},
            "train": train,
        }

    def make(run: Path, cfg: dict) -> DiscreteLMTrainer:
        cleanup_distributed()
        _t.manual_seed(0)
        return DiscreteLMTrainer(
            FakeDiscreteSpeechLM(), loader(), None, cfg, _t.device("cpu"),
            enforce_clean_git=False,
        )

    # baseline run with a stamped run_manifest hash
    base = tmp / "b7a"
    cfg = cfg_for(base / "run", run_manifest_hash="rm_a")
    t = make(base, cfg)
    t.train_steps(1, stage="S1")
    ck = base / "run" / "last.pt"
    stored = _t.load(ck, map_location="cpu", weights_only=False)["hashes"]
    rec("B7", "INFO", f"checkpoint hashes keys={sorted(stored)}")
    have_all = all(
        k in stored
        for k in (
            "config", "model_config", "speaker_map", "subset",
            "eval_contract", "recipe", "recipe_source_manifest", "run_manifest",
        )
    )
    rec("B7", "PASS" if have_all else "FAIL",
        f"checkpoint stores all 8 hash keys: {have_all} (run_manifest={stored.get('run_manifest')})")

    # (a) run_manifest mismatch must be refused
    c2 = cfg_for(tmp / "b7b" / "run", run_manifest_hash="rm_b")
    c2["train"]["resume"] = str(ck)
    try:
        make(tmp / "b7b", c2)
        rec("B7", "FAIL", "run_manifest mismatch NOT refused on resume")
    except ResumeMismatchError as exc:
        rec("B7", "PASS", f"run_manifest mismatch refused: {exc!s:.90}")

    # (b) config mismatch must be refused
    c3 = cfg_for(tmp / "b7c" / "run", run_manifest_hash="rm_a", lr=9e-4)
    c3["train"]["lr"] = 9e-4
    c3["train"]["resume"] = str(ck)
    try:
        make(tmp / "b7c", c3)
        rec("B7", "FAIL", "config mismatch NOT refused on resume")
    except ResumeMismatchError as exc:
        rec("B7", "PASS", f"config mismatch refused: {exc!s:.90}")

    # (c) ADVERSARIAL: dropping the hash from the *new* config makes the check vanish
    c4 = cfg_for(tmp / "b7d" / "run")  # no run_manifest_hash at all
    c4["train"]["resume"] = str(ck)
    try:
        make(tmp / "b7d", c4)
        rec(
            "B7",
            "FAIL",
            "fail-OPEN: resume accepted after dropping run_manifest_hash from config "
            "(current=None -> check skipped)",
        )
    except ResumeMismatchError as exc:
        rec("B7", "PASS", f"unstamped run_manifest still refused: {exc!s:.90}")

    # (d) checkpoint missing the key while config stamps it must be refused
    payload = _t.load(ck, map_location="cpu", weights_only=False)
    payload["hashes"] = {k: v for k, v in payload["hashes"].items() if k != "run_manifest"}
    stripped = tmp / "b7e_last.pt"
    stripped.parent.mkdir(parents=True, exist_ok=True)
    _t.save(payload, stripped)
    c5 = cfg_for(tmp / "b7e" / "run", run_manifest_hash="rm_a")
    c5["train"]["resume"] = str(stripped)
    try:
        make(tmp / "b7e", c5)
        rec("B7", "FAIL", "checkpoint lacking run_manifest hash NOT refused")
    except ResumeMismatchError as exc:
        rec("B7", "PASS", f"checkpoint lacking run_manifest refused: {exc!s:.90}")


# -------------------------------------------------------------- B8 manifest
def b8_manifest() -> None:
    from minispeech.data.d1b_oracles import load_eval_contract
    from minispeech.data.discrete_oracles import file_sha256

    contract = load_eval_contract(ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
    manifest = contract.get("recipe_source_manifest") or {}
    if not isinstance(manifest, dict) or not manifest:
        rec("B8", "FAIL", f"recipe_source_manifest unusable: {type(manifest).__name__}")
        return
    mismatches: list[tuple[str, str, str]] = []
    missing: list[str] = []
    for rel, declared in manifest.items():
        p = ROOT / rel
        if not p.is_file():
            missing.append(rel)
            continue
        got = file_sha256(p)
        if got != declared:
            mismatches.append((rel, declared, got))
    rec(
        "B8",
        "PASS" if not mismatches and not missing else "FAIL",
        f"n_locked={len(manifest)} missing={missing} mismatches="
        + json.dumps([[r, d[:12], g[:12]] for r, d, g in mismatches]),
    )
    # recompute the aggregate manifest digest with the freezer's own procedure
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "freeze77", ROOT / "scripts/77_freeze_d1c_oracles.py"
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        fn = None
        for cand in ("source_manifest_sha256", "manifest_sha256", "recipe_source_manifest_sha256"):
            if hasattr(mod, cand):
                fn = getattr(mod, cand)
                break
        if fn is not None:
            got = fn(manifest) if callable(fn) else None
            rec("B8", "INFO", f"freezer-recomputed manifest digest={got}")
        else:
            import hashlib

            canon = json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")).encode("utf-8")
            got = hashlib.sha256(canon).hexdigest()
            declared = contract.get("recipe_source_manifest_sha256")
            rec(
                "B8",
                "PASS" if got == declared else "INFO",
                f"canonical-json manifest digest={got} declared={declared} match={got == declared}",
            )
    except Exception as exc:  # noqa: BLE001
        rec("B8", "INFO", f"manifest digest recompute skipped: {type(exc).__name__}: {exc}")
    # coverage gaps: d1c-critical modules that are NOT hash-locked
    critical = [
        "src/minispeech/data/discrete_oracles.py",
        "src/minispeech/data/d1b_oracles.py",
        "src/minispeech/eval/asr.py",
        "src/minispeech/data/codec_seq_protocol.py",
        "src/minispeech/constants.py",
        "configs/eval_contracts/schemas/discrete_eval_report_v2.schema.json",
        "configs/train_subsets/d1c_dev32.json",
        "configs/asr.yaml",
    ]
    unlocked = [c for c in critical if c not in manifest and (ROOT / c).exists()]
    rec(
        "B8",
        "PASS" if not unlocked else "FAIL",
        f"d1c-critical files NOT in frozen manifest: {unlocked}",
    )


def main() -> int:
    import tempfile

    print(f"ROOT={ROOT}")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for fn, args in ((b5_batched, ()), (b7_dynamic, (tmp,)), (b8_manifest, ())):
            try:
                fn(*args)
            except Exception as exc:  # noqa: BLE001
                import traceback

                rec(fn.__name__, "ERROR", f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
    print("\n=== REAUDIT PROBE2 SUMMARY ===")
    for b, s, d in RESULTS:
        print(f"{s:5s} {b:4s} {d}")
    fails = [r for r in RESULTS if r[1] in ("FAIL", "ERROR")]
    print(f"\nTOTAL checks={len(RESULTS)} fail_or_error={len(fails)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
