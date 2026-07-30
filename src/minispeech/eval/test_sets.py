"""Frozen mel eval test-set loading (phrases / utt_ids / YAML suite)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from minispeech.utils import expand, load_yaml

TestSetKind = Literal["phrases", "utt_ids"]


@dataclass(frozen=True)
class EvalItem:
    utt_id: str
    text: str
    note: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalTestSet:
    id: str
    kind: TestSetKind
    items: tuple[EvalItem, ...]
    description: str = ""
    path: str | None = None

    @property
    def n(self) -> int:
        return len(self.items)

    def texts(self) -> list[str]:
        return [i.text for i in self.items]

    def utt_ids(self) -> list[str]:
        return [i.utt_id for i in self.items]


def _as_items_phrases(raw: dict[str, Any]) -> list[EvalItem]:
    items: list[EvalItem] = []
    for i, row in enumerate(raw.get("items") or []):
        text = str(row.get("text") or "").strip()
        if not text:
            raise ValueError(f"phrases item {i} missing text")
        utt_id = str(row.get("utt_id") or f"{raw.get('id', 'set')}_{i:02d}")
        items.append(
            EvalItem(
                utt_id=utt_id,
                text=text,
                note=str(row.get("note") or ""),
                meta={k: v for k, v in row.items() if k not in {"utt_id", "text", "note"}},
            )
        )
    return items


def _as_items_utt_ids(raw: dict[str, Any]) -> list[EvalItem]:
    items: list[EvalItem] = []
    # Prefer explicit items with text; else bare utt_ids (text filled later from manifest)
    if raw.get("items"):
        for i, row in enumerate(raw["items"]):
            utt_id = str(row.get("utt_id") or "")
            if not utt_id:
                raise ValueError(f"utt_ids item {i} missing utt_id")
            items.append(
                EvalItem(
                    utt_id=utt_id,
                    text=str(row.get("text") or ""),
                    note=str(row.get("note") or ""),
                )
            )
        return items
    for i, utt_id in enumerate(raw.get("utt_ids") or []):
        items.append(EvalItem(utt_id=str(utt_id), text=""))
    return items


def load_test_set(path: str | Path) -> EvalTestSet:
    """Load a frozen JSON test set (``kind``: phrases | utt_ids)."""
    p = expand(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    kind = str(raw.get("kind") or "phrases")
    if kind not in ("phrases", "utt_ids"):
        raise ValueError(f"unsupported test set kind {kind!r} in {p}")
    if kind == "phrases":
        items = _as_items_phrases(raw)
    else:
        items = _as_items_utt_ids(raw)
    return EvalTestSet(
        id=str(raw.get("id") or p.stem),
        kind=kind,  # type: ignore[arg-type]
        items=tuple(items),
        description=str(raw.get("description") or ""),
        path=str(p),
    )


def load_suite_config(path: str | Path = "configs/eval_mel_suite.yaml") -> dict[str, Any]:
    cfg = load_yaml(expand(path))
    return cfg if isinstance(cfg, dict) else {}


def list_suite_test_sets(suite_cfg: dict[str, Any] | None = None) -> dict[str, Path]:
    """Return ``{set_id: path}`` from suite YAML."""
    cfg = suite_cfg if suite_cfg is not None else load_suite_config()
    raw = cfg.get("test_sets") or {}
    out: dict[str, Path] = {}
    for k, v in raw.items():
        out[str(k)] = expand(v)
    return out


def load_suite_test_sets(
    suite_cfg: dict[str, Any] | None = None,
    *,
    only: list[str] | None = None,
) -> dict[str, EvalTestSet]:
    paths = list_suite_test_sets(suite_cfg)
    if only:
        paths = {k: v for k, v in paths.items() if k in only}
    return {k: load_test_set(v) for k, v in paths.items()}


def aggregate_cer_wer(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Aggregate per-utt CER/WER rows into mean/median/n (CPU-only)."""
    cers = [float(r["cer"]) for r in rows if r.get("cer") is not None]
    wers = [float(r["wer"]) for r in rows if r.get("wer") is not None]
    if not cers:
        return {
            "n": 0,
            "mean_cer": None,
            "median_cer": None,
            "mean_wer": None,
            "median_wer": None,
        }
    cers_sorted = sorted(cers)
    wers_sorted = sorted(wers) if wers else []
    mid = len(cers_sorted) // 2
    if len(cers_sorted) % 2:
        med_cer = cers_sorted[mid]
    else:
        med_cer = 0.5 * (cers_sorted[mid - 1] + cers_sorted[mid])
    med_wer: float | None
    if wers_sorted:
        midw = len(wers_sorted) // 2
        if len(wers_sorted) % 2:
            med_wer = wers_sorted[midw]
        else:
            med_wer = 0.5 * (wers_sorted[midw - 1] + wers_sorted[midw])
    else:
        med_wer = None
    return {
        "n": len(cers),
        "mean_cer": float(sum(cers) / len(cers)),
        "median_cer": float(med_cer),
        "mean_wer": float(sum(wers) / len(wers)) if wers else None,
        "median_wer": med_wer,
    }
