from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch

from minispeech.data.manifest import Utterance, write_manifest
from minispeech.eval.test_sets import EvalItem, EvalTestSet

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "external_baseline_eval",
    ROOT / "scripts" / "39_external_baseline_eval.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_silero_text_translates_canonical_stress() -> None:
    assert MODULE._silero_text("Открыть замо́к и осмотреть за́мок.") == (
        "Открыть зам+ок и осмотреть з+амок."
    )


def test_synth_enables_silero_russian_frontend() -> None:
    class FakeModel:
        kwargs: dict[str, Any]

        def apply_tts(self, **kwargs: Any) -> np.ndarray:
            self.kwargs = kwargs
            return np.ones(16, dtype=np.float32)

    model = FakeModel()
    waveform = MODULE._synth_one(
        model,
        "з+амок",
        voice="xenia",
        sample_rate=24_000,
    )
    assert waveform.shape == (16,)
    assert model.kwargs["put_accent"] is True
    assert model.kwargs["put_yo"] is True


def test_synth_retries_unsupported_native_text_with_shared_frontend() -> None:
    class FakeModel:
        inputs: list[str]

        def __init__(self) -> None:
            self.inputs = []

        def apply_tts(self, **kwargs: Any) -> np.ndarray:
            text = str(kwargs["text"])
            self.inputs.append(text)
            if any("A" <= char <= "Z" or "a" <= char <= "z" for char in text):
                raise ValueError
            return np.ones(16, dtype=np.float32)

    model = FakeModel()
    waveform, model_input, primary_error = MODULE._synth_with_frontend_fallback(
        model,
        "Visa 4555 4412 3456 7890.",
        voice="xenia",
        sample_rate=24_000,
        input_mode="raw",
    )

    assert waveform.shape == (16,)
    assert primary_error == "ValueError: "
    assert model_input == model.inputs[-1]
    assert not any("A" <= char <= "Z" or "a" <= char <= "z" for char in model_input)


def test_input_mode_frontend_always_verbalizes() -> None:
    class FakeModel:
        inputs: list[str]

        def __init__(self) -> None:
            self.inputs = []

        def apply_tts(self, **kwargs: Any) -> np.ndarray:
            self.inputs.append(str(kwargs["text"]))
            return np.ones(8, dtype=np.float32)

    model = FakeModel()
    _, model_input, err = MODULE._synth_with_frontend_fallback(
        model,
        "Открытие в 09:05",
        voice="xenia",
        sample_rate=24_000,
        input_mode="frontend",
    )
    assert err is None
    assert model_input == model.inputs[0]
    assert "09:05" not in model_input
    assert "ноль" in model_input


def test_load_silero_accepts_packaged_model_without_eval(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    class PackagedModel:
        device: str | None = None

        def to(self, device: str) -> None:
            self.device = device

    model = PackagedModel()
    repo = tmp_path / "silero"
    repo.mkdir()
    (repo / "hubconf.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(torch.hub, "load", lambda **_: (model, "example"))

    loaded, info = MODULE._load_silero(
        local_repo=str(repo),
        allow_download=False,
        model_id="v4_ru",
        device="cpu",
    )

    assert loaded is model
    assert model.device == "cpu"
    assert info["status"] == "ok"


def test_fill_utt_texts_uses_manifest_schema(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest_path,
        [
            Utterance(
                utt_id="utt_1",
                wav="unused.wav",
                text_raw="сырой текст",
                text_stressed="те́кст",
            )
        ],
    )
    test_set = EvalTestSet(
        id="val",
        kind="utt_ids",
        items=(EvalItem(utt_id="utt_1", text=""),),
    )

    filled = MODULE._fill_utt_texts(test_set, str(manifest_path))

    assert filled.items[0].text == "те́кст"
