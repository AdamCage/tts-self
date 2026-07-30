"""Eval package — import submodules directly to avoid heavy optional deps at import time."""

__all__ = [
    "compute_cer",
    "compute_wer",
    "delta_wer",
    "text_mask_test",
    "text_shuffle_test",
]


def __getattr__(name: str):
    if name in ("compute_cer", "compute_wer", "delta_wer"):
        from minispeech.eval.asr_wer import compute_cer, compute_wer, delta_wer

        return {"compute_cer": compute_cer, "compute_wer": compute_wer, "delta_wer": delta_wer}[name]
    if name in ("text_mask_test", "text_shuffle_test"):
        from minispeech.eval.text_usage import text_mask_test, text_shuffle_test

        return {"text_mask_test": text_mask_test, "text_shuffle_test": text_shuffle_test}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
