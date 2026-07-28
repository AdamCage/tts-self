"""Text-usage diagnostics (S1 gate)."""

from __future__ import annotations

from typing import Any

import torch

from minispeech.train.losses import masked_l1


@torch.no_grad()
def text_shuffle_test(model, batch: dict[str, Any], device) -> dict[str, float]:
    """Shuffle texts within batch; loss must rise if text is used."""
    model.eval()
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    out0 = model(
        batch["z_input"],
        batch["text_hidden"],
        frame_mask=batch["frame_mask"],
        text_mask=batch["text_mask"],
        z_target=batch["z_target"],
    )
    if model.mode == "regression":
        base = masked_l1(out0.z_pred, batch["z_target"], batch["frame_mask"]).item()
    else:
        from minispeech.train.losses import flow_mse

        base = flow_mse(out0.extras["v_pred"], out0.extras["v_target"], batch["frame_mask"]).item()

    perm = torch.randperm(batch["text_hidden"].shape[0], device=device)
    th = batch["text_hidden"][perm]
    tm = batch["text_mask"][perm]
    out1 = model(
        batch["z_input"],
        th,
        frame_mask=batch["frame_mask"],
        text_mask=tm,
        z_target=batch["z_target"],
    )
    if model.mode == "regression":
        shuffled = masked_l1(out1.z_pred, batch["z_target"], batch["frame_mask"]).item()
    else:
        from minispeech.train.losses import flow_mse

        shuffled = flow_mse(out1.extras["v_pred"], out1.extras["v_target"], batch["frame_mask"]).item()

    rise = (shuffled - base) / max(base, 1e-8)
    return {"base_loss": base, "shuffled_loss": shuffled, "relative_rise": rise, "pass": rise >= 0.20}


@torch.no_grad()
def text_mask_test(model, batch: dict[str, Any], device) -> dict[str, float]:
    model.eval()
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    out0 = model(
        batch["z_input"],
        batch["text_hidden"],
        frame_mask=batch["frame_mask"],
        text_mask=batch["text_mask"],
        z_target=batch["z_target"],
    )
    out1 = model(
        batch["z_input"],
        batch["text_hidden"],
        frame_mask=batch["frame_mask"],
        text_mask=batch["text_mask"],
        z_target=batch["z_target"],
        mask_text=True,
    )
    if model.mode == "regression":
        base = masked_l1(out0.z_pred, batch["z_target"], batch["frame_mask"]).item()
        masked = masked_l1(out1.z_pred, batch["z_target"], batch["frame_mask"]).item()
    else:
        from minispeech.train.losses import flow_mse

        base = flow_mse(out0.extras["v_pred"], out0.extras["v_target"], batch["frame_mask"]).item()
        masked = flow_mse(out1.extras["v_pred"], out1.extras["v_target"], batch["frame_mask"]).item()
    rise = (masked - base) / max(base, 1e-8)
    return {"base_loss": base, "masked_loss": masked, "relative_rise": rise, "pass": rise >= 0.20}
