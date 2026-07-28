"""Text-usage diagnostics (S1/S2 gates)."""

from __future__ import annotations

from typing import Any

import torch

from minispeech.train.losses import masked_l1


@torch.no_grad()
def text_shuffle_test(
    model, batch: dict[str, Any], device, *, min_rise: float = 0.20
) -> dict[str, float]:
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
    return {"base_loss": base, "shuffled_loss": shuffled, "relative_rise": rise, "pass": rise >= min_rise}


@torch.no_grad()
def text_mask_test(
    model, batch: dict[str, Any], device, *, min_rise: float = 0.20
) -> dict[str, float]:
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
    return {"base_loss": base, "masked_loss": masked, "relative_rise": rise, "pass": rise >= min_rise}


@torch.no_grad()
def text_prefix_counterfactual_test(
    model,
    batch: dict[str, Any],
    device,
    *,
    prefix_frames: int = 8,
    min_ratio: float = 3.0,
    max_cont: int = 24,
) -> dict[str, Any]:
    """
    Same acoustic GT prefix + two different texts → free-running continuations.
    Pass if L1(cont_a, cont_b) / L1(cont_a, cont_a_repeat) ≥ min_ratio.
    """
    model.eval()
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    if batch["text_hidden"].shape[0] < 2:
        return {"pass": False, "error": "need batch size >= 2", "ratio": 0.0}

    z_gt = batch["z_target"][0:1]
    k = min(prefix_frames, int(z_gt.shape[1]))
    prefix = z_gt[:, :k]
    th_a, th_b = batch["text_hidden"][0:1], batch["text_hidden"][1:2]
    tm_a, tm_b = batch["text_mask"][0:1], batch["text_mask"][1:2]

    def continue_from_prefix(th, tm):
        text_memory = model.encode_text(th)
        past = None
        tout, past = model.talker.forward_step(
            model.talker.bos.view(1, 1, -1).to(device=device, dtype=th.dtype),
            text_memory,
            past,
            text_mask=tm,
            position=0,
        )
        for j in range(k):
            z_in = prefix[:, j : j + 1]
            tout, past = model.talker.forward_step(
                z_in, text_memory, past, text_mask=tm, position=j + 1
            )
        # Hidden after GT prefix z_{k-1}; AR-predict continuations
        h_t = tout.hidden
        z_prev = prefix[:, -1:]
        cont = []
        for j in range(max_cont):
            if model.mode == "regression":
                z_t = model.head(h_t, z_prev=z_prev)
            else:
                from minispeech.models.heads_flow import sample_rectified_flow

                z_t = sample_rectified_flow(model.head, h_t, steps=4, method="heun", cfg_scale=1.0)
            cont.append(z_t)
            z_prev = z_t
            tout, past = model.talker.forward_step(
                z_t, text_memory, past, text_mask=tm, position=k + 1 + j
            )
            h_t = tout.hidden
        return torch.cat(cont, dim=1)

    cont_a = continue_from_prefix(th_a, tm_a)
    cont_b = continue_from_prefix(th_b, tm_b)
    cont_a2 = continue_from_prefix(th_a, tm_a)

    div_ab = (cont_a - cont_b).abs().mean().item()
    div_ctrl = (cont_a - cont_a2).abs().mean().item() + 1e-8
    ratio = div_ab / div_ctrl
    return {
        "div_ab": div_ab,
        "div_control": div_ctrl,
        "ratio": ratio,
        "prefix_frames": float(k),
        "pass": bool(ratio >= min_ratio),
    }
