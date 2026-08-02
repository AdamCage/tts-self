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


# --- N2 (Qwen latent backbone + diffusion head) ports of the same gate keys ---


def _n2_batch_to_device(batch: dict[str, Any], device) -> dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def n2_text_shuffle_test(
    model, batch: dict[str, Any], device, *, min_rise: float = 0.20
) -> dict[str, float]:
    """Shuffle ``input_ids`` within batch; diffusion train-loss must rise."""
    model.eval()
    batch = _n2_batch_to_device(batch, device)
    out0 = model(
        batch["input_ids"],
        batch["z_input"],
        batch["z_target"],
        text_mask=batch["text_mask"],
        frame_mask=batch["frame_mask"],
    )
    base = float(out0.loss.item())
    perm = torch.randperm(batch["input_ids"].shape[0], device=device)
    out1 = model(
        batch["input_ids"][perm],
        batch["z_input"],
        batch["z_target"],
        text_mask=batch["text_mask"][perm],
        frame_mask=batch["frame_mask"],
    )
    shuffled = float(out1.loss.item())
    rise = (shuffled - base) / max(base, 1e-8)
    return {
        "base_loss": base,
        "shuffled_loss": shuffled,
        "relative_rise": rise,
        "pass": rise >= min_rise,
    }


@torch.no_grad()
def n2_text_mask_test(
    model, batch: dict[str, Any], device, *, min_rise: float = 0.20
) -> dict[str, float]:
    model.eval()
    batch = _n2_batch_to_device(batch, device)
    out0 = model(
        batch["input_ids"],
        batch["z_input"],
        batch["z_target"],
        text_mask=batch["text_mask"],
        frame_mask=batch["frame_mask"],
    )
    out1 = model(
        batch["input_ids"],
        batch["z_input"],
        batch["z_target"],
        text_mask=batch["text_mask"],
        frame_mask=batch["frame_mask"],
        mask_text=True,
    )
    base = float(out0.loss.item())
    masked = float(out1.loss.item())
    rise = (masked - base) / max(base, 1e-8)
    return {
        "base_loss": base,
        "masked_loss": masked,
        "relative_rise": rise,
        "pass": rise >= min_rise,
    }


@torch.no_grad()
def n2_text_prefix_counterfactual_test(
    model,
    batch: dict[str, Any],
    device,
    *,
    prefix_frames: int = 8,
    min_ratio: float = 3.0,
    max_cont: int = 24,
) -> dict[str, Any]:
    """Same GT latent prefix + two texts → AR continuations (N2 generate path)."""
    model.eval()
    batch = _n2_batch_to_device(batch, device)
    if batch["input_ids"].shape[0] < 2:
        return {"pass": False, "error": "need batch size >= 2", "ratio": 0.0}

    z_gt = batch["z_target"][0:1]
    n_gt = int(batch["frame_mask"][0].sum().item())
    k = min(prefix_frames, n_gt)
    prefix = z_gt[:, :k]
    ids_a, ids_b = batch["input_ids"][0:1], batch["input_ids"][1:2]
    tm_a, tm_b = batch["text_mask"][0:1], batch["text_mask"][1:2]

    def continue_from_prefix(input_ids, text_mask):
        z_list = [prefix[:, j, :] for j in range(k)]
        for _ in range(max_cont):
            T_cur = len(z_list) + 1
            z_input = torch.zeros(
                1, T_cur, model.backbone.latent_dim, device=device, dtype=prefix.dtype
            )
            bos = model.backbone.bos_latent.to(device=device, dtype=z_input.dtype)
            z_input[:, 0] = bos
            for j, zj in enumerate(z_list):
                z_input[:, j + 1] = zj
            frame_mask = torch.ones(1, T_cur, dtype=torch.bool, device=device)
            bb = model.backbone(
                input_ids, z_input, text_mask=text_mask, frame_mask=frame_mask
            )
            from minispeech.models.head_diffusion import sample_ddim

            head_dtype = next(model.head.parameters()).dtype
            h_t = bb.acoustic_hidden[:, -1:, :].to(dtype=head_dtype)
            z_t = sample_ddim(
                model.head,
                h_t,
                steps=min(4, int(model.sample_steps)),
                cfg_scale=float(model.cfg_scale),
                frame_mask=torch.ones(1, 1, dtype=torch.bool, device=device),
            )
            z_list.append(z_t[:, 0, :])
        cont = torch.stack(z_list[k:], dim=1)
        return cont

    cont_a = continue_from_prefix(ids_a, tm_a)
    cont_b = continue_from_prefix(ids_b, tm_b)
    cont_a2 = continue_from_prefix(ids_a, tm_a)
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


@torch.no_grad()
def n2_attention_diagonality(model, batch: dict[str, Any], device) -> float | None:
    """Acoustic→text attention diagonality from the last backbone layer.

    Decision (A5): use the last transformer layer, mean over heads, submatrix
    where queries are acoustic positions and keys are text positions; then reuse
    ``guided_attn.attention_diagonality``. Returns ``None`` if attentions are
    unavailable.
    """
    from minispeech.train.guided_attn import attention_diagonality

    model.eval()
    batch = _n2_batch_to_device(batch, device)
    bb = model.backbone
    if bb.backbone is None:
        return None
    embeds, masks = bb.build_inputs_embeds(
        batch["input_ids"],
        batch["z_input"],
        text_mask=batch["text_mask"],
        frame_mask=batch["frame_mask"],
    )
    attn_hf = masks.seq_mask.long()
    try:
        out = bb.backbone(
            inputs_embeds=embeds,
            attention_mask=attn_hf,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    except Exception:
        return None
    attns = getattr(out, "attentions", None)
    if not attns:
        return None
    # last layer: [B, H, S, S]
    a = attns[-1]
    if a is None:
        return None
    L = int(masks.max_text_len)
    T = int(masks.max_acoustic_len)
    # acoustic queries, text keys → [B, H, T, L]
    sub = a[:, :, L : L + T, :L]
    cross = sub.mean(dim=1)  # [B, T, L]
    return attention_diagonality(cross, batch["frame_mask"], batch["text_mask"])
