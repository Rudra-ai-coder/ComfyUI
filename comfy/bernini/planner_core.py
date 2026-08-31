# SPDX-License-Identifier: Apache-2.0
"""Bernini semantic planning core (sample_vit_embed + renderer feature extraction)."""

import math

import numpy as np
import torch

from comfy.ldm.bernini.connector import MLPConnector


def format_mllm_inputs_embeds(mllm, input_ids, visual_embeds, visual_input_mask, visual_output_mask):
    """Scatter VIT features into token embeddings (BerniniModel.format_mllm_inputs_embeds)."""
    inputs_embeds = mllm.get_input_embeddings()(input_ids).to(dtype=torch.bfloat16)

    if visual_embeds is not None and visual_embeds.numel() > 0:
        visual_mask = visual_input_mask | visual_output_mask
        n_visual_tokens = visual_mask.sum().long().item()
        n_visual_features = visual_embeds.shape[0]
        if n_visual_tokens != n_visual_features:
            raise ValueError(
                f"Visual tokens/features mismatch: tokens={n_visual_tokens}, features={n_visual_features}"
            )
        visual_mask = visual_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        visual_embeds = visual_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(visual_mask, visual_embeds)
    return inputs_embeds


def post_process_input_embeds(input_embeds, visual_output_mask, mask_tokens, inference=True):
    """Replace target VIT tokens with mask tokens for planning (inference masks all)."""
    target_vit_embed_mask = visual_output_mask.squeeze(0) if visual_output_mask.ndim > 1 else visual_output_mask
    target_vit_embeds = input_embeds[:, target_vit_embed_mask, :]
    target_vit_embeds_gt = target_vit_embeds.clone()
    mask_token = mask_tokens[:, :1]

    if inference:
        all_vit_token_num = int(target_vit_embed_mask.sum().item())
        target_vit_embeds[:, :, :] = mask_token.expand(1, all_vit_token_num, -1)
        input_embeds[:, target_vit_embed_mask, :] = target_vit_embeds
        diff_loss_mask = torch.ones(all_vit_token_num, device=target_vit_embeds.device)
    else:
        all_vit_token_num = int(target_vit_embed_mask.sum().item())
        diff_loss_mask = torch.zeros(all_vit_token_num, device=target_vit_embeds.device)

    return {
        "input_embeds": input_embeds,
        "diff_loss_mask": diff_loss_mask,
        "target_vit_embeds": target_vit_embeds_gt,
    }


def feat_from_planner_to_renderer(hidden_states, visual_output_mask, connector: MLPConnector, inference=True):
    """Project planner hidden states to 4096-d renderer contexts."""
    pred_vit_embed_mask = visual_output_mask.squeeze(0) if visual_output_mask.ndim > 1 else visual_output_mask
    pred_vit_embeds = hidden_states[:, pred_vit_embed_mask, :].clone()
    txt_and_vit_token_mask = visual_output_mask.squeeze(0).logical_not() if visual_output_mask.ndim > 1 else visual_output_mask.logical_not()

    if not inference:
        raise NotImplementedError("Training path not implemented in ComfyUI")

    cond_embed_mask = txt_and_vit_token_mask | pred_vit_embed_mask
    diff_mllm_context_txt_mask = txt_and_vit_token_mask[cond_embed_mask]
    diff_mllm_context_vit_mask = pred_vit_embed_mask[cond_embed_mask]

    connector_param = next(connector.parameters())
    if connector_param.device != hidden_states.device or connector_param.dtype != hidden_states.dtype:
        connector.to(device=hidden_states.device, dtype=hidden_states.dtype)

    diff_mllm_contexts = hidden_states[:, cond_embed_mask, :]
    diff_mllm_contexts = connector.for_gen(diff_mllm_contexts)

    return {
        "diff_mllm_contexts": diff_mllm_contexts,
        "pred_vit_embeds": pred_vit_embeds,
        "diff_mllm_context_txt_mask": diff_mllm_context_txt_mask,
        "diff_mllm_context_vit_mask": diff_mllm_context_vit_mask,
    }


@torch.no_grad()
def sample_vit_decoder(vit_decoder, vit_embed, uncond_vit_embed, imgcond_vit_embed, vit_txt_cfg, sample_steps, vit_img_cfg=None, verbose=False):
    """Flow-match VIT decoder with optional txt/img CFG."""
    dtype = vit_embed.dtype
    if vit_img_cfg is not None and vit_txt_cfg > 1.0:
        vit_embed = torch.cat([vit_embed, uncond_vit_embed, imgcond_vit_embed], dim=1)
    elif vit_txt_cfg > 1.0:
        vit_embed = torch.cat([vit_embed, uncond_vit_embed], dim=1)

    vit_embed = (
        vit_decoder.sample(
            z=vit_embed[0],
            cfg=vit_txt_cfg,
            img_cfg=vit_img_cfg,
            num_inference_steps=sample_steps,
            verbose=verbose,
        )
        .unsqueeze(0)
        .to(dtype)
    )

    if vit_img_cfg is not None and vit_txt_cfg > 1.0:
        vit_embed = vit_embed[:, : vit_embed.shape[1] // 3, :]
    elif vit_txt_cfg > 1.0:
        vit_embed = vit_embed[:, : vit_embed.shape[1] // 2, :]
    return vit_embed


@torch.no_grad()
def sample_vit_embed(
    mllm,
    connector: MLPConnector,
    vit_decoder,
    mask_tokens: torch.Tensor,
    input_embeds,
    position_ids,
    attention_mask_4d,
    visual_output_token_mask,
    uncond_input_embeds,
    uncond_position_ids,
    uncond_attention_mask_4d,
    uncond_visual_output_token_mask,
    imgcond_input_embeds,
    imgcond_position_ids,
    imgcond_attention_mask_4d,
    imgcond_visual_output_token_mask,
    planning_step=25,
    vit_denoising_step=3,
    vit_txt_cfg=1.4,
    vit_img_cfg=1.2,
    feature_type="masked_tgt_embed_with_qwen_txt_vit_tokens",
    verbose=False,
):
    """MaskGIT planning loop ported from BerniniPipeline.sample_vit_embed."""
    device = input_embeds.device

    def _bool_mask(m):
        """Convert (B, L, L) float 0/-inf mask → (B, 1, L, L) bool for HF v5 SDPA.

        HF transformers ≥5 passes attention_mask through create_causal_mask →
        sdpa_mask → and_mask which does `causal_bool & our_mask`.  Bitwise AND
        is only defined for bool tensors, so a float mask raises:
          NotImplementedError: "bitwise_and_cuda" not implemented for 'Float'.
        Converting to bool (True = attend, False = masked) fixes this; PyTorch
        SDPA then treats True positions as attended.
        """
        if m is None:
            return None
        b = (m > float("-inf"))       # (B, L, L) bool
        if b.dim() == 3:
            b = b.unsqueeze(1)         # (B, 1, L, L)
        return b

    def mask_ratio_generator_infer(step, totals):
        return np.cos(math.pi / 2.0 * (step + 1) / totals)

    n_query_tokens = int(visual_output_token_mask.sum().item())
    order = np.array(list(range(n_query_tokens)))
    np.random.shuffle(order)
    order = torch.tensor(order, device=device, dtype=torch.long)
    mask = torch.ones(n_query_tokens, device=device)

    if position_ids.shape[1] == 3:
        position_ids = position_ids.transpose(0, 1).contiguous()
    if uncond_position_ids.shape[1] == 3:
        uncond_position_ids = uncond_position_ids.transpose(0, 1).contiguous()
    if imgcond_position_ids.shape[1] == 3:
        imgcond_position_ids = imgcond_position_ids.transpose(0, 1).contiguous()

    if vit_decoder is not None:
        for step in range(planning_step):
            if connector is not None:
                connector_param = next(connector.parameters())
                if connector_param.device != input_embeds.device or connector_param.dtype != input_embeds.dtype:
                    connector.to(device=input_embeds.device, dtype=input_embeds.dtype)

            hidden_state = mllm(
                inputs_embeds=input_embeds.clone(),
                position_ids=position_ids.clone(),
                attention_mask=_bool_mask(attention_mask_4d),
                output_hidden_states=True,
            ).hidden_states[-2]
            uncond_hidden_state = mllm(
                inputs_embeds=uncond_input_embeds.clone(),
                position_ids=uncond_position_ids.clone(),
                attention_mask=_bool_mask(uncond_attention_mask_4d),
                output_hidden_states=True,
            ).hidden_states[-2]
            imgcond_hidden_state = mllm(
                inputs_embeds=imgcond_input_embeds.clone(),
                position_ids=imgcond_position_ids.clone(),
                attention_mask=_bool_mask(imgcond_attention_mask_4d),
                output_hidden_states=True,
            ).hidden_states[-2]

            cond_vit_embed = hidden_state[:, visual_output_token_mask, :]
            uncond_vit_embed = uncond_hidden_state[:, uncond_visual_output_token_mask, :]
            imgcond_vit_embed = imgcond_hidden_state[:, imgcond_visual_output_token_mask, :]
            pred_vit_embed_mllm = connector.for_vit(cond_vit_embed)
            uncond_pred_vit_embed_mllm = connector.for_vit(uncond_vit_embed)
            imgcond_pred_vit_embed_mllm = connector.for_vit(imgcond_vit_embed)

            mask_ratio = mask_ratio_generator_infer(step, planning_step)
            mask_len = torch.tensor([np.floor(n_query_tokens * mask_ratio)], device=device)
            mask_len = torch.maximum(
                torch.tensor([1.0], device=device),
                torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len),
            )
            mask_next = torch.zeros_like(mask)
            mask_next = torch.scatter(
                mask_next,
                dim=-1,
                index=order[: mask_len.long()],
                src=torch.ones_like(mask),
            ).bool()
            if step >= planning_step - 1:
                mask_to_pred = mask.bool()
            else:
                mask_to_pred = torch.logical_xor(mask.bool(), mask_next)
            mask = mask_next

            if mask_to_pred.nonzero(as_tuple=True)[0].sum() == 0:
                continue

            pred_idx = mask_to_pred.nonzero(as_tuple=True)[0]
            cond_pred_vit_embed = pred_vit_embed_mllm[:, pred_idx]
            uncond_pred_vit_embed = uncond_pred_vit_embed_mllm[:, pred_idx]
            imgcond_pred_vit_embed = imgcond_pred_vit_embed_mllm[:, pred_idx]
            cur_pred_vit_embed = sample_vit_decoder(
                vit_decoder,
                vit_embed=cond_pred_vit_embed,
                uncond_vit_embed=uncond_pred_vit_embed,
                imgcond_vit_embed=imgcond_pred_vit_embed,
                vit_txt_cfg=vit_txt_cfg,
                vit_img_cfg=vit_img_cfg,
                sample_steps=vit_denoising_step,
                verbose=verbose,
            )

            all_target_vit_embed = input_embeds[:, visual_output_token_mask, :]
            all_target_vit_embed[:, pred_idx] = cur_pred_vit_embed
            input_embeds[:, visual_output_token_mask] = all_target_vit_embed
            uncond_input_embeds[:, uncond_visual_output_token_mask] = all_target_vit_embed
            imgcond_input_embeds[:, imgcond_visual_output_token_mask] = all_target_vit_embed

    pred_vit_embed_diff = input_embeds[:, visual_output_token_mask, :]

    outputs = mllm(
        inputs_embeds=input_embeds.clone(),
        position_ids=position_ids.clone(),
        attention_mask=_bool_mask(attention_mask_4d),
        output_hidden_states=True,
    )
    uncond_outputs = mllm(
        inputs_embeds=uncond_input_embeds.clone(),
        position_ids=uncond_position_ids.clone(),
        attention_mask=_bool_mask(uncond_attention_mask_4d),
        output_hidden_states=True,
    )

    cond_outputs = feat_from_planner_to_renderer(
        outputs.hidden_states[-2],
        visual_output_token_mask,
        connector,
        inference=True,
    )
    uncond_out = feat_from_planner_to_renderer(
        uncond_outputs.hidden_states[-2],
        uncond_visual_output_token_mask,
        connector,
        inference=True,
    )

    if feature_type in ["masked_tgt_embed_with_qwen_txt_tokens"]:
        cond_embeds_wtxt_wvit = cond_outputs["diff_mllm_contexts"]
        cond_embeds_wtxt_wovit = None
        cond_embeds_wotxt_wvit = None
        cond_embeds_wotxt_wovit = uncond_out["diff_mllm_contexts"]
    else:
        uncond_cond_embeds = uncond_out["diff_mllm_contexts"]
        diff_mllm_context_txt_mask = uncond_out["diff_mllm_context_txt_mask"]
        cond_embeds_wotxt_wovit = uncond_cond_embeds[:, diff_mllm_context_txt_mask]
        diff_mllm_context_txt_mask = cond_outputs["diff_mllm_context_txt_mask"]
        diff_mllm_context_vit_mask = cond_outputs["diff_mllm_context_vit_mask"]
        cond_embeds_wtxt_wvit = cond_outputs["diff_mllm_contexts"]
        cond_embeds_wtxt_wovit = cond_embeds_wtxt_wvit[:, diff_mllm_context_txt_mask]
        cond_embeds_wotxt_wvit = cond_embeds_wtxt_wvit[:, diff_mllm_context_vit_mask]

    return {
        "cond_embeds_wtxt_wvit": cond_embeds_wtxt_wvit,
        "cond_embeds_wtxt_wovit": cond_embeds_wtxt_wovit,
        "cond_embeds_wotxt_wvit": cond_embeds_wotxt_wvit,
        "cond_embeds_wotxt_wovit": cond_embeds_wotxt_wovit,
        "pred_vit_embed": pred_vit_embed_diff,
    }
