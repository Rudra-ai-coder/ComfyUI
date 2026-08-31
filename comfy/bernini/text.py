# SPDX-License-Identifier: Apache-2.0
"""Bernini T5 + planner text merge helpers (pipeline.py ~1083-1115)."""

from typing import Optional

import torch

TEXT_BRANCH_KEYS = {
    "wtxt_wvit": None,
    "wtxt_wovit": "bernini_text_wtxt_wovit",
    "wotxt_wvit": "bernini_text_wotxt_wvit",
    "wotxt_wovit": "bernini_text_wotxt_wovit",
}


def pad_and_truncate_feat(
    feat: Optional[torch.Tensor],
    max_sequence_length: int = 512,
    truncate: bool = True,
) -> Optional[torch.Tensor]:
    if feat is None:
        return None
    if feat.ndim == 2:
        feat = feat.unsqueeze(0)
    if feat.shape[1] < max_sequence_length:
        feat = torch.cat(
            [feat, feat.new_zeros((feat.shape[0], max_sequence_length - feat.shape[1], feat.shape[-1]))],
            dim=1,
        )
    if truncate and feat.shape[1] > max_sequence_length:
        feat = feat[:, :max_sequence_length, :]
    return feat


def _trim_t5_padding(embeds: torch.Tensor) -> torch.Tensor:
    """Trim zero-padding rows added by ComfyUI's Wan tokenizer (min_length=512, zero_out_masked=True).

    The original Bernini pipeline receives variable-length T5 embeddings (actual token
    count, no padding).  ComfyUI's Wan T5 pre-pads to 512 with zero vectors.  When
    planner embeddings are appended before truncating to 512, those zero rows fill the
    budget and the planner tokens are cut off.  Trimming to the last non-zero row
    restores the original behaviour: T5 actual tokens + planner tokens → pad/truncate
    to 512.
    """
    if embeds.ndim == 2:
        embeds = embeds.unsqueeze(0)
    # Zero-padded positions have exactly zero L2 norm (zero_out_masked=True).
    norms = embeds.norm(dim=-1).squeeze(0)  # [seq_len]
    nonzero = (norms > 1e-6).nonzero(as_tuple=False)
    if nonzero.numel() == 0:
        return embeds[:, :1, :]  # degenerate prompt: keep at least BOS token
    actual_len = int(nonzero[-1].item()) + 1
    return embeds[:, :actual_len, :]


def merge_t5_planner(
    t5_embeds: torch.Tensor,
    planner_embeds: Optional[torch.Tensor],
    max_sequence_length: int = 512,
    truncate: bool = True,
) -> torch.Tensor:
    if planner_embeds is not None:
        if planner_embeds.ndim == 2:
            planner_embeds = planner_embeds.unsqueeze(0)
        # Trim ComfyUI's 512-padded T5 to actual token length so the planner
        # embeddings are not pushed out by zero-padding when truncated to 512.
        t5_trimmed = _trim_t5_padding(t5_embeds)
        merged = torch.cat([t5_trimmed, planner_embeds], dim=1)
    else:
        merged = t5_embeds
    return pad_and_truncate_feat(merged, max_sequence_length, truncate)


def extract_cross_attn(conditioning) -> Optional[torch.Tensor]:
    """First cross-attn tensor from node-time or sampler-time conditioning."""
    if conditioning is None or len(conditioning) == 0:
        return None
    entry = conditioning[0]
    if isinstance(entry, dict):
        return entry.get("cross_attn")
    return entry[0]


def set_merged_conditioning(conditioning, cross_attn_tensor, extra_pooled=None):
    """Replace cross-attn tensor and optional pooled keys for Bernini branches."""
    if conditioning is None:
        return None
    out = []
    if isinstance(conditioning[0], dict):
        for entry in conditioning:
            p = entry.copy()
            p["cross_attn"] = cross_attn_tensor
            if extra_pooled:
                p.update(extra_pooled)
            out.append(p)
        return out
    for _t, pooled in conditioning:
        p = pooled.copy()
        if extra_pooled:
            p.update(extra_pooled)
        out.append([cross_attn_tensor, p])
    return out
