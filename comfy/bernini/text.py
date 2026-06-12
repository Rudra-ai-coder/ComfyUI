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


def merge_t5_planner(
    t5_embeds: torch.Tensor,
    planner_embeds: Optional[torch.Tensor],
    max_sequence_length: int = 512,
    truncate: bool = True,
) -> torch.Tensor:
    if planner_embeds is not None:
        if planner_embeds.ndim == 2:
            planner_embeds = planner_embeds.unsqueeze(0)
        merged = torch.cat([t5_embeds, planner_embeds], dim=1)
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
