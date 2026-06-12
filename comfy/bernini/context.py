# SPDX-License-Identifier: Apache-2.0
"""Bernini in-context latent branch helpers for chained guidance."""

from typing import List, Optional, Tuple

import comfy.conds
import node_helpers

from .text import TEXT_BRANCH_KEYS


def _iter_pooled(conditioning):
    if conditioning is None:
        return
    if len(conditioning) == 0:
        return
    if isinstance(conditioning[0], dict):
        for entry in conditioning:
            yield entry.get("cross_attn"), entry
        return
    for t, pooled in conditioning:
        yield t, pooled


def get_pooled_value(conditioning, key, default=None):
    for _, pooled in _iter_pooled(conditioning):
        if key in pooled:
            return pooled[key]
    return default


def get_context_latents(conditioning) -> Optional[list]:
    ctx = get_pooled_value(conditioning, "context_latents", None)
    if ctx is not None:
        return list(ctx)
    if conditioning and isinstance(conditioning[0], dict):
        mc = conditioning[0].get("model_conds", {})
        wrapped = mc.get("context_latents")
        if wrapped is not None and hasattr(wrapped, "cond"):
            return list(wrapped.cond)
    return None


def make_source_ids(num_sources: int, max_trained_src_id: int = 5, interpolate: bool = True) -> List[float]:
    if num_sources <= 0:
        return []
    if interpolate and num_sources > max_trained_src_id:
        import torch
        return torch.linspace(1.0, float(max_trained_src_id), num_sources).tolist()
    return [float(i) for i in range(1, num_sources + 1)]


def strip_context_latents(conditioning):
    if conditioning is None:
        return None
    if isinstance(conditioning[0], dict):
        out = []
        for entry in conditioning:
            c = entry.copy()
            c.pop("context_latents", None)
            mc = dict(c.get("model_conds", {}))
            mc.pop("context_latents", None)
            c["model_conds"] = mc
            out.append(c)
        return out
    out = []
    for t, pooled in conditioning:
        p = {k: v for k, v in pooled.items() if k != "context_latents"}
        out.append([t, p])
    return out


def apply_context_latents(conditioning, context_latents: Optional[list]):
    if conditioning is None:
        return None
    if isinstance(conditioning[0], dict):
        return build_branch_cond_list(conditioning, context_latents)
    if context_latents is None or len(context_latents) == 0:
        return strip_context_latents(conditioning)
    return node_helpers.conditioning_set_values(conditioning, {"context_latents": context_latents})


def split_context_branches(
    context_latents: Optional[list],
    num_videos: int,
) -> Tuple[Optional[list], Optional[list], Optional[list]]:
    """Return (none, video_only, all) context lists for guidance branches."""
    if context_latents is None or len(context_latents) == 0:
        return None, None, None
    all_ctx = list(context_latents)
    video_ctx = all_ctx[:num_videos] if num_videos > 0 else []
    return None, video_ctx if video_ctx else None, all_ctx


def get_branch_cross_attn(cond_list, branch: str):
    if not cond_list:
        return None
    key = TEXT_BRANCH_KEYS.get(branch)
    if key:
        for entry in cond_list:
            val = entry.get(key)
            if val is not None:
                return val.cond if hasattr(val, "cond") else val
    entry = cond_list[0]
    cross = entry.get("cross_attn")
    if cross is not None:
        return cross.cond if hasattr(cross, "cond") else cross
    return None


def build_branch_cond_list(cond_list, context_latents: Optional[list], cross_attn=None):
    if cond_list is None:
        return None
    out = []
    for entry in cond_list:
        bc = entry.copy()
        mc = dict(bc.get("model_conds", {}))
        if context_latents is None or len(context_latents) == 0:
            mc.pop("context_latents", None)
            bc.pop("context_latents", None)
        else:
            mc["context_latents"] = comfy.conds.CONDList(list(context_latents))
        if cross_attn is not None:
            bc["cross_attn"] = cross_attn
            if "c_crossattn" in mc:
                mc["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)
        bc["model_conds"] = mc
        out.append(bc)
    return out
