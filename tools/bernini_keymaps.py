# SPDX-License-Identifier: Apache-2.0
"""Diffusers WanTransformer3DModel → ComfyUI WanModel key remapping.

Inverse of HuggingFace diffusers scripts/convert_wan_to_diffusers.py,
aligned with Wan2GP models/wan/convert_wan.py rename_key_universal.
"""

import re
from typing import Dict, Optional

_RE_BLOCK = re.compile(r"^blocks\.(\d+)\.")

DIT_STRIP_PREFIXES = (
    "diff_dec.transformer.",
    "diff_dec_low.transformer_2.",
    "transformer.",
    "transformer_2.",
)


def strip_dit_prefix(key: str) -> Optional[str]:
    """Return inner diffusers Wan key, or None if not a DiT tensor key."""
    for prefix in DIT_STRIP_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return None


def _common_block_renames(s: str) -> str:
    s = re.sub(r"blocks\.(\d+)\.attn1\.", r"blocks.\1.self_attn.", s)
    s = re.sub(r"blocks\.(\d+)\.attn2\.", r"blocks.\1.cross_attn.", s)
    s = re.sub(r"(\b[^.\s]*attn[^.\s]*\b.*?\.)to_q\.", r"\1q.", s)
    s = re.sub(r"(\b[^.\s]*attn[^.\s]*\b.*?\.)to_k\.", r"\1k.", s)
    s = re.sub(r"(\b[^.\s]*attn[^.\s]*\b.*?\.)to_v\.", r"\1v.", s)
    s = re.sub(r"(\b[^.\s]*attn[^.\s]*\b.*?\.)to_out\.0\.", r"\1o.", s)
    s = re.sub(r"blocks\.(\d+)\.ffn\.net\.0\.proj\.", r"blocks.\1.ffn.0.", s)
    s = re.sub(r"blocks\.(\d+)\.ffn\.net\.2\.", r"blocks.\1.ffn.2.", s)
    return s


def diffusers_wan_to_comfy(key: str) -> str:
    """Map a diffusers-style Wan transformer key to ComfyUI WanModel naming."""
    s = _common_block_renames(key)

    s = re.sub(r"blocks\.(\d+)\.cross_attn\.add_k_proj\.", r"blocks.\1.cross_attn.k_img.", s)
    s = re.sub(r"blocks\.(\d+)\.cross_attn\.add_v_proj\.", r"blocks.\1.cross_attn.v_img.", s)
    s = re.sub(r"blocks\.(\d+)\.cross_attn\.norm_added_k\.", r"blocks.\1.cross_attn.norm_k_img.", s)

    s = re.sub(r"blocks\.(\d+)\.scale_shift_table$", r"blocks.\1.modulation", s)
    # diffusers norm2 (cross_attn, affine) → Comfy norm3
    s = re.sub(r"blocks\.(\d+)\.norm2\b", r"blocks.\1.norm3", s)

    s = re.sub(r"^condition_embedder\.text_embedder\.linear_1\.", "text_embedding.0.", s)
    s = re.sub(r"^condition_embedder\.text_embedder\.linear_2\.", "text_embedding.2.", s)
    s = re.sub(r"^condition_embedder\.time_embedder\.linear_1\.", "time_embedding.0.", s)
    s = re.sub(r"^condition_embedder\.time_embedder\.linear_2\.", "time_embedding.2.", s)
    s = re.sub(r"^condition_embedder\.time_proj\.", "time_projection.1.", s)

    s = re.sub(r"^condition_embedder\.image_embedder\.norm1\.", "img_emb.proj.0.", s)
    s = re.sub(r"^condition_embedder\.image_embedder\.ff\.net\.0\.proj\.", "img_emb.proj.1.", s)
    s = re.sub(r"^condition_embedder\.image_embedder\.ff\.net\.2\.", "img_emb.proj.3.", s)
    s = re.sub(r"^condition_embedder\.image_embedder\.norm2\.", "img_emb.proj.4.", s)

    s = re.sub(r"^proj_out\.", "head.head.", s)
    if s == "scale_shift_table":
        s = "head.modulation"

    return s


def remap_dit_state_dict(state_dict: Dict[str, object], comfy_prefix: str = "") -> Dict[str, object]:
    """Convert diffusers Wan transformer state dict to Comfy WanModel keys."""
    out: Dict[str, object] = {}
    for key, tensor in state_dict.items():
        new_key = diffusers_wan_to_comfy(key)
        if comfy_prefix:
            new_key = f"{comfy_prefix}{new_key}"
        if new_key in out:
            raise KeyError(f"Duplicate Comfy key after remap: {new_key} (from {key})")
        out[new_key] = tensor
    return out


BUCKET_PLANNER = "planner"
BUCKET_VIT_DECODER = "vit_decoder"
BUCKET_MLLM = "mllm"
BUCKET_HIGH_NOISE = "high_noise"
BUCKET_LOW_NOISE = "low_noise"
# Bundled in joint shard for official diffusers; ComfyUI loads UMT5 via CLIPLoader (wan) from HF t5_text_encoder/
BUCKET_T5_BUNDLED = "t5_bundled"
BUCKET_UNKNOWN = "unknown"

PLANNER_EXACT = frozenset({"mask_tokens"})


def classify_bernini_key(key: str) -> str:
    if key.startswith("t5_text_encoder."):
        return BUCKET_T5_BUNDLED
    if key.startswith("mllm."):
        return BUCKET_MLLM
    if key.startswith("vit_decoder."):
        return BUCKET_VIT_DECODER
    if key.startswith("connector.") or key in PLANNER_EXACT:
        return BUCKET_PLANNER
    if key.startswith("diff_dec_low.transformer_2."):
        return BUCKET_LOW_NOISE
    if key.startswith("diff_dec.transformer."):
        return BUCKET_HIGH_NOISE
    return BUCKET_UNKNOWN


def export_key_for_bucket(key: str, bucket: str) -> str:
    if bucket == BUCKET_MLLM and key.startswith("mllm."):
        return key[len("mllm."):]
    return key
