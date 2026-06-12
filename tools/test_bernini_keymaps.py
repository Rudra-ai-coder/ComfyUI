#!/usr/bin/env python3
"""Unit tests for bernini_keymaps (no shard files required)."""

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS))

from bernini_keymaps import (
    BUCKET_HIGH_NOISE,
    BUCKET_LOW_NOISE,
    BUCKET_MLLM,
    BUCKET_PLANNER,
    BUCKET_T5_BUNDLED,
    BUCKET_VIT_DECODER,
    classify_bernini_key,
    diffusers_wan_to_comfy,
    export_key_for_bucket,
    strip_dit_prefix,
)


def test_classify():
    assert classify_bernini_key("connector.proj_gen.0.weight") == BUCKET_PLANNER
    assert classify_bernini_key("mask_tokens") == BUCKET_PLANNER
    assert classify_bernini_key("vit_decoder.net.input_proj.weight") == BUCKET_VIT_DECODER
    assert classify_bernini_key("mllm.model.layers.0.self_attn.q_proj.weight") == BUCKET_MLLM
    assert classify_bernini_key("diff_dec.transformer.patch_embedding.weight") == BUCKET_HIGH_NOISE
    assert classify_bernini_key("diff_dec_low.transformer_2.patch_embedding.weight") == BUCKET_LOW_NOISE
    assert classify_bernini_key("t5_text_encoder.encoder.block.0.layer.0.SelfAttention.q.weight") == BUCKET_T5_BUNDLED


def test_strip_and_remap():
    src = "diff_dec.transformer.blocks.0.attn1.to_q.weight"
    inner = strip_dit_prefix(src)
    assert inner == "blocks.0.attn1.to_q.weight"
    assert diffusers_wan_to_comfy(inner) == "blocks.0.self_attn.q.weight"

    src2 = "diff_dec.transformer.condition_embedder.text_embedder.linear_1.weight"
    inner2 = strip_dit_prefix(src2)
    assert diffusers_wan_to_comfy(inner2) == "text_embedding.0.weight"

    src3 = "diff_dec.transformer.scale_shift_table"
    assert diffusers_wan_to_comfy(strip_dit_prefix(src3)) == "head.modulation"

    src4 = "diff_dec.transformer.proj_out.weight"
    assert diffusers_wan_to_comfy(strip_dit_prefix(src4)) == "head.head.weight"


def test_mllm_export():
    assert export_key_for_bucket("mllm.model.norm.weight", BUCKET_MLLM) == "model.norm.weight"
    assert export_key_for_bucket("mllm.visual.patch_embed.proj.weight", BUCKET_MLLM) == "model.visual.patch_embed.proj.weight"


def main():
    test_classify()
    test_strip_and_remap()
    test_mllm_export()
    print("All bernini_keymaps tests passed.")


if __name__ == "__main__":
    main()
