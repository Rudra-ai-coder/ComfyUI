#!/usr/bin/env python3
"""Unit tests for comfy.bernini guidance + context + text merge (no GPU)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from comfy.bernini.context import make_source_ids, split_context_branches, build_branch_cond_list, get_branch_cross_attn
from comfy.bernini.guidance import (
    apg_delta,
    chained_cfg_rv2v,
    normalized_guidance,
    vae_txt_vit_wapg,
    vae_txt_vit_wapg_from_x0,
    velocity_to_x0,
    x0_to_velocity,
)
from comfy.bernini.text import merge_t5_planner, pad_and_truncate_feat


def test_split_context():
    ctx = [torch.zeros(1), torch.ones(1)]
    none, vid, all_ctx = split_context_branches(ctx, num_videos=1)
    assert none is None
    assert len(vid) == 1
    assert len(all_ctx) == 2
    none2, vid2, all2 = split_context_branches(ctx, num_videos=0)
    assert vid2 is None


def test_chained_cfg():
    a = torch.tensor(1.0)
    b = torch.tensor(2.0)
    c = torch.tensor(3.0)
    d = torch.tensor(4.0)
    out = chained_cfg_rv2v(a, b, c, d, 1.0, 1.0, 1.0)
    assert out.item() == 4.0


def test_apg():
    cond = torch.ones(1, 4, 8, 8)
    uncond = torch.zeros(1, 4, 8, 8)
    out = normalized_guidance(cond, uncond, 2.0, eta=1.0, norm_threshold=0.0)
    assert out.shape == cond.shape


def test_apg_delta():
    delta = torch.tensor([[1.0, 2.0]])
    ref = torch.tensor([[1.0, 0.0]])
    out = apg_delta(delta, ref, parallel_scale=0.2, orthogonal_scale=1.0)
    assert out.shape == delta.shape


def test_vae_txt_vit_wapg():
    base = torch.zeros(1, 2, 2)
    img = torch.ones(1, 2, 2)
    txt = torch.full((1, 2, 2), 2.0)
    vit = torch.full((1, 2, 2), 3.0)
    out = vae_txt_vit_wapg(base, img, txt, vit, 1.0, 1.0, 1.0)
    assert out.shape == base.shape


def test_x0_velocity_roundtrip():
    x = torch.randn(1, 4, 8, 8)
    v = torch.randn_like(x)
    sigma = torch.tensor(0.7)
    x0 = velocity_to_x0(x, v, sigma)
    v2 = x0_to_velocity(x, x0, sigma)
    assert torch.allclose(v, v2, atol=1e-5)


def test_vae_txt_vit_wapg_velocity_vs_x0_space():
    """APG is not affine-invariant: velocity-space result must differ from naive x0 APG."""
    torch.manual_seed(0)
    x = torch.randn(1, 2, 4, 4)
    sigma = torch.tensor(0.6)
    v_base = torch.randn_like(x)
    v_img = v_base + 0.3 * torch.randn_like(x)
    v_txt = v_img + 0.3 * torch.randn_like(x)
    v_vit = v_txt + 0.3 * torch.randn_like(x)
    x0_base = velocity_to_x0(x, v_base, sigma)
    x0_img = velocity_to_x0(x, v_img, sigma)
    x0_txt = velocity_to_x0(x, v_txt, sigma)
    x0_vit = velocity_to_x0(x, v_vit, sigma)

    x0_official = vae_txt_vit_wapg_from_x0(
        x, sigma, x0_base, x0_img, x0_txt, x0_vit, 1.0, 1.0, 1.0,
        parallel_scale=0.2, orthogonal_scale=1.0,
    )
    x0_naive = vae_txt_vit_wapg(
        x0_base, x0_img, x0_txt, x0_vit, 1.0, 1.0, 1.0,
        parallel_scale=0.2, orthogonal_scale=1.0,
    )
    assert not torch.allclose(x0_official, x0_naive, atol=1e-4)

    # With parallel=orthogonal=1, APG is identity → spaces agree after roundtrip.
    x0_full = vae_txt_vit_wapg_from_x0(
        x, sigma, x0_base, x0_img, x0_txt, x0_vit, 1.0, 1.0, 1.0,
        parallel_scale=1.0, orthogonal_scale=1.0,
    )
    x0_plain = (
        x0_base
        + 1.0 * (x0_img - x0_base)
        + 1.0 * (x0_txt - x0_img)
        + 1.0 * (x0_vit - x0_txt)
    )
    assert torch.allclose(x0_full, x0_plain, atol=1e-5)


def test_merge_pad_truncate():
    t5 = torch.randn(1, 100, 4096)
    planner = torch.randn(1, 450, 4096)
    merged = merge_t5_planner(t5, planner, max_sequence_length=512, truncate=True)
    assert merged.shape == (1, 512, 4096)
    short = merge_t5_planner(t5[:, :10], None, max_sequence_length=512, truncate=True)
    assert short.shape == (1, 512, 4096)


def test_make_source_ids():
    assert make_source_ids(3) == [1.0, 2.0, 3.0]
    ids = make_source_ids(8, max_trained_src_id=5, interpolate=True)
    assert len(ids) == 8
    assert ids[0] == 1.0
    assert ids[-1] == 5.0


def test_branch_cross_attn():
    t = torch.randn(1, 8, 4096)
    cond = [{"cross_attn": t, "bernini_text_wtxt_wovit": torch.randn(1, 8, 4096)}]
    assert torch.equal(get_branch_cross_attn(cond, "wtxt_wvit"), t)
    assert get_branch_cross_attn(cond, "wtxt_wovit").shape == (1, 8, 4096)


def test_build_branch_cond_list():
    t = torch.randn(1, 4, 4096)
    cond = [{"cross_attn": t, "model_conds": {}}]
    none = build_branch_cond_list(cond, None, cross_attn=t)
    assert "context_latents" not in none[0]["model_conds"]


def main():
    test_split_context()
    test_chained_cfg()
    test_apg()
    test_apg_delta()
    test_vae_txt_vit_wapg()
    test_x0_velocity_roundtrip()
    test_vae_txt_vit_wapg_velocity_vs_x0_space()
    test_merge_pad_truncate()
    test_make_source_ids()
    test_branch_cross_attn()
    test_build_branch_cond_list()
    print("All bernini guidance tests passed.")


if __name__ == "__main__":
    main()
