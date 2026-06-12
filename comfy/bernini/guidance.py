# SPDX-License-Identifier: Apache-2.0
"""Bernini chained CFG / APG helpers (ported from Bernini wan_diffusion.py)."""

import torch
import torch.nn.functional as F


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        self.running_average = update_value + self.momentum * self.running_average


def _normalize_diff(diff, base_pred, momentum_buffer, eta, norm_threshold):
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -4], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    v0, v1 = diff.double(), base_pred.double()
    v1 = F.normalize(v1, dim=[-1, -2, -4])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -4], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    diff_parallel = v0_parallel.to(diff.dtype)
    diff_orthogonal = v0_orthogonal.to(diff.dtype)
    return diff_orthogonal + eta * diff_parallel


def normalized_guidance(
    pred_cond,
    pred_uncond,
    guidance_scale,
    momentum_buffer=None,
    eta=1.0,
    norm_threshold=0.0,
):
    nd = _normalize_diff(pred_cond - pred_uncond, pred_cond, momentum_buffer, eta, norm_threshold)
    return pred_uncond + guidance_scale * nd


def normalized_guidance_chain(
    pred_uncond,
    preds,
    scales,
    momentum_buffers,
    eta,
    norm_thresholds,
):
    bases = [pred_uncond] + list(preds)
    result = pred_uncond
    for i, cond in enumerate(preds):
        nt = norm_thresholds[i] if isinstance(norm_thresholds, (list, tuple)) else norm_thresholds
        mb = momentum_buffers[i] if momentum_buffers is not None else None
        nd = _normalize_diff(cond - bases[i], cond, mb, eta, nt)
        result = result + scales[i] * nd
    return result


def chained_cfg_rv2v(eps_none, eps_v, eps_vi, eps_vti, omega_vid, omega_img, omega_txt):
    return (
        eps_none
        + omega_vid * (eps_v - eps_none)
        + omega_img * (eps_vi - eps_v)
        + omega_txt * (eps_vti - eps_vi)
    )


def chained_cfg_v2v_chain(eps_none, eps_v, eps_vti, omega_vid, omega_txt):
    return eps_none + omega_vid * (eps_v - eps_none) + omega_txt * (eps_vti - eps_v)


def apg_delta(
    delta: torch.Tensor,
    ref: torch.Tensor,
    parallel_scale: float = 0.2,
    orthogonal_scale: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Adaptive projected guidance delta (wan_diffusion.py)."""
    b = delta.shape[0]
    delta_f = delta.reshape(b, -1)
    ref_f = ref.reshape(b, -1)
    ref_norm_sq = (ref_f * ref_f).sum(dim=1, keepdim=True).clamp_min(eps)
    proj_coeff = (delta_f * ref_f).sum(dim=1, keepdim=True) / ref_norm_sq
    delta_parallel_f = proj_coeff * ref_f
    delta_orthogonal_f = delta_f - delta_parallel_f
    return (
        parallel_scale * delta_parallel_f.reshape_as(delta)
        + orthogonal_scale * delta_orthogonal_f.reshape_as(delta)
    )


def vae_txt_vit_wapg(
    eps_base,
    eps_img,
    eps_txt,
    eps_vit,
    omega_img,
    omega_txt,
    omega_tgt,
    parallel_scale=0.2,
    orthogonal_scale=1.0,
):
    delta_img = apg_delta(eps_img - eps_base, ref=eps_img, parallel_scale=parallel_scale, orthogonal_scale=orthogonal_scale)
    delta_txt = apg_delta(eps_txt - eps_img, ref=eps_txt, parallel_scale=parallel_scale, orthogonal_scale=orthogonal_scale)
    delta_vit = apg_delta(eps_vit - eps_txt, ref=eps_vit, parallel_scale=parallel_scale, orthogonal_scale=orthogonal_scale)
    return eps_base + omega_img * delta_img + omega_txt * delta_txt + omega_tgt * delta_vit
