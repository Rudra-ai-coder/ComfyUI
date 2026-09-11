import torch

import comfy.nested_tensor
from comfy_extras.nodes_minimax_h3 import (
    MiniMaxH3ContinueAV,
    _empty_av_latent,
    _pixel_frames_from_latent_t,
    temporal_shape,
    video_latent_t,
)


def _context(length=22, width=32, height=32):
    latent, frames = _empty_av_latent(width, height, length)
    video, audio = latent["samples"].unbind()
    video = torch.arange(video.numel(), dtype=video.dtype, device=video.device).reshape(video.shape)
    audio = torch.arange(audio.numel(), dtype=audio.dtype, device=audio.device).reshape(audio.shape) + 1000
    latent["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
    return latent, frames


def test_continue_freezes_prefix_and_snaps_total_grid():
    context, ctx_frames = _context(22)
    extra_len = 22
    out, cond = MiniMaxH3ContinueAV.execute(context, extra_len).result
    assert cond is None
    video, audio = out["samples"].unbind()
    v_mask, a_mask = out["noise_mask"].unbind()
    ctx_t = video_latent_t(ctx_frames)
    total_frames, total_t, total_a = temporal_shape(ctx_frames + extra_len)
    assert video.shape[2] == total_t
    assert audio.shape[-1] == total_a
    assert _pixel_frames_from_latent_t(total_t) == total_frames
    assert torch.count_nonzero(v_mask[:, :, :ctx_t]) == 0
    assert torch.count_nonzero(v_mask[:, :, ctx_t:] == 0) == 0
    assert torch.count_nonzero(a_mask[..., :context["samples"].unbind()[1].shape[-1]]) == 0
    ctx_v = context["samples"].unbind()[0]
    torch.testing.assert_close(video[:, :, :ctx_t], ctx_v)


def test_continue_uses_optional_remaining_latent():
    context, ctx_frames = _context(5)
    remaining, extra_frames = _empty_av_latent(32, 32, 22)
    rv, ra = remaining["samples"].unbind()
    rv.fill_(7)
    ra.fill_(8)
    remaining["samples"] = comfy.nested_tensor.NestedTensor((rv, ra))
    out, _ = MiniMaxH3ContinueAV.execute(context, 124, latent=remaining).result
    video, audio = out["samples"].unbind()
    ctx_t = video_latent_t(ctx_frames)
    assert (video[:, :, ctx_t:ctx_t + rv.shape[2]] == 7).all()
    extra_a = min(ra.shape[-1], audio.shape[-1] - context["samples"].unbind()[1].shape[-1])
    assert extra_a > 0
    assert (audio[..., context["samples"].unbind()[1].shape[-1]:context["samples"].unbind()[1].shape[-1] + extra_a] == 8).all()
    assert extra_frames == 22


def test_continue_shifts_keyframe_indices():
    context, ctx_frames = _context(22)
    positive = [[None, {"minimax_keyframes": [{"resolved_frame_index": 0}, {"resolved_frame_index": 21}]}]]
    _, cond = MiniMaxH3ContinueAV.execute(context, 22, positive=positive).result
    idxs = [kf["resolved_frame_index"] for kf in cond[0][1]["minimax_keyframes"]]
    assert idxs == [ctx_frames, ctx_frames + 21]
    assert positive[0][1]["minimax_keyframes"][0]["resolved_frame_index"] == 0


def test_continue_rejects_non_h3_latent():
    bad = {"samples": torch.zeros(1, 4, 8, 8)}
    try:
        MiniMaxH3ContinueAV.execute(bad, 22)
    except ValueError as e:
        assert "H3 AV" in str(e)
    else:
        raise AssertionError("expected ValueError")
