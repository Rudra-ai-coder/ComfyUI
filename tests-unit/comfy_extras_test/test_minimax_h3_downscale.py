import torch

import comfy.nested_tensor
from comfy_extras.nodes_minimax_h3 import _empty_av_latent, temporal_shape, video_latent_t
from comfy_extras.nodes_minimax_h3_upscale import MiniMaxH3DownscaleLatent, MiniMaxH3TemporalUpscaleLatent


def test_downscale_halves_spatial_keeps_time_and_audio():
    latent, _ = _empty_av_latent(64, 64, 22)
    video, audio = latent["samples"].unbind()
    video = torch.arange(video.numel(), dtype=torch.float32).reshape(video.shape)
    audio = torch.arange(audio.numel(), dtype=torch.float32).reshape(audio.shape) + 100
    latent["samples"] = comfy.nested_tensor.NestedTensor((video, audio))

    out = MiniMaxH3DownscaleLatent.execute(latent, 0.5, 0, 0, "area").result[0]
    v, a = out["samples"].unbind()
    assert v.shape == (1, 24, video.shape[2], 2, 2)
    torch.testing.assert_close(a, audio)


def test_downscale_resizes_video_noise_mask():
    latent, _ = _empty_av_latent(64, 64, 22)
    video, audio = latent["samples"].unbind()
    v_mask = torch.ones_like(video[:, :1])
    a_mask = torch.ones_like(audio[:, :1])
    latent["noise_mask"] = comfy.nested_tensor.NestedTensor((v_mask, a_mask))

    out = MiniMaxH3DownscaleLatent.execute(latent, 0.5, 0, 0, "nearest-exact").result[0]
    vm, am = out["noise_mask"].unbind()
    assert vm.shape[-2:] == (2, 2)
    assert am.shape == a_mask.shape


def test_temporal_upscale_doubles_duration_on_frame_grid():
    latent, frames = _empty_av_latent(64, 64, 124)
    video, audio = latent["samples"].unbind()
    video = video.clone()
    video[:, :, 0] = 1
    video[:, :, -1] = 9
    audio = audio.clone()
    audio[..., 0] = 2
    audio[..., -1] = 8
    latent["samples"] = comfy.nested_tensor.NestedTensor((video, audio))

    out = MiniMaxH3TemporalUpscaleLatent.execute(latent, 2.0, 0, "linear").result[0]
    v, a = out["samples"].unbind()
    dst_frames = 17 * round((frames * 2 - 5) / 17) + 5
    _, dst_t, dst_a = temporal_shape(dst_frames)
    assert dst_t == video_latent_t(dst_frames)
    assert v.shape[2] == dst_t
    assert a.shape[-1] == dst_a
    torch.testing.assert_close(v[:, :, 0], video[:, :, 0])
    torch.testing.assert_close(v[:, :, -1], video[:, :, -1])
    torch.testing.assert_close(a[..., 0], audio[..., 0])
    torch.testing.assert_close(a[..., -1], audio[..., -1])
