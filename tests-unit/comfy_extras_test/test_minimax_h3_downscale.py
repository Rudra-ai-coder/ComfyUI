import torch

import comfy.nested_tensor
from comfy_extras.nodes_minimax_h3 import _empty_av_latent
from comfy_extras.nodes_minimax_h3_upscale import MiniMaxH3DownscaleLatent


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
