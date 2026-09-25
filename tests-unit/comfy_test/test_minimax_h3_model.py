import torch
from torch import nn

from comfy.ldm.minimax.model import MiniMaxH3Model, time_shift_sigma
from comfy.model_sampling import CONST


def make_model(video_output, audio_output):
    model = MiniMaxH3Model.__new__(MiniMaxH3Model)
    nn.Module.__init__(model)
    model.sigma_shift_video = 12.0
    model.sigma_shift_audio = 3.0
    model._forward = lambda *args, **kwargs: [video_output.clone(), audio_output.clone()]
    return model


def test_forward_scales_velocity_to_mask_timestep():
    video_output = torch.full((1, 2, 1, 2, 2), 2.0)
    audio_output = torch.full((1, 2, 2, 3), 3.0)
    video_mask = torch.tensor([[[[[1.0, 0.75], [0.5, 0.25]]]]])
    audio_mask = torch.tensor([[[[1.0, 0.5, 0.25], [0.75, 0.5, 0.0]]]])
    sigma = torch.tensor([0.5])
    clean = torch.arange(video_output.numel(), dtype=torch.float32).reshape_as(video_output)
    model_input = clean + sigma.reshape(1, 1, 1, 1, 1) * video_mask * video_output
    model = make_model(video_output, audio_output)

    out = model(
        [model_input, torch.zeros_like(audio_output)],
        sigma * 1000.0,
        torch.empty(1, 1, 1),
        minimax_payload={"audio_scale": 1.0},
        denoise_mask=video_mask,
        audio_denoise_mask=audio_mask,
    )

    torch.testing.assert_close(out[0], video_output * video_mask)
    torch.testing.assert_close(out[1], audio_output * audio_mask)
    denoised = CONST.calculate_denoised(None, sigma, out[0], model_input)
    torch.testing.assert_close(denoised, clean)


def test_forward_scales_audio_velocity_before_carry_conversion():
    video_output = torch.ones((1, 1, 1, 1, 1))
    audio_output = torch.full((1, 1, 2, 2), 3.0)
    audio_src = torch.full_like(audio_output, 2.0)
    audio_mask = torch.tensor([[[[0.75, 0.5], [0.25, 0.0]]]])
    model = make_model(video_output, audio_output)
    sigma_v = torch.tensor(0.5)
    sigma_a = time_shift_sigma(sigma_v, 12.0, 3.0)
    carry = sigma_a / sigma_v

    out = model(
        [torch.zeros_like(video_output), audio_src],
        sigma_v.reshape(1) * 1000.0,
        torch.empty(1, 1, 1),
        minimax_payload={"audio_scale": 4.0},
        audio_denoise_mask=audio_mask,
    )

    expected = -3.0 * audio_src * carry + (1.0 + 3.0 * sigma_a) * audio_output * audio_mask
    torch.testing.assert_close(out[1], expected)


def _rope_stub():
    model = MiniMaxH3Model.__new__(MiniMaxH3Model)
    model.rope = nn.Module()
    model.rope.register_buffer("inv_freq", torch.ones(16, dtype=torch.float32))
    return model


def test_rope_freqs_spatial_scale_only_visual_hw():
    model = _rope_stub()
    pos = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=torch.float64)
    base = model.rope_freqs(pos, "cpu")
    scaled = model.rope_freqs(pos, "cpu", spatial_scale=0.5, spatial_idx=torch.tensor([0]))

    t, h, w = 16, 32, 48
    torch.testing.assert_close(scaled[0, :t], base[0, :t])
    torch.testing.assert_close(scaled[0, t:h], base[0, t:h] * 0.5)
    torch.testing.assert_close(scaled[0, h:w], base[0, h:w] * 0.5)
    torch.testing.assert_close(scaled[0, w:], scaled[0, :w])
    torch.testing.assert_close(scaled[1], base[1])
    torch.testing.assert_close(model.rope_freqs(pos, "cpu"), base)
