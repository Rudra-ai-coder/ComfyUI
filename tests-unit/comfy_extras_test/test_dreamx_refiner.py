import torch

from comfy.ldm.wan.flash_latent_up import FlashLatentUpsampler
from comfy_extras.nodes_dreamx import _unwrap_checkpoint, _flash_from_state_dict


def test_flash_latent_upsampler_doubles_spatial():
    m = FlashLatentUpsampler(in_channels=48, out_channels=48, mid_channels=128, num_blocks=2, upsample_scale=2)
    x = torch.randn(1, 48, 3, 8, 8)
    y = m(x)
    assert y.shape == (1, 48, 3, 16, 16)


def test_unwrap_nested_generator_keys():
    inner = {"pre_shuffle.0.weight": torch.zeros(1), "model.blocks.0.x": torch.zeros(1)}
    sd = {"generator": inner}
    out = _unwrap_checkpoint(sd)
    assert "pre_shuffle.0.weight" in out


def test_flash_from_state_dict_roundtrip():
    m = FlashLatentUpsampler(in_channels=48, out_channels=48, mid_channels=128, num_blocks=2, upsample_scale=2)
    loaded = _flash_from_state_dict(m.state_dict())
    x = torch.randn(1, 48, 2, 4, 4)
    assert loaded(x).shape == (1, 48, 2, 8, 8)
