import torch
import torch.nn as nn


def _conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


class FlashLatentMemBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv = nn.Sequential(
            _conv3x3(in_channels * 2, out_channels),
            nn.ReLU(inplace=False),
            _conv3x3(out_channels, out_channels),
            nn.ReLU(inplace=False),
            _conv3x3(out_channels, out_channels),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=False)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], dim=1)) + self.skip(x))


class FlashLatentUpsampler(nn.Module):
    """Wan2.2 48ch spatial latent upsampler from DreamX FlashLatent."""

    def __init__(self, in_channels=48, out_channels=48, mid_channels=128, num_blocks=8,
                 upsample_scale=2, memory_init="replicate"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.upsample_scale = upsample_scale
        self.memory_init = memory_init
        r2 = upsample_scale * upsample_scale
        self.pre_shuffle = nn.Sequential(
            _conv3x3(in_channels, mid_channels * r2),
            nn.PixelShuffle(upsample_scale),
            nn.ReLU(inplace=False),
        )
        self.blocks = nn.ModuleList(
            [FlashLatentMemBlock(mid_channels, mid_channels) for _ in range(num_blocks)]
        )
        self.final = _conv3x3(mid_channels, out_channels)

    def forward(self, latent):
        if latent.ndim != 5:
            raise ValueError(f"Expected latent shape [B, C, T, H, W], got {tuple(latent.shape)}")
        b, c, t, h, w = latent.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {c}")
        feat = latent.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        feat = self.pre_shuffle(feat)
        _, mid, hh, ww = feat.shape
        for block in self.blocks:
            frames = feat.reshape(b, t, mid, hh, ww)
            if self.memory_init == "replicate":
                first = frames[:, :1]
            else:
                first = torch.zeros_like(frames[:, :1])
            past = torch.cat([first, frames[:, :-1]], dim=1).reshape(feat.shape)
            feat = block(feat, past)
        feat = self.final(feat)
        out_c = feat.shape[1]
        return feat.reshape(b, t, out_c, hh, ww).permute(0, 2, 1, 3, 4)
