# Minimal DiffLoss_FM + SimpleMLPAdaLN for Bernini VIT flow-matching inference.
# Parameter layout matches vit_decoder.* keys in bernini_vit_decoder.safetensors.

import math

import torch
import torch.nn as nn
from tqdm import tqdm


class FlowMatchScheduler:
    """Flow-matching scheduler for VIT decoder sampling (inference only)."""

    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        inverse_timesteps: bool = False,
        extra_one_step: bool = False,
        reverse_sigmas: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        shift: float = None,
        device=None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        if shift is not None:
            self.shift = shift
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps + 1, device=device, dtype=dtype
            )[:-1]
        else:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps, device=device, dtype=dtype
            )
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps

    def step(self, model_output, timestep, sample, to_final: bool = False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(device=sample.device, non_blocking=True)
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        return sample + model_output * (sigma_ - sigma)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(t.dtype))
        return t_emb


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.out_norm = None
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        out = gate_mlp * h
        if self.out_norm is not None:
            out = self.out_norm(out)
        return x + out


class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """Diffusion MLP for VIT flow matching.

    Keys in bernini_vit_decoder.safetensors:
      vit_decoder.net.input_proj.*
      vit_decoder.net.time_embed.mlp.*
      vit_decoder.net.cond_embed.*
      vit_decoder.net.res_blocks.{i}.*
      vit_decoder.net.final_layer.*
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)

        self.res_blocks = nn.ModuleList(
            ResBlock(model_channels) for _ in range(num_res_blocks)
        )
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)
        y = t + c

        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def forward_with_txt_img_cfg(self, x, t, c, txt_cfg_scale, img_cfg_scale):
        part = x[: len(x) // 3]
        combined = torch.cat([part, part, part], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps, imgcond_eps = torch.split(eps, len(eps) // 3, dim=0)
        part_eps = (
            uncond_eps
            + img_cfg_scale * (imgcond_eps - uncond_eps)
            + txt_cfg_scale * (cond_eps - imgcond_eps)
        )
        eps = torch.cat([part_eps, part_eps, part_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class DiffLoss_FM(nn.Module):
    """VIT flow-matching decoder (inference sample() only)."""

    def __init__(
        self,
        target_channels=3584,
        z_channels=3584,
        depth=16,
        width=4096,
        diff_net="SimpleMLPAdaLN",
        scheduler_type="FlowMatchScheduler",
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=2.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        extra_one_step=False,
        diffusion_batch_mul=1,
    ):
        super().__init__()
        self.diffusion_batch_mul = diffusion_batch_mul
        self.in_channels = target_channels
        out_channels = target_channels

        if diff_net == "SimpleMLPAdaLN":
            self.net = SimpleMLPAdaLN(
                in_channels=target_channels,
                model_channels=width,
                out_channels=out_channels,
                z_channels=z_channels,
                num_res_blocks=depth,
                grad_checkpointing=False,
            )
        else:
            raise NotImplementedError(f"Unknown diff_net: {diff_net}")

        self.num_inference_steps = num_inference_steps
        if scheduler_type == "FlowMatchScheduler":
            self.scheduler = FlowMatchScheduler(
                num_inference_steps=num_inference_steps,
                num_train_timesteps=num_train_timesteps,
                shift=shift,
                sigma_max=sigma_max,
                sigma_min=sigma_min,
                extra_one_step=extra_one_step,
            )
        else:
            raise NotImplementedError(f"Unknown scheduler_type: {scheduler_type}")

    def sample(self, z, cfg, num_inference_steps, img_cfg=None, verbose=True):
        device = z.device

        if img_cfg is not None and cfg > 1.0:
            noise = torch.randn(z.shape[0] // 3, self.in_channels, device=device)
            noise = torch.cat([noise, noise, noise], dim=0)
            model_kwargs = dict(c=z, txt_cfg_scale=cfg, img_cfg_scale=img_cfg)
            sample_fn = self.net.forward_with_txt_img_cfg
        elif cfg > 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels, device=device)
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels, device=device)
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        try:
            self.scheduler.set_timesteps(num_inference_steps, training=False)
        except TypeError:
            self.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps.to(device)
        samples = noise.to(z.dtype)
        progress_bar = tqdm(timesteps, desc=f"Vit diffusion with cfg={cfg}") if verbose else None

        for t in timesteps:
            timestep = t.unsqueeze(0).to(dtype=z.dtype, device=device)
            noise_pred = sample_fn(x=samples, t=timestep, **model_kwargs)
            samples = self.scheduler.step(model_output=noise_pred, timestep=timestep, sample=samples)
            if not isinstance(samples, torch.Tensor):
                samples = samples.prev_sample
            if verbose:
                progress_bar.update(1)

        if verbose and progress_bar is not None:
            progress_bar.close()

        return samples
