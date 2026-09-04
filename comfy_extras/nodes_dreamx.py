"""DreamX-Creator Autoregressive 1-Step 2K refiner (SR-DiT 5B + Flash latent upsampler).

Weights: https://huggingface.co/GD-ML/DreamX-Creator
Place `sr_dit_5b.pt` in models/diffusion_models and `latent_upsampler_flash.pt`
in models/latent_upscale_models. Use Wan 2.2 VAE + UMT5 (CLIPLoader type wan).
"""

import json
import math

import torch
from typing_extensions import override

import folder_paths
import latent_preview
import comfy.model_management
import comfy.model_patcher
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
from comfy.ldm.wan.flash_latent_up import FlashLatentUpsampler
from comfy_api.latest import ComfyExtension, io

DREAMX_SR_STEPS = (1000, 750, 500, 250)


def _unwrap_checkpoint(sd):
    if not isinstance(sd, dict) or len(sd) == 0:
        raise ValueError("DreamX checkpoint is empty or not a state dict")
    for wrap in ("generator", "generator_ema", "model", "state_dict"):
        inner = sd.get(wrap)
        if isinstance(inner, dict) and len(inner) > 0 and any(torch.is_tensor(v) for v in inner.values()):
            sd = inner
            break
    cleaned = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        nk = k.replace("_fsdp_wrapped_module.", "")
        if nk.startswith("module."):
            nk = nk[7:]
        cleaned[nk] = v
    if len(cleaned) == 0:
        raise ValueError("DreamX checkpoint has no tensor weights")
    sample = next(iter(cleaned))
    if sample.startswith("model.") and "model.patch_embedding.weight" not in cleaned:
        if any(k.startswith("model.blocks.") or k.startswith("model.patch_embedding.") for k in cleaned):
            cleaned = {k[6:] if k.startswith("model.") else k: v for k, v in cleaned.items()}
    return cleaned


def _flash_from_state_dict(sd):
    sd = _unwrap_checkpoint(sd)
    if "pre_shuffle.0.weight" not in sd or "final.weight" not in sd:
        raise ValueError("Not a DreamX FlashLatentUpsampler checkpoint (missing pre_shuffle/final)")
    mid = sd["blocks.0.conv.0.weight"].shape[1] // 2
    conv0 = sd["pre_shuffle.0.weight"]
    r2 = conv0.shape[0] // mid
    scale = int(round(math.sqrt(r2)))
    num_blocks = 0
    while f"blocks.{num_blocks}.conv.0.weight" in sd:
        num_blocks += 1
    model = FlashLatentUpsampler(
        in_channels=conv0.shape[1],
        out_channels=sd["final.weight"].shape[0],
        mid_channels=mid,
        num_blocks=num_blocks,
        upsample_scale=scale,
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise ValueError(f"FlashLatentUpsampler missing keys: {missing[:8]}")
    return model


def _sr_sigmas(model, sigma_start):
    max_t = float(sigma_start) * 1000.0
    kept = [t for t in DREAMX_SR_STEPS if t <= max_t + 1e-3]
    if len(kept) == 0:
        kept = [DREAMX_SR_STEPS[-1]]
    ms = model.get_model_object("model_sampling")
    vals = [float(ms.sigma(torch.tensor(t, dtype=torch.float32))) for t in kept]
    vals.append(0.0)
    return torch.FloatTensor(vals)


class DreamXFlashLatentUpsamplerLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DreamXFlashLatentUpsamplerLoader",
            display_name="Load DreamX Flash Latent Upsampler",
            category="model/loaders",
            inputs=[
                io.Combo.Input("model_name", options=folder_paths.get_filename_list("latent_upscale_models")),
            ],
            outputs=[io.LatentUpscaleModel.Output()],
        )

    @classmethod
    def execute(cls, model_name) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise("latent_upscale_models", model_name)
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        model = _flash_from_state_dict(sd)
        dtype = comfy.model_management.vae_dtype(allowed_dtypes=[torch.bfloat16, torch.float32])
        model.to(dtype=dtype)
        comfy.model_management.archive_model_dtypes(model)
        patcher = comfy.model_patcher.CoreModelPatcher(
            model,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device(),
        )
        return io.NodeOutput(patcher)


class DreamXSRDiTLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DreamXSRDiTLoader",
            display_name="Load DreamX SR-DiT",
            category="model/loaders",
            inputs=[
                io.Combo.Input("unet_name", options=folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("weight_dtype", options=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], default="default"),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, unet_name, weight_dtype="default") -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        sd = _unwrap_checkpoint(sd)
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        metadata = {"config": json.dumps({"transformer": {"causal_ar": True}})}
        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata)
        if model is None:
            raise RuntimeError(f"Could not detect DreamX SR-DiT from {unet_name}")
        return io.NodeOutput(model)


class DreamXUpsampleLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DreamXUpsampleLatent",
            display_name="DreamX Upsample Latent",
            category="model/latent/dreamx",
            inputs=[
                io.Latent.Input("samples"),
                io.LatentUpscaleModel.Input("upscale_model"),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, samples, upscale_model) -> io.NodeOutput:
        latents = samples["samples"]
        if latents.ndim != 5:
            raise ValueError(f"DreamX latent upsample expects 5D [B,C,T,H,W], got {tuple(latents.shape)}")
        memory_required = math.prod(latents.shape) * 64.0
        comfy.model_management.load_models_gpu([upscale_model], memory_required=memory_required)
        device = upscale_model.load_device
        model = upscale_model.model
        in_dtype = latents.dtype
        latents = latents.to(device=device, dtype=comfy.model_management.vae_dtype(allowed_dtypes=[torch.bfloat16, torch.float32]))
        up = model(latents)
        out = samples.copy()
        out["samples"] = up.to(device=comfy.model_management.intermediate_device(), dtype=in_dtype)
        out.pop("noise_mask", None)
        return io.NodeOutput(out)


class DreamXRefine(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DreamXRefine",
            display_name="DreamX 2K Refine",
            category="sampling/dreamx",
            description="Causal few-step SR on an upsampled Wan2.2 latent. Mixes at sigma_start then denoises remaining SR steps with Sampler AR Video.",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1),
                io.Float.Input("sigma_start", default=0.6251, min=0.0, max=1.0, step=0.0001),
                io.Int.Input("num_frame_per_block", default=3, min=1, max=64),
                io.Int.Input("kv_len", default=9, min=0, max=1024,
                             tooltip="Latent frames kept in KV cache. 0 = full history."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, model, positive, negative, latent, seed, cfg, sigma_start,
                num_frame_per_block, kv_len) -> io.NodeOutput:
        samples = latent["samples"]
        if samples.ndim != 5:
            raise ValueError(f"DreamX refine expects 5D Wan2.2 latents [B,C,T,H,W], got {tuple(samples.shape)}")
        sigmas = _sr_sigmas(model, sigma_start)
        sampler = comfy.samplers.ksampler("ar_video", {
            "num_frame_per_block": num_frame_per_block,
            "kv_len": kv_len,
        })
        samples = comfy.sample.fix_empty_latent_channels(model, samples)
        process_in = model.get_model_object("process_latent_in")
        process_out = model.get_model_object("process_latent_out")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        lr = process_in(samples)
        noise_m = torch.randn(lr.shape, generator=generator, dtype=torch.float32).to(device=lr.device, dtype=lr.dtype)
        mixed = float(sigma_start) * noise_m + (1.0 - float(sigma_start)) * lr
        latent_image = process_out(mixed)
        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        out_samples = comfy.sample.sample_custom(
            model, mixed, cfg, sampler, sigmas, positive, negative, latent_image,
            noise_mask=latent.get("noise_mask"), callback=callback, disable_pbar=disable_pbar, seed=seed,
        )
        out = latent.copy()
        out["samples"] = out_samples
        out.pop("noise_mask", None)
        return io.NodeOutput(out)


class DreamXExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            DreamXFlashLatentUpsamplerLoader,
            DreamXSRDiTLoader,
            DreamXUpsampleLatent,
            DreamXRefine,
        ]


async def comfy_entrypoint() -> DreamXExtension:
    return DreamXExtension()
