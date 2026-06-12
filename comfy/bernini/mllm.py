# SPDX-License-Identifier: Apache-2.0
"""Bernini Qwen2.5-VL-7B MLLM wrapper (HF folder or bernini_mllm.safetensors)."""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import PIL.Image
import torch

import comfy.model_management
import folder_paths

LOG = logging.getLogger("bernini.mllm")

# Architecture + processor fallback when only weights safetensors are local.
MLLM_HUB_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_PROCESSOR_SUBDIR = "mllm_processor"


def _tensor_to_pil(image_tensor: torch.Tensor) -> PIL.Image.Image:
    """Convert ComfyUI IMAGE [H,W,C] float 0-1 to PIL."""
    arr = (image_tensor.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    return PIL.Image.fromarray(arr)


def _sample_video_frames(video: torch.Tensor, max_frames: int, frame_factor: int = 2) -> List[PIL.Image.Image]:
    """Uniformly sample frames from ComfyUI video batch [T,H,W,C]."""
    total = video.shape[0]
    if total <= max_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, max_frames).astype(int).tolist()
    if frame_factor > 1 and len(indices) > 1:
        n = (len(indices) // frame_factor) * frame_factor
        indices = indices[: max(n, frame_factor)]
    return [_tensor_to_pil(video[i]) for i in indices]


def resolve_bernini_model_path(name_or_path: str) -> str:
    """Resolve a models/bernini filename or absolute path."""
    if os.path.isfile(name_or_path):
        return name_or_path
    return folder_paths.get_full_path_or_raise("bernini", name_or_path)


def resolve_mllm_processor_dir(processor_name: str) -> str:
    """Processor folder: models/bernini/<subdir>, absolute path, or HF hub id."""
    if os.path.isdir(processor_name):
        return processor_name
    bernini_roots = folder_paths.get_folder_paths("bernini")
    if bernini_roots:
        candidate = os.path.join(bernini_roots[0], processor_name)
        if os.path.isdir(candidate):
            return candidate
    return processor_name


class BerniniMLLM:
    """Qwen2.5-VL-7B from Bernini-Diffusers mllm/ folder or bernini_mllm.safetensors."""

    def __init__(self, model, processor, path: str, config_path: str):
        self.model = model
        self.processor = processor
        self.path = path
        self.config_path = config_path
        self.dtype = torch.bfloat16

    @classmethod
    def load(
        cls,
        path: str,
        processor_path: Optional[str] = None,
        device=None,
    ):
        from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLModel

        if device is None:
            device = comfy.model_management.unet_offload_device()

        if os.path.isdir(path):
            return cls._load_hf_folder(path, device)

        weights_path = resolve_bernini_model_path(path)
        proc_dir = resolve_mllm_processor_dir(processor_path or DEFAULT_PROCESSOR_SUBDIR)
        if not os.path.isdir(proc_dir):
            LOG.warning(
                "MLLM processor folder not found at %s — falling back to %s (needs HF cache or network)",
                proc_dir,
                MLLM_HUB_ID,
            )
            proc_dir = MLLM_HUB_ID

        LOG.info("Loading Bernini MLLM weights %s, processor/config %s -> %s", weights_path, proc_dir, device)
        processor = AutoProcessor.from_pretrained(proc_dir, padding_side="right", trust_remote_code=True)
        config = AutoConfig.from_pretrained(proc_dir, trust_remote_code=True)
        model = Qwen2_5_VLModel.from_config(config, torch_dtype=torch.bfloat16)

        from safetensors.torch import load_file

        state_dict = load_file(weights_path, device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            LOG.warning("Bernini MLLM missing keys: %s", missing[:8])
        if unexpected:
            LOG.warning("Bernini MLLM unexpected keys: %s", unexpected[:8])

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(device)
        return cls(model, processor, weights_path, proc_dir)

    @classmethod
    def _load_hf_folder(cls, path: str, device):
        from transformers import AutoProcessor, Qwen2_5_VLModel

        LOG.info("Loading Bernini MLLM from HF folder %s -> %s", path, device)
        processor = AutoProcessor.from_pretrained(path, padding_side="right", trust_remote_code=True)
        model = Qwen2_5_VLModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(device)
        return cls(model, processor, path, path)

    def to(self, device, dtype=None):
        if dtype is None:
            dtype = self.dtype
        self.model.to(device=device, dtype=dtype)
        return self

    @torch.no_grad()
    def get_vit_features(self, pixel_values, grid_thw) -> Tuple[torch.Tensor, ...]:
        """Extract split VIT embeddings (matches Bernini get_vit_features)."""
        pixel_values = pixel_values.type(self.model.dtype).to(self.model.device)
        grid_thw = grid_thw.to(self.model.device)
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
            image_embeds = self.model.visual(pixel_values, grid_thw=grid_thw)
        split_sizes = (
            grid_thw.prod(-1) // self.model.visual.spatial_merge_size ** 2
        ).tolist()
        return torch.split(image_embeds, split_sizes)

    @torch.no_grad()
    def encode_images(
        self,
        images: List[torch.Tensor],
        vit_min_pixels: int = 3136,
        vit_max_pixels: int = 50176,
    ):
        """Encode ComfyUI IMAGE tensors -> list of VIT embed tensors + grid_thw."""
        if not images:
            return [], []
        pil_images = [_tensor_to_pil(img[0] if img.ndim == 4 else img) for img in images]
        image_inputs = self.processor.image_processor(
            images=pil_images,
            return_tensors="pt",
            min_pixels=vit_min_pixels,
            max_pixels=vit_max_pixels,
        )
        pixel_values = image_inputs["pixel_values"]
        image_grid_thw = image_inputs["image_grid_thw"]
        embeds = self.get_vit_features(pixel_values, image_grid_thw)
        return list(embeds), image_grid_thw.numpy().tolist()

    @torch.no_grad()
    def encode_videos(
        self,
        videos: List[torch.Tensor],
        vit_min_pixels: int = 3136,
        vit_max_pixels: int = 50176,
        vit_fps: int = 2,
        max_frames: int = 81,
    ):
        """Encode ComfyUI video batches -> list of VIT embed tensors + grid_thw."""
        if not videos:
            return [], []
        all_embeds = []
        all_grids = []
        for video in videos:
            frames = _sample_video_frames(video, max_frames=max_frames, frame_factor=2)
            video_inputs = self.processor.video_processor(
                videos=[frames],
                return_tensors="pt",
                size={"shortest_edge": vit_min_pixels, "longest_edge": vit_max_pixels},
            )
            pixel_values = video_inputs["pixel_values_videos"]
            grid_thw = video_inputs["video_grid_thw"]
            embeds = self.get_vit_features(pixel_values, grid_thw)
            all_embeds.extend(embeds)
            all_grids.extend(grid_thw.numpy().tolist())
        return all_embeds, all_grids

    def make_chat_template(self):
        from comfy.bernini.planner_template import BerniniTemplate

        return BerniniTemplate(self.processor.tokenizer)

    def make_position_id_func(self):
        from comfy.bernini.planner_process import make_position_id_func

        return make_position_id_func(self.config_path)

    def offload(self):
        offload = comfy.model_management.unet_offload_device()
        self.to(offload)
        comfy.model_management.soft_empty_cache()
