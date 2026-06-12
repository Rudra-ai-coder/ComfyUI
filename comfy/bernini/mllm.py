# SPDX-License-Identifier: Apache-2.0
"""Bernini Qwen2.5-VL-7B MLLM wrapper (transformers HF layout)."""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import PIL.Image
import torch

import comfy.model_management

LOG = logging.getLogger("bernini.mllm")


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


class BerniniMLLM:
    """Qwen2.5-VL-7B loaded from Bernini-Diffusers mllm/ folder."""

    def __init__(self, model, processor, path: str):
        self.model = model
        self.processor = processor
        self.path = path
        self.dtype = torch.bfloat16

    @classmethod
    def load(cls, path: str, device=None):
        from transformers import AutoProcessor, Qwen2_5_VLModel

        if not os.path.isdir(path):
            raise FileNotFoundError(f"MLLM path not found: {path}")

        if device is None:
            device = comfy.model_management.unet_offload_device()

        LOG.info("Loading Bernini MLLM from %s -> %s", path, device)
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
        return cls(model, processor, path)

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

        return make_position_id_func(self.path)

    def offload(self):
        offload = comfy.model_management.unet_offload_device()
        self.to(offload)
        comfy.model_management.soft_empty_cache()
