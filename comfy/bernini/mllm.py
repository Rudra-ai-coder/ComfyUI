# SPDX-License-Identifier: Apache-2.0
"""Bernini Qwen2.5-VL-7B MLLM wrapper (HF folder or bernini_mllm.safetensors)."""

import logging
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch

import comfy.model_management
import folder_paths

LOG = logging.getLogger("bernini.mllm")

# Processor/config fallback (tokenizer + preprocessor JSON, not weight shards).
BERNINI_DIFFUSERS_REPO = "ByteDance/Bernini-Diffusers"
BERNINI_MLLM_SUBFOLDER = "mllm"
MLLM_HUB_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_PROCESSOR_SUBDIR = "mllm_processor"


def _tensor_to_pil(image_tensor: torch.Tensor) -> PIL.Image.Image:
    arr = (image_tensor.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    return PIL.Image.fromarray(arr)


def _sample_video_frames(video: torch.Tensor, max_frames: int, frame_factor: int = 2) -> List[PIL.Image.Image]:
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
    if os.path.isfile(name_or_path):
        return name_or_path
    return folder_paths.get_full_path_or_raise("bernini", name_or_path)


def _processor_dir_candidates(processor_name: str) -> List[str]:
    name = processor_name or DEFAULT_PROCESSOR_SUBDIR
    candidates: List[str] = []
    if os.path.isdir(name):
        candidates.append(name)
    bernini_roots = folder_paths.get_folder_paths("bernini")
    if bernini_roots:
        root = bernini_roots[0]
        candidates.extend([
            os.path.join(root, name),
            os.path.join(root, "mllm"),
            os.path.join(root, DEFAULT_PROCESSOR_SUBDIR),
        ])
    candidates.extend([
        "/workspace/data/Bernini-Diffusers/mllm",
        os.path.expanduser("~/Bernini-Diffusers/mllm"),
    ])
    if os.environ.get("BERNINI_DIFFUSERS_PATH"):
        candidates.append(os.path.join(os.environ["BERNINI_DIFFUSERS_PATH"], "mllm"))
    seen = set()
    out = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _find_local_processor_dir(processor_name: str) -> Optional[str]:
    for candidate in _processor_dir_candidates(processor_name):
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate
    return None


def _load_processor_and_config(processor_name: str):
    from transformers import AutoConfig, AutoProcessor

    local = _find_local_processor_dir(processor_name)
    if local is not None:
        LOG.info("Loading MLLM processor/config from local %s", local)
        processor = AutoProcessor.from_pretrained(local, padding_side="right", trust_remote_code=True)
        config = AutoConfig.from_pretrained(local, trust_remote_code=True)
        return processor, config, local

    LOG.info(
        "No local mllm_processor/ found under models/bernini/ — downloading processor metadata from %s (subfolder %s)",
        BERNINI_DIFFUSERS_REPO,
        BERNINI_MLLM_SUBFOLDER,
    )
    processor = AutoProcessor.from_pretrained(
        BERNINI_DIFFUSERS_REPO,
        subfolder=BERNINI_MLLM_SUBFOLDER,
        padding_side="right",
        trust_remote_code=True,
    )
    config = AutoConfig.from_pretrained(
        BERNINI_DIFFUSERS_REPO,
        subfolder=BERNINI_MLLM_SUBFOLDER,
        trust_remote_code=True,
    )
    return processor, config, BERNINI_DIFFUSERS_REPO


def _create_qwen_mllm_from_config(config, dtype=torch.bfloat16):
    """Match official Bernini: Qwen2_5_VLForConditionalGeneration._from_config."""
    from transformers import Qwen2_5_VLForConditionalGeneration

    if hasattr(Qwen2_5_VLForConditionalGeneration, "_from_config"):
        return Qwen2_5_VLForConditionalGeneration._from_config(config, torch_dtype=dtype)
    if hasattr(Qwen2_5_VLForConditionalGeneration, "from_config"):
        return Qwen2_5_VLForConditionalGeneration.from_config(config, torch_dtype=dtype)
    model = Qwen2_5_VLForConditionalGeneration(config)
    return model.to(dtype=dtype)


def _get_visual_module(model):
    if hasattr(model, "visual") and getattr(model, "visual", None) is not None:
        return model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    raise AttributeError("Could not find Qwen2.5-VL visual module on MLLM model")


class BerniniMLLM:
    """Qwen2.5-VL-7B from bernini_mllm.safetensors + processor metadata, or full HF mllm/ folder."""

    def __init__(self, model, processor, path: str, config_path: str, mllm_config):
        self.model = model
        self.processor = processor
        self.path = path
        self.config_path = config_path
        self._mllm_config = mllm_config
        self.dtype = torch.bfloat16

    @classmethod
    def load(
        cls,
        path: str,
        processor_path: Optional[str] = None,
        device=None,
    ):
        from transformers import Qwen2_5_VLForConditionalGeneration

        if device is None:
            device = comfy.model_management.unet_offload_device()

        if os.path.isdir(path):
            return cls._load_hf_folder(path, device)

        if not path.endswith(".safetensors"):
            raise FileNotFoundError(
                f"Bernini MLLM weights must be bernini_mllm.safetensors in models/bernini/, "
                f"a .safetensors path, or an HF mllm/ directory — got: {path!r}. "
                f"Leave hf_folder empty on BerniniMLLMLoader when using the safetensors file."
            )

        weights_path = resolve_bernini_model_path(path)
        processor, config, config_path = _load_processor_and_config(processor_path or DEFAULT_PROCESSOR_SUBDIR)

        LOG.info("Loading Bernini MLLM weights %s -> %s", weights_path, device)
        model = _create_qwen_mllm_from_config(config, dtype=torch.bfloat16)

        from safetensors.torch import load_file

        state_dict = load_file(weights_path, device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            LOG.warning("Bernini MLLM missing keys (%d): %s", len(missing), missing[:8])
        if unexpected:
            LOG.warning("Bernini MLLM unexpected keys (%d): %s", len(unexpected), unexpected[:8])

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(device)
        return cls(model, processor, weights_path, config_path, config)

    @classmethod
    def _load_hf_folder(cls, path: str, device):
        from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

        LOG.info("Loading Bernini MLLM from HF folder %s -> %s", path, device)
        processor = AutoProcessor.from_pretrained(path, padding_side="right", trust_remote_code=True)
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(device)
        return cls(model, processor, path, path, config)

    def to(self, device, dtype=None):
        if dtype is None:
            dtype = self.dtype
        self.model.to(device=device, dtype=dtype)
        return self

    @torch.no_grad()
    def get_vit_features(self, pixel_values, grid_thw) -> Tuple[torch.Tensor, ...]:
        visual = _get_visual_module(self.model)
        pixel_values = pixel_values.type(self.model.dtype).to(self.model.device)
        grid_thw = grid_thw.to(self.model.device)
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
            image_embeds = visual(pixel_values, grid_thw=grid_thw)
        split_sizes = (grid_thw.prod(-1) // visual.spatial_merge_size ** 2).tolist()
        return torch.split(image_embeds, split_sizes)

    @torch.no_grad()
    def encode_images(
        self,
        images: List[torch.Tensor],
        vit_min_pixels: int = 3136,
        vit_max_pixels: int = 50176,
    ):
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
        from comfy.bernini.planner_process import make_position_id_func_from_config

        return make_position_id_func_from_config(self._mllm_config)

    def offload(self):
        offload = comfy.model_management.unet_offload_device()
        self.to(offload)
        comfy.model_management.soft_empty_cache()
