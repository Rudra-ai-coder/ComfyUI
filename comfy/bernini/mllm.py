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
# Qwen2.5-VL-7B bf16 ≈ 14GB — used for load-device heuristics.
MLLM_WEIGHT_BYTES = 14 * 1024 ** 3

# One BerniniMLLM per (weights path, processor folder) for the ComfyUI process.
_MLLM_CACHE: dict[tuple[str, str], "BerniniMLLM"] = {}


def _mllm_load_device():
    """GPU when VRAM allows (prefer direct GPU read over CPU staging)."""
    load_device = comfy.model_management.text_encoder_device()
    offload_device = comfy.model_management.text_encoder_offload_device()
    if comfy.model_management.args.gpu_only:
        return load_device
    if load_device.type == "cuda":
        free = comfy.model_management.get_free_memory(load_device)
        if free > MLLM_WEIGHT_BYTES * 1.15:
            return load_device
    return comfy.model_management.text_encoder_initial_device(
        load_device, offload_device, model_size=MLLM_WEIGHT_BYTES
    )


def _safetensors_device(target: torch.device) -> str:
    if target.type == "cuda":
        return str(target)
    return "cpu"


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
        return Qwen2_5_VLForConditionalGeneration._from_config(config, dtype=dtype)
    if hasattr(Qwen2_5_VLForConditionalGeneration, "from_config"):
        return Qwen2_5_VLForConditionalGeneration.from_config(config, dtype=dtype)
    model = Qwen2_5_VLForConditionalGeneration(config)
    return model.to(dtype=dtype)


def _remap_mllm_state_dict_key(key: str) -> str:
    """Map bernini_mllm export keys to HF Qwen2_5_VLForConditionalGeneration."""
    if key.startswith("lm_head."):
        return key
    if not key.startswith("model."):
        key = "model." + key
    # Bernini shard: model.visual.* + model.layers.* (Qwen2_5_VLModel language stack).
    # HF ForConditionalGeneration: model.visual.* + model.language_model.layers.*
    if (
        key.startswith("model.layers.")
        or key.startswith("model.embed_tokens.")
        or key == "model.norm.weight"
    ):
        key = "model.language_model." + key[len("model."):]
    return key


def _remap_bernini_mllm_state_dict(state_dict: dict) -> dict:
    out = {}
    for key, tensor in state_dict.items():
        out[_remap_mllm_state_dict_key(key)] = tensor
    return out


def _extract_visual_embeds(output) -> torch.Tensor:
    """HF transformers visual returns BaseModelOutputWithPooling: pooler_output is post-merger (official Bernini path)."""
    if torch.is_tensor(output):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state
    raise TypeError(f"Unexpected Qwen2.5-VL visual forward output type: {type(output)}")


def _get_visual_module(model):
    if hasattr(model, "visual") and getattr(model, "visual", None) is not None:
        return model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    raise AttributeError("Could not find Qwen2.5-VL visual module on MLLM model")


def _mllm_cache_key(path: str, processor_name: str) -> tuple[str, str]:
    proc = processor_name or DEFAULT_PROCESSOR_SUBDIR
    if os.path.isdir(path):
        return (os.path.abspath(path), proc)
    return (resolve_bernini_model_path(path), proc)


class BerniniMLLM:
    """Qwen2.5-VL-7B from bernini_mllm.safetensors + processor metadata, or full HF mllm/ folder."""

    def __init__(
        self,
        processor,
        config,
        config_path: str,
        weights_path: Optional[str] = None,
        hf_folder_path: Optional[str] = None,
        model=None,
    ):
        self.processor = processor
        self._mllm_config = config
        self.config_path = config_path
        self._weights_path = weights_path
        self._hf_folder_path = hf_folder_path
        self.path = weights_path or hf_folder_path or config_path
        self.model = model
        self._weights_loaded = model is not None
        self.dtype = torch.bfloat16

    @classmethod
    def open(cls, path: str, processor_path: Optional[str] = None):
        """
        Fast open: processor/config only. ~15GB weights load on first ensure_weights_on_device().
        Cached per process so re-queued workflows skip disk I/O after the first planning pass.
        """
        key = _mllm_cache_key(path, processor_path or DEFAULT_PROCESSOR_SUBDIR)
        cached = _MLLM_CACHE.get(key)
        if cached is not None:
            LOG.info("Reusing cached Bernini MLLM (%s)", key[0])
            return cached

        processor_name = processor_path or DEFAULT_PROCESSOR_SUBDIR
        if os.path.isdir(path):
            processor, config, config_path = _load_processor_and_config(processor_name)
            hf_path = os.path.abspath(path)
            LOG.info(
                "Bernini MLLM opened from HF folder %s (weights deferred until planning)",
                hf_path,
            )
            inst = cls(
                processor=processor,
                config=config,
                config_path=config_path,
                hf_folder_path=hf_path,
            )
        else:
            weights_path = resolve_bernini_model_path(path)
            processor, config, config_path = _load_processor_and_config(processor_name)
            size_gb = os.path.getsize(weights_path) / (1024 ** 3)
            LOG.info(
                "Bernini MLLM opened (%s, %.1f GB on disk — weights deferred until planning)",
                weights_path,
                size_gb,
            )
            inst = cls(
                processor=processor,
                config=config,
                config_path=config_path,
                weights_path=weights_path,
            )

        _MLLM_CACHE[key] = inst
        return inst

    @classmethod
    def load(cls, path: str, processor_path: Optional[str] = None, device=None):
        """Load weights immediately (legacy). Prefer open() + ensure_weights_on_device()."""
        mllm = cls.open(path, processor_path=processor_path)
        mllm.ensure_weights_on_device(device)
        return mllm

    def _current_device(self) -> Optional[torch.device]:
        if self.model is None:
            return None
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return None

    def ensure_weights_on_device(self, device=None):
        """Load weights from disk on first call; later calls only move between devices (skip if already there)."""
        if device is None:
            device = _mllm_load_device()
        if not self._weights_loaded:
            self._load_weights(device)
            return
        if self.model is not None:
            cur = self._current_device()
            if cur is None or cur.type != device.type or (device.type == "cuda" and cur.index != device.index):
                LOG.info("MLLM: moving %s → %s", cur, device)
                self.model.to(device)

    def _load_weights(self, device):
        if self._hf_folder_path is not None:
            self._load_hf_weights(self._hf_folder_path, device)
            return

        if not self._weights_path or not self._weights_path.endswith(".safetensors"):
            raise FileNotFoundError(
                f"Bernini MLLM weights must be bernini_mllm.safetensors in models/bernini/, "
                f"a .safetensors path, or an HF mllm/ directory — got: {self._weights_path!r}."
            )

        weights_path = self._weights_path
        size_gb = os.path.getsize(weights_path) / (1024 ** 3)
        sd_device = _safetensors_device(device)
        LOG.info(
            "Loading Bernini MLLM weights %s (%.1f GB on disk) -> %s (safetensors read on %s)",
            weights_path,
            size_gb,
            device,
            sd_device,
        )

        if device.type == "cuda":
            comfy.model_management.soft_empty_cache()

        model = _create_qwen_mllm_from_config(self._mllm_config, dtype=torch.bfloat16)
        if device.type != "cpu":
            model = model.to(device=device)

        from safetensors.torch import load_file

        state_dict = _remap_bernini_mllm_state_dict(
            load_file(weights_path, device=sd_device)
        )
        LOG.info("MLLM safetensors read complete (%d tensors), applying to model...", len(state_dict))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        LOG.info("MLLM state dict applied (missing=%d, unexpected=%d)", len(missing), len(unexpected))
        if missing:
            LOG.warning("Bernini MLLM missing keys (%d): %s", len(missing), missing[:8])
        if unexpected:
            LOG.warning("Bernini MLLM unexpected keys (%d): %s", len(unexpected), unexpected[:8])

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if device.type == "cpu":
            model.to(device)

        self.model = model
        self._weights_loaded = True

    def _load_hf_weights(self, path: str, device):
        from transformers import Qwen2_5_VLForConditionalGeneration

        LOG.info("Loading Bernini MLLM from HF folder %s -> %s", path, device)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(device)
        self.model = model
        self._weights_loaded = True

    def to(self, device, dtype=None):
        self.ensure_weights_on_device(device)
        if dtype is None:
            dtype = self.dtype
        self.model.to(device=device, dtype=dtype)
        return self

    @torch.no_grad()
    def get_vit_features(self, pixel_values, grid_thw) -> Tuple[torch.Tensor, ...]:
        self.ensure_weights_on_device()
        visual = _get_visual_module(self.model)
        pixel_values = pixel_values.type(self.model.dtype).to(self.model.device)
        grid_thw = grid_thw.to(self.model.device)
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
            image_embeds = _extract_visual_embeds(visual(pixel_values, grid_thw=grid_thw))
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
        if not self._weights_loaded or self.model is None:
            return
        self.to(comfy.model_management.text_encoder_offload_device())
        comfy.model_management.soft_empty_cache()
