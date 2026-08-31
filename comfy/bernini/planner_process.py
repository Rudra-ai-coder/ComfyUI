# SPDX-License-Identifier: Apache-2.0
"""Bernini bernini_process_sample inference subset for ComfyUI."""

import json
import os
import random
from typing import Any, Callable, Dict, Optional

import torch
from transformers import AutoConfig, Qwen2_5_VLModel

from comfy.bernini.planner_template import BerniniTemplate


def get_drop_condition(text_dropout_rate, img_dropout_rate, video_dropout_rate):
    drop_text, drop_video, drop_img = 0, 0, 0
    if random.random() < text_dropout_rate:
        drop_text = 1
    if random.random() < img_dropout_rate:
        drop_img = 1
    if random.random() < video_dropout_rate:
        drop_video = 1
    return drop_text, drop_video, drop_img


class _QwenRopeIndexHelper:
    """Delegate HF Qwen2_5_VLModel.get_rope_index (needs get_vision_position_ids on self)."""

    def __init__(self, config):
        self.config = config
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id

    get_vision_position_ids = Qwen2_5_VLModel.get_vision_position_ids
    get_rope_index = Qwen2_5_VLModel.get_rope_index


def _build_mm_token_type_ids(input_ids: torch.Tensor, image_token_id: int, video_token_id: int) -> torch.Tensor:
    """Match HF processor: text=0, image=1, video=2 (after BerniniTemplate rewrites visual pads)."""
    mm = torch.zeros_like(input_ids, dtype=torch.int32)
    mm[input_ids == image_token_id] = 1
    mm[input_ids == video_token_id] = 2
    return mm


def bernini_process_sample(
    sample: Dict[str, Any],
    processor,
    chat_template: BerniniTemplate,
    position_id_func: Callable,
    text_dropout_rate: float = 0.0,
    img_dropout_rate: float = 0.0,
    video_dropout_rate: float = 0.0,
    vit_mask_ratio: float = 1.0,
    neg_prompt: Optional[str] = "",
    **kwargs,
):
    """Process one Bernini sample into planner tensors (inference path)."""
    task_name = kwargs.get("source_name", "v2v").split("$")[0].lower()
    drop_text, drop_video, drop_img = get_drop_condition(
        text_dropout_rate, img_dropout_rate, video_dropout_rate
    )

    conversations = json.loads(sample["inputs"])
    token_num_inputs = {}

    raw_image_embeds = sample.get("image_embeds") or []
    raw_image_grid_thw = sample.get("image_grid_thw") or []
    if raw_image_embeds:
        merge_length = processor.image_processor.merge_size ** 2
        token_num_inputs["image"] = (
            torch.tensor(raw_image_grid_thw).prod(dim=-1) // merge_length
        )

    raw_video_embeds = sample.get("video_embeds") or []
    raw_video_grid_thw = sample.get("video_grid_thw") or []
    if raw_video_embeds:
        merge_length = processor.image_processor.merge_size ** 2
        token_num_inputs["video"] = (
            torch.tensor(raw_video_grid_thw).prod(dim=-1) // merge_length
        )

    tokenized_example = chat_template.encode_messages(
        conversations,
        token_num_inputs,
        task_name,
        drop_text=drop_text,
        drop_video=drop_video,
        drop_img=drop_img,
        vit_mask_ratio=vit_mask_ratio,
        neg_prompt=neg_prompt,
        **kwargs,
    )

    vit_type_list = tokenized_example.pop("vit_type_list")
    vit_img_and_vid_id_list = tokenized_example.pop("vit_img_and_vid_id_list")

    visual_embeds = []
    image_grid_thw, video_grid_thw = [], []
    for vit_type, vit_id in zip(vit_type_list, vit_img_and_vid_id_list):
        if vit_type == 0:
            image_grid_thw.append(raw_image_grid_thw[vit_id])
            visual_embeds.append(raw_image_embeds[vit_id])
        elif vit_type == 1:
            video_grid_thw.append(raw_video_grid_thw[vit_id])
            visual_embeds.append(raw_video_embeds[vit_id])

    if image_grid_thw:
        image_grid_thw = torch.tensor(image_grid_thw)
    else:
        image_grid_thw = None
    if video_grid_thw:
        video_grid_thw = torch.tensor(video_grid_thw)
    else:
        video_grid_thw = None

    if visual_embeds:
        tgt_device = visual_embeds[0].device
        tokenized_example["visual_embeds"] = torch.cat(
            [e.to(tgt_device) for e in visual_embeds], dim=0
        )
    else:
        tokenized_example["visual_embeds"] = torch.zeros(0, 3584)

    input_ids = tokenized_example["input_ids"]
    tokenized_example["position_ids"] = position_id_func(
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=tokenized_example["attention_mask"].unsqueeze(0),
    )[0].squeeze(1).clone()
    tokenized_example["mllm_seqlen"] = tokenized_example["attention_mask"].sum().reshape(1)
    tokenized_example["task_name"] = task_name
    return tokenized_example


def make_position_id_func_from_config(config):
    """Build Qwen2.5-VL rope index function from a loaded config."""
    helper = _QwenRopeIndexHelper(config)

    def position_id_func(
        input_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        rope_kwargs = {
            "input_ids": input_ids,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "attention_mask": attention_mask,
            "second_per_grid_ts": second_per_grid_ts,
        }
        mm_token_type_ids = _build_mm_token_type_ids(
            input_ids, config.image_token_id, config.video_token_id
        )
        rope_kwargs["mm_token_type_ids"] = mm_token_type_ids
        try:
            return helper.get_rope_index(**rope_kwargs)
        except TypeError:
            rope_kwargs.pop("mm_token_type_ids", None)
            return helper.get_rope_index(**rope_kwargs)

    return position_id_func


def make_position_id_func(mllm_path: str):
    """Build Qwen2.5-VL rope index function from mllm config path or HF repo id."""
    from comfy.bernini.mllm import BERNINI_DIFFUSERS_REPO, BERNINI_MLLM_SUBFOLDER

    if os.path.isdir(mllm_path):
        config = AutoConfig.from_pretrained(mllm_path, trust_remote_code=True)
    elif mllm_path == BERNINI_DIFFUSERS_REPO:
        config = AutoConfig.from_pretrained(
            BERNINI_DIFFUSERS_REPO, subfolder=BERNINI_MLLM_SUBFOLDER, trust_remote_code=True
        )
    else:
        config = AutoConfig.from_pretrained(mllm_path, trust_remote_code=True)
    return make_position_id_func_from_config(config)


def transform_planner_sample(
    sample: dict,
    processor,
    chat_template: BerniniTemplate,
    position_id_func: Callable,
    task_name: str = "v2v",
    neg_prompt: str = "",
    use_qwen_neg_prompt: bool = True,
) -> dict:
    """Build cond / uncond / imgcond tokenized branches (official transform_inputs)."""

    def _run(drop_text, drop_img, drop_video, neg=None):
        return bernini_process_sample(
            sample.copy(),
            processor=processor,
            chat_template=chat_template,
            position_id_func=position_id_func,
            text_dropout_rate=drop_text,
            img_dropout_rate=drop_img,
            video_dropout_rate=drop_video,
            source_name=task_name,
            neg_prompt=neg or "",
        )

    tokenized = _run(0.0, 0.0, 0.0)
    imgcond = _run(0.0, 1.0, 1.0)
    uncond_neg = neg_prompt if use_qwen_neg_prompt else ""
    uncond = _run(1.0, 1.0, 1.0, neg=uncond_neg)

    return {
        "inputs": tokenized,
        "uncond_inputs": uncond,
        "imgcond_inputs": imgcond,
        "task_name": task_name,
    }
