# SPDX-License-Identifier: Apache-2.0
"""Bernini planner input structure builder (v2v / i2v / r2v inference)."""

import json
from typing import List, Optional


def generate_unified_inputs(
    prompt: str,
    input_image_paths=None,
    input_video_paths=None,
    has_video_input: bool = False,
    output_t: int = 81,
    output_h: int = 480,
    output_w: int = 832,
) -> str:
    """Build Bernini conversation JSON matching official generate_unified_inputs."""
    input_image_paths = [p for p in (input_image_paths or []) if p is not None]
    if input_video_paths is None:
        input_video_paths = [None] if has_video_input else []
    else:
        input_video_paths = [p for p in input_video_paths if p is not None]

    video_index = 0
    inputs_structure = [{"type": "special_token", "text": "[CLS]", "has_loss": 0}]

    for _ in input_video_paths:
        inputs_structure.append({
            "type": "video",
            "video_index": video_index,
            "decode_mode": "video",
        })
        video_index += 1

    h, w = output_h, output_w
    for i, _img in enumerate(input_image_paths):
        idx = i + len(input_video_paths)
        inputs_structure.append({
            "type": "image",
            "image_index": idx,
            "height": h,
            "width": w,
        })

    inputs_structure.append({"type": "text", "text": prompt, "has_loss": 0})

    if output_t == 1:
        inputs_structure.extend([{"type": "special_token", "text": "[SOG]", "has_loss": 1}])
        target_idx = len(input_image_paths) if input_image_paths else 0
        inputs_structure.append({
            "type": "image_gen",
            "image_index": target_idx,
            "height": output_h,
            "width": output_w,
            "has_loss": 1,
        })
        inputs_structure.extend([{"type": "special_token", "text": "[EOG]", "has_loss": 1}])
    else:
        inputs_structure.extend([{"type": "special_token", "text": "[SOV]", "has_loss": 1}])
        inputs_structure.append({
            "type": "video_gen",
            "video_index": video_index,
            "decode_mode": "video",
        })
        inputs_structure.extend([{"type": "special_token", "text": "[EOV]", "has_loss": 1}])

    inputs_structure.append({"type": "special_token", "text": "[EOS]", "has_loss": 1})
    return json.dumps(inputs_structure, ensure_ascii=False)


def generate_unified_inputs_from_counts(
    prompt: str,
    num_input_videos: int = 0,
    num_input_images: int = 0,
    output_t: int = 81,
    output_h: int = 480,
    output_w: int = 832,
) -> str:
    """Like generate_unified_inputs but uses counts instead of paths."""
    video_paths = [True] * num_input_videos
    image_paths = [True] * num_input_images
    return generate_unified_inputs(
        prompt,
        input_image_paths=image_paths,
        input_video_paths=video_paths,
        has_video_input=num_input_videos > 0,
        output_t=output_t,
        output_h=output_h,
        output_w=output_w,
    )
