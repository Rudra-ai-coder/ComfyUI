# SPDX-License-Identifier: Apache-2.0
"""Bernini planner stack loaders + Phase 3 planning pipeline nodes."""

import logging
import os

import folder_paths
import comfy.model_management
import node_helpers
import torch
from comfy.bernini.text import extract_cross_attn, merge_t5_planner, set_merged_conditioning
from comfy.ldm.bernini import DiffLoss_FM, MLPConnector
from safetensors.torch import load_file
from typing_extensions import override

from comfy.bernini.mllm import BerniniMLLM
from comfy.bernini.planner_core import format_mllm_inputs_embeds, post_process_input_embeds, sample_vit_embed
from comfy.bernini.planner_inputs import generate_unified_inputs_from_counts
from comfy.bernini.planner_process import transform_planner_sample
from comfy_api.latest import ComfyExtension, io

LOG = logging.getLogger("bernini.planner")

TASK_NAMES = ["v2v", "rv2v", "i2v", "r2v", "t2v", "t2i"]


class BerniniPlannerWeights:
    """Connector + mask_tokens loaded from bernini_planner.safetensors."""

    def __init__(self, connector: MLPConnector, mask_tokens: torch.Tensor):
        self.connector = connector
        self.mask_tokens = mask_tokens

    def to(self, device, dtype=None):
        self.connector = self.connector.to(device=device, dtype=dtype or torch.bfloat16)
        self.mask_tokens = self.mask_tokens.to(device=device, dtype=dtype or torch.bfloat16)
        return self

    def offload(self):
        offload = comfy.model_management.unet_offload_device()
        self.to(offload)


class BerniniVitDecoderWeights:
    """VIT flow-matching decoder."""

    def __init__(self, vit_decoder: DiffLoss_FM):
        self.vit_decoder = vit_decoder

    def to(self, device, dtype=None):
        self.vit_decoder = self.vit_decoder.to(device=device, dtype=dtype or torch.bfloat16)
        return self

    def offload(self):
        offload = comfy.model_management.unet_offload_device()
        self.to(offload)


class BerniniPlannerInputs:
    """CPU-resident packed tensors for BerniniSemanticPlanning."""

    def __init__(self, data: dict):
        self.data = data


class BerniniPlannerEmbeds:
    """Four renderer planner embedding branches (pre-T5 concat)."""

    def __init__(self, wtxt_wvit, wtxt_wovit, wotxt_wvit, wotxt_wovit, pred_vit_embed=None):
        self.wtxt_wvit = wtxt_wvit
        self.wtxt_wovit = wtxt_wovit
        self.wotxt_wvit = wotxt_wvit
        self.wotxt_wovit = wotxt_wovit
        self.pred_vit_embed = pred_vit_embed


def _load_split_state_dict(path: str) -> dict:
    if not os.path.isfile(path):
        path = folder_paths.get_full_path_or_raise("bernini", path)
    return load_file(path, device="cpu")


def _move_tensors_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: _move_tensors_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_tensors_to_cpu(v) for v in obj]
    return obj


def _move_tensors_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_tensors_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_tensors_to_device(v, device) for v in obj]
    return obj


class BerniniMLLMLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniMLLMLoader",
            display_name="Bernini MLLM Loader",
            category="loaders/bernini",
            description="Load Qwen2.5-VL-7B from models/bernini/bernini_mllm.safetensors "
                        "(convert script output) or a full HF mllm/ directory path. "
                        "Tokenizer/processor: models/bernini/mllm_processor/ (no weights).",
            inputs=[
                io.Combo.Input(
                    "weights",
                    options=folder_paths.get_filename_list("bernini"),
                    tooltip="Place in ComfyUI/models/bernini/. Use bernini_mllm.safetensors from convert script.",
                ),
                io.String.Input(
                    "hf_folder",
                    default="",
                    optional=True,
                    tooltip="Optional: absolute path to full HF mllm/ folder (overrides weights combo).",
                ),
                io.String.Input(
                    "processor_folder",
                    default="mllm_processor",
                    optional=True,
                    tooltip="Subfolder under models/bernini/ with tokenizer + preprocessor JSON "
                            "(no weight shards). Download from Bernini-Diffusers mllm/ or Qwen hub.",
                ),
            ],
            outputs=[io.Custom("BERNINI_MLLM").Output(display_name="mllm")],
        )

    @classmethod
    def execute(cls, weights, processor_folder="mllm_processor", hf_folder="") -> io.NodeOutput:
        path = hf_folder.strip() if hf_folder and hf_folder.strip() else weights
        mllm = BerniniMLLM.load(path, processor_path=processor_folder)
        return io.NodeOutput(mllm)


class BerniniPlannerLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniPlannerLoader",
            display_name="Bernini Planner Loader",
            category="loaders/bernini",
            description="Load connector + mask_tokens from bernini_planner.safetensors "
                        "(output of tools/convert_bernini_weights.py).",
            inputs=[
                io.String.Input("path", default="bernini_planner.safetensors"),
            ],
            outputs=[io.Custom("BERNINI_PLANNER").Output(display_name="planner")],
        )

    @classmethod
    def execute(cls, path) -> io.NodeOutput:
        sd = _load_split_state_dict(path)
        connector_sd = {k.removeprefix("connector."): v for k, v in sd.items() if k.startswith("connector.")}
        mask = sd.get("mask_tokens")
        if mask is None:
            raise KeyError("mask_tokens not found in planner checkpoint")

        connector = MLPConnector(in_dim=3584)
        missing, unexpected = connector.load_state_dict(connector_sd, strict=False)
        if missing:
            LOG.warning("Bernini planner connector missing keys: %s", missing[:8])
        if unexpected:
            LOG.warning("Bernini planner connector unexpected keys: %s", unexpected[:8])

        connector.eval()
        for p in connector.parameters():
            p.requires_grad_(False)

        offload = comfy.model_management.unet_offload_device()
        planner = BerniniPlannerWeights(connector.to(device=offload, dtype=torch.bfloat16), mask)
        return io.NodeOutput(planner)


class BerniniVitDecoderLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniVitDecoderLoader",
            display_name="Bernini VIT Decoder Loader",
            category="loaders/bernini",
            description="Load DiffLoss_FM VIT decoder from bernini_vit_decoder.safetensors.",
            inputs=[
                io.String.Input("path", default="bernini_vit_decoder.safetensors"),
            ],
            outputs=[io.Custom("BERNINI_VIT_DECODER").Output(display_name="vit_decoder")],
        )

    @classmethod
    def execute(cls, path) -> io.NodeOutput:
        sd = _load_split_state_dict(path)
        vit_sd = {k.removeprefix("vit_decoder."): v for k, v in sd.items() if k.startswith("vit_decoder.")}
        vit_decoder = DiffLoss_FM(
            z_channels=3584,
            target_channels=3584,
            depth=16,
            width=4096,
            shift=2.0,
        )
        missing, unexpected = vit_decoder.load_state_dict(vit_sd, strict=False)
        if missing:
            LOG.warning("Bernini vit_decoder missing keys: %s", missing[:8])
        if unexpected:
            LOG.warning("Bernini vit_decoder unexpected keys: %s", unexpected[:8])

        vit_decoder.eval()
        for p in vit_decoder.parameters():
            p.requires_grad_(False)

        offload = comfy.model_management.unet_offload_device()
        wrapper = BerniniVitDecoderWeights(vit_decoder.to(device=offload, dtype=torch.bfloat16))
        return io.NodeOutput(wrapper)


class BerniniPreparePlannerInputs(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniPreparePlannerInputs",
            display_name="Bernini Prepare Planner Inputs",
            category="conditioning/bernini",
            description="Pack prompt + optional images/video into Bernini planner tensors "
                        "(cond / uncond / imgcond branches). Outputs CPU tensors.",
            inputs=[
                io.Custom("BERNINI_MLLM").Input("mllm"),
                io.String.Input("prompt", multiline=True, default=""),
                io.Combo.Input("task_name", options=TASK_NAMES, default="v2v"),
                io.Int.Input("width", default=832, min=16, max=8192, step=16),
                io.Int.Input("height", default=480, min=16, max=8192, step=16),
                io.Int.Input("length", default=81, min=1, max=8192, step=4),
                io.String.Input("neg_prompt", multiline=True, default="", optional=True),
                io.Image.Input("source_video", optional=True, tooltip="Source video for v2v/rv2v."),
                io.Autogrow.Input(
                    "reference_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("reference_image"),
                        prefix="reference_image_",
                        min=0,
                        max=8,
                    ),
                ),
                io.Int.Input("vit_min_pixels", default=3136, min=256, max=1048576, advanced=True),
                io.Int.Input("vit_max_pixels", default=50176, min=256, max=1048576, advanced=True),
            ],
            outputs=[io.Custom("BERNINI_PLANNER_INPUTS").Output(display_name="planner_inputs")],
        )

    @classmethod
    def execute(
        cls,
        mllm: BerniniMLLM,
        prompt,
        task_name,
        width,
        height,
        length,
        neg_prompt="",
        source_video=None,
        reference_images=None,
        vit_min_pixels=3136,
        vit_max_pixels=50176,
    ) -> io.NodeOutput:
        device = comfy.model_management.get_torch_device()
        load_device = comfy.model_management.text_encoder_device()
        mllm.to(load_device)

        ref_image_list = []
        if reference_images:
            for name in sorted(reference_images):
                imgs = reference_images[name]
                if imgs is not None:
                    for i in range(imgs.shape[0]):
                        ref_image_list.append(imgs[i : i + 1])

        num_videos = 0
        video_embeds, video_grid_thw = [], []
        if source_video is not None:
            num_videos = 1
            ve, vg = mllm.encode_videos(
                [source_video],
                vit_min_pixels=vit_min_pixels,
                vit_max_pixels=vit_max_pixels,
                max_frames=length,
            )
            video_embeds.extend(ve)
            video_grid_thw.extend(vg)

        image_embeds, image_grid_thw = [], []
        if ref_image_list:
            ie, ig = mllm.encode_images(
                ref_image_list,
                vit_min_pixels=vit_min_pixels,
                vit_max_pixels=vit_max_pixels,
            )
            image_embeds.extend(ie)
            image_grid_thw.extend(ig)

        output_t = 1 if length == 1 else length

        # Target output VIT placeholder (official pipeline duplicates source or uses fake).
        if output_t == 1:
            if ref_image_list or source_video is not None:
                placeholder = ref_image_list[0] if ref_image_list else source_video[0:1]
            else:
                placeholder = torch.zeros((1, height, width, 3))
            ie, ig = mllm.encode_images(
                [placeholder],
                vit_min_pixels=vit_min_pixels,
                vit_max_pixels=vit_max_pixels,
            )
            image_embeds.extend(ie)
            image_grid_thw.extend(ig)
        else:
            if source_video is not None:
                ve, vg = mllm.encode_videos(
                    [source_video],
                    vit_min_pixels=vit_min_pixels,
                    vit_max_pixels=vit_max_pixels,
                    max_frames=length,
                )
            else:
                fake_vid = torch.zeros((length, height, width, 3))
                ve, vg = mllm.encode_videos(
                    [fake_vid],
                    vit_min_pixels=vit_min_pixels,
                    vit_max_pixels=vit_max_pixels,
                    max_frames=length,
                )
            video_embeds.extend(ve)
            video_grid_thw.extend(vg)
        inputs_json = generate_unified_inputs_from_counts(
            prompt,
            num_input_videos=num_videos,
            num_input_images=len(ref_image_list),
            output_t=output_t,
            output_h=height,
            output_w=width,
        )

        sample = {
            "inputs": inputs_json,
            "image_embeds": image_embeds,
            "image_grid_thw": image_grid_thw,
            "video_embeds": video_embeds,
            "video_grid_thw": video_grid_thw,
        }

        chat_template = mllm.make_chat_template()
        position_id_func = mllm.make_position_id_func()
        packed = transform_planner_sample(
            sample,
            processor=mllm.processor,
            chat_template=chat_template,
            position_id_func=position_id_func,
            task_name=task_name,
            neg_prompt=neg_prompt or "",
        )
        packed = _move_tensors_to_cpu(packed)
        mllm.offload()
        return io.NodeOutput(BerniniPlannerInputs(packed))


class BerniniSemanticPlanning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniSemanticPlanning",
            display_name="Bernini Semantic Planning",
            category="conditioning/bernini",
            description="Run MaskGIT planning (~25 steps) with Qwen + connector + VIT decoder. "
                        "Outputs four 4096-d embedding branches for renderer guidance. "
                        "Offloads planner models to CPU after execution.",
            inputs=[
                io.Custom("BERNINI_MLLM").Input("mllm"),
                io.Custom("BERNINI_PLANNER").Input("planner"),
                io.Custom("BERNINI_VIT_DECODER").Input("vit_decoder"),
                io.Custom("BERNINI_PLANNER_INPUTS").Input("planner_inputs"),
                io.Int.Input("planning_step", default=25, min=1, max=100),
                io.Float.Input("vit_txt_cfg", default=1.4, min=1.0, max=10.0, step=0.05),
                io.Float.Input("vit_img_cfg", default=1.2, min=1.0, max=10.0, step=0.05),
                io.Int.Input("vit_denoising_step", default=3, min=1, max=20),
            ],
            outputs=[
                io.Custom("BERNINI_PLANNER_EMBEDS").Output(display_name="planner_embeds"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mllm: BerniniMLLM,
        planner: BerniniPlannerWeights,
        vit_decoder: BerniniVitDecoderWeights,
        planner_inputs: BerniniPlannerInputs,
        planning_step=25,
        vit_txt_cfg=1.4,
        vit_img_cfg=1.2,
        vit_denoising_step=3,
    ) -> io.NodeOutput:
        device = comfy.model_management.get_torch_device()
        dtype = torch.bfloat16

        mllm.to(device, dtype=dtype)
        planner.to(device, dtype=dtype)
        vit_decoder.to(device, dtype=dtype)

        data = _move_tensors_to_device(planner_inputs.data, device)
        inputs = data["inputs"]
        uncond_inputs = data["uncond_inputs"]
        imgcond_inputs = data["imgcond_inputs"]

        def _prepare_branch(branch):
            input_embeds = format_mllm_inputs_embeds(
                mllm.model,
                branch["input_ids"].unsqueeze(0),
                branch["visual_embeds"],
                branch["visual_input_token_mask"],
                branch["visual_output_token_mask"],
            )
            post = post_process_input_embeds(
                input_embeds.unsqueeze(0),
                branch["visual_output_token_mask"],
                planner.mask_tokens,
                inference=True,
            )
            return post["input_embeds"].squeeze(0)

        inputs_embed = _prepare_branch(inputs)
        uncond_inputs_embed = _prepare_branch(uncond_inputs)
        imgcond_inputs_embed = _prepare_branch(imgcond_inputs)

        ret = sample_vit_embed(
            mllm.model,
            planner.connector,
            vit_decoder.vit_decoder,
            planner.mask_tokens,
            input_embeds=inputs_embed.unsqueeze(0),
            position_ids=inputs["position_ids"].unsqueeze(0),
            attention_mask_4d=inputs["attention_mask_4d"],
            visual_output_token_mask=inputs["visual_output_token_mask"],
            uncond_input_embeds=uncond_inputs_embed.unsqueeze(0),
            uncond_position_ids=uncond_inputs["position_ids"].unsqueeze(0),
            uncond_attention_mask_4d=uncond_inputs["attention_mask_4d"],
            uncond_visual_output_token_mask=uncond_inputs["visual_output_token_mask"],
            imgcond_input_embeds=imgcond_inputs_embed.unsqueeze(0),
            imgcond_position_ids=imgcond_inputs["position_ids"].unsqueeze(0),
            imgcond_attention_mask_4d=imgcond_inputs["attention_mask_4d"],
            imgcond_visual_output_token_mask=imgcond_inputs["visual_output_token_mask"],
            planning_step=planning_step,
            vit_denoising_step=vit_denoising_step,
            vit_txt_cfg=vit_txt_cfg,
            vit_img_cfg=vit_img_cfg,
        )

        embeds = BerniniPlannerEmbeds(
            wtxt_wvit=_move_tensors_to_cpu(ret["cond_embeds_wtxt_wvit"]),
            wtxt_wovit=_move_tensors_to_cpu(ret["cond_embeds_wtxt_wovit"]) if ret["cond_embeds_wtxt_wovit"] is not None else None,
            wotxt_wvit=_move_tensors_to_cpu(ret["cond_embeds_wotxt_wvit"]) if ret["cond_embeds_wotxt_wvit"] is not None else None,
            wotxt_wovit=_move_tensors_to_cpu(ret["cond_embeds_wotxt_wovit"]),
            pred_vit_embed=_move_tensors_to_cpu(ret["pred_vit_embed"]),
        )

        mllm.offload()
        planner.offload()
        vit_decoder.offload()
        comfy.model_management.soft_empty_cache()

        return io.NodeOutput(embeds)


class BerniniMergePlannerText(io.ComfyNode):
    """Concat UMT5 cross-attn with four planner branches for full Bernini renderer."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniMergePlannerText",
            display_name="Bernini Merge Planner Text",
            category="conditioning/bernini",
            description="Merge UMT5 embeddings with planner branches (wtxt_wvit, wtxt_wovit, "
                        "wotxt_wvit, wotxt_wovit), pad/truncate to max_sequence_length for vae_txt_vit_wapg.",
            inputs=[
                io.Conditioning.Input("positive", tooltip="UMT5-encoded positive prompt."),
                io.Conditioning.Input("negative", tooltip="UMT5-encoded negative prompt."),
                io.Custom("BERNINI_PLANNER_EMBEDS").Input("planner_embeds"),
                io.Int.Input("max_sequence_length", default=512, min=64, max=2048, step=1),
                io.Boolean.Input("truncate", default=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        planner_embeds: BerniniPlannerEmbeds,
        max_sequence_length=512,
        truncate=True,
    ) -> io.NodeOutput:
        pos_t5 = extract_cross_attn(positive)
        neg_t5 = extract_cross_attn(negative)
        if pos_t5 is None or neg_t5 is None:
            raise ValueError("BerniniMergePlannerText requires UMT5 cross_attn on positive and negative conditioning.")

        merged = {
            "wtxt_wvit": merge_t5_planner(pos_t5, planner_embeds.wtxt_wvit, max_sequence_length, truncate),
            "wtxt_wovit": merge_t5_planner(pos_t5, planner_embeds.wtxt_wovit, max_sequence_length, truncate),
            "wotxt_wvit": merge_t5_planner(neg_t5, planner_embeds.wotxt_wvit, max_sequence_length, truncate),
            "wotxt_wovit": merge_t5_planner(neg_t5, planner_embeds.wotxt_wovit, max_sequence_length, truncate),
        }

        positive = set_merged_conditioning(
            positive,
            merged["wtxt_wvit"],
            {
                "bernini_text_wtxt_wvit": merged["wtxt_wvit"],
                "bernini_text_wtxt_wovit": merged["wtxt_wovit"],
                "bernini_text_wotxt_wvit": merged["wotxt_wvit"],
                "bernini_text_wotxt_wovit": merged["wotxt_wovit"],
            },
        )
        negative = set_merged_conditioning(negative, merged["wotxt_wovit"])
        return io.NodeOutput(positive, negative)


class BerniniPlannerExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            BerniniMLLMLoader,
            BerniniPlannerLoader,
            BerniniVitDecoderLoader,
            BerniniPreparePlannerInputs,
            BerniniSemanticPlanning,
            BerniniMergePlannerText,
        ]


async def comfy_entrypoint() -> BerniniPlannerExtension:
    return BerniniPlannerExtension()
