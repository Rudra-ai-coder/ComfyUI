import hashlib
import json
import os

import torch
from typing_extensions import override

import comfy.nested_tensor
import comfy.utils
import folder_paths
from comfy.cli_args import args
from comfy_api.latest import ComfyExtension, io, ui


def _pack_value(value, tensors):
    if torch.is_tensor(value):
        key = "t{}".format(len(tensors))
        tensors[key] = value.contiguous().cpu()
        return {"t": "tensor", "k": key}
    if getattr(value, "is_nested", False):
        return {"t": "nested", "s": [_pack_value(x, tensors) for x in value.unbind()]}
    if isinstance(value, dict):
        return {"t": "dict", "v": {k: _pack_value(v, tensors) for k, v in value.items()}}
    if isinstance(value, list):
        return {"t": "list", "v": [_pack_value(x, tensors) for x in value]}
    if isinstance(value, tuple):
        return {"t": "tuple", "v": [_pack_value(x, tensors) for x in value]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"t": "lit", "v": value}
    raise TypeError("Cannot save conditioning value of type {}".format(type(value).__name__))


def _unpack_value(spec, tensors):
    kind = spec["t"]
    if kind == "tensor":
        return tensors[spec["k"]]
    if kind == "nested":
        return comfy.nested_tensor.NestedTensor([_unpack_value(s, tensors) for s in spec["s"]])
    if kind == "dict":
        return {k: _unpack_value(v, tensors) for k, v in spec["v"].items()}
    if kind == "list":
        return [_unpack_value(x, tensors) for x in spec["v"]]
    if kind == "tuple":
        return tuple(_unpack_value(x, tensors) for x in spec["v"])
    if kind == "lit":
        return spec["v"]
    raise ValueError("Unknown conditioning pack type {!r}".format(kind))


def pack_conditioning(conditioning):
    tensors = {}
    entries = []
    for cond, extra in conditioning:
        entries.append({"cond": _pack_value(cond, tensors), "extra": _pack_value(extra, tensors)})
    if not tensors:
        tensors["t0"] = torch.empty(0)
    return tensors, {"version": 1, "entries": entries}


def unpack_conditioning(tensors, schema):
    if schema.get("version") != 1:
        raise ValueError("Unsupported conditioning file version {}".format(schema.get("version")))
    out = []
    for entry in schema["entries"]:
        out.append([_unpack_value(entry["cond"], tensors), _unpack_value(entry["extra"], tensors)])
    return out


class CLIPTextEncodeControlnet(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CLIPTextEncodeControlnet",
            display_name="CLIP Text Encode (Controlnet)",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.Conditioning.Input("conditioning"),
                io.String.Input("text", multiline=True, dynamic_prompts=True),
            ],
            outputs=[io.Conditioning.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clip, conditioning, text) -> io.NodeOutput:
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]['cross_attn_controlnet'] = cond
            n[1]['pooled_output_controlnet'] = pooled
            c.append(n)
        return io.NodeOutput(c)

class T5TokenizerOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T5TokenizerOptions",
            display_name="T5 Tokenizer Options",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.Int.Input("min_padding", default=0, min=0, max=10000, step=1),
                io.Int.Input("min_length", default=0, min=0, max=10000, step=1),
            ],
            outputs=[io.Clip.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clip, min_padding, min_length) -> io.NodeOutput:
        clip = clip.clone()
        for t5_type in ["t5xxl", "pile_t5xl", "t5base", "mt5xl", "umt5xxl"]:
            clip.set_tokenizer_option("{}_min_padding".format(t5_type), min_padding)
            clip.set_tokenizer_option("{}_min_length".format(t5_type), min_length)

        return io.NodeOutput(clip)


class SaveConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SaveConditioning",
            display_name="Save Conditioning",
            search_aliases=["export conditioning", "cache prompt", "save clip"],
            category="model/conditioning",
            description="Save CONDITIONING (embeddings and extras) to skip re-encoding later. Copy the file into input/ to load it.",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.String.Input("filename_prefix", default="conditioning/ComfyUI"),
            ],
            outputs=[io.Conditioning.Output()],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, conditioning, filename_prefix="conditioning/ComfyUI") -> io.NodeOutput:
        tensors, schema = pack_conditioning(conditioning)
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory())
        file = "{}_{:05}_.conditioning".format(filename, counter)
        metadata = {"comfy_conditioning": json.dumps(schema)}
        if not args.disable_metadata:
            if cls.hidden.prompt is not None:
                metadata["prompt"] = json.dumps(cls.hidden.prompt)
            if cls.hidden.extra_pnginfo is not None:
                for x in cls.hidden.extra_pnginfo:
                    metadata[x] = json.dumps(cls.hidden.extra_pnginfo[x])
        comfy.utils.save_torch_file(tensors, os.path.join(full_output_folder, file), metadata=metadata)
        return io.NodeOutput(
            conditioning,
            ui={"files": [ui.SavedResult(file, subfolder, io.FolderType.output)]},
        )


class LoadConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".conditioning")]
        return io.Schema(
            node_id="LoadConditioning",
            display_name="Load Conditioning",
            search_aliases=["import conditioning", "open conditioning", "cached prompt"],
            category="model/conditioning",
            description="Load CONDITIONING saved by Save Conditioning. Place the .conditioning file in input/.",
            inputs=[
                io.Combo.Input("conditioning", options=sorted(files)),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, conditioning) -> io.NodeOutput:
        path = folder_paths.get_annotated_filepath(conditioning)
        if not path.endswith(".conditioning"):
            raise ValueError("Invalid conditioning file: {}".format(conditioning))
        tensors, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        if metadata is None or "comfy_conditioning" not in metadata:
            raise ValueError("Not a ComfyUI conditioning file: {}".format(conditioning))
        schema = json.loads(metadata["comfy_conditioning"])
        return io.NodeOutput(unpack_conditioning(tensors, schema))

    @classmethod
    def fingerprint_inputs(cls, conditioning):
        path = folder_paths.get_annotated_filepath(conditioning)
        m = hashlib.sha256()
        with open(path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def validate_inputs(cls, conditioning):
        if not isinstance(conditioning, str) or not conditioning.endswith(".conditioning"):
            return "Invalid conditioning file: {}".format(conditioning)
        if not folder_paths.exists_annotated_filepath(conditioning):
            return "Invalid conditioning file: {}".format(conditioning)
        return True


class CondExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CLIPTextEncodeControlnet,
            T5TokenizerOptions,
            SaveConditioning,
            LoadConditioning,
        ]


async def comfy_entrypoint() -> CondExtension:
    return CondExtension()
