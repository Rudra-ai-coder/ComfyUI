#!/usr/bin/env python3
"""Extract a ComfyUI LoRA from a MiniMax H3 base vs fine-tuned pair.

SVD of (fine_tune - base) on matching 2D weights. Load with the normal
Load LoRA node on the *base* MiniMax H3 model.

Both files must be the same architecture (same PDD head count, hidden size).
Run from the ComfyUI repo root:

  python tools/extract_minimax_h3_lora.py \\
    --base models/diffusion_models/minimax_h3_base.safetensors \\
    --finetune models/diffusion_models/minimax_h3_ft.safetensors \\
    --output models/loras/minimax_h3_ft.safetensors \\
    --rank 64
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from safetensors import safe_open
from tqdm.auto import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import comfy.utils
from comfy_extras.nodes_lora_extract import extract_lora

PREFIXES = ("diffusion_model.", "model.diffusion_model.", "model.")
SKIP_SUFFIXES = (
    ".weight_scale",
    ".weight_scale_2",
    ".comfy_quant",
    ".absmax",
    ".quant_map",
)


def strip_prefix(key):
    for prefix in PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def index_keys(keys):
    index = {}
    for key in keys:
        if any(key.endswith(s) for s in SKIP_SUFFIXES):
            continue
        index[strip_prefix(key)] = key
    return index


def is_weight(key):
    return key.endswith(".weight") and not any(key.endswith(s) for s in SKIP_SUFFIXES)


def is_bias(key):
    return key.endswith(".bias")


class WeightStore:
    def __init__(self, path):
        self.path = path
        self._cm = None
        self._handle = None
        self._sd = None
        if path.lower().endswith((".safetensors", ".sft")):
            self._cm = safe_open(path, framework="pt")
            self._handle = self._cm.__enter__()
        else:
            self._sd = comfy.utils.load_torch_file(path, safe_load=True)

    def keys(self):
        if self._handle is not None:
            return list(self._handle.keys())
        return list(self._sd.keys())

    def get(self, key, device):
        if self._handle is not None:
            t = self._handle.get_tensor(key)
        else:
            t = self._sd[key]
        return t.to(device=device, dtype=torch.float32)

    def close(self):
        if self._cm is not None:
            self._cm.__exit__(None, None, None)
            self._cm = None
            self._handle = None


def extract_pair(base_get, ft_get, shared, rank, min_diff, bias, device):
    out = {}
    skipped = []
    for bare in tqdm(shared, desc="extract", unit="tensor"):
        if not (is_weight(bare) or (bias and is_bias(bare))):
            continue
        bw = base_get(bare, device)
        fw = ft_get(bare, device)
        if tuple(bw.shape) != tuple(fw.shape):
            skipped.append((bare, tuple(bw.shape), tuple(fw.shape)))
            continue
        diff = fw - bw
        if float(diff.abs().max()) < min_diff:
            continue
        if is_weight(bare):
            module = "diffusion_model.{}".format(bare[:-len(".weight")])
        else:
            module = "diffusion_model.{}".format(bare[:-len(".bias")])
        if is_bias(bare) or diff.ndim < 2:
            key = "{}.diff_b".format(module) if is_bias(bare) else "{}.diff".format(module)
            out[key] = diff.contiguous().half().cpu()
            continue
        used_rank = min(rank, diff.shape[0], diff.shape[1])
        try:
            up, down = extract_lora(diff, used_rank)
        except Exception:
            skipped.append((bare, tuple(bw.shape), "svd_failed"))
            continue
        out["{}.lora_up.weight".format(module)] = up.contiguous().half().cpu()
        out["{}.lora_down.weight".format(module)] = down.contiguous().half().cpu()
        out["{}.alpha".format(module)] = torch.tensor(float(used_rank), dtype=torch.float16)
    return out, skipped


def extract_from_dicts(base_sd, ft_sd, rank=8, min_diff=1e-6, bias=False, device="cpu"):
    device = torch.device(device)
    base_index = index_keys(base_sd.keys())
    ft_index = index_keys(ft_sd.keys())
    shared = sorted(set(base_index) & set(ft_index))
    return extract_pair(
        lambda bare, dev: base_sd[base_index[bare]].to(device=dev, dtype=torch.float32),
        lambda bare, dev: ft_sd[ft_index[bare]].to(device=dev, dtype=torch.float32),
        shared, rank, min_diff, bias, device)


def extract_files(base_path, ft_path, rank, min_diff, bias, device):
    base = WeightStore(base_path)
    ft = WeightStore(ft_path)
    try:
        base_index = index_keys(base.keys())
        ft_index = index_keys(ft.keys())
        shared = sorted(set(base_index) & set(ft_index))
        only_ft = sorted(set(ft_index) - set(base_index))
        out, skipped = extract_pair(
            lambda bare, dev: base.get(base_index[bare], dev),
            lambda bare, dev: ft.get(ft_index[bare], dev),
            shared, rank, min_diff, bias, device)
        return out, skipped, only_ft
    finally:
        base.close()
        ft.close()


def main():
    p = argparse.ArgumentParser(description="Extract a MiniMax H3 LoRA from base vs fine-tuned weights")
    p.add_argument("--base", required=True, help="Base MiniMax H3 diffusion checkpoint")
    p.add_argument("--finetune", required=True, help="Fine-tuned MiniMax H3 diffusion checkpoint")
    p.add_argument("--output", required=True, help="Output .safetensors LoRA path")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--min-diff", type=float, default=1e-6, help="Skip tensors whose max abs delta is below this")
    p.add_argument("--bias", action="store_true", help="Also store bias deltas as .diff_b")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out, skipped, only_ft = extract_files(
        args.base, args.finetune, args.rank, args.min_diff, args.bias, torch.device(args.device))
    if not out:
        raise SystemExit("no LoRA tensors extracted; check that both files are MiniMax H3 DiT weights")
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    comfy.utils.save_torch_file(out, args.output, metadata={
        "format": "comfyui_lora",
        "model": "minimax_h3",
        "rank": str(args.rank),
    })
    n_up = sum(1 for k in out if k.endswith(".lora_up.weight"))
    print("wrote {} ({} modules)".format(args.output, n_up))
    if skipped:
        print("skipped {} shape/svd mismatches (same architecture required, including PDD heads):".format(len(skipped)))
        for item in skipped[:20]:
            print("  ", item)
    if only_ft:
        print("{} keys only in the fine-tune (not extracted):".format(len(only_ft)))
        for k in only_ft[:20]:
            print("  ", k)


if __name__ == "__main__":
    main()
