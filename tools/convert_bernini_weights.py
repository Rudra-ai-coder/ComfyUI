#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Split ByteDance/Bernini-Diffusers joint bernini/ shards into ComfyUI-ready components.

Input (after hf download):
  Bernini-Diffusers/bernini/model.safetensors.index.json
  Bernini-Diffusers/bernini/model-00001-of-00038.safetensors ...

Outputs (default names in --output-dir):
  bernini_planner.safetensors       connector.*, mask_tokens
  bernini_vit_decoder.safetensors   vit_decoder.*
  bernini_mllm.safetensors          mllm.* (strip mllm. prefix for HF Qwen layout)
  bernini_high_noise.safetensors    diff_dec.transformer.* → Comfy WanModel keys
  bernini_low_noise.safetensors     diff_dec_low.transformer_2.* → Comfy WanModel keys

Requirements: pip install safetensors torch

Example:
  python tools/convert_bernini_weights.py \\
    --input-dir /data/Bernini-Diffusers/bernini \\
    --output-dir /data/bernini-comfy \\
    --dtype bfloat16

Dry-run (no shard read):
  python tools/convert_bernini_weights.py --input-dir ... --dry-run
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError as exc:
    print("Missing dependency. Install with: pip install safetensors torch", file=sys.stderr)
    raise SystemExit(1) from exc

# Allow running as script from repo root or tools/
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from bernini_keymaps import (  # noqa: E402
    BUCKET_HIGH_NOISE,
    BUCKET_LOW_NOISE,
    BUCKET_MLLM,
    BUCKET_PLANNER,
    BUCKET_T5_BUNDLED,
    BUCKET_UNKNOWN,
    BUCKET_VIT_DECODER,
    classify_bernini_key,
    diffusers_wan_to_comfy,
    export_key_for_bucket,
    strip_dit_prefix,
)

LOG = logging.getLogger("convert_bernini_weights")

DEFAULT_OUTPUT_NAMES = {
    BUCKET_PLANNER: "bernini_planner.safetensors",
    BUCKET_VIT_DECODER: "bernini_vit_decoder.safetensors",
    BUCKET_MLLM: "bernini_mllm.safetensors",
    BUCKET_HIGH_NOISE: "bernini_high_noise.safetensors",
    BUCKET_LOW_NOISE: "bernini_low_noise.safetensors",
}

ALL_BUCKETS = tuple(DEFAULT_OUTPUT_NAMES.keys())


def resolve_index_path(input_dir: Path) -> Path:
    """Find model.safetensors.index.json or any *.safetensors.index.json in input_dir."""
    preferred = input_dir / "model.safetensors.index.json"
    if preferred.is_file():
        return preferred
    candidates = sorted(input_dir.glob("*.safetensors.index.json"))
    if not candidates:
        raise FileNotFoundError(f"No *.safetensors.index.json in {input_dir}")
    return candidates[0]


def load_weight_map(index_path: Path) -> Dict[str, str]:
    with open(index_path, encoding="utf-8") as f:
        data = json.load(f)
    weight_map = data.get("weight_map")
    if not weight_map:
        raise ValueError(f"No weight_map in {index_path}")
    return weight_map


def parse_dtype(name: str) -> Optional[torch.dtype]:
    if name is None or name.lower() in ("none", "unchanged", ""):
        return None
    n = name.lower().strip()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if n not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[n]


def maybe_cast(tensor: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.Tensor:
    if dtype is not None and tensor.dtype != dtype:
        return tensor.to(dtype=dtype)
    return tensor


def iter_tensors_from_shard(
    shard_path: Path,
    keys: Iterable[str],
    dtype: Optional[torch.dtype],
) -> Iterable[Tuple[str, torch.Tensor]]:
    """Load one tensor at a time to limit transient RAM within a shard read."""
    keys_list = list(keys)
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        available = set(f.keys())
        for key in keys_list:
            if key not in available:
                raise KeyError(f"Key {key} missing from shard {shard_path}")
            yield key, maybe_cast(f.get_tensor(key), dtype)


def export_tensor_key(src_key: str, bucket: str, comfy_prefix: str) -> Optional[str]:
    if bucket in (BUCKET_HIGH_NOISE, BUCKET_LOW_NOISE):
        inner = strip_dit_prefix(src_key)
        if inner is None:
            LOG.warning("Skipping non-DiT key in %s bucket: %s", bucket, src_key)
            return None
        dst_key = diffusers_wan_to_comfy(inner)
        if comfy_prefix:
            dst_key = f"{comfy_prefix}{dst_key}"
        return dst_key
    return export_key_for_bucket(src_key, bucket)


def convert_one_bucket(
    bucket: str,
    keys: List[str],
    weight_map: Dict[str, str],
    input_dir: Path,
    output_dir: Path,
    dtype: Optional[torch.dtype],
    comfy_prefix: str,
    meta_base: Dict[str, str],
) -> Optional[Tuple[str, int, float]]:
    """Read shards once per bucket; peak RAM ≈ one output file (+ one shard batch)."""
    if not keys:
        LOG.warning("Bucket %s is empty", bucket)
        return None

    shard_to_keys: Dict[str, List[str]] = defaultdict(list)
    for key in keys:
        shard_to_keys[weight_map[key]].append(key)

    state_dict: Dict[str, torch.Tensor] = {}
    shards_sorted = sorted(shard_to_keys.keys())
    LOG.info(
        "Bucket %s (%s): %d keys across %d shards",
        bucket,
        DEFAULT_OUTPUT_NAMES[bucket],
        len(keys),
        len(shards_sorted),
    )

    for shard_name in shards_sorted:
        shard_path = input_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing shard: {shard_path}")

        shard_keys = shard_to_keys[shard_name]
        LOG.info("  %s (%d tensors)", shard_name, len(shard_keys))

        for src_key, tensor in iter_tensors_from_shard(shard_path, shard_keys, dtype):
            dst_key = export_tensor_key(src_key, bucket, comfy_prefix)
            if dst_key is None:
                continue
            if dst_key in state_dict:
                raise KeyError(f"Duplicate export key {dst_key} in bucket {bucket} (from {src_key})")
            state_dict[dst_key] = tensor

        gc.collect()

    if bucket in (BUCKET_HIGH_NOISE, BUCKET_LOW_NOISE):
        meta = {
            **meta_base,
            "bucket": bucket,
            "comfy_format": "wan_model",
            "comfy_prefix": comfy_prefix,
        }
    else:
        meta = {**meta_base, "bucket": bucket}

    out_path = output_dir / DEFAULT_OUTPUT_NAMES[bucket]
    n, nbytes = save_bucket(out_path, state_dict, meta)
    gb = nbytes / (1024 ** 3)
    LOG.info("Wrote %s (%d tensors, %.2f GB)", out_path, n, gb)

    del state_dict
    gc.collect()

    return out_path.name, n, gb


def save_bucket(
    path: Path,
    state_dict: Dict[str, torch.Tensor],
    metadata: Dict[str, str],
) -> Tuple[int, int]:
    if not state_dict:
        LOG.warning("Empty state dict, not writing %s", path)
        return 0, 0
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(path), metadata=metadata)
    num_tensors = len(state_dict)
    total_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
    return num_tensors, total_bytes


def run_conversion(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    index_path = resolve_index_path(input_dir)
    weight_map = load_weight_map(index_path)
    dtype = parse_dtype(args.dtype)

    # Classify all keys
    by_bucket: Dict[str, List[str]] = defaultdict(list)
    for key in weight_map:
        bucket = classify_bernini_key(key)
        by_bucket[bucket].append(key)

    t5_bundled = by_bucket.pop(BUCKET_T5_BUNDLED, [])
    unknown = by_bucket.pop(BUCKET_UNKNOWN, [])
    LOG.info("Index: %s (%d keys)", index_path, len(weight_map))
    LOG.info("Classification: %s", dict(Counter({k: len(v) for k, v in by_bucket.items()})))
    if t5_bundled:
        LOG.info(
            "%d bundled t5_text_encoder keys skipped (expected — use HF t5_text_encoder/ with CLIPLoader wan)",
            len(t5_bundled),
        )
    if unknown:
        LOG.warning("%d unknown keys (will be skipped): %s", len(unknown), unknown[:10])
        if len(unknown) > 10:
            LOG.warning("... and %d more", len(unknown) - 10)

    buckets_to_run = _resolve_buckets(args.only)

    if args.dry_run:
        print("\n=== Dry run — buckets ===")
        for bucket in buckets_to_run:
            filename = DEFAULT_OUTPUT_NAMES[bucket]
            keys = by_bucket.get(bucket, [])
            print(f"  {filename}: {len(keys)} keys")
        if t5_bundled:
            print(
                f"  (bundled t5_text_encoder, skipped): {len(t5_bundled)} keys "
                "— download t5_text_encoder/ from Bernini-Diffusers for CLIPLoader"
            )
        if unknown:
            print(f"  (skipped unknown): {len(unknown)} keys")
        if not args.only:
            print("\nLow RAM tip: use --only <bucket> per run (see README).")
        print("\nNo files written (--dry-run).")
        return 0

    meta_base = {
        "format": "bernini_comfy_split",
        "source_index": str(index_path),
        "script": "convert_bernini_weights.py",
        "dtype": args.dtype or "unchanged",
    }

    summary: List[str] = []
    comfy_prefix = args.comfy_prefix or ""

    for bucket in buckets_to_run:
        result = convert_one_bucket(
            bucket,
            by_bucket.get(bucket, []),
            weight_map,
            input_dir,
            output_dir,
            dtype,
            comfy_prefix,
            meta_base,
        )
        if result is not None:
            name, n, gb = result
            summary.append(f"  {name}: {n} tensors, {gb:.2f} GB")

    print("\n=== Conversion complete ===")
    print("\n".join(summary))
    print(f"\nOutput directory: {output_dir}")
    print("\nNext steps:")
    print("  - Load bernini_high_noise / bernini_low_noise via ComfyUI UNETLoader")
    print("  - Use bernini_planner + bernini_vit_decoder in Bernini planner nodes (upcoming)")
    print("  - bernini_mllm: HF Qwen2.5-VL keys (or use Bernini-Diffusers/mllm/ folder)")
    return 0


def _resolve_buckets(only: Optional[str]) -> Tuple[str, ...]:
    if only is None:
        return ALL_BUCKETS
    if only not in DEFAULT_OUTPUT_NAMES:
        raise ValueError(f"Invalid bucket: {only}")
    return (only,)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Split Bernini-Diffusers bernini/ shards for ComfyUI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing model.safetensors.index.json and shard files "
        "(e.g. Bernini-Diffusers/bernini)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <input-dir>/comfy_split)",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Cast tensors when reading shards: bfloat16, float16, float32, or none",
    )
    p.add_argument(
        "--comfy-prefix",
        type=str,
        default="",
        help="Optional prefix for DiT keys (e.g. model.diffusion_model.)",
    )
    p.add_argument(
        "--only",
        type=str,
        choices=ALL_BUCKETS,
        default=None,
        metavar="BUCKET",
        help="Convert a single bucket to limit peak RAM (~50GB systems). "
        "Choices: planner, vit_decoder, mllm, high_noise, low_noise. "
        "Run once per bucket; use --dtype float16 for DiT buckets.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only classify keys from index; do not read shards or write files",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        LOG.error("Input directory does not exist: %s", input_dir)
        return 1

    if args.output_dir is None:
        args.output_dir = str(input_dir / "comfy_split")

    return run_conversion(args)


if __name__ == "__main__":
    raise SystemExit(main())
