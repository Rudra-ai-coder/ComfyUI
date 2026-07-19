#!/usr/bin/env python3
"""Merge Bernini DiT trunk with Bernini-R-S2V audio injectors (fp8/fp16 safetensors).

Keeps S2V-only modules from the S2V checkpoint and replaces every other tensor
with the matching key from a Bernini (or Bernini-R) trunk checkpoint.

S2V modules (ComfyUI WAN22_S2V detection + WanModel_S2V):
  - audio_injector
  - casual_audio_encoder
  - cond_encoder
  - frame_packer
  - trainable_cond_mask

Designed for ~50 GB system RAM: tensors are streamed from memory-mapped
safetensors; only the output dict is fully materialized (~one checkpoint).

Example (RunPod):
  python3 merge_bernini_s2v_audio.py \\
    --bernini  /workspace/models/wan2.2_bernini_high_noise_fp8_scaled.safetensors \\
    --s2v      /workspace/models/wan2.2_bernini_r_high_noise_fp8_scaled_s2v.safetensors \\
    --output   /workspace/models/wan2.2_bernini_high_noise_fp8_scaled_s2v.safetensors

  # Dry-run first (lists key plan, no write):
  python3 merge_bernini_s2v_audio.py --bernini ... --s2v ... --output ... --dry-run
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple


# Prefixes that identify Wan2.2 S2V / audio modules (and their fp8 scale siblings).
AUDIO_PREFIXES: Tuple[str, ...] = (
    "audio_injector",
    "casual_audio_encoder",
    "cond_encoder",
    "frame_packer",
    "trainable_cond_mask",
)


def _strip_known_prefixes(key: str) -> str:
    for p in ("model.diffusion_model.", "diffusion_model.", "model."):
        if key.startswith(p):
            return key[len(p) :]
    return key


def is_audio_key(key: str) -> bool:
    k = _strip_known_prefixes(key)
    return any(k == p or k.startswith(p + ".") for p in AUDIO_PREFIXES)


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return -1.0


def list_keys(path: str) -> List[str]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        return list(f.keys())


def tensor_meta(path: str, key: str):
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        t = f.get_slice(key)
        return tuple(t.get_shape()), str(t.get_dtype())


def classify_keys(
    bernini_keys: Sequence[str],
    s2v_keys: Sequence[str],
) -> Dict[str, object]:
    bset, sset = set(bernini_keys), set(s2v_keys)
    shared = bset & sset
    only_bernini = bset - sset
    only_s2v = sset - bset

    audio_from_s2v = sorted(k for k in sset if is_audio_key(k))
    trunk_from_bernini = sorted(k for k in shared if not is_audio_key(k))
    # Shared audio keys should still come from S2V (keep injectors).
    shared_audio = sorted(k for k in shared if is_audio_key(k))
    # S2V-only non-audio (unexpected if trunks match).
    s2v_only_nonaudio = sorted(k for k in only_s2v if not is_audio_key(k))
    # Audio keys that somehow exist only on Bernini (should be empty).
    bernini_audio = sorted(k for k in only_bernini if is_audio_key(k))

    return {
        "audio_from_s2v": audio_from_s2v,
        "shared_audio": shared_audio,
        "trunk_from_bernini": trunk_from_bernini,
        "only_bernini": sorted(only_bernini),
        "only_s2v": sorted(only_s2v),
        "s2v_only_nonaudio": s2v_only_nonaudio,
        "bernini_audio": bernini_audio,
        "n_bernini": len(bset),
        "n_s2v": len(sset),
        "n_shared": len(shared),
    }


def plan_sources(info: Dict[str, object]) -> Dict[str, str]:
    """Map output_key -> source ('bernini' | 's2v')."""
    plan: Dict[str, str] = {}
    for k in info["trunk_from_bernini"]:
        plan[k] = "bernini"
    for k in info["audio_from_s2v"]:
        plan[k] = "s2v"
    # Keep unexpected S2V-only non-audio so we don't drop tensors silently.
    for k in info["s2v_only_nonaudio"]:
        plan[k] = "s2v"
    # Optionally include Bernini-only trunk keys (rare; usually empty).
    for k in info["only_bernini"]:
        if not is_audio_key(k):
            plan[k] = "bernini"
    return plan


def validate_shapes(
    bernini_path: str,
    s2v_path: str,
    plan: Dict[str, str],
) -> List[str]:
    errors: List[str] = []
    from safetensors import safe_open

    with safe_open(bernini_path, framework="pt", device="cpu") as fb, safe_open(
        s2v_path, framework="pt", device="cpu"
    ) as fs:
        for key, src in plan.items():
            if src != "bernini":
                continue
            if key not in fb.keys() or key not in fs.keys():
                continue
            sb, db = tuple(fb.get_slice(key).get_shape()), fb.get_slice(key).get_dtype()
            ss, ds = tuple(fs.get_slice(key).get_shape()), fs.get_slice(key).get_dtype()
            if sb != ss:
                errors.append(f"SHAPE MISMATCH {key}: bernini {sb} vs s2v {ss}")
            if db != ds:
                errors.append(f"DTYPE MISMATCH {key}: bernini {db} vs s2v {ds}")
    return errors


def merge(
    bernini_path: str,
    s2v_path: str,
    output_path: str,
    *,
    dry_run: bool = False,
    allow_shape_mismatch: bool = False,
    include_bernini_only: bool = True,
) -> None:
    print(f"Bernini trunk : {bernini_path}")
    print(f"S2V (audio)   : {s2v_path}")
    print(f"Output        : {output_path}")
    print(f"RSS at start  : {_rss_gb():.2f} GB")

    t0 = time.time()
    bernini_keys = list_keys(bernini_path)
    s2v_keys = list_keys(s2v_path)
    info = classify_keys(bernini_keys, s2v_keys)
    plan = plan_sources(info)
    if not include_bernini_only:
        plan = {k: v for k, v in plan.items() if k in set(s2v_keys) or is_audio_key(k)}

    n_audio = sum(1 for k, s in plan.items() if s == "s2v" and is_audio_key(k))
    n_trunk = sum(1 for k, s in plan.items() if s == "bernini")
    n_s2v_keep = sum(1 for k, s in plan.items() if s == "s2v" and not is_audio_key(k))

    print()
    print("=== Key plan ===")
    print(f"  Bernini keys          : {info['n_bernini']}")
    print(f"  S2V keys              : {info['n_s2v']}")
    print(f"  Shared                : {info['n_shared']}")
    print(f"  Trunk ← Bernini       : {n_trunk}")
    print(f"  Audio ← S2V           : {n_audio}")
    print(f"  Extra non-audio ← S2V : {n_s2v_keep}")
    print(f"  Only in Bernini       : {len(info['only_bernini'])}")
    print(f"  Only in S2V           : {len(info['only_s2v'])}")

    if info["audio_from_s2v"]:
        # Show a few audio prefixes actually present
        prefixes_hit = sorted(
            {
                next(p for p in AUDIO_PREFIXES if _strip_known_prefixes(k) == p or _strip_known_prefixes(k).startswith(p + "."))
                for k in info["audio_from_s2v"]
            }
        )
        print(f"  Audio prefixes found  : {prefixes_hit}")
    else:
        print("  WARNING: no audio keys found on S2V checkpoint — wrong file?")

    if info["s2v_only_nonaudio"]:
        print(f"  WARNING: {len(info['s2v_only_nonaudio'])} S2V-only non-audio keys will be kept from S2V")
        for k in info["s2v_only_nonaudio"][:12]:
            print(f"    - {k}")
        if len(info["s2v_only_nonaudio"]) > 12:
            print(f"    ... +{len(info['s2v_only_nonaudio']) - 12} more")

    if info["only_bernini"] and include_bernini_only:
        print(f"  NOTE: {len(info['only_bernini'])} Bernini-only keys will be added to output")

    # Detection sanity: ComfyUI looks for casual_audio_encoder.encoder.final_linear.weight
    detect_suffix = "casual_audio_encoder.encoder.final_linear.weight"
    has_detect = any(_strip_known_prefixes(k).endswith(detect_suffix) or _strip_known_prefixes(k) == detect_suffix for k in plan)
    print(f"  ComfyUI S2V detect key present in plan: {has_detect}")

    print()
    print("Validating shapes on shared trunk keys...")
    errors = validate_shapes(bernini_path, s2v_path, plan)
    if errors:
        for e in errors[:30]:
            print(" ", e)
        if len(errors) > 30:
            print(f"  ... +{len(errors) - 30} more")
        if not allow_shape_mismatch:
            raise SystemExit(f"Aborting: {len(errors)} shape/dtype mismatches. Use --allow-shape-mismatch to force.")
        print("  Continuing despite mismatches (--allow-shape-mismatch).")
    else:
        print("  All shared trunk keys match shape/dtype.")

    if dry_run:
        report = {
            "bernini": bernini_path,
            "s2v": s2v_path,
            "output": output_path,
            "counts": {
                "trunk_from_bernini": n_trunk,
                "audio_from_s2v": n_audio,
                "extra_from_s2v": n_s2v_keep,
            },
            "audio_keys_sample": info["audio_from_s2v"][:40],
            "s2v_only_nonaudio": info["s2v_only_nonaudio"],
            "only_bernini_sample": info["only_bernini"][:40],
        }
        report_path = output_path + ".merge_plan.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nDry-run done in {time.time() - t0:.1f}s. Plan written to {report_path}")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    print()
    print("Loading / merging tensors (this is the RAM peak)...")
    from safetensors import safe_open
    from safetensors.torch import save_file

    out: Dict[str, object] = {}
    n_done = 0
    n_total = len(plan)

    with safe_open(bernini_path, framework="pt", device="cpu") as fb, safe_open(
        s2v_path, framework="pt", device="cpu"
    ) as fs:
        for key, src in plan.items():
            handle = fb if src == "bernini" else fs
            if key not in handle.keys():
                print(f"  SKIP missing {key} from {src}")
                continue
            out[key] = handle.get_tensor(key)
            n_done += 1
            if n_done % 200 == 0 or n_done == n_total:
                print(f"  {n_done}/{n_total} tensors  RSS={_rss_gb():.2f} GB")

    print(f"Saving {len(out)} tensors → {output_path}")
    metadata = {
        "format": "pt",
        "merged_by": "merge_bernini_s2v_audio.py",
        "trunk_source": os.path.basename(bernini_path),
        "audio_source": os.path.basename(s2v_path),
        "audio_prefixes": ",".join(AUDIO_PREFIXES),
    }
    save_file(out, output_path, metadata=metadata)
    del out
    gc.collect()

    size = os.path.getsize(output_path)
    print()
    print(f"Done in {time.time() - t0:.1f}s")
    print(f"  Output size : {_human_bytes(size)}")
    print(f"  Final RSS   : {_rss_gb():.2f} GB")
    print()
    print("Place the file in ComfyUI/models/diffusion_models/ and load as usual.")
    print("ComfyUI should detect WAN22_S2V because casual_audio_encoder keys are present.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Swap Bernini trunk weights into Bernini-R-S2V while keeping audio injectors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--bernini", required=True, help="Bernini (or Bernini-R) trunk safetensors")
    p.add_argument("--s2v", required=True, help="Bernini-R-S2V safetensors (audio source)")
    p.add_argument("--output", required=True, help="Output merged safetensors path")
    p.add_argument("--dry-run", action="store_true", help="Only print/write key plan, do not merge")
    p.add_argument(
        "--allow-shape-mismatch",
        action="store_true",
        help="Do not abort when a shared trunk key has different shape/dtype",
    )
    p.add_argument(
        "--no-bernini-only",
        action="store_true",
        help="Do not add keys that exist only in Bernini (default: add them)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    for path, label in ((args.bernini, "bernini"), (args.s2v, "s2v")):
        if not os.path.isfile(path):
            raise SystemExit(f"{label} file not found: {path}")
    merge(
        args.bernini,
        args.s2v,
        args.output,
        dry_run=args.dry_run,
        allow_shape_mismatch=args.allow_shape_mismatch,
        include_bernini_only=not args.no_bernini_only,
    )


if __name__ == "__main__":
    main()
