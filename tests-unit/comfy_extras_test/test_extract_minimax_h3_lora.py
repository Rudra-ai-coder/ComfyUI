import importlib.util
from pathlib import Path

import torch

_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "extract_minimax_h3_lora.py"
_spec = importlib.util.spec_from_file_location("extract_minimax_h3_lora", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
extract_from_dicts = _mod.extract_from_dicts


def test_extract_minimax_h3_lora_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(16, 8)
    delta = torch.randn(16, 8) * 0.1
    base = {"diffusion_model.blocks.0.attn.qkv_proj.weight": w}
    ft = {"diffusion_model.blocks.0.attn.qkv_proj.weight": w + delta}
    out, skipped = extract_from_dicts(base, ft, rank=4, min_diff=1e-8)
    assert skipped == []
    up = out["diffusion_model.blocks.0.attn.qkv_proj.lora_up.weight"]
    down = out["diffusion_model.blocks.0.attn.qkv_proj.lora_down.weight"]
    recon = up.float() @ down.float()
    err = (recon - delta.float()).norm() / delta.float().norm()
    assert err < 0.15


def test_extract_skips_shape_mismatch():
    base = {"diffusion_model.final_layer.video_out.weight": torch.randn(96, 8)}
    ft = {"diffusion_model.final_layer.video_out.weight": torch.randn(192, 8)}
    out, skipped = extract_from_dicts(base, ft, rank=4)
    assert out == {}
    assert skipped and skipped[0][0] == "final_layer.video_out.weight"
