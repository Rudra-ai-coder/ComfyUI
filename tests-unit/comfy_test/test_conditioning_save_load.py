import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.nested_tensor  # noqa: E402
from comfy_extras.nodes_cond import pack_conditioning, unpack_conditioning  # noqa: E402


def test_pack_unpack_minimax_style_conditioning():
    cond = torch.randn(1, 8, 16)
    tags = torch.ones(8, dtype=torch.long)
    kf_latent = torch.randn(1, 24, 1, 4, 6)
    extra = {
        "pooled_output": None,
        "minimax_token_tags": tags,
        "minimax_frame_count": 124,
        "minimax_keyframes": [{"resolved_frame_index": 0, "latent": kf_latent}],
        "minimax_refs": [{"kind": "image", "latent_h": 4, "latent_w": 6, "latent": kf_latent}],
    }
    tensors, schema = pack_conditioning([[cond, extra]])
    loaded = unpack_conditioning(tensors, schema)
    assert torch.equal(loaded[0][0], cond)
    assert loaded[0][1]["minimax_frame_count"] == 124
    assert loaded[0][1]["pooled_output"] is None
    assert torch.equal(loaded[0][1]["minimax_token_tags"], tags)
    assert loaded[0][1]["minimax_keyframes"][0]["resolved_frame_index"] == 0
    assert torch.equal(loaded[0][1]["minimax_keyframes"][0]["latent"], kf_latent)
    assert loaded[0][1]["minimax_refs"][0]["kind"] == "image"


def test_pack_unpack_nested_tensor_extra():
    cond = torch.randn(1, 4, 8)
    nested = comfy.nested_tensor.NestedTensor((torch.randn(1, 2, 3), torch.randn(1, 4, 5)))
    tensors, schema = pack_conditioning([[cond, {"av": nested}]])
    loaded = unpack_conditioning(tensors, schema)
    a, b = loaded[0][1]["av"].unbind()
    oa, ob = nested.unbind()
    assert torch.equal(a, oa)
    assert torch.equal(b, ob)
