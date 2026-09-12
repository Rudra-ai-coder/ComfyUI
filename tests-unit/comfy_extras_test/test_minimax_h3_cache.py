import torch

import comfy.nested_tensor
from comfy_extras.nodes_minimax_h3 import pack_h3_cache, unpack_h3_cache, _validate_h3_cache_combo


def _i2v_positive():
    cond = torch.randn(1, 8, 16)
    tags = torch.ones(8, dtype=torch.long)
    kf = torch.randn(1, 24, 1, 4, 6)
    extra = {
        "pooled_output": None,
        "minimax_token_tags": tags,
        "minimax_keyframes": [{"resolved_frame_index": 0, "latent": kf}],
    }
    return [[cond, extra]]


def _ref2va_positive():
    cond = torch.randn(1, 4, 8)
    z = torch.randn(1, 24, 2, 4, 6)
    audio = torch.randn(1, 32, 2, 8)
    extra = {
        "minimax_refs": [
            {"kind": "video_audio", "latent_t": 2, "latent_h": 4, "latent_w": 6,
             "ref_audio_t": 8, "latent": z, "audio_latent": audio},
        ],
    }
    return [[cond, extra]]


def _av_latent():
    video = torch.randn(1, 24, 7, 4, 6)
    audio = torch.randn(1, 32, 2, 40)
    mask_v = torch.ones(1, 1, 7, 4, 6)
    mask_a = torch.zeros(1, 1, 2, 40)
    return {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": comfy.nested_tensor.NestedTensor((mask_v, mask_a)),
        "h3_frozen_video_t": 2,
        "h3_frozen_audio_t": 8,
    }


def test_cache_i2v_conditioning_and_latent():
    positive = _i2v_positive()
    latent = _av_latent()
    tensors, schema = pack_h3_cache(positive, latent)
    loaded_pos, loaded_lat = unpack_h3_cache(tensors, schema)
    assert torch.equal(loaded_pos[0][0], positive[0][0])
    assert torch.equal(loaded_pos[0][1]["minimax_token_tags"], positive[0][1]["minimax_token_tags"])
    assert loaded_pos[0][1]["minimax_keyframes"][0]["resolved_frame_index"] == 0
    assert torch.equal(loaded_pos[0][1]["minimax_keyframes"][0]["latent"],
                       positive[0][1]["minimax_keyframes"][0]["latent"])
    ov, oa = latent["samples"].unbind()
    lv, la = loaded_lat["samples"].unbind()
    assert torch.equal(lv, ov)
    assert torch.equal(la, oa)
    assert loaded_lat["h3_frozen_video_t"] == 2
    mv, ma = loaded_lat["noise_mask"].unbind()
    assert torch.equal(mv, latent["noise_mask"].unbind()[0])
    assert torch.equal(ma, latent["noise_mask"].unbind()[1])


def test_cache_ref2va_conditioning_only():
    positive = _ref2va_positive()
    tensors, schema = pack_h3_cache(positive, None)
    loaded_pos, loaded_lat = unpack_h3_cache(tensors, schema)
    assert loaded_lat is None
    ref = loaded_pos[0][1]["minimax_refs"][0]
    src = positive[0][1]["minimax_refs"][0]
    assert ref["kind"] == "video_audio"
    assert torch.equal(ref["latent"], src["latent"])
    assert torch.equal(ref["audio_latent"], src["audio_latent"])


def test_cache_latent_only():
    latent = _av_latent()
    tensors, schema = pack_h3_cache(None, latent)
    loaded_pos, loaded_lat = unpack_h3_cache(tensors, schema)
    assert loaded_pos is None
    assert torch.equal(loaded_lat["samples"].unbind()[0], latent["samples"].unbind()[0])


def test_cache_requires_something_to_save():
    try:
        pack_h3_cache(None, None)
    except ValueError as e:
        assert "conditioning or a latent" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_validate_h3_cache_rejects_bad_combo():
    assert _validate_h3_cache_combo("../secret.h3cache") is not True
    assert _validate_h3_cache_combo("clip.latent") is not True
    assert _validate_h3_cache_combo(None) is not True
