"""MiniMax H3 local upscale / in-context regenerate helpers.

Official H3-Regenerate-2K is not open-sourced. These nodes approximate it:
- Encode AV: IMAGE frames + AUDIO -> NestedTensor AV latent
- Upscale Latent: spatially upscale a previous AV NestedTensor for a low-sigma refine
- Regenerate: decode the previous sample, attach as Ref2VA <Audio 1>/<Video 1>,
  and prepare a target-resolution latent (upscaled or empty)

Does not modify the core MiniMax H3 conditioning nodes.
"""

import torch
import torchaudio

import nodes
import comfy.nested_tensor
import comfy.utils
import node_helpers
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_audio import vae_decode_audio
from comfy_extras import nodes_minimax_h3 as h3

REGENERATE_HINT = (
    "Regenerate <Video 1> (soundtrack <Audio 1>) at higher resolution with sharper "
    "detail. Preserve motion, identity, composition, and audio-visual sync.\n\n"
)


def _target_pixel_size(lh, lw, scale_by, width, height):
    m = h3.CANVAS_MULTIPLE
    if width <= 0 and height <= 0:
        tw = max(m, round(lw * 16 * scale_by / m) * m)
        th = max(m, round(lh * 16 * scale_by / m) * m)
    elif width <= 0:
        th = max(m, round(height / m) * m)
        tw = max(m, round(lw * 16 * th / (lh * 16) / m) * m)
    elif height <= 0:
        tw = max(m, round(width / m) * m)
        th = max(m, round(lh * 16 * tw / (lw * 16) / m) * m)
    else:
        tw = max(m, round(width / m) * m)
        th = max(m, round(height / m) * m)
    return tw, th


def _align_frame_count_down(n):
    """Snap frame count down to the H3 17k+5 grid (same as reference video packing)."""
    n = int(n)
    if n < 5:
        raise ValueError("MiniMax H3 needs at least 5 frames (~0.2s at 24 fps)")
    while n % 17 != 5:
        n -= 1
    if n < 5:
        raise ValueError("MiniMax H3 needs at least 5 frames on the 17k+5 grid")
    return n


def _prepare_stereo_waveform(audio, sample_rate, num_samples):
    """Resample to sample_rate, force stereo [1, 2, L], trim/pad to num_samples."""
    waveform = audio["waveform"][:1]  # [1, C, L]
    sr = audio["sample_rate"]
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    elif waveform.shape[1] > 2:
        waveform = waveform[:, :2]
    cur = waveform.shape[-1]
    if cur < num_samples:
        waveform = torch.nn.functional.pad(waveform, (0, num_samples - cur))
    elif cur > num_samples:
        waveform = waveform[..., :num_samples]
    return waveform


def _encode_ref_audio(audio_vae, audio):
    waveform = audio["waveform"]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    return z, z.shape[-1]


def _pack_ref_video(vae, frames, frame_count, soundtrack=None, audio_vae=None):
    vh, vw = frames.shape[1], frames.shape[2]
    cw, ch = h3.adapt_canvas(vw, vh)
    if vw * vh < cw * ch:
        m = h3.CANVAS_MULTIPLE
        cw = max(m, round(vw / m) * m)
        ch = max(m, round(vh / m) * m)
    frames = h3._resize(frames, cw, ch, "disabled")
    if frames.shape[0] > frame_count:
        frames = frames[:frame_count]
    n = frames.shape[0]
    if n < 5:
        raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
    while n % 17 != 5:
        n -= 1
    frames = frames[:n]
    z = vae.encode(frames)

    ref_items = []
    audio_latent, ref_audio_t = (None, 0)
    if soundtrack is not None and audio_vae is not None:
        audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
        ref_items.append({"type": "audio"})
    sample_idx = list(range(0, frames.shape[0], h3.FPS // 2))
    ref_items.append({"type": "video", "data": frames[sample_idx],
                      "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
    ref_blocks = [{"kind": "video_audio" if ref_audio_t else "video",
                   "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                   "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent}]
    return ref_items, ref_blocks


def _frame_count_from_video_latent(latent_t):
    if latent_t <= 2:
        return h3.align_frame_count(5)
    return h3.align_frame_count(((latent_t - 2) // 5) * 17 + 5)


class MiniMaxH3EncodeAV(io.ComfyNode):
    """Encode IMAGE frames + AUDIO into a MiniMax H3 NestedTensor AV latent."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3EncodeAV",
            display_name="MiniMax H3 Encode AV",
            search_aliases=["minimax encode", "video audio to latent", "av encode"],
            category="model/latent/minimax",
            description="Encode video frames and audio into a joint MiniMax H3 AV latent.",
            inputs=[
                io.Image.Input("images", tooltip="Video frames at 24 fps (IMAGE batch)"),
                io.Audio.Input("audio", tooltip="Soundtrack paired with the video"),
                io.Vae.Input("vae", tooltip="MiniMax H3 video VAE"),
                io.Vae.Input("audio_vae", tooltip="MiniMax H3 audio VAE"),
                io.Int.Input("width", default=0, min=0, max=nodes.MAX_RESOLUTION, step=32,
                    tooltip="Target pixel width (0 = use image width, rounded to ×32)"),
                io.Int.Input("height", default=0, min=0, max=nodes.MAX_RESOLUTION, step=32,
                    tooltip="Target pixel height (0 = use image height, rounded to ×32)"),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, images, audio, vae, audio_vae, width=0, height=0) -> io.NodeOutput:
        if images is None or images.shape[0] < 5:
            raise ValueError("MiniMax H3 Encode AV needs at least 5 frames (~0.2s at 24 fps)")
        if audio is None:
            raise ValueError("MiniMax H3 Encode AV requires audio")

        n = _align_frame_count_down(images.shape[0])
        frames = images[:n]
        ih, iw = frames.shape[1], frames.shape[2]
        m = h3.CANVAS_MULTIPLE
        if width <= 0:
            width = max(m, round(iw / m) * m)
        else:
            width = max(m, round(width / m) * m)
        if height <= 0:
            height = max(m, round(ih / m) * m)
        else:
            height = max(m, round(height / m) * m)
        if frames.shape[1] != height or frames.shape[2] != width:
            frames = h3._resize(frames, width, height, "disabled")

        video_z = vae.encode(frames)

        _, _, audio_t = h3.temporal_shape(n)
        vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
        # exact multiple of hop (sr / AUDIO_LATENT_FPS) so encode T matches empty-AV audio_t
        waveform = _prepare_stereo_waveform(audio, vae_sr, audio_t * (vae_sr // h3.AUDIO_LATENT_FPS))
        # same encode convention as MiniMaxH3ReferenceToVideo / VAEEncodeAudio
        audio_z = audio_vae.encode(waveform.movedim(1, -1))

        return io.NodeOutput({"samples": comfy.nested_tensor.NestedTensor((video_z, audio_z))})


class MiniMaxH3UpscaleLatent(io.ComfyNode):
    """Spatially upscale a previous H3 AV latent for a low-sigma refine pass."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3UpscaleLatent",
            display_name="MiniMax H3 Upscale Latent",
            search_aliases=["minimax upscale", "h3 regenerate", "in-context upscale"],
            category="model/latent/minimax",
            description="Upscale a MiniMax H3 AV latent for a low-sigma refine pass.",
            inputs=[
                io.Latent.Input("samples", tooltip="Previous H3 AV NestedTensor latent (video + audio)"),
                io.Float.Input("scale_by", default=2.0, min=1.0, max=8.0, step=0.5,
                    tooltip="Spatial scale factor when width/height are 0 (2 = 2×, 3 = 3×)"),
                io.Int.Input("width", default=0, min=0, max=nodes.MAX_RESOLUTION, step=32,
                    tooltip="Target pixel width (0 = use scale_by). Rounded to a multiple of 32."),
                io.Int.Input("height", default=0, min=0, max=nodes.MAX_RESOLUTION, step=32,
                    tooltip="Target pixel height (0 = use scale_by). Rounded to a multiple of 32."),
                io.Combo.Input("upscale_method", options=["bilinear", "bicubic", "nearest-exact", "area", "bislerp"],
                    default="bilinear"),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, samples, scale_by, width, height, upscale_method) -> io.NodeOutput:
        av = samples["samples"]
        if not getattr(av, "is_nested", False):
            raise ValueError("MiniMax H3 Upscale Latent expects a NestedTensor AV latent")
        video, audio = av.unbind()
        tw, th = _target_pixel_size(video.shape[-2], video.shape[-1], scale_by, width, height)
        video_up = comfy.utils.common_upscale(video, tw // 16, th // 16, upscale_method, "disabled")
        out = samples.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((video_up, audio))
        out.pop("noise_mask", None)
        return io.NodeOutput(out)


class MiniMaxH3Regenerate(io.ComfyNode):
    """Local in-context regenerate from a previous H3 AV sample (Ref2VA path)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Regenerate",
            display_name="MiniMax H3 Regenerate",
            search_aliases=["minimax 2k", "h3 regenerate", "in-context regenerate", "upscale regenerate"],
            category="model/conditioning/minimax",
            description="Build Ref2VA in-context conditioning from a previous H3 AV sample for high-res regeneration.",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae", tooltip="MiniMax H3 video VAE"),
                io.Vae.Input("audio_vae", tooltip="MiniMax H3 audio VAE"),
                io.Latent.Input("samples", tooltip="Previous H3 AV NestedTensor latent to regenerate from"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                    tooltip="Same prompt as the base run. Mentions of <Video 1> / <Audio 1> help; regenerate_hint adds one if enabled."),
                io.Float.Input("scale_by", default=2.0, min=1.0, max=8.0, step=0.5,
                    tooltip="Spatial scale when width/height are 0"),
                io.Int.Input("width", default=0, min=0, max=nodes.MAX_RESOLUTION, step=32,
                    tooltip="Target pixel width (0 = use scale_by)"),
                io.Int.Input("height", default=0, min=0, max=nodes.MAX_RESOLUTION, step=32,
                    tooltip="Target pixel height (0 = use scale_by)"),
                io.Combo.Input("latent_source", options=["upscaled", "empty"], default="upscaled",
                    tooltip="upscaled: start from spatially upscaled previous sample (use with low_sigmas). empty: fresh noise at target size (full regenerate with ICL ref only)."),
                io.Combo.Input("upscale_method", options=["bilinear", "bicubic", "nearest-exact", "area", "bislerp"],
                    default="bilinear"),
                io.Boolean.Input("regenerate_hint", default=True,
                    tooltip="Prepend a short instruction that references <Video 1> / <Audio 1>"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="latent"),
                io.Image.Output(display_name="ref_video", tooltip="Decoded previous frames used as the in-context reference"),
                io.Audio.Output(display_name="ref_audio", tooltip="Decoded previous soundtrack used as the in-context reference"),
            ],
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, samples, prompt, scale_by, width, height,
                latent_source="upscaled", upscale_method="bilinear", regenerate_hint=True) -> io.NodeOutput:
        av = samples["samples"]
        if not getattr(av, "is_nested", False):
            raise ValueError("MiniMax H3 Regenerate expects a NestedTensor AV latent")
        video_z, audio_z = av.unbind()

        frames = vae.decode(video_z)
        if frames.ndim == 5:
            frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
        soundtrack = vae_decode_audio(audio_vae, {"samples": audio_z})

        tw, th = _target_pixel_size(video_z.shape[-2], video_z.shape[-1], scale_by, width, height)
        frame_count = _frame_count_from_video_latent(video_z.shape[2])

        if latent_source == "upscaled":
            video_up = comfy.utils.common_upscale(video_z, tw // 16, th // 16, upscale_method, "disabled")
            latent = samples.copy()
            latent["samples"] = comfy.nested_tensor.NestedTensor((video_up, audio_z))
            latent.pop("noise_mask", None)
        else:
            latent, frame_count = h3._empty_av_latent(tw, th, frame_count, batch_size=video_z.shape[0])

        text = (REGENERATE_HINT + prompt) if regenerate_hint else prompt
        ref_items, ref_blocks = _pack_ref_video(vae, frames, frame_count, soundtrack, audio_vae)
        tokens = clip.tokenize(text, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
        return io.NodeOutput(cond, latent, frames, soundtrack)


class MiniMaxH3UpscaleExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            MiniMaxH3EncodeAV,
            MiniMaxH3UpscaleLatent,
            MiniMaxH3Regenerate,
        ]


async def comfy_entrypoint() -> MiniMaxH3UpscaleExtension:
    return MiniMaxH3UpscaleExtension()
