import torch
from PIL import Image
from comfy.cli_args import args, LatentPreviewMethod
from comfy.taesd.taesd import TAESD
from comfy.sd import VAE
import comfy.model_management
import folder_paths
import comfy.utils
import logging

default_preview_method = args.preview_method

MAX_PREVIEW_RESOLUTION = args.preview_size
VIDEO_TAES = ["taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5", "taeltx_2"]

def preview_to_image(latent_image, do_scale=True):
        if do_scale:
            latents_ubyte = (((latent_image + 1.0) / 2.0).clamp(0, 1)  # change scale from -1..1 to 0..1
                                .mul(0xFF)  # to 0..255
                                )
        else:
            latents_ubyte = (latent_image.clamp(0, 1)
                                .mul(0xFF)  # to 0..255
                                )
        if comfy.model_management.directml_enabled:
                latents_ubyte = latents_ubyte.to(dtype=torch.uint8)
        latents_ubyte = latents_ubyte.to(device="cpu", dtype=torch.uint8, non_blocking=comfy.model_management.device_supports_non_blocking(latent_image.device))

        return Image.fromarray(latents_ubyte.numpy())

class LatentPreviewer:
    def decode_latent_to_preview(self, x0):
        pass

    def decode_latent_to_preview_image(self, preview_format, x0):
        preview_image = self.decode_latent_to_preview(x0)
        return ("JPEG", preview_image, MAX_PREVIEW_RESOLUTION)

class TAESDPreviewerImpl(LatentPreviewer):
    def __init__(self, taesd):
        self.taesd = taesd

    def decode_latent_to_preview(self, x0):
        x_sample = self.taesd.decode(x0[:1])[0].movedim(0, 2)
        return preview_to_image(x_sample)

def _decode_taehv_vae_frame_hwc(taesd_vae, latent_frame):
    """Decode one video latent frame (1,C,1,H,W) to an (H,W,C) tensor."""
    out = taesd_vae.decode(latent_frame)
    if out.ndim == 5:
        # (B, C, T, H, W)
        img = out[0, :, 0]
    elif out.ndim == 4:
        img = out[0]
    else:
        img = out
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        return img.movedim(0, -1)
    if img.ndim == 2:
        return img.unsqueeze(-1)
    return img


def _decode_taehv_vae_batch_nhwc(taesd_vae, x0):
    """Decode a (N,C,H,W) latent batch to (N,H,W,C) for VHS animated previews."""
    frames = []
    for i in range(x0.shape[0]):
        latent = x0[i:i + 1].unsqueeze(2)
        frames.append(_decode_taehv_vae_frame_hwc(taesd_vae, latent))
    return torch.stack(frames, dim=0)


def _normalize_vhs_preview_tensor(image_tensor):
    """VHS process_previews expects (N,H,W,C). TAEHV batch decode can yield 5D layouts."""
    if image_tensor.ndim == 5:
        if image_tensor.shape[-1] in (1, 3, 4):
            image_tensor = image_tensor.reshape(
                -1, image_tensor.shape[-3], image_tensor.shape[-2], image_tensor.shape[-1]
            )
        elif image_tensor.shape[1] in (1, 3, 4):
            # (N, C, T, H, W) -> per-frame NHWC
            image_tensor = image_tensor.movedim(1, -1).reshape(
                -1, image_tensor.shape[-3], image_tensor.shape[-2], image_tensor.shape[-1]
            )
    if image_tensor.ndim == 3:
        if image_tensor.shape[-1] in (1, 3, 4):
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.shape[0] in (1, 3, 4):
            image_tensor = image_tensor.movedim(0, -1).unsqueeze(0)
        else:
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(-1)
    if image_tensor.ndim == 4 and image_tensor.shape[1] in (1, 3, 4) and image_tensor.shape[-1] not in (1, 3, 4):
        image_tensor = image_tensor.movedim(1, -1)
    return image_tensor


class TAEHVPreviewerImpl(TAESDPreviewerImpl):
    def decode_latent_to_preview(self, x0):
        x_sample = self.taesd.decode(x0[:1, :, :1])[0][0]
        return preview_to_image(x_sample, do_scale=False)

    def decode_latent_to_preview_image(self, preview_format, x0):
        """Filmstrip of evenly-spaced frames (static preview when VHS animated mode is off)."""
        T = x0.shape[2] if x0.ndim == 5 else 1
        N = min(T, 9)
        if N <= 1:
            return super().decode_latent_to_preview_image(preview_format, x0)

        indices = [round(i * (T - 1) / (N - 1)) for i in range(N)]
        frames = []
        for fi in indices:
            x_sample = self.taesd.decode(x0[:1, :, fi:fi + 1])[0][0]
            frames.append(preview_to_image(x_sample, do_scale=False))

        w, h = frames[0].size
        strip = Image.new("RGB", (w * N, h))
        for i, frame in enumerate(frames):
            strip.paste(frame, (i * w, 0))
        return ("JPEG", strip, MAX_PREVIEW_RESOLUTION)


class Latent2RGBPreviewer(LatentPreviewer):
    def __init__(self, latent_rgb_factors, latent_rgb_factors_bias=None, latent_rgb_factors_reshape=None):
        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu").transpose(0, 1)
        self.latent_rgb_factors_bias = None
        if latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = torch.tensor(latent_rgb_factors_bias, device="cpu")
        self.latent_rgb_factors_reshape = latent_rgb_factors_reshape

    def decode_latent_to_preview(self, x0):
        if self.latent_rgb_factors_reshape is not None:
            x0 = self.latent_rgb_factors_reshape(x0)
        self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
        if self.latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)

        if x0.ndim == 5:
            x0 = x0[0, :, 0]
        else:
            x0 = x0[0]

        latent_image = torch.nn.functional.linear(x0.movedim(0, -1), self.latent_rgb_factors, bias=self.latent_rgb_factors_bias)
        # latent_image = x0[0].permute(1, 2, 0) @ self.latent_rgb_factors

        return preview_to_image(latent_image)


def get_previewer(device, latent_format):
    _patch_vhs_taehv_preview()
    previewer = None
    method = args.preview_method
    if method != LatentPreviewMethod.NoPreviews:
        # TODO previewer methods
        taesd_decoder_path = None
        if latent_format.taesd_decoder_name is not None:
            taesd_decoder_path = next(
                (fn for fn in folder_paths.get_filename_list("vae_approx")
                    if fn.startswith(latent_format.taesd_decoder_name)),
                ""
            )
            taesd_decoder_path = folder_paths.get_full_path("vae_approx", taesd_decoder_path)

        if method == LatentPreviewMethod.Auto:
            method = LatentPreviewMethod.Latent2RGB

        if method == LatentPreviewMethod.TAESD:
            if taesd_decoder_path:
                if latent_format.taesd_decoder_name in VIDEO_TAES:
                    taesd = VAE(comfy.utils.load_torch_file(taesd_decoder_path))
                    taesd.first_stage_model.show_progress_bar = False
                    previewer = TAEHVPreviewerImpl(taesd)
                else:
                    taesd = TAESD(None, taesd_decoder_path, latent_channels=latent_format.latent_channels).to(device)
                    previewer = TAESDPreviewerImpl(taesd)
            else:
                logging.warning("Warning: TAESD previews enabled, but could not find models/vae_approx/{}".format(latent_format.taesd_decoder_name))

        if previewer is None:
            if latent_format.latent_rgb_factors is not None:
                previewer = Latent2RGBPreviewer(latent_format.latent_rgb_factors, latent_format.latent_rgb_factors_bias, latent_format.latent_rgb_factors_reshape)
    return previewer

def prepare_callback(model, steps, x0_output_dict=None):
    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = get_previewer(model.load_device, model.model.latent_format)

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0

        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)
    return callback

def set_preview_method(override: str = None):
    if override and override != "default":
        method = LatentPreviewMethod.from_string(override)
        if method is not None:
            args.preview_method = method
            return
    args.preview_method = default_preview_method


def _patch_vhs_taehv_preview():
    """Fix VHS + lighttaew/TAEHV: batch decode yields 5D tensors that break F.interpolate."""
    global _vhs_taehv_fix_applied
    if _vhs_taehv_fix_applied:
        return

    WrappedPreviewer = None
    vhs_rates_table = {}
    for module_name in (
        "videohelpersuite.latent_preview",
        "custom_nodes.comfyui-videohelpersuite.videohelpersuite.latent_preview",
    ):
        try:
            import importlib
            mod = importlib.import_module(module_name)
            WrappedPreviewer = getattr(mod, "WrappedPreviewer", None)
            vhs_rates_table = getattr(mod, "rates_table", {})
            break
        except ImportError:
            continue

    if WrappedPreviewer is None:
        return

    import io
    import struct
    from PIL import Image
    import torch.nn.functional as F
    from server import PromptServer
    import server
    serv = PromptServer.instance

    _orig_decode = WrappedPreviewer.decode_latent_to_preview

    def _patched_decode_latent_to_preview(self, x0):
        if hasattr(self, "taesd"):
            if x0.ndim == 4 and x0.shape[0] > 0:
                return _decode_taehv_vae_batch_nhwc(self.taesd, x0)
            out = _orig_decode(self, x0)
            return _normalize_vhs_preview_tensor(out)
        return _orig_decode(self, x0)

    def _patched_process_previews(self, image_tensor, ind, leng):
        image_tensor = _patched_decode_latent_to_preview(self, image_tensor)
        image_tensor = _normalize_vhs_preview_tensor(image_tensor)
        if image_tensor.size(1) > 512 or image_tensor.size(2) > 512:
            image_tensor = image_tensor.movedim(-1, 0)
            if image_tensor.size(2) < image_tensor.size(3):
                height = (512 * image_tensor.size(2)) // image_tensor.size(3)
                image_tensor = F.interpolate(image_tensor, (height, 512), mode="bilinear")
            else:
                width = (512 * image_tensor.size(3)) // image_tensor.size(2)
                image_tensor = F.interpolate(image_tensor, (512, width), mode="bilinear")
            image_tensor = image_tensor.movedim(0, -1)
        previews_ubyte = (
            ((image_tensor + 1.0) / 2.0).clamp(0, 1).mul(0xFF)
        ).to(device="cpu", dtype=torch.uint8)
        for preview in previews_ubyte:
            i = Image.fromarray(preview.numpy())
            message = io.BytesIO()
            message.write((1).to_bytes(length=4, byteorder="big") * 2)
            message.write(ind.to_bytes(length=4, byteorder="big"))
            message.write(struct.pack("16p", serv.last_node_id.encode("ascii")))
            i.save(message, format="JPEG", quality=95, compress_level=1)
            serv.send_sync(
                server.BinaryEventTypes.PREVIEW_IMAGE,
                message.getvalue(),
                serv.client_id,
            )
            ind = (ind + 1) % leng

    WrappedPreviewer.decode_latent_to_preview = _patched_decode_latent_to_preview
    WrappedPreviewer.process_previews = _patched_process_previews

    # Skip VHS wrap for TAEHV — use core filmstrip preview (avoids broken animated path).
    hooked = get_previewer
    if hasattr(hooked, "__wrapped__"):
        _vhs_inner = hooked.__wrapped__

        def _patched_vhs_get_previewer(device, latent_format, *args, **kwargs):
            previewer = _vhs_inner(device, latent_format, *args, **kwargs)
            if isinstance(previewer, TAEHVPreviewerImpl):
                return previewer
            try:
                from server import PromptServer
                serv = PromptServer.instance
                extra_info = next(serv.prompt_queue.currently_running.values().__iter__())[3]["extra_pnginfo"]["workflow"]["extra"]
                prev_setting = extra_info.get("VHS_latentpreview", False)
                if extra_info.get("VHS_latentpreviewrate", 0) != 0:
                    rate_setting = extra_info["VHS_latentpreviewrate"]
                else:
                    rate_setting = vhs_rates_table.get(latent_format.__class__.__name__, 8)
            except Exception:
                prev_setting = False
                rate_setting = 8
            if not prev_setting or not hasattr(previewer, "decode_latent_to_preview"):
                return previewer
            return WrappedPreviewer(previewer, rate_setting)

        _patched_vhs_get_previewer.__wrapped__ = _vhs_inner
        import sys
        sys.modules[__name__].get_previewer = _patched_vhs_get_previewer

    _vhs_taehv_fix_applied = True
    logging.info("Bernini/ComfyUI: patched VideoHelperSuite preview for lighttaew/TAEHV")


_vhs_taehv_fix_applied = False

