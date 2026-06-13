# SPDX-License-Identifier: Apache-2.0
"""Bernini chained guidance guiders (Bernini-R + full Bernini renderer)."""

from typing import List

import comfy.samplers
from comfy.bernini.context import (
    build_branch_cond_list,
    get_branch_cross_attn,
    get_context_latents,
    get_processed_context_latents,
    get_pooled_value,
    split_context_branches,
)
from comfy.bernini.guidance import (
    MomentumBuffer,
    chained_cfg_rv2v,
    chained_cfg_v2v_chain,
    normalized_guidance,
    vae_txt_vit_wapg,
)
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

GUIDANCE_MODES = [
    "rv2v",
    "v2v",
    "v2v_chain",
    "t2v",
    "v2v_apg",
    "t2v_apg",
    "rv2v_wapg",
    "vae_txt_vit_wapg",
]


def _calc_one(model, cond, x, timestep, model_options):
    if cond is None:
        return None
    # calc_cond_batch expects list[list[dict]] (outer = branches, inner = cond entries).
    # Wrap the single branch in an outer list so conds[0] is the list[dict], not a dict.
    return comfy.samplers.calc_cond_batch(model, [cond], x, timestep, model_options)[0]


class Guider_Bernini(comfy.samplers.CFGGuider):
    def set_bernini(
        self,
        guidance_mode: str,
        omega_vid: float,
        omega_img: float,
        omega_txt: float,
        eta: float = 0.5,
        norm_threshold: float = 50.0,
        momentum: float = 0.0,
        omega_tgt: float = 1.0,
        omega_scale: float = 1.0,
        switch_dit_boundary: float = 0.875,
        num_train_timesteps: int = 1000,
        apg_parallel_scale: float = 0.2,
        apg_orthogonal_scale: float = 1.0,
    ):
        self.guidance_mode = guidance_mode
        self.omega_vid = omega_vid
        self.omega_img = omega_img
        self.omega_txt = omega_txt
        self.omega_tgt = omega_tgt
        self.eta = eta
        self.norm_threshold = norm_threshold
        self.momentum = momentum
        self.omega_scale = omega_scale
        self.switch_dit_boundary = switch_dit_boundary
        self.num_train_timesteps = num_train_timesteps
        self.apg_parallel_scale = apg_parallel_scale
        self.apg_orthogonal_scale = apg_orthogonal_scale
        self._momentum_buffers: List[MomentumBuffer] = []
        self._omega_scaled = False

    def set_conds(self, positive, negative):
        self._num_videos = int(get_pooled_value(positive, "bernini_num_videos", 0) or 0)
        self.inner_set_conds({"positive": positive, "negative": negative})

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        self._omega_scaled = False
        return super().sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)

    def _scaled_omegas(self, timestep):
        txt, tgt, img, vid = self.omega_txt, self.omega_tgt, self.omega_img, self.omega_vid
        if self.omega_scale != 1.0 and not self._omega_scaled:
            boundary = self.switch_dit_boundary * self.num_train_timesteps
            t_val = float(timestep.flatten()[0])
            if t_val < boundary:
                txt *= self.omega_scale
                tgt *= self.omega_scale
                img *= self.omega_scale
                vid *= self.omega_scale
                self._omega_scaled = True
        return vid, img, txt, tgt

    def _build_conds(self, positive, negative, ctx_none, ctx_video, ctx_all):
        pos = self.conds.get("positive")
        neg = self.conds.get("negative")
        return {
            "none_neg": build_branch_cond_list(neg, ctx_none),
            "v_neg": build_branch_cond_list(neg, ctx_video),
            "vi_neg": build_branch_cond_list(neg, ctx_all),
            "vi_pos": build_branch_cond_list(pos, ctx_all),
            "t_pos": build_branch_cond_list(pos, ctx_none),
        }

    def _build_wvitcfg_conds(self, positive):
        pos = self.conds.get("positive")
        raw_ctx = get_processed_context_latents(positive)
        ctx_all = list(raw_ctx) if raw_ctx else None
        txt_wtxt_wvit = get_branch_cross_attn(pos, "wtxt_wvit")
        txt_wtxt_wovit = get_branch_cross_attn(pos, "wtxt_wovit")
        txt_wotxt_wovit = get_branch_cross_attn(pos, "wotxt_wovit")
        return {
            "base": build_branch_cond_list(pos, None, txt_wotxt_wovit),
            "img": build_branch_cond_list(pos, ctx_all, txt_wotxt_wovit),
            "txt": build_branch_cond_list(pos, ctx_all, txt_wtxt_wovit),
            "vit": build_branch_cond_list(pos, ctx_all, txt_wtxt_wvit),
        }

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        positive_cond = self.conds.get("positive", None)
        negative_cond = self.conds.get("negative", None)
        # Use model_conds-processed latents (already normalised by process_latent_in).
        # Branch cond lists bypass extra_conds so they must carry pre-scaled tensors.
        proc_ctx = get_processed_context_latents(positive_cond)
        ctx_none, ctx_video, ctx_all = split_context_branches(proc_ctx, self._num_videos)
        mode = self.guidance_mode
        # Call once so all four omegas are consistently scaled for this step.
        vid_omega, img_omega, txt_omega, tgt_omega = self._scaled_omegas(timestep)

        if mode == "vae_txt_vit_wapg":
            branches = self._build_wvitcfg_conds(positive_cond)
            eps_base = _calc_one(self.inner_model, branches["base"], x, timestep, model_options)
            if img_omega > 0.0:
                eps_img = _calc_one(self.inner_model, branches["img"], x, timestep, model_options)
            else:
                eps_img = eps_base
            if txt_omega > 0.0:
                eps_txt = _calc_one(self.inner_model, branches["txt"], x, timestep, model_options)
            else:
                eps_txt = eps_img
            if tgt_omega > 0.0:
                eps_vit = _calc_one(self.inner_model, branches["vit"], x, timestep, model_options)
            else:
                eps_vit = eps_txt
            return vae_txt_vit_wapg(
                eps_base,
                eps_img,
                eps_txt,
                eps_vit,
                img_omega,
                txt_omega,
                tgt_omega,
                parallel_scale=self.apg_parallel_scale,
                orthogonal_scale=self.apg_orthogonal_scale,
            )

        branches = self._build_conds(positive_cond, negative_cond, ctx_none, ctx_video, ctx_all)

        if mode == "t2v":
            eps_u = _calc_one(self.inner_model, branches["none_neg"], x, timestep, model_options)
            eps_t = _calc_one(self.inner_model, branches["t_pos"], x, timestep, model_options)
            return eps_u + txt_omega * (eps_t - eps_u)

        if mode == "t2v_apg":
            eps_u = _calc_one(self.inner_model, branches["none_neg"], x, timestep, model_options)
            eps_t = _calc_one(self.inner_model, branches["t_pos"], x, timestep, model_options)
            mb = MomentumBuffer(self.momentum)
            return normalized_guidance(
                eps_t, eps_u, txt_omega, mb, self.eta, self.norm_threshold
            )

        if mode == "v2v":
            eps_u = _calc_one(self.inner_model, branches["vi_neg"], x, timestep, model_options)
            eps_t = _calc_one(self.inner_model, branches["vi_pos"], x, timestep, model_options)
            return eps_u + txt_omega * (eps_t - eps_u)

        if mode == "v2v_apg":
            eps_u = _calc_one(self.inner_model, branches["vi_neg"], x, timestep, model_options)
            eps_t = _calc_one(self.inner_model, branches["vi_pos"], x, timestep, model_options)
            mb = MomentumBuffer(self.momentum)
            return normalized_guidance(
                eps_t, eps_u, txt_omega, mb, self.eta, self.norm_threshold
            )

        if mode == "v2v_chain":
            eps_none = _calc_one(self.inner_model, branches["none_neg"], x, timestep, model_options)
            eps_v = _calc_one(self.inner_model, branches["v_neg"], x, timestep, model_options)
            eps_vti = _calc_one(self.inner_model, branches["vi_pos"], x, timestep, model_options)
            return chained_cfg_v2v_chain(eps_none, eps_v, eps_vti, vid_omega, txt_omega)

        if mode == "rv2v":
            eps_none = _calc_one(self.inner_model, branches["none_neg"], x, timestep, model_options)
            eps_v = _calc_one(self.inner_model, branches["v_neg"], x, timestep, model_options)
            eps_vi = _calc_one(self.inner_model, branches["vi_neg"], x, timestep, model_options)
            eps_vti = _calc_one(self.inner_model, branches["vi_pos"], x, timestep, model_options)
            return chained_cfg_rv2v(
                eps_none, eps_v, eps_vi, eps_vti,
                vid_omega, img_omega, txt_omega,
            )

        if mode == "rv2v_wapg":
            # Plain chained CFG with 4 omega terms — matching sample_one_step in the original.
            # Differs from rv2v only in the split of positive into txt-only and txt+VIT branches
            # so that omega_tgt controls the VIT/planner link independently.
            eps_none = _calc_one(self.inner_model, branches["none_neg"], x, timestep, model_options)
            eps_v = _calc_one(self.inner_model, branches["v_neg"], x, timestep, model_options)
            eps_vi = _calc_one(self.inner_model, branches["vi_neg"], x, timestep, model_options)
            # eps_vti: video+img context, positive text, NO VIT tokens (wtxt_wovit)
            # eps_vtic: video+img context, positive text + VIT tokens (wtxt_wvit)
            pos = self.conds.get("positive")
            raw_ctx = get_processed_context_latents(positive_cond)
            ctx_all_list = list(raw_ctx) if raw_ctx else None
            txt_wtxt_wovit = get_branch_cross_attn(pos, "wtxt_wovit")
            txt_wtxt_wvit = get_branch_cross_attn(pos, "wtxt_wvit")
            vi_pos_txt = build_branch_cond_list(pos, ctx_all_list, txt_wtxt_wovit)
            vi_pos_vit = build_branch_cond_list(pos, ctx_all_list, txt_wtxt_wvit)
            eps_vti = _calc_one(self.inner_model, vi_pos_txt, x, timestep, model_options)
            eps_vtic = _calc_one(self.inner_model, vi_pos_vit, x, timestep, model_options)
            return (
                eps_none
                + vid_omega * (eps_v   - eps_none)
                + img_omega * (eps_vi  - eps_v)
                + txt_omega * (eps_vti - eps_vi)
                + tgt_omega * (eps_vtic - eps_vti)
            )

        raise ValueError(f"Unknown Bernini guidance_mode: {mode}")


class BerniniGuider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniGuider",
            display_name="Bernini Guider",
            category="model/sampling/guiders",
            description="Chained Bernini guidance (rv2v, v2v, t2v, APG, vae_txt_vit_wapg). "
                        "Use with BerniniConditioning + BerniniMergePlannerText and SamplerCustomAdvanced.",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Combo.Input("guidance_mode", options=GUIDANCE_MODES, default="rv2v"),
                io.Float.Input("omega_vid", default=3.0, min=0.0, max=20.0, step=0.05,
                               tooltip="Video context guidance scale (official default 3.0)."),
                io.Float.Input("omega_img", default=3.0, min=0.0, max=20.0, step=0.05,
                               tooltip="Image/reference context guidance scale (official default 3.0)."),
                io.Float.Input("omega_txt", default=4.0, min=0.0, max=20.0, step=0.05,
                               tooltip="Text guidance scale (official default 4.0)."),
                io.Float.Input("omega_tgt", default=4.0, min=0.0, max=20.0, step=0.05,
                               tooltip="VIT / planner branch scale (vae_txt_vit_wapg, official default 4.0)."),
                io.Float.Input("omega_scale", default=0.75, min=0.0, max=5.0, step=0.05, advanced=True,
                               tooltip="Scale omegas after dual-expert switch (low-noise expert). Official default 0.75."),
                io.Float.Input("switch_dit_boundary", default=0.875, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Int.Input("num_train_timesteps", default=1000, min=1, max=10000, advanced=True),
                io.Float.Input("eta", default=0.5, min=0.0, max=10.0, step=0.01, advanced=True,
                               tooltip="APG parallel scale (APG modes only)."),
                io.Float.Input("norm_threshold", default=50.0, min=0.0, max=500.0, step=0.1, advanced=True),
                io.Float.Input("momentum", default=0.0, min=-5.0, max=1.0, step=0.01, advanced=True),
                io.Float.Input("apg_parallel_scale", default=0.2, min=0.0, max=2.0, step=0.01, advanced=True),
                io.Float.Input("apg_orthogonal_scale", default=1.0, min=0.0, max=2.0, step=0.01, advanced=True),
            ],
            outputs=[io.Guider.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        positive,
        negative,
        guidance_mode,
        omega_vid,
        omega_img,
        omega_txt,
        omega_tgt=4.0,
        omega_scale=0.75,
        switch_dit_boundary=0.875,
        num_train_timesteps=1000,
        eta=0.5,
        norm_threshold=50.0,
        momentum=0.0,
        apg_parallel_scale=0.2,
        apg_orthogonal_scale=1.0,
    ) -> io.NodeOutput:
        guider = Guider_Bernini(model)
        guider.set_conds(positive, negative)
        guider.set_bernini(
            guidance_mode=guidance_mode,
            omega_vid=omega_vid,
            omega_img=omega_img,
            omega_txt=omega_txt,
            omega_tgt=omega_tgt,
            omega_scale=omega_scale,
            switch_dit_boundary=switch_dit_boundary,
            num_train_timesteps=num_train_timesteps,
            eta=eta,
            norm_threshold=norm_threshold,
            momentum=momentum,
            apg_parallel_scale=apg_parallel_scale,
            apg_orthogonal_scale=apg_orthogonal_scale,
        )
        guider.set_cfg(1.0)
        return io.NodeOutput(guider)


class BerniniSamplerExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [BerniniGuider]


async def comfy_entrypoint() -> BerniniSamplerExtension:
    return BerniniSamplerExtension()
