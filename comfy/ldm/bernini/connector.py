# Minimal MLPConnector + RMSNorm for Bernini planner inference.
# Parameter layout matches connector.* keys in bernini_planner.safetensors.

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


class MLPConnector(nn.Module):
    """Connector between MLLM hidden states and gen/vit branches.

    Keys in bernini_planner.safetensors:
      connector.proj_gen.{0,2,3}.*
      connector.pred_vit.{0,2,3,4}.*
    """

    def __init__(
        self,
        in_dim,
        num_layers_for_gen=1,
        out_dim_for_gen=4096,
        enable_gen_branch=True,
        gen_head_type="mlp",
        num_layers_for_vit=1,
        out_dim_for_vit=3584,
        enable_vit_branch=True,
    ):
        super().__init__()
        self.enable_gen_branch = enable_gen_branch
        self.enable_vit_branch = enable_vit_branch
        if enable_gen_branch:
            self.proj_gen = nn.Sequential(
                nn.Linear(in_dim, out_dim_for_gen),
                nn.GELU(),
                RMSNorm(out_dim_for_gen),
                nn.Linear(out_dim_for_gen, out_dim_for_gen),
            )
        if enable_vit_branch:
            self.pred_vit = nn.Sequential(
                nn.Linear(in_dim, out_dim_for_vit),
                nn.GELU(),
                nn.Linear(out_dim_for_vit, out_dim_for_vit),
                RMSNorm(out_dim_for_vit),
                nn.Linear(out_dim_for_vit, out_dim_for_vit),
            )

    @staticmethod
    def _run_projection(proj, x):
        param = next(proj.parameters(), None)
        if param is not None and (x.device != param.device or x.dtype != param.dtype):
            x = x.to(device=param.device, dtype=param.dtype)
        return proj(x)

    def for_gen(self, x):
        return self._run_projection(self.proj_gen, x)

    def for_vit(self, x):
        return self._run_projection(self.pred_vit, x)
