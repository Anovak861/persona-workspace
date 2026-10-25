"""Option 1: side-car persona stream.

A narrow residual stream s (d_side) runs alongside the base stream h. After base layer l:
    s <- s + MLP_l(norm(s) + R_l norm(h))
    h <- h + scale * W_l s          (W_l zero-initialized, so step 0 == base model)

v0 is per-position (no attention inside the side-car): context reaches s only through the
base stream it reads. That keeps generation cache-free; a causal-attention block is a later option.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from .core import PersonaModule


@dataclass
class SideCarConfig:
    d_model: int
    n_layers: int
    d_side: int = 256
    mlp_mult: int = 4
    layers: tuple | None = None  # which base layers to read/write; None = all


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SideCarLayer(nn.Module):
    def __init__(self, d_model, d_side, mlp_mult):
        super().__init__()
        self.norm_h = RMSNorm(d_model)
        self.norm_s = RMSNorm(d_side)
        self.read = nn.Linear(d_model, d_side, bias=False)
        self.up = nn.Linear(d_side, mlp_mult * d_side, bias=False)
        self.down = nn.Linear(mlp_mult * d_side, d_side, bias=False)
        self.write = nn.Linear(d_side, d_model, bias=False)
        nn.init.zeros_(self.write.weight)

    def forward(self, h, s):
        x = self.norm_s(s) + self.read(self.norm_h(h))
        s = s + self.down(nn.functional.silu(self.up(x)))
        return s, self.write(s)


class SideCar(PersonaModule):
    def __init__(self, cfg: SideCarConfig):
        super().__init__()
        self.cfg = cfg
        self.layer_ids = list(cfg.layers) if cfg.layers is not None else list(range(cfg.n_layers))
        self.blocks = nn.ModuleDict(
            {str(i): SideCarLayer(cfg.d_model, cfg.d_side, cfg.mlp_mult) for i in self.layer_ids}
        )

    def uses_layer(self, idx):
        return str(idx) in self.blocks

    def init_state(self, h):
        return torch.zeros(*h.shape[:-1], self.cfg.d_side, device=h.device, dtype=torch.float32)

    def step(self, idx, h, s, ctx):
        s, w = self.blocks[str(idx)](h.float(), s)
        return h + (ctx.scale * w).to(h.dtype), s, s
