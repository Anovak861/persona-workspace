"""Option 2: global workspace bottleneck.

K slots of width d_ws, computed causally per position (slot state z_t depends only on tokens <= t).

  read  (after base layers in `read_layers`): each slot queries the prefix positions of that
        layer's residual (the *sources*) and adds what it retrieves to its state. Sources compete
        for each slot through the softmax. What earlier reads retrieved is carried in z, so
        later reads query with it. (Pooling sources across read layers would need an explicit
        T*K x S mask and doesn't fit the fused kernels at T=2048.)
  process: slot self-attention + MLP among the K slots, per position.
  write (after base layers in `write_layers`): each position attends over its own K slots and
        writes the result back into the residual.
          soft: h <- h + scale * W z                     (W zero-init: step 0 == base)
          hard: h <- h - P P^T h + scale * P f(z)        (P: d_model x r orthonormal, frozen)
                The base's own content in subspace P is removed at write layers and replaced
                by workspace content, so downstream layers and the head see P only through the
                workspace. P is meant to be the base's role/persona subspace
                (scripts/fit_role_subspace.py); with no P given, a random one is used.

Access log (ctx.record): per read layer, slot->position attention weights (B, T, K, T), mean
over heads, plus slot states and per-position broadcast weights over slots. Recording uses an explicit attention path, so keep sequences short.

Generation: source keys/values persist in ctx.cache across decoding steps. Assumes right
padding (or batch size 1), since padded prefix positions would otherwise be attendable sources.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import PersonaModule
from .sidecar import RMSNorm


@dataclass
class WorkspaceConfig:
    d_model: int
    n_layers: int
    n_slots: int = 16
    d_ws: int = 512
    n_heads: int = 8
    mlp_mult: int = 4
    read_layers: tuple = (3, 7, 11)
    write_layers: tuple = (5, 9, 13)
    hard: bool = False
    hard_rank: int = 64


class SlotProcess(nn.Module):
    def __init__(self, d, n_heads, mlp_mult):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True, bias=False)
        self.up = nn.Linear(d, mlp_mult * d, bias=False)
        self.down = nn.Linear(mlp_mult * d, d, bias=False)

    def forward(self, z):  # (B, T, K, d)
        B, T, K, d = z.shape
        x = z.reshape(B * T, K, d)
        y = self.n1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        x = x + self.down(F.silu(self.up(self.n2(x))))
        return x.reshape(B, T, K, d)


class Read(nn.Module):
    def __init__(self, cfg: WorkspaceConfig):
        super().__init__()
        d = cfg.d_ws
        self.h = cfg.n_heads
        self.norm_h, self.norm_z = RMSNorm(cfg.d_model), RMSNorm(d)
        self.k = nn.Linear(cfg.d_model, d, bias=False)
        self.v = nn.Linear(cfg.d_model, d, bias=False)
        self.q = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.process = SlotProcess(d, cfg.n_heads, cfg.mlp_mult)

    def sources(self, h):
        x = self.norm_h(h)
        return self.k(x), self.v(x)

    def forward(self, z, slot_emb, src_k, src_v, q_pos, src_pos, attn_dtype, explicit=False):
        """z (B,T,K,d); src_k/v (B,S,d); q_pos (T,), src_pos (S,) absolute positions.
        Slots are folded into query heads (B, H*K, T, dh) sharing the H source heads (GQA),
        so the full-sequence case is plain causal attention on the fused kernels."""
        B, T, K, d = z.shape
        H, dh = self.h, d // self.h
        S = src_k.shape[1]
        q = self.q(self.norm_z(z + slot_emb))  # (B,T,K,d)
        q = q.reshape(B, T, K, H, dh).permute(0, 3, 2, 1, 4).reshape(B, H * K, T, dh)  # head = h*K + k
        k = src_k.reshape(B, S, H, dh).transpose(1, 2)
        v = src_v.reshape(B, S, H, dh).transpose(1, 2)
        weights = None
        if explicit:
            mask = src_pos[None, :] <= q_pos[:, None]  # (T, S)
            kk, vv = k.repeat_interleave(K, 1), v.repeat_interleave(K, 1)
            s = (q @ kk.transpose(-1, -2)) / math.sqrt(dh)
            a = s.masked_fill(~mask, float("-inf")).softmax(-1)  # (B, H*K, T, S)
            out = a @ vv
            weights = a.reshape(B, H, K, T, S).mean(1).permute(0, 2, 1, 3)  # (B, T, K, S)
        else:
            q, k, v = q.to(attn_dtype), k.to(attn_dtype), v.to(attn_dtype)
            if T == S:  # full sequence from position 0
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            else:  # decoding against cached sources
                mask = src_pos[None, :] <= q_pos[:, None]
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
            out = out.float()
        out = out.reshape(B, H, K, T, dh).permute(0, 3, 2, 1, 4).reshape(B, T, K, d)
        z = z + self.o(out)
        return self.process(z), weights


class Write(nn.Module):
    def __init__(self, cfg: WorkspaceConfig, P: torch.Tensor | None):
        super().__init__()
        d = cfg.d_ws
        self.hard = cfg.hard
        self.norm_h, self.norm_z = RMSNorm(cfg.d_model), RMSNorm(d)
        self.q = nn.Linear(cfg.d_model, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, cfg.hard_rank if cfg.hard else cfg.d_model, bias=False)
        nn.init.zeros_(self.out.weight)
        if cfg.hard:
            self.register_buffer("P", P)

    def forward(self, h, z, scale):
        if self.hard:  # base content in P is removed before anything (incl. the query) sees it
            h = h - (h @ self.P) @ self.P.T
        q = self.q(self.norm_h(h))  # (B,T,d)
        zn = self.norm_z(z)
        k, v = self.k(zn), self.v(zn)  # (B,T,K,d)
        a = torch.einsum("btd,btkd->btk", q, k).div(math.sqrt(q.shape[-1])).softmax(-1)
        y = self.out(torch.einsum("btk,btkd->btd", a, v))
        if self.hard:
            y = y @ self.P.T
        return h + scale * y, a


class Workspace(PersonaModule):
    def __init__(self, cfg: WorkspaceConfig, P: dict | None = None):
        """P: {write_layer: (d_model, hard_rank) orthonormal} for the hard variant."""
        super().__init__()
        self.cfg = cfg
        assert min(cfg.write_layers) > min(cfg.read_layers), "first write must come after first read"
        self.slot_emb = nn.Parameter(torch.randn(cfg.n_slots, cfg.d_ws) * cfg.d_ws ** -0.5)
        self.reads = nn.ModuleDict({str(i): Read(cfg) for i in cfg.read_layers})
        if cfg.hard and P is None:
            g = torch.Generator().manual_seed(0)
            P = {i: torch.linalg.qr(torch.randn(cfg.d_model, cfg.hard_rank, generator=g))[0]
                 for i in cfg.write_layers}
        self.writes = nn.ModuleDict({str(i): Write(cfg, P[i] if cfg.hard else None) for i in cfg.write_layers})

    def uses_layer(self, idx):
        return str(idx) in self.reads or str(idx) in self.writes

    def init_state(self, h):
        B, T, _ = h.shape
        return torch.zeros(B, T, self.cfg.n_slots, self.cfg.d_ws, device=h.device, dtype=torch.float32)

    def step(self, idx, h, z, ctx):
        # Pure in (h, z), so checkpoint recompute is safe. ctx.cache only carries this read
        # layer's sources between decoding calls and is written only without grad.
        hf = h.float()
        rec = {}
        if str(idx) in self.reads:
            read = self.reads[str(idx)]
            T = h.shape[1]
            pos = ctx.cache_position.to(h.device) if ctx.cache_position is not None else torch.arange(T, device=h.device)
            k, v = read.sources(hf)
            past = ctx.cache.get(idx)
            if past is not None:
                k, v, pos_all = torch.cat([past[0], k], 1), torch.cat([past[1], v], 1), torch.cat([past[2], pos])
            else:
                pos_all = pos
            if not torch.is_grad_enabled():
                ctx.cache[idx] = (k, v, pos_all)
            z, w = read(z, self.slot_emb, k, v, pos, pos_all, h.dtype, explicit=ctx.record)
            if ctx.record:
                rec["access"] = w
                rec["slots"] = z
        if str(idx) in self.writes:
            hf, a = self.writes[str(idx)](hf, z, ctx.scale)
            if ctx.record:
                rec["broadcast"] = a
        return hf.to(h.dtype), z, rec or None
