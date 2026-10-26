"""Attach a persona module to a frozen HF decoder.

Each decoder layer is wrapped in a PersonaLayer. The module carries a per-forward state
(e.g. the side-car stream) from layer to layer through a context object; the per-layer step
is a pure function of (h, state), so it can be activation-checkpointed together with the base
layer without the recompute double-updating the state.

Usage:
    persona = SideCar(cfg)
    handle = attach(model, persona)
    handle.enabled = False          # ablate
    handle.scale = 2.0              # steer
    handle.record = True            # collect per-layer states in handle.records
    handle.load(path)               # transplant
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class PersonaModule(nn.Module):
    """Interface. Subclasses implement init_state and step."""

    def init_state(self, h: torch.Tensor):
        raise NotImplementedError

    def step(self, idx: int, h: torch.Tensor, state, ctx: "PersonaContext"):
        """Called after base layer `idx`. Returns (new_h, new_state, record_or_None).
        Must apply ctx.scale to whatever it writes into h."""
        raise NotImplementedError

    def uses_layer(self, idx: int) -> bool:
        return True


@dataclass
class PersonaContext:
    enabled: bool = True
    scale: float = 1.0
    record: bool = False
    checkpointing: bool = False
    state: object = None
    role_ids: torch.Tensor | None = None  # (B, T) 0=other 1=user 2=assistant, for role-routed modules
    records: dict = field(default_factory=dict)
    cache_position: torch.Tensor | None = None  # absolute positions of this forward's tokens
    cache: dict = field(default_factory=dict)  # module-owned state that persists across decoding steps
    pad_mask: torch.Tensor | None = None  # (B, T_prompt) 1=real token, for left-padded batched generation


def _split(out):
    return (out[0], out[1:]) if isinstance(out, tuple) else (out, None)


def _join(h, rest):
    return h if rest is None else (h,) + tuple(rest)


class PersonaLayer(nn.Module):
    def __init__(self, base: nn.Module, idx: int, persona: PersonaModule, ctx: PersonaContext):
        super().__init__()
        self.base = base
        self.idx = idx
        self.persona = [persona]  # list: keep out of this module's parameter tree
        self.ctx = ctx

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("base"), name)

    def forward(self, hidden_states, *args, **kwargs):
        ctx, persona = self.ctx, self.persona[0]
        if not ctx.enabled or not persona.uses_layer(self.idx):
            return self.base(hidden_states, *args, **kwargs)

        if ctx.state is None:
            ctx.cache_position = kwargs.get("cache_position")
            if ctx.cache_position is None or int(ctx.cache_position[0]) == 0:
                ctx.cache.clear()
            ctx.state = persona.init_state(hidden_states)

        rest_box = []

        def fn(h, state):
            h, rest = _split(self.base(h, *args, **kwargs))
            rest_box[:] = [rest]
            return persona.step(self.idx, h, state, ctx)

        if ctx.checkpointing and torch.is_grad_enabled():
            # recompute would append to the KV cache twice; training forwards don't need it
            kwargs.pop("past_key_values", None)
            kwargs.pop("past_key_value", None)
            h, state, rec = checkpoint(fn, hidden_states, ctx.state, use_reentrant=False)
        else:
            h, state, rec = fn(hidden_states, ctx.state)
        ctx.state = state
        if ctx.record and rec is not None:
            if not isinstance(rec, dict):
                rec = {"state": rec}
            for k, v in rec.items():
                ctx.records.setdefault(self.idx, {}).setdefault(k, []).append(v.detach().float().cpu())
        return _join(h, rest_box[0])


class PersonaHandle:
    """User-facing switches. Attribute writes go to the shared context."""

    def __init__(self, model, persona, ctx):
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "persona", persona)
        object.__setattr__(self, "ctx", ctx)

    def __getattr__(self, name):
        return getattr(self.ctx, name)

    def __setattr__(self, name, value):
        setattr(self.ctx, name, value)

    def clear_records(self):
        self.ctx.records.clear()

    def save(self, path):
        torch.save(self.persona.state_dict(), path)

    def load(self, path_or_sd):
        sd = torch.load(path_or_sd, map_location="cpu") if isinstance(path_or_sd, str) else path_or_sd
        self.persona.load_state_dict(sd)


def _layers(model):
    return model.model.layers


def attach(model, persona: PersonaModule, checkpointing: bool = False) -> PersonaHandle:
    """Freeze `model`, wrap its decoder layers, register `persona` as `model.persona`."""
    model.requires_grad_(False)
    ctx = PersonaContext(checkpointing=checkpointing)
    layers = _layers(model)
    for i in range(len(layers)):
        layers[i] = PersonaLayer(layers[i], i, persona, ctx)
    model.persona = persona
    persona.to(next(model.parameters()).device)

    def reset(*_args, **_kwargs):
        ctx.state = None

    model.model.register_forward_pre_hook(reset)
    model.persona_handle = PersonaHandle(model, persona, ctx)
    return model.persona_handle


def n_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
