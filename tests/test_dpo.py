"""The DPO reference: functional_call with a frozen copy of the module params must reproduce the
initial module after the live params have moved, and must not touch the live params."""
import torch
from torch.func import functional_call

from persona_ws import attach

from test_core import kind, perturb, sidecar_for, tiny, x  # noqa: F401 (kind is a fixture)


def test_functional_call_reference(kind):
    model = tiny(kind)
    attach(model, sidecar_for(model), checkpointing=True)
    perturb(model.persona)
    named = {n: q for n, q in model.named_parameters() if q.requires_grad}
    assert named and all(n.startswith("persona.") for n in named)
    ref_params = {n: q.detach().clone() for n, q in named.items()}
    ref = model(x).logits.detach()

    perturb(model.persona)  # the policy moves away from the reference
    moved = model(x).logits.detach()
    assert not torch.allclose(moved, ref)
    with torch.no_grad():
        again = functional_call(model, ref_params, (), {"input_ids": x}, strict=False).logits
    torch.testing.assert_close(again, ref)
    torch.testing.assert_close(model(x).logits.detach(), moved)  # live params untouched


def test_reference_grads_flow_only_to_policy(kind):
    model = tiny(kind)
    attach(model, sidecar_for(model), checkpointing=True)
    perturb(model.persona)
    model.train()
    named = {n: q for n, q in model.named_parameters() if q.requires_grad}
    ref_params = {n: q.detach().clone() for n, q in named.items()}
    with torch.no_grad():
        r = functional_call(model, ref_params, (), {"input_ids": x}, strict=False).logits.sum()
    pol = model(input_ids=x).logits.sum()
    (pol - r).backward()
    assert all(q.grad is not None for q in named.values())
