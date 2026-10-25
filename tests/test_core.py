import copy

import pytest
import torch
from transformers import Olmo2Config, Olmo2ForCausalLM, OlmoeConfig, OlmoeForCausalLM

from persona_ws import SideCar, SideCarConfig, attach, n_trainable


def tiny(kind):
    torch.manual_seed(0)
    common = dict(vocab_size=128, hidden_size=64, intermediate_size=64, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    if kind == "olmo2":
        return Olmo2ForCausalLM(Olmo2Config(**common)).eval()
    return OlmoeForCausalLM(OlmoeConfig(**common, num_experts=4, num_experts_per_tok=2)).eval()


def sidecar_for(model, **kw):
    c = model.config
    return SideCar(SideCarConfig(d_model=c.hidden_size, n_layers=c.num_hidden_layers, d_side=16, **kw))


def perturb(persona):
    with torch.no_grad():
        for b in persona.blocks.values():
            b.write.weight.normal_(0, 0.5)


@pytest.fixture(params=["olmo2", "olmoe"])
def kind(request):
    return request.param


x = torch.randint(0, 128, (2, 10))


def test_zero_init_matches_base(kind):
    base = tiny(kind)
    ref = base(x).logits
    model = copy.deepcopy(base)
    attach(model, sidecar_for(model))
    torch.testing.assert_close(model(x).logits, ref)
    assert n_trainable(model) == n_trainable(model.persona) > 0


def test_toggle_and_scale(kind):
    base = tiny(kind)
    ref = base(x).logits
    model = copy.deepcopy(base)
    h = attach(model, sidecar_for(model))
    perturb(model.persona)
    on = model(x).logits
    assert not torch.allclose(on, ref)
    h.enabled = False
    torch.testing.assert_close(model(x).logits, ref)
    h.enabled, h.scale = True, 0.0
    torch.testing.assert_close(model(x).logits, ref)


def test_layer_subset(kind):
    model = tiny(kind)
    attach(model, sidecar_for(model, layers=(1,)))
    assert list(model.persona.blocks) == ["1"]
    model(x)


def test_checkpointing_same_grads(kind):
    grads = []
    for ckpt in (False, True):
        model = tiny(kind)
        torch.manual_seed(1)
        attach(model, sidecar_for(model), checkpointing=ckpt)
        perturb(model.persona)
        model(x, labels=x).loss.backward()
        grads.append([p.grad.clone() for p in model.persona.parameters()])
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b)


def test_records(kind):
    model = tiny(kind)
    h = attach(model, sidecar_for(model))
    h.record = True
    model(x)
    assert set(h.records) == {0, 1, 2}
    assert h.records[0][0].shape == (2, 10, 16)


def test_swap(kind, tmp_path):
    model = tiny(kind)
    h = attach(model, sidecar_for(model))
    perturb(model.persona)
    a = model(x).logits
    h.save(tmp_path / "a.pt")
    perturb(model.persona)
    assert not torch.allclose(model(x).logits, a)
    h.load(str(tmp_path / "a.pt"))
    torch.testing.assert_close(model(x).logits, a)


def test_generate_with_cache(kind):
    base = tiny(kind)
    ref = base.generate(x[:1], max_new_tokens=5, do_sample=False)
    model = copy.deepcopy(base)
    attach(model, sidecar_for(model))
    assert torch.equal(model.generate(x[:1], max_new_tokens=5, do_sample=False), ref)
    perturb(model.persona)
    # per-position side-car: cached generation must equal full recompute
    out = model.generate(x[:1], max_new_tokens=5, do_sample=False)
    full = x[:1]
    for _ in range(5):
        full = torch.cat([full, model(full).logits[:, -1:].argmax(-1)], 1)
    assert torch.equal(out, full)
