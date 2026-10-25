import copy

import pytest
import torch

from persona_ws import Workspace, WorkspaceConfig, attach
from test_core import tiny

x = torch.randint(0, 128, (2, 10))


def ws_for(model, hard=False, **kw):
    c = model.config
    cfg = WorkspaceConfig(d_model=c.hidden_size, n_layers=c.num_hidden_layers, n_slots=4, d_ws=16, n_heads=2,
                          read_layers=(0, 1), write_layers=(1, 2), hard=hard, hard_rank=8, **kw)
    return Workspace(cfg)


def perturb(ws):
    torch.manual_seed(3)
    with torch.no_grad():
        for w in ws.writes.values():
            w.out.weight.normal_(0, 0.5)


@pytest.fixture(params=["olmo2", "olmoe"])
def kind(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["soft", "hard"])
def hard(request):
    return request.param


def test_soft_zero_init_matches_base(kind):
    base = tiny(kind)
    ref = base(x).logits
    model = copy.deepcopy(base)
    attach(model, ws_for(model))
    torch.testing.assert_close(model(x).logits, ref)


def test_hard_differs_at_init_and_disable_restores(kind):
    base = tiny(kind)
    ref = base(x).logits
    model = copy.deepcopy(base)
    h = attach(model, ws_for(model, hard=True))
    assert not torch.allclose(model(x).logits, ref)
    h.enabled = False
    torch.testing.assert_close(model(x).logits, ref)


def test_hard_write_ignores_base_content_in_P(kind):
    model = tiny(kind)
    ws = ws_for(model, hard=True)
    perturb(ws)
    w = ws.writes["1"]
    P = w.P
    assert torch.allclose(P.T @ P, torch.eye(P.shape[1]), atol=1e-5)
    hh = torch.randn(2, 10, P.shape[0])
    z = torch.randn(2, 10, 4, 16)
    out1, _ = w(hh, z, 1.0)
    out2, _ = w(hh + torch.randn(2, 10, P.shape[1]) @ P.T, z, 1.0)
    torch.testing.assert_close(out1, out2)


def test_causal(kind, hard):
    model = tiny(kind)
    attach(model, ws_for(model, hard=hard))
    perturb(model.persona)
    a = model(x).logits
    x2 = x.clone()
    x2[:, 7] = (x2[:, 7] + 1) % 128
    b = model(x2).logits
    torch.testing.assert_close(a[:, :7], b[:, :7])
    assert not torch.allclose(a[:, 7:], b[:, 7:])


def test_explicit_path_matches_sdpa_and_records(kind, hard):
    model = tiny(kind)
    h = attach(model, ws_for(model, hard=hard))
    perturb(model.persona)
    a = model(x).logits
    h.record = True
    b = model(x).logits
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
    assert h.records[0]["access"][0].shape == (2, 10, 4, 10)
    acc = h.records[1]["access"][0]
    assert torch.all(acc[:, 3, :, 4:] == 0)  # no future sources
    torch.testing.assert_close(acc.sum(-1), torch.ones(2, 10, 4))
    assert h.records[2]["broadcast"][0].shape == (2, 10, 4)


def test_checkpointing_same_grads(kind, hard):
    grads = []
    for ckpt in (False, True):
        model = tiny(kind)
        attach(model, ws_for(model, hard=hard), checkpointing=ckpt)
        perturb(model.persona)
        model(x, labels=x).loss.backward()
        grads.append([p.grad.clone() for p in model.persona.parameters() if p.grad is not None])
    assert len(grads[0]) == len(grads[1]) > 0
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b)


def test_generate_with_cache_matches_full(kind, hard):
    model = tiny(kind)
    attach(model, ws_for(model, hard=hard))
    perturb(model.persona)
    out = model.generate(x[:1], max_new_tokens=6, do_sample=False)
    full = x[:1]
    with torch.no_grad():
        for _ in range(6):
            full = torch.cat([full, model(full).logits[:, -1:].argmax(-1)], 1)
    assert torch.equal(out, full)


def test_swap(kind, tmp_path):
    model = tiny(kind)
    h = attach(model, ws_for(model))
    perturb(model.persona)
    a = model(x).logits
    h.save(str(tmp_path / "ws.pt"))
    model2 = tiny(kind)
    h2 = attach(model2, ws_for(model2))
    h2.load(str(tmp_path / "ws.pt"))
    torch.testing.assert_close(model2(x).logits, a)
