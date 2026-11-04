"""Jacobian lens (J-lens) and J-space, after "Verbalizable Representations Form a Global Workspace in Language
Models" (Anthropic 2026).

J_l = E_{context, t, t' >= t} [ d h_{L,t'} / d h_{l,t} ], with h_l the residual after decoder layer l and h_L the
final residual (before the final norm), averaged over generic text. The J-lens reads an activation as
W_U diag(g) J_l h (g = final norm weight; the norm's scale doesn't change the token ranking). The J-lens vectors at
layer l are the rows of W_U diag(g) J_l, one per vocabulary token. J-space = sparse non-negative combinations of at most
k of them; an activation's J-space part is found by non-negative orthogonal matching pursuit.

Fitting: one backward pass gives one row of J_l at every layer for every sequence in the batch. Sequence i gets
its own output coordinate k_i and the loss is sum_{t'} h_{L,t'}[k_i], so the gradient at h_{l,t} is
sum_{t' >= t} J_{t,t'}[k_i, :]. Summing over t and dividing by the number of (t, t') pairs gives the average. BOS
(t = 0, huge norms in OLMo-2) is dropped as source and target.
"""
import random

import torch
from datasets import load_dataset


def wiki_contexts(tok, n, T=128, seed=0):
    d = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train[:60000]")
    text = [t for t in d["text"] if len(t) > 400]
    random.Random(seed).shuffle(text)
    out = []
    for t in text:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) >= T - 1:
            out.append([tok.bos_token_id] + ids[: T - 1])
        if len(out) == n:
            break
    return torch.tensor(out)


def fit(model, tok, n_per_row=32, bs=64, T=128, n_ctx=2048, seed=0):
    """{layer: J_l (d, d) fp32 on cpu}, for every decoder layer. model's persona module (if any) is as set."""
    layers = model.model.layers
    L = len(layers)
    d = model.config.hidden_size
    ctx = wiki_contexts(tok, n_ctx, T, seed).to(model.device)
    J = {l: torch.zeros(d, d, device=model.device) for l in range(L)}
    cnt = torch.zeros(d, device=model.device)
    rows = [k for _ in range(n_per_row) for k in range(d)]
    random.Random(seed).shuffle(rows)
    g = torch.Generator().manual_seed(seed)
    model.eval()
    model.requires_grad_(False)
    pairs = (T - 1) * T / 2  # (t, t') pairs with 1 <= t <= t' <= T-1
    for s in range(0, len(rows), bs):
        ks = torch.tensor(rows[s:s + bs], device=model.device)
        ids = ctx[torch.randint(len(ctx), (len(ks),), generator=g)]
        outs = {}
        hk = [model.model.embed_tokens.register_forward_hook(lambda m, a, o: o.requires_grad_(True))]
        hk += [layers[l].register_forward_hook(
            lambda m, a, o, l=l: outs.__setitem__(l, o[0] if isinstance(o, tuple) else o)) for l in range(L)]
        with torch.enable_grad():
            model(input_ids=ids)
            for h in hk:
                h.remove()
            hL = outs[L - 1]
            loss = hL[torch.arange(len(ks)), 1:, :].gather(-1, ks[:, None, None].expand(-1, T - 1, 1)).float().sum()
            grads = torch.autograd.grad(loss, [outs[l] for l in range(L - 1)], allow_unused=True)
        for l, gr in enumerate(grads):
            gsum = gr[:, 1:].float().sum(1)  # (B, d): sum over source positions
            J[l].index_add_(0, ks, gsum / pairs)
        cnt.index_add_(0, ks, torch.ones_like(ks, dtype=torch.float))
        del outs, grads
    for l in range(L - 1):
        J[l] = (J[l] / cnt[:, None]).cpu()
    J[L - 1] = torch.eye(d)
    return J


class Lens:
    """J-lens readout and J-space decomposition for one model, given its fitted J."""

    def __init__(self, model, J):
        self.U = model.get_output_embeddings().weight.float().detach()  # (V, d)
        self.g = model.model.norm.weight.float().detach()
        self.J = J
        self._atoms = {}

    def atoms(self, l):
        """Unit J-lens vectors at layer l, (V, d) on gpu; row v = direction that promotes token v."""
        if l not in self._atoms:
            A = (self.U * self.g) @ self.J[l].to(self.U.device)
            self._atoms = {l: A / A.norm(dim=1, keepdim=True).clamp_min(1e-8)}  # keep one layer cached
        return self._atoms[l]

    def logits(self, l, h):
        """J-lens logits of h (..., d)."""
        return (h.float().to(self.U.device) @ self.J[l].T.to(self.U.device) * self.g) @ self.U.T

    def decompose(self, l, h, k=25):
        """Non-negative OMP of h (n, d) on layer-l atoms, batched: each step adds the atom most correlated with
        the residual, refits least squares on the selected atoms and clamps negative weights to zero (an
        approximation to NNLS). Returns (idx (n, k), coef (n, k), resid_frac (n,))."""
        A = self.atoms(l)
        h = h.float().to(A.device)
        n = h.shape[0]
        idx = torch.zeros(n, k, dtype=torch.long, device=A.device)
        r, w = h.clone(), None
        for j in range(k):
            c = r @ A.T
            if j:
                c.scatter_(1, idx[:, :j], -1e9)
            idx[:, j] = c.argmax(1)
            X = A[idx[:, : j + 1]].transpose(1, 2)  # (n, d, j+1)
            w = torch.linalg.lstsq(X, h[..., None]).solution.clamp_min(0)
            r = h - (X @ w)[..., 0]
        return idx, w[..., 0], r.norm(dim=1) ** 2 / h.norm(dim=1).clamp_min(1e-8) ** 2

    def random_control(self, l, h, k=25, seed=0):
        """Residual fraction after least squares on k random unit directions (the paper's control)."""
        g = torch.Generator(device=self.U.device).manual_seed(seed)
        R = torch.randn(k, h.shape[1], device=self.U.device, generator=g)
        R = R / R.norm(dim=1, keepdim=True)
        h = h.float().to(self.U.device)
        w = torch.linalg.lstsq(R.T.expand(h.shape[0], -1, -1), h[..., None]).solution
        r = h - (R.T @ w)[..., 0]
        return r.norm(dim=1) ** 2 / h.norm(dim=1) ** 2


# concept token sets for persona readouts (single-token words, with and without a leading space)
CONCEPTS = {
    "ai_self": ["AI", "assistant", "Assistant", "language", "model", "bot", "chatbot", "virtual", "machine", "computer",
                "program", "software", "algorithm", "OpenAI", "artificial"],
    "pretend": ["pretend", "role", "character", "fiction", "fictional", "story", "acting", "imagine", "play", "persona",
                "roleplay", "costume", "scene", "script", "stage"],
    "human": ["human", "person", "born", "family", "childhood", "parents", "wife", "husband", "job", "hometown"],
    "refusal": ["sorry", "cannot", "unable", "refuse", "harmful", "illegal", "unethical", "dangerous", "inappropriate",
                "apologize", "against", "safety", "decline", "Sorry"],
}


def concept_ids(tok):
    out = {}
    for c, words in CONCEPTS.items():
        ids = set()
        for w in words:
            for s in (w, " " + w):
                t = tok(s, add_special_tokens=False)["input_ids"]
                if len(t) == 1:
                    ids.add(t[0])
        out[c] = torch.tensor(sorted(ids))
    return out


def concept_mass(lens, l, H, cids, k=25):
    """Per concept: mean fraction of J-space coefficient mass on its tokens, and fraction of rows where present."""
    idx, coef, res = lens.decompose(l, H, k)
    tot = coef.sum(1).clamp_min(1e-8)
    out = {"resid_frac": float(res.mean())}
    for c, ids in cids.items():
        on = torch.isin(idx, ids.to(idx.device))
        out[f"{c}/mass"] = float(((coef * on).sum(1) / tot).mean())
        out[f"{c}/present"] = float(on.any(1).float().mean())
    return out, idx, coef
