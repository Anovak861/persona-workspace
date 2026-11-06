"""How well does a role space capture the base model's role play? (CPU; reads role_space_v2.py outputs)

python scripts/analyze_role_space.py --spaces <a.pt> [<b.pt> ...] --out <json> [--basis_out <dir>]

Per space file, per layer, for answer-mean ('mean') and last-prompt-token ('last') role vectors:
1. Spectrum: PCA of the centred role vectors; variance explained by top-k; k for 70% / 90%.
2. Split-half reliability: role vectors from even vs odd questions (disjoint rollouts); mean cosine of the centred
   halves, and the Spearman-Brown corrected reliability of the full-data vectors.
3. Held-out-role completeness (5-fold over roles): PCA on the training roles; for each held-out role, the share of
   its centred vector captured by the top-k training PCs. Baselines: a random k-dim subspace (k/d) and the
   within-role sampling noise ((v_A - v_B) / 2, held-out roles), which no role space should capture beyond chance.
4. Assistant axis (Lu et al.): default minus the mean of the other roles; cosine with PC1; roles at each end.
5. v1 vs v2: share of the v1 40-role PC8 / PC32 inside the v2 top-k space, and vice versa.
Across files with the same model (e.g. plain vs chat format): subspace overlap ||U_a^T U_b||_F^2 / k and
cross-format held-out capture (roles of b captured by the top-k PCs of a).
--basis_out saves, per file, the full-role PCA basis, eigenvalues and axis per layer for persona_locus.py.
"""
import argparse
import json
import os

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--spaces", nargs="+", required=True)
p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
p.add_argument("--out", required=True)
p.add_argument("--basis_out", default=None)
args = p.parse_args()
rng = np.random.default_rng(0)


def pca(X):
    """X (n, d) rows already centred -> (U (d, r), explained variance ratios (r,))"""
    _, S, Vt = torch.linalg.svd(X, full_matrices=False)
    ev = S ** 2
    return Vt.T, (ev / ev.sum())


def capture(V, U):
    """mean over rows of ||U^T v||^2 / ||v||^2"""
    return float(((V @ U) ** 2).sum(1).div((V ** 2).sum(1).clamp_min(1e-12)).mean())


def k_for(evr, q):
    return int((torch.cumsum(evr, 0) < q).sum().item() + 1)


def analyse(f):
    D = torch.load(f, weights_only=False)
    names, layers = D["roles"], D["layers"]
    is_def = np.array([r == "default" for r in names])
    out = {"file": f, "n_roles": len(names), "n_kept": D["n_kept"], "format": D["args"]["format"],
           "base": D["args"]["base"], "kept_frac": float(np.mean([x["kept"] for x in D["rollouts"]]))}
    d = D["vectors"]["mean"].shape[-1]
    basis = {}
    for pos in ("mean", "last"):
        for l in layers:
            X = D["vectors"][pos][:, l].float()
            XA, XB = D["vectors"][f"{pos}_A"][:, l].float(), D["vectors"][f"{pos}_B"][:, l].float()
            mu = X.mean(0)
            U, evr = pca(X - mu)
            r = {f"evr_top{k}": float(evr[:k].sum()) for k in args.ks if k <= len(evr)}
            r["k70"], r["k90"] = k_for(evr, .7), k_for(evr, .9)
            # split-half reliability
            ca, cb = XA - XA.mean(0), XB - XB.mean(0)
            rel = float(torch.nn.functional.cosine_similarity(ca, cb, dim=1).mean())
            r["split_half_cos"] = rel
            r["reliability_sb"] = 2 * rel / (1 + rel) if rel > -1 else float("nan")
            # held-out-role completeness
            perm = rng.permutation(len(names))
            folds = np.array_split(perm, 5)
            cap = {k: [] for k in args.ks}
            noise = {k: [] for k in args.ks}
            for te in folds:
                tr = np.setdiff1d(perm, te)
                mtr = X[tr].mean(0)
                Utr, _ = pca(X[tr] - mtr)
                Vte = X[te] - mtr
                Nte = (XA[te] - XB[te]) / 2
                for k in args.ks:
                    if k <= Utr.shape[1]:
                        cap[k].append(capture(Vte, Utr[:, :k]))
                        noise[k].append(capture(Nte, Utr[:, :k]))
            for k in args.ks:
                if cap[k]:
                    r[f"heldout_capture_k{k}"] = float(np.mean(cap[k]))
                    r[f"heldout_noise_capture_k{k}"] = float(np.mean(noise[k]))
                    r[f"random_k{k}"] = k / d
            # assistant axis
            if is_def.any():
                ax = X[is_def].mean(0) - X[~is_def].mean(0)
                ax = ax / ax.norm()
                r["axis_cos_pc1"] = float(abs(ax @ U[:, 0]))
                proj = ((X[~is_def] - mu) @ ax).numpy()
                o = np.argsort(proj)
                others = [n for n in names if n != "default"]
                r["axis_near_assistant"] = [others[i] for i in o[::-1][:8]]
                r["axis_far"] = [others[i] for i in o[:8]]
            else:
                ax = None
            # v1 vs v2
            for fmt, v1 in D.get("v1", {}).items():
                P1 = v1["pcs"][l].float()
                for k1 in (8, 32):
                    for k2 in (8, 32, r["k90"]):
                        r[f"v1{fmt}_pc{k1}_in_v2_top{k2}"] = float(((P1[:, :k1].T @ U[:, :k2]) ** 2).sum() / k1)
                r[f"v1{fmt}_axis_cos_v2_axis"] = float(abs(v1["axis"][l].float() @ ax)) if ax is not None else None
            out[f"{pos}/L{l}"] = r
            if pos == "mean":
                basis[l] = {"U": U, "evr": evr, "axis": ax, "mu": mu}
    return out, basis, D


report, bases, data = {"args": vars(args)}, {}, {}
for f in args.spaces:
    tag = os.path.basename(f)[:-3]
    report[tag], bases[tag], data[tag] = analyse(f)
    L = data[tag]["layers"]
    for l in (L[len(L) // 4], L[len(L) // 2], L[3 * len(L) // 4], L[-1]):
        r = report[tag][f"mean/L{l}"]
        print(tag, f"L{l}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                             if k.startswith(("evr_top8", "k70", "k90", "reliab", "heldout_capture_k8", "heldout_capture_k32",
                                              "heldout_noise_capture_k32", "random_k32", "axis_cos", "v1chat_pc32_in_v2_top32",
                                              "v1plain_pc32_in_v2_top32"))}, flush=True)

# cross-file (same base model): subspace overlap and cross-format capture
tags = list(bases)
for i, a in enumerate(tags):
    for b in tags[i + 1:]:
        if data[a]["args"]["base"] != data[b]["args"]["base"]:
            continue
        common = [r for r in data[a]["roles"] if r in data[b]["roles"]]
        ia = [data[a]["roles"].index(r) for r in common]
        ib = [data[b]["roles"].index(r) for r in common]
        cr = {}
        for l in data[a]["layers"]:
            Ua, Ub = bases[a][l]["U"], bases[b][l]["U"]
            Xb = data[b]["vectors"]["mean"][ib, l].float()
            Xa = data[a]["vectors"]["mean"][ia, l].float()
            for k in (8, 32):
                cr[f"L{l}/overlap_top{k}"] = float(((Ua[:, :k].T @ Ub[:, :k]) ** 2).sum() / k)
                cr[f"L{l}/{b}_captured_by_{a}_top{k}"] = capture(Xb - Xb.mean(0), Ua[:, :k])
                cr[f"L{l}/{a}_captured_by_{b}_top{k}"] = capture(Xa - Xa.mean(0), Ub[:, :k])
            va = Xa - Xa.mean(0)
            vb = Xb - Xb.mean(0)
            cr[f"L{l}/role_vector_corr"] = float(torch.nn.functional.cosine_similarity(va, vb, dim=1).mean())
        report[f"cross/{a}__{b}"] = cr
        mid = data[a]["mid"]
        print("cross", a, b, {k: round(v, 3) for k, v in cr.items() if k.startswith(f"L{mid}/")}, flush=True)

json.dump(report, open(args.out, "w"), indent=1)
if args.basis_out:
    os.makedirs(args.basis_out, exist_ok=True)
    for tag, B in bases.items():
        torch.save({"layers": list(B), "U": {l: b["U"] for l, b in B.items()}, "evr": {l: b["evr"] for l, b in B.items()},
                    "axis": {l: b["axis"] for l, b in B.items()}, "roles": data[tag]["roles"]},
                   f"{args.basis_out}/{tag}-basis.pt")
print("wrote", args.out)
