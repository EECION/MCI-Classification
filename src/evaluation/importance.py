"""Edge-level feature importance on the hold-out test set.

The paper reports importance distributions across the 100 random hold-out
experiments without specifying an algorithm; two model-agnostic options are
provided:

- permutation (default): for each upper-triangle edge, permute its values
  across the test subjects of a given class and record the drop in the
  class-recognition rate (per_class scope -> stage-specific top-20, paper
  Section 3.5) or the drop in overall accuracy (overall scope).
- gradient: mean absolute gradient of the class logit w.r.t. the edge.
"""
import numpy as np
import torch

from src.data.dataset import CLASS_NAMES
from src.utils.edges import edge_list


@torch.no_grad()
def permutation_importance(model, X, y, edges, device):
    model.eval()
    X = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
    y = np.asarray(y)
    base_pred = model(X).argmax(dim=1).cpu().numpy()

    rows_per_class = [np.where(y == c)[0] for c in range(len(CLASS_NAMES))]
    base_rate = {name: float((base_pred[rows] == c).mean()) if len(rows) else float("nan")
                 for c, (name, rows) in enumerate(zip(CLASS_NAMES, rows_per_class))}
    empty = [name for name, rows in zip(CLASS_NAMES, rows_per_class) if len(rows) < 2]
    for name in empty:
        base_rate[name] = float("nan")

    gen = torch.Generator(device="cpu").manual_seed(0)
    drops = {name: np.zeros(len(edges)) for name in CLASS_NAMES}
    for e, (i, j) in enumerate(edges):
        permuted = X.clone()
        for rows in rows_per_class:
            if len(rows) < 2:
                continue
            order = torch.randperm(len(rows), generator=gen).to(device)
            vals = X[torch.as_tensor(rows, device=device)][:, i, j][order]
            permuted[torch.as_tensor(rows, device=device), i, j] = vals
            permuted[torch.as_tensor(rows, device=device), j, i] = vals
        pred = model(permuted).argmax(dim=1).cpu().numpy()
        for c, (name, rows) in enumerate(zip(CLASS_NAMES, rows_per_class)):
            if len(rows) < 2:
                continue
            drops[name][e] = base_rate[name] - float((pred[rows] == c).mean())
    return {name: [round(float(v), 5) for v in vals] for name, vals in drops.items()}


def gradient_importance(model, X, y, edges, device):
    model.eval()
    X_dev = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
    y = np.asarray(y)
    out = {}
    for c, name in enumerate(CLASS_NAMES):
        rows = np.where(y == c)[0]
        if len(rows) == 0:
            out[name] = [0.0] * len(edges)
            continue
        Xg = X_dev.clone().requires_grad_(True)
        logits = model(Xg)
        grad = torch.autograd.grad(
            logits[torch.as_tensor(rows, device=device), c].sum(), Xg)[0].abs()
        # symmetrized edge importance: average |dL/dx_ij| and |dL/dx_ji| over class rows
        rows_dev = torch.as_tensor(rows, device=device)
        imp = 0.5 * (grad[rows_dev].mean(0) + grad[rows_dev].mean(0).T)
        out[name] = [round(float(imp[i, j]), 5) for i, j in edges]
    return out


def compute_importance(model, X_test, y_test, imp_cfg, device):
    edges = edge_list()
    max_edges = imp_cfg.get("max_edges")
    if max_edges:
        edges = edges[:int(max_edges)]
    method = imp_cfg.get("method", "permutation")
    if method == "permutation":
        scores = permutation_importance(model, X_test, y_test, edges, device)
    elif method == "gradient":
        scores = gradient_importance(model, X_test, y_test, edges, device)
    else:
        raise ValueError(f"unknown importance method: {method}")
    return {"method": method, "edges": [f"{i}-{j}" for i, j in edges], "scores": scores}
