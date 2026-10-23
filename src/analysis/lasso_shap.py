"""Lasso clinical-score prediction + SHAP interpretation
(Supplementary Methods S7, Eqs. S28-S32).
"""
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


def lasso_shap(X, y, feature_names, top_n=12, cv=10, seed=0):
    """Fit LassoCV (alpha by CV), explain with linear SHAP.

    Returns a dict with the fitted model, per-participant SHAP values,
    mean|SHAP| importance ranking and top-`top_n` relative contributions
    (percent of the total mean|SHAP|, remainder pooled as C_others, Eq. S32).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    model = LassoCV(cv=cv, random_state=seed).fit(Xs, y)
    explainer = shap.LinearExplainer(model, Xs)
    shap_values = explainer.shap_values(Xs)

    importance = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-importance)
    total = importance.sum()
    contrib = importance / total * 100.0 if total > 0 else np.zeros_like(importance)

    top_idx = order[:top_n]
    rows = [{"feature": feature_names[j], "mean_abs_shap": float(importance[j]),
             "contribution_pct": float(contrib[j])} for j in top_idx]
    rows.append({"feature": "C_others", "mean_abs_shap": float(importance[order[top_n:]].sum()),
                 "contribution_pct": float(contrib[order[top_n:]].sum())})

    return {
        "model": model,
        "scaler": scaler,
        "alpha": float(model.alpha_),
        "r2": float(model.score(Xs, y)),
        "shap_values": shap_values,
        "feature_names": list(feature_names),
        "importance": importance,
        "ranking": [feature_names[j] for j in order],
        "top_contributions": pd.DataFrame(rows),
    }
