"""Mantel test between connection MD features and clinical scores
(Supplementary Methods S6.1, Eqs. S24-S26).

Connection features use Euclidean distances; the scalar clinical score uses
absolute differences. The statistic is the Pearson correlation between the
upper-triangular elements; significance uses `n_perm` permutations, and BH-FDR
is applied across the three scores within each diagnostic stage.
"""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr


def mantel_statistic(X, y):
    d_conn = pdist(np.asarray(X, dtype=float), metric="euclidean")
    d_clin = pdist(np.asarray(y, dtype=float).reshape(-1, 1), metric="cityblock")
    return float(pearsonr(d_conn, d_clin)[0])


def mantel_test(X, y, n_perm=1000, seed=0):
    """Return (r_obs, one-sided permutation p)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    r_obs = mantel_statistic(X, y)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(int(n_perm)):
        r_perm = mantel_statistic(X, rng.permutation(y))
        if r_perm >= r_obs:
            count += 1
    return r_obs, (count + 1) / (n_perm + 1)


def benjamini_hochberg(pvals):
    """Return BH-adjusted q-values (step-up FDR)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.clip(ranked, 0, 1)
    return q


def mantel_stage(X_md, clinical_df, scores, n_perm=1000, seed=0):
    """Run Mantel tests for all clinical scores within one stage + FDR.

    X_md: (N, E) MD features of the selected connections.
    clinical_df: DataFrame indexed/aligned with X_md rows.
    Returns list of dicts {score, r, p, q, significant}.
    """
    results = []
    for score in scores:
        y = clinical_df[score].to_numpy(dtype=float)
        r, p = mantel_test(X_md, y, n_perm=n_perm, seed=seed)
        results.append({"score": score, "r": r, "p": p})
    qs = benjamini_hochberg([res["p"] for res in results])
    for res, q in zip(results, qs):
        res["q"] = float(q)
        res["significant"] = bool(q < 0.05)
    return results
