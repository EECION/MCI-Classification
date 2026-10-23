"""Pearson correlation between selected connections (Supplementary Methods S6.2, Eq. S27)."""
import numpy as np
import pandas as pd


def connection_correlation(X_md, feature_names):
    """X_md: (N, E) participant-level MD values; returns (corr DataFrame, pval DataFrame)."""
    from scipy.stats import pearsonr

    n_feat = X_md.shape[1]
    corr = np.eye(n_feat)
    pval = np.zeros((n_feat, n_feat))
    for a in range(n_feat):
        for b in range(a + 1, n_feat):
            r, p = pearsonr(X_md[:, a], X_md[:, b])
            corr[a, b] = corr[b, a] = r
            pval[a, b] = pval[b, a] = p
    names = list(feature_names)
    return (pd.DataFrame(corr, index=names, columns=names),
            pd.DataFrame(pval, index=names, columns=names))
