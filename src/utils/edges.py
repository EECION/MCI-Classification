from pathlib import Path

import pandas as pd

AAL_CSV = Path(__file__).resolve().parents[2] / "resources" / "aal90.csv"
N_REGIONS = 90


def load_aal_labels():
    """Return (abbr_list, full_name_list), each of length 90 in AAL order."""
    df = pd.read_csv(AAL_CSV)
    return list(df["abbr"]), list(df["full_name"])


def edge_list(n=N_REGIONS):
    """Upper-triangle edges as (i, j) tuples, i < j. 4005 edges for n=90."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def edge_keys(n=N_REGIONS):
    return [f"{i}-{j}" for i, j in edge_list(n)]


def edge_names(n=N_REGIONS):
    """Connection labels like 'REC.L-PHG.R' (paper / Table S1 nomenclature)."""
    abbr, _ = load_aal_labels()
    return [f"{abbr[i]}-{abbr[j]}" for i, j in edge_list(n)]


def flatten_upper(matrix, n=N_REGIONS):
    """(..., n, n) -> (..., n*(n-1)/2) upper-triangle values in edge_list order."""
    import numpy as np
    iu = np.triu_indices(n, k=1)
    return np.asarray(matrix)[..., iu[0], iu[1]]
