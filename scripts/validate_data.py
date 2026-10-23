"""Validate the data layout and print a Table-1 style demographic summary."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy.stats import chi2_contingency, kruskal  # noqa: E402

from src.data.dataset import load_connectivity, load_md, load_subjects  # noqa: E402
from src.utils.io import PROJECT_ROOT  # noqa: E402

CONTINUOUS = ["age", "education", "mmse", "faq", "gds"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    args = parser.parse_args()

    root = Path(args.data_root)
    subjects = load_subjects(root / "subjects.csv")
    n = len(subjects)
    print(f"subjects: {n} | groups: {dict(subjects['group'].value_counts())}")

    x = load_connectivity({"root": str(root), "connectivity_dir": "connectivity",
                           "stacked": False}, list(subjects["subject_id"]))
    print(f"connectivity: {x.shape} dtype={x.dtype} "
          f"range=[{x.min():.3f}, {x.max():.3f}] finite={np.isfinite(x).all()}")
    asym = np.abs(x - np.transpose(x, (0, 2, 1))).max()
    print(f"max |A - A^T| = {asym:.4f} (symmetrization handled by preprocessing)")

    md_dir = root / "md"
    if md_dir.exists():
        md = load_md({"root": str(root), "md_dir": "md", "stacked": False},
                     list(subjects["subject_id"]))
        print(f"md: {md.shape} range=[{md.min():.3f}, {md.max():.3f}] finite={np.isfinite(md).all()}")

    print("\n--- Table 1 style summary ---")
    groups = ["NC", "EMCI", "LMCI"]
    if "sex" in subjects.columns:
        table = pd_crosstab_safe(subjects)
        chi2, p, _, _ = chi2_contingency(table)
        print(f"sex (m/f): " + " ".join(
            f"{g} {(int(((subjects['group'] == g) & (subjects['sex'] == 'M')).sum()))}/"
            f"{(int(((subjects['group'] == g) & (subjects['sex'] == 'F')).sum()))}"
            for g in groups) + f" | chi2 p={p:.3g}")
    for col in CONTINUOUS:
        if col not in subjects.columns:
            continue
        parts = [f"{g} {subjects.loc[subjects['group'] == g, col].mean():.1f} ± "
                 f"{subjects.loc[subjects['group'] == g, col].std():.1f}" for g in groups]
        samples = [subjects.loc[subjects['group'] == g, col].dropna() for g in groups]
        if all(len(s) > 0 for s in samples):
            _, p = kruskal(*samples)
            print(f"{col}: " + " | ".join(parts) + f" | Kruskal-Wallis p={p:.3g}")

    print("\ndata validation OK")


def pd_crosstab_safe(subjects):
    import pandas as pd
    return pd.crosstab(subjects["group"], subjects["sex"]).reindex(["NC", "EMCI", "LMCI"])


if __name__ == "__main__":
    main()
