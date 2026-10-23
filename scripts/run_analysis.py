"""Clinical association and explainability pipeline (paper Section 2.5, S6-S7).

Inputs: top-k connection tables from scripts/summarize.py + MD matrices + clinical
scores. Outputs: Mantel tests with FDR, Pearson correlation matrices, per
(stage x score) Lasso + SHAP importance / beeswarm data / top-12 contributions.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import prepare_config  # noqa: E402
from src.analysis.correlation import connection_correlation  # noqa: E402
from src.analysis.lasso_shap import lasso_shap  # noqa: E402
from src.analysis.mantel import mantel_stage  # noqa: E402
from src.data.dataset import load_md, load_subjects  # noqa: E402
from src.utils.io import PROJECT_ROOT  # noqa: E402


def edge_key_to_index(key):
    i, j = key.split("-")
    return int(i), int(j)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--summary-dir", default=str(PROJECT_ROOT / "outputs" / "summaries"))
    parser.add_argument("--fast", action="store_true",
                        help="reduce Mantel permutations for debugging")
    args = parser.parse_args()

    cfg = prepare_config(args.config)
    acfg = cfg["analysis"]
    top_k = acfg.get("top_k", 20)
    scores = acfg.get("clinical_scores", ["mmse", "faq", "gds"])
    stages = acfg.get("stages", ["EMCI", "LMCI"])
    n_perm = 100 if args.fast else acfg.get("mantel_permutations", 1000)
    shap_top = acfg.get("shap_top_n", 12)
    summary_dir = Path(args.summary_dir)
    out_dir = summary_dir / "shap"
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = load_subjects(Path(cfg["data"]["root"]) / cfg["data"]["subjects_csv"])

    mantel_rows = []
    for stage in stages:
        top_path = summary_dir / f"top{top_k}_{stage}.csv"
        if not top_path.exists():
            print(f"missing {top_path}; run scripts/summarize.py first")
            continue
        top = pd.read_csv(top_path)
        edges = [edge_key_to_index(k) for k in top["edge"]]

        stage_subjects = subjects[subjects["group"] == stage]
        md = load_md({"root": str(cfg["data"]["root"]), "md_dir": cfg["data"].get("md_dir", "md"),
                      "stacked": cfg["data"].get("stacked", False)},
                     list(stage_subjects["subject_id"]))
        X_md = np.stack([md[:, i, j] for i, j in edges], axis=1)
        names = list(top["connection"])
        print(f"[{stage}] {len(stage_subjects)} subjects, {len(edges)} MD features")

        results = mantel_stage(X_md, stage_subjects, scores,
                               n_perm=n_perm, seed=cfg["experiment"].get("seed", 42))
        for res in results:
            mantel_rows.append({"stage": stage, **res,
                                "score_r": res["r"], "p": res["p"],
                                "q_fdr": res["q"], "significant_q<0.05": res["significant"]})
        corr, pval = connection_correlation(X_md, names)
        corr.round(4).to_csv(summary_dir / f"corr_matrix_{stage}.csv")
        pval.round(4).to_csv(summary_dir / f"corr_pvalues_{stage}.csv")

        for score in scores:
            y = stage_subjects[score].to_numpy(dtype=float)
            result = lasso_shap(X_md, y, names, top_n=shap_top,
                                cv=acfg.get("lasso", {}).get("cv", 10),
                                seed=cfg["experiment"].get("seed", 42))
            sub = out_dir / f"{stage}_{score}"
            sub.mkdir(parents=True, exist_ok=True)
            result["top_contributions"].round(4).to_csv(sub / "top_contributions.csv", index=False)
            np.save(sub / "shap_values.npy", result["shap_values"])
            pd.DataFrame({"connection": names,
                          "mean_abs_shap": result["importance"].round(6)}
                         ).to_csv(sub / "importance.csv", index=False)
            print(f"[{stage} {score}] alpha={result['alpha']:.4f} R2={result['r2']:.3f} | "
                  f"top: {result['top_contributions'].iloc[0]['feature']} "
                  f"({result['top_contributions'].iloc[0]['contribution_pct']:.1f}%)")

    if mantel_rows:
        pd.DataFrame([{k: v for k, v in r.items() if k != "score_r"} for r in mantel_rows]
                     ).to_csv(summary_dir / "mantel_results.csv", index=False)
        print(f"mantel results -> {summary_dir / 'mantel_results.csv'}")


if __name__ == "__main__":
    main()
