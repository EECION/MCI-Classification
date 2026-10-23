"""Aggregation of hold-out run records into paper-style outputs (Table 2,
Fig 5 distribution data, Fig 6 ROC data, Fig 7/8 top-20 connections, paired
Base-vs-Enhanced statistics).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.models.registry import PAIRS
from src.utils.edges import edge_names
from src.utils.io import load_json

METRICS = ("acc", "auc", "sen", "spe")
STAGES = ("NC", "EMCI", "LMCI")
BACKBONE_NAMES = {"transformer": "Transformer", "resnet": "ResNet",
                  "fcn": "FCN", "lstm": "LSTM"}


def collect_runs(outputs_root, exp_name):
    run_dir = Path(outputs_root) / exp_name / "runs"
    if not run_dir.exists():
        return []
    return [load_json(p) for p in sorted(run_dir.glob("run_*.json"))]


def _mean_std(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return "n/a"
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    return f"{arr.mean() * 100:.1f} ± {std * 100:.1f}" if arr.mean() <= 1.0 else f"{arr.mean():.3f} ± {std:.3f}"


def build_table2(outputs_root, out_dir):
    """Table 2 layout: Stage x Backbone x {Base, Enh} x {ACC, AUC, SEN, SPE}."""
    rows = []
    for base, enh in PAIRS:
        base_runs = collect_runs(outputs_root, base)
        enh_runs = collect_runs(outputs_root, enh)
        if not base_runs or not enh_runs:
            continue
        by_seed_e = {r["seed"]: r for r in enh_runs}
        for stage in STAGES:
            row = {"Stage": stage, "Backbone": BACKBONE_NAMES.get(base, base),
                   "n_runs": f"{len(base_runs)}/{len(enh_runs)}"}
            for metric in METRICS:
                base_vals = [r["metrics"][stage][metric] for r in base_runs if stage in r["metrics"]]
                enh_vals = [r["metrics"][stage][metric] for r in enh_runs if stage in r["metrics"]]
                row[f"{metric.upper()} (Base)"] = _mean_std(base_vals)
                row[f"{metric.upper()} (Enh.)"] = _mean_std(enh_vals)
            rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "table2.csv", index=False)
    with open(out_dir / "table2.md", "w", encoding="utf-8") as f:
        f.write("# Table 2. Performance of baseline and enhanced models (mean ± std over runs)\n\n")
        f.write(df.to_markdown(index=False))
    return df


def export_distribution(outputs_root, out_dir):
    """Per-run OvR metric values -> metrics_distribution.csv (Fig 5 data)."""
    rows = []
    for run_dir in sorted(Path(outputs_root).glob("*/runs")):
        exp = run_dir.parent.name
        for p in sorted(run_dir.glob("run_*.json")):
            run = load_json(p)
            for stage in STAGES:
                for metric in METRICS:
                    rows.append({"experiment": exp, "seed": run["seed"], "stage": stage,
                                 "metric": metric.upper(),
                                 "value": run["metrics"][stage][metric]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics_distribution.csv", index=False)
    return df


def export_roc_data(outputs_root, out_dir):
    """Pooled test-set labels and softmax probabilities per experiment (Fig 6 data)."""
    payload = {}
    for run_dir in sorted(Path(outputs_root).glob("*/runs")):
        exp = run_dir.parent.name
        ys, ps, seeds = [], [], []
        for p in sorted(run_dir.glob("run_*.json")):
            run = load_json(p)
            ys.append(np.asarray(run["y_true"]))
            ps.append(np.asarray(run["y_prob"]))
            seeds.append(np.full(len(run["y_true"]), run["seed"]))
        if ys:
            payload[f"{exp}__y_true"] = np.concatenate(ys)
            payload[f"{exp}__y_prob"] = np.concatenate(ps)
            payload[f"{exp}__seed"] = np.concatenate(seeds)
    if payload:
        np.savez_compressed(out_dir / "roc_data.npz", **payload)
    return payload


def top20_connections(outputs_root, out_dir, importance_variant, top_k=20):
    """Aggregate per-run importance of one variant into stage-specific top-k
    connections with selection frequencies, plus shared/specific sets (Fig 7/8)."""
    runs = collect_runs(outputs_root, importance_variant)
    runs = [r for r in runs if "importance" in r]
    if not runs:
        return None, None
    all_names = edge_names()
    key_to_name = {f"{i}-{j}": all_names[k]
                   for k, (i, j) in enumerate((a, b) for a in range(90) for b in range(a + 1, 90))}

    top_tables, per_stage_sets = {}, {}
    for stage in ("EMCI", "LMCI"):
        matrices, edges_refs = [], None
        for run in runs:
            imp = run["importance"]
            edges_refs = imp["edges"]
            matrices.append(np.asarray(imp["scores"][stage], dtype=float))
        mean_imp = np.mean(np.stack(matrices), axis=0)
        order = np.argsort(-mean_imp)[:top_k]

        freq = []
        top_per_run = [set(np.argsort(-m)[:top_k].tolist()) for m in matrices]
        for idx in order:
            freq.append(float(np.mean([idx in tops for tops in top_per_run])))

        df = pd.DataFrame({
            "rank": np.arange(1, len(order) + 1),
            "edge": [edges_refs[i] for i in order],
            "connection": [key_to_name.get(edges_refs[i], edges_refs[i]) for i in order],
            "mean_importance": mean_imp[order],
            "selection_frequency": freq,
        })
        df.to_csv(out_dir / f"top{top_k}_{stage}.csv", index=False)
        top_tables[stage] = df
        per_stage_sets[stage] = set(df["connection"])

    if len(per_stage_sets) == 2:
        shared = per_stage_sets["EMCI"] & per_stage_sets["LMCI"]
        rows = ([{"type": "shared", "stage": "EMCI&LMCI", "connection": c} for c in sorted(shared)]
                + [{"type": "specific", "stage": "EMCI", "connection": c}
                   for c in sorted(per_stage_sets["EMCI"] - shared)]
                + [{"type": "specific", "stage": "LMCI", "connection": c}
                   for c in sorted(per_stage_sets["LMCI"] - shared)])
        pd.DataFrame(rows).to_csv(out_dir / "shared_vs_specific.csv", index=False)
    return top_tables, per_stage_sets


def paired_statistics(outputs_root, out_dir):
    """Wilcoxon signed-rank Base vs Enh on paired seeds (bonus for review)."""
    rows = []
    for base, enh in PAIRS:
        base_runs = {r["seed"]: r for r in collect_runs(outputs_root, base)}
        enh_runs = {r["seed"]: r for r in collect_runs(outputs_root, enh)}
        common = sorted(set(base_runs) & set(enh_runs))
        if len(common) < 5:
            continue
        for stage in STAGES:
            for metric in METRICS:
                b = np.array([base_runs[s]["metrics"][stage][metric] for s in common])
                e = np.array([enh_runs[s]["metrics"][stage][metric] for s in common])
                try:
                    stat, p = wilcoxon(e, b)
                except ValueError:
                    stat, p = np.nan, np.nan
                rows.append({"pair": f"{base}->{enh}", "stage": stage, "metric": metric.upper(),
                             "n": len(common), "mean_base": b.mean(), "mean_enh": e.mean(),
                             "delta": (e - b).mean(), "wilcoxon_p": p})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "paired_stats.csv", index=False)
    return df


def summarize_all(outputs_root, summary_dir, importance_variant="transformer_cbam", top_k=20):
    outputs_root = Path(outputs_root)
    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    table2 = build_table2(outputs_root, summary_dir)
    export_distribution(outputs_root, summary_dir)
    export_roc_data(outputs_root, summary_dir)
    top20_connections(outputs_root, summary_dir, importance_variant, top_k)
    paired_statistics(outputs_root, summary_dir)
    return table2
