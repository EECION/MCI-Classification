"""Aggregate hold-out runs into Table 2, distribution/ROC data, top-20 connections."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import prepare_config  # noqa: E402
from src.analysis.tables import summarize_all  # noqa: E402
from src.utils.io import PROJECT_ROOT  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "summaries"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"),
                        help="default.yaml carries the analysis section (top_k etc.)")
    parser.add_argument("--importance-variant", default=None,
                        help="override analysis.importance_variant")
    args = parser.parse_args()

    cfg = prepare_config(args.config)
    acfg = cfg["analysis"]
    variant = args.importance_variant or acfg.get("importance_variant", "transformer_cbam")
    top_k = acfg.get("top_k", 20)

    table2 = summarize_all(args.outputs, args.out, variant, top_k)
    if table2 is None:
        print("no complete pairs of runs found; run scripts/run_holdout.py first")
        return
    print(f"summaries written to {args.out}")
    print(table2.to_string(index=False))


if __name__ == "__main__":
    main()
