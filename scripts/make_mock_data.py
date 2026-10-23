"""Generate deterministic mock data (NC=54, EMCI=42, LMCI=54) under data/."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mock_data import generate  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    out = Path(args.out)
    out = out if out.is_absolute() else Path(__file__).resolve().parents[1] / out
    generate(out, seed=args.seed)
    print(f"mock data written to {out}")
