"""100 repeated random hold-out experiments (paper Section 2.4).

Pass a baseline and its enhanced config together to obtain paired splits:
    python scripts/run_holdout.py --configs configs/transformer.yaml configs/transformer_cbam.yaml
Shard across two GPUs:
    (GPU0) ... --shard 0/2      (GPU1, CUDA_VISIBLE_DEVICES=1) ... --shard 1/2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_device_arg, prepare_config, resolve_cfg_device  # noqa: E402
from src.data.dataset import load_preprocessed_data  # noqa: E402
from src.training.holdout import HoldoutRunner  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def parse_shard(text):
    if not text:
        return None
    index, count = text.split("/")
    index, count = int(index), int(count)
    assert 0 <= index < count, "--shard must be i/n with 0 <= i < n"
    return index, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True,
                        help="one or more variant configs; identical splits are shared")
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--shard", default=None, help="e.g. 0/2 runs seeds where seed %% n == 0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fast", action="store_true")
    add_device_arg(parser)
    args = parser.parse_args()

    configs = [prepare_config(p, fast=args.fast) for p in args.configs]
    roots = {str(c["data"]["root"]) for c in configs}
    assert len(roots) == 1, "all configs must use the same data root"
    device = resolve_cfg_device(configs[0], args.device)

    first = configs[0]
    exp0 = first["experiment"]["name"]
    logger = get_logger("holdout", first["experiment"]["output_dir"] / exp0 / "logs" / "holdout.log")
    logger.info(f"variants={[c['experiment']['name'] for c in configs]} device={device} "
                f"shard={args.shard} overwrite={args.overwrite}")

    subjects, X = load_preprocessed_data(first["data"], logger)
    y = subjects["y"].to_numpy()

    n_runs = args.n_runs or first["holdout"].get("n_runs", 100)
    seeds = list(range(n_runs))
    shard = parse_shard(args.shard)
    if shard:
        index, count = shard
        seeds = [s for s in seeds if s % count == index]
    logger.info(f"{len(seeds)} seeds to run: {seeds[0]}..{seeds[-1]}")

    runner = HoldoutRunner(configs, device, logger, overwrite=args.overwrite)
    runner.run(X, y, seeds)
    logger.info("hold-out done")


if __name__ == "__main__":
    main()
