"""Two-stage Optuna Bayesian tuning with 10-fold CV (paper Section 2.4)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_device_arg, prepare_config, resolve_cfg_device  # noqa: E402
from src.data.dataset import load_preprocessed_data  # noqa: E402
from src.training.tuner import run_tuning  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fast", action="store_true")
    add_device_arg(parser)
    args = parser.parse_args()

    cfg = prepare_config(args.config, fast=args.fast)
    device = resolve_cfg_device(cfg, args.device)
    exp = cfg["experiment"]["name"]
    logger = get_logger("tune", cfg["experiment"]["output_dir"] / exp / "logs" / "tuning.log")
    logger.info(f"variant={cfg['model']['variant']} device={device}")

    subjects, X = load_preprocessed_data(cfg["data"], logger)
    y = subjects["y"].to_numpy()
    run_tuning(cfg, X, y, device, logger)


if __name__ == "__main__":
    main()
