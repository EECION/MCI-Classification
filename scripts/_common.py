"""Shared helpers for entry scripts (path bootstrap, config preparation, --fast)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import PROJECT_ROOT, load_config, resolve_device  # noqa: E402


def prepare_config(path, fast=False):
    cfg = load_config(path)
    exp = cfg["experiment"]
    out_dir = Path(exp.get("output_dir", "outputs"))
    exp["output_dir"] = (out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir).resolve()
    root = Path(cfg["data"].get("root", "data"))
    cfg["data"]["root"] = (root if root.is_absolute() else PROJECT_ROOT / root).resolve()
    if fast:
        apply_fast(cfg)
    return cfg


def apply_fast(cfg):
    """Shrink the pipeline for quick debugging (VSCode breakpoints)."""
    cfg["training"].update({"epochs": 3, "warmup_epochs": 1, "early_stopping_patience": 2})
    cfg["tuning"].update({"n_trials_stage1": 2, "n_trials_stage2": 1, "cv_folds": 3,
                          "max_epochs_per_trial": 3})
    cfg["holdout"].update({"n_runs": 2})
    imp = dict(cfg["holdout"].get("importance", {}))
    imp["max_edges"] = 300
    cfg["holdout"]["importance"] = imp


def add_device_arg(parser):
    parser.add_argument("--device", default=None,
                        help="override training.device, e.g. cuda:0 / cpu / auto")
    return parser


def resolve_cfg_device(cfg, arg):
    return resolve_device(arg or cfg["training"].get("device", "auto"))
