"""Single training run for debugging (one split, full training, test metrics)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_device_arg, prepare_config, resolve_cfg_device  # noqa: E402
from src.data.dataset import load_preprocessed_data, make_loader, ConnectivityDataset  # noqa: E402
from src.data.splits import carve_validation, holdout_split  # noqa: E402
from src.evaluation.metrics import ovr_metrics  # noqa: E402
from src.models.registry import build_model  # noqa: E402
from src.training.holdout import HoldoutRunner  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast", action="store_true")
    add_device_arg(parser)
    args = parser.parse_args()

    cfg = prepare_config(args.config, fast=args.fast)
    device = resolve_cfg_device(cfg, args.device)
    exp = cfg["experiment"]["name"]
    logger = get_logger("train", cfg["experiment"]["output_dir"] / exp / "logs" / "train.log")
    logger.info(f"variant={cfg['model']['variant']} device={device} seed={args.seed}")

    subjects, X = load_preprocessed_data(cfg["data"], logger)
    y = subjects["y"].to_numpy()
    train_idx, test_idx = holdout_split(y, args.seed,
                                        cfg["holdout"].get("test_fraction", 0.2))
    sub_train, val_idx = carve_validation(y, train_idx,
                                          cfg["training"].get("val_fraction", 0.1), args.seed)
    logger.info(f"train {len(sub_train)} | val {len(val_idx)} | test {len(test_idx)}")

    hp = HoldoutRunner([cfg], device)._resolve_hp(cfg)
    logger.info(f"hyperparameters: {hp}")

    set_seed(args.seed)
    model = build_model(cfg["model"])
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model parameters: {n_params:,}")

    nw = cfg["training"].get("num_workers", 0)
    tr = make_loader(ConnectivityDataset(X[sub_train], y[sub_train]), hp["batch_size"], True, nw)
    va = make_loader(ConnectivityDataset(X[val_idx], y[val_idx]), hp["batch_size"], False, nw)
    te = make_loader(ConnectivityDataset(X[test_idx], y[test_idx]), hp["batch_size"], False, nw)

    train_cfg = dict(cfg["training"])
    train_cfg.update({"lr": hp["lr"], "weight_decay": hp["weight_decay"]})
    trainer = Trainer(model, device, train_cfg, logger=logger)
    best_state, history = trainer.fit(tr, va)
    model.load_state_dict(best_state)
    model.to(device)
    _, test_acc, probs, labels = trainer.evaluate(te)
    metrics = ovr_metrics(labels, probs)
    logger.info(f"test overall acc {test_acc:.3f}")
    for stage in ("NC", "EMCI", "LMCI"):
        m = metrics[stage]
        logger.info(f"  {stage}: acc {m['acc']:.3f} auc {m['auc']:.3f} "
                    f"sen {m['sen']:.3f} spe {m['spe']:.3f}")

    out = cfg["experiment"]["output_dir"] / exp / "single" / f"run_seed{args.seed}.json"
    save_json({"seed": args.seed, "hyperparameters": hp, "n_params": n_params,
               "history": history, "metrics": metrics,
               "y_true": labels.tolist(), "y_prob": probs.round(6).tolist()}, out)
    logger.info(f"saved {out}")


if __name__ == "__main__":
    main()
