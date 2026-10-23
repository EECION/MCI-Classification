"""100 repeated random hold-out runner (paper Sections 2.4, 3.3-3.5).

For each seed: a stratified 80:20 split (identical across all variants passed
to the runner -> paired Base/Enh comparisons), training with the tuned or
configured hyperparameters, OvR test metrics, and optional edge importance.
Each run is a self-contained JSON; existing runs are skipped so interrupted
sweeps resume, and `--shard i/n` splits seeds across GPUs/processes.
"""
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import ConnectivityDataset, make_loader
from src.data.splits import carve_validation, holdout_split
from src.evaluation.importance import compute_importance
from src.evaluation.metrics import ovr_metrics, summarize_runs
from src.models.registry import build_model
from src.training.trainer import Trainer
from src.utils.io import load_json, save_json
from src.utils.seed import set_seed


class HoldoutRunner:
    def __init__(self, configs, device, logger=None, overwrite=False):
        self.configs = configs
        self.device = device
        self.logger = logger
        self.overwrite = overwrite
        self._warned_default_hp = set()

    def _resolve_hp(self, cfg):
        mode = cfg["holdout"].get("best_params", "auto")
        exp = cfg["experiment"]["name"]
        tcfg = cfg["training"]
        if mode in (False, "none"):
            return {"lr": tcfg["lr"], "weight_decay": tcfg["weight_decay"],
                    "batch_size": tcfg["batch_size"]}
        if mode == "auto":
            path = Path(cfg["experiment"]["output_dir"]) / exp / "tuning" / "best_params.json"
            if path.exists():
                bp = load_json(path)
                return {"lr": bp["lr"], "weight_decay": bp["weight_decay"],
                        "batch_size": bp["batch_size"]}
            if exp not in self._warned_default_hp and self.logger:
                self.logger.warning(f"[{exp}] no tuned params found ({path}); "
                                    f"using training-section defaults")
                self._warned_default_hp.add(exp)
            return {"lr": tcfg["lr"], "weight_decay": tcfg["weight_decay"],
                    "batch_size": tcfg["batch_size"]}
        bp = load_json(mode)  # explicit path
        return {"lr": bp["lr"], "weight_decay": bp["weight_decay"], "batch_size": bp["batch_size"]}

    def _run_one(self, cfg, X, y, seed, train_idx, val_idx, test_idx):
        exp = cfg["experiment"]["name"]
        hp = self._resolve_hp(cfg)
        set_seed(seed)
        model = build_model(cfg["model"])
        nw = cfg["training"].get("num_workers", 0)
        tr_loader = make_loader(ConnectivityDataset(X[train_idx], y[train_idx]),
                                hp["batch_size"], shuffle=True, num_workers=nw)
        va_loader = make_loader(ConnectivityDataset(X[val_idx], y[val_idx]),
                                hp["batch_size"], shuffle=False, num_workers=nw)
        te_loader = make_loader(ConnectivityDataset(X[test_idx], y[test_idx]),
                                hp["batch_size"], shuffle=False, num_workers=nw)
        train_cfg = dict(cfg["training"])
        train_cfg.update({"lr": hp["lr"], "weight_decay": hp["weight_decay"]})
        trainer = Trainer(model, self.device, train_cfg)
        best_state, history = trainer.fit(tr_loader, va_loader)
        model.load_state_dict(best_state)
        model.to(self.device)
        _, test_acc, probs, labels = trainer.evaluate(te_loader)
        metrics = ovr_metrics(labels, probs)

        run = {
            "experiment": exp,
            "variant": cfg["model"]["variant"],
            "seed": int(seed),
            "hyperparameters": {k: (float(v) if isinstance(v, float) else v) for k, v in hp.items()},
            "split": {"train": train_idx.tolist(), "val": val_idx.tolist(),
                      "test": test_idx.tolist()},
            "history": history,
            "metrics": metrics,
            "y_true": labels.tolist(),
            "y_prob": [[round(float(p), 6) for p in row] for row in probs],
        }
        imp_cfg = cfg["holdout"].get("importance", {})
        every = int(imp_cfg.get("every_n_runs", 1))
        if imp_cfg.get("enabled", True) and seed % every == 0:
            run["importance"] = compute_importance(model, X[test_idx], y[test_idx],
                                                    imp_cfg, self.device)
        return run

    def run(self, X, y, seeds):
        n = len(y)
        for seed in seeds:
            train_idx, test_idx = holdout_split(y, seed,
                                                self.configs[0]["holdout"].get("test_fraction", 0.2))
            val_fraction = self.configs[0]["training"].get("val_fraction", 0.1)
            sub_train, val_idx = carve_validation(y, train_idx, val_fraction, seed)
            for cfg in self.configs:
                exp = cfg["experiment"]["name"]
                run_path = Path(cfg["experiment"]["output_dir"]) / exp / "runs" / f"run_{seed:03d}.json"
                if run_path.exists() and not self.overwrite:
                    continue
                run = self._run_one(cfg, X, y, seed, sub_train, val_idx, test_idx)
                save_json(run, run_path)
                if self.logger:
                    self.logger.info(
                        f"[{exp}] seed {seed:3d} done | overall acc {run['metrics']['overall_acc']:.3f}"
                        f" | " + " ".join(
                            f"{c}: acc {run['metrics'][c]['acc']:.3f} auc {run['metrics'][c]['auc']:.3f}"
                            for c in ("NC", "EMCI", "LMCI")))
            self._log_progress(seed)
        return self

    def _log_progress(self, seed):
        if not self.logger:
            return
        for cfg in self.configs:
            exp = cfg["experiment"]["name"]
            run_dir = Path(cfg["experiment"]["output_dir"]) / exp / "runs"
            done = sorted(run_dir.glob("run_*.json"))
            if not done:
                continue
            accs = [load_json(p)["metrics"]["overall_acc"] for p in done]
            stats = summarize_runs(accs)
            self.logger.info(f"[{exp}] progress: {stats['n']}/? runs | overall acc "
                             f"{stats['mean']:.4f} ± {stats['std']:.4f}")
