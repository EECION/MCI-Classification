"""Two-stage Optuna Bayesian tuning with 10-fold CV objective (paper Section 2.4).

Stage 1: 20 TPE trials over the full space
         lr ~ log-U(1e-5, 1e-3), weight_decay ~ log-U(1e-6, 1e-3), batch in {16,32,64}.
Stage 2: 10 further trials with the continuous ranges shrunk around the top-3
         stage-1 trials (+/- `stage2_shrink_log10` decades, clipped to the
         original bounds); batch sizes restricted to those seen in the top-3.
The best of all 30 trials is written to <output>/<variant>/tuning/best_params.json.
"""
import numpy as np
import optuna
import pandas as pd

from src.data.dataset import ConnectivityDataset, make_loader
from src.data.splits import stratified_folds
from src.models.registry import build_model
from src.training.trainer import Trainer
from src.utils.io import save_json
from src.utils.seed import set_seed

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _fold_score(cfg, X, y, train_idx, val_idx, hp, tuning_cfg, device, seed):
    set_seed(seed)
    model = build_model(cfg["model"])
    train_cfg = dict(cfg["training"])
    train_cfg.update({"lr": hp["lr"], "weight_decay": hp["weight_decay"],
                      "epochs": tuning_cfg.get("max_epochs_per_trial", 50)})
    trainer = Trainer(model, device, train_cfg)
    tr_loader = make_loader(ConnectivityDataset(X[train_idx], y[train_idx]),
                            hp["batch_size"], shuffle=True,
                            num_workers=cfg["training"].get("num_workers", 0))
    va_loader = make_loader(ConnectivityDataset(X[val_idx], y[val_idx]),
                            hp["batch_size"], shuffle=False,
                            num_workers=cfg["training"].get("num_workers", 0))
    best_state, _ = trainer.fit(tr_loader, va_loader)
    model.load_state_dict(best_state)
    model.to(device)
    val_loss, val_acc, _, _ = trainer.evaluate(va_loader)
    return val_loss, val_acc


def run_tuning(cfg, X, y, device, logger=None):
    tcfg = cfg["tuning"]
    space = tcfg["search_space"]
    seed = cfg["experiment"].get("seed", 42)
    objective_name = tcfg.get("objective", "val_acc")
    folds = stratified_folds(y, tcfg.get("cv_folds", 10), seed)
    base_epochs = tcfg.get("max_epochs_per_trial", 50)

    def make_objective(bounds):
        def objective(trial):
            hp = {
                "lr": trial.suggest_float("lr", *bounds["lr"], log=True),
                "weight_decay": trial.suggest_float("weight_decay", *bounds["weight_decay"], log=True),
                "batch_size": trial.suggest_categorical("batch_size", list(bounds["batch_size"])),
            }
            losses, accs = [], []
            for k, (tr, va) in enumerate(folds):
                set_seed(seed + 1000 + k)
                val_loss, val_acc = _fold_score(cfg, X, y, tr, va, hp,
                                                {"max_epochs_per_trial": base_epochs},
                                                device, seed)
                losses.append(val_loss)
                accs.append(val_acc)
            trial.set_user_attr("mean_val_loss", float(np.mean(losses)))
            if objective_name == "val_loss":
                return -float(np.mean(losses))   # maximize -> minimize loss
            return float(np.mean(accs))
        return objective

    if logger:
        logger.info(f"tuning stage 1: {tcfg['n_trials_stage1']} trials, "
                    f"space lr{space['lr']} wd{space['weight_decay']} batch{space['batch_size']}, "
                    f"{len(folds)}-fold CV")
    study1 = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study1.optimize(make_objective(space), n_trials=tcfg["n_trials_stage1"])

    top3 = sorted(study1.trials, key=lambda t: t.value, reverse=True)[:3]
    shrink = tcfg.get("stage2_shrink_log10", 0.5)

    def shrink_bounds(values, lo, hi):
        lo2 = max(lo, 10 ** (np.log10(min(values)) - shrink))
        hi2 = min(hi, 10 ** (np.log10(max(values)) + shrink))
        if lo2 >= hi2:
            center = np.log10(np.median(values))
            lo2, hi2 = max(lo, 10 ** (center - shrink)), min(hi, 10 ** (center + shrink))
            if lo2 >= hi2:
                lo2, hi2 = lo, hi
        return float(lo2), float(hi2)

    bounds2 = {
        "lr": shrink_bounds([t.params["lr"] for t in top3], *space["lr"]),
        "weight_decay": shrink_bounds([t.params["weight_decay"] for t in top3], *space["weight_decay"]),
        "batch_size": sorted({t.params["batch_size"] for t in top3}),
    }
    if logger:
        logger.info(f"tuning stage 2: {tcfg['n_trials_stage2']} trials, shrunk space "
                    f"lr{[f'{v:.2e}' for v in bounds2['lr']]} wd{[f'{v:.2e}' for v in bounds2['weight_decay']]} "
                    f"batch{bounds2['batch_size']}")
    study2 = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=seed + 1))
    study2.optimize(make_objective(bounds2), n_trials=tcfg["n_trials_stage2"])

    rows = []
    for stage, study in ((1, study1), (2, study2)):
        for t in study.trials:
            rows.append({"stage": stage, "trial": t.number,
                         **t.params, "objective": t.value,
                         "mean_val_loss": t.user_attrs.get("mean_val_loss")})
    trials_df = pd.DataFrame(rows).sort_values("objective", ascending=False)
    best = trials_df.iloc[0]
    best_params = {
        "lr": float(best["lr"]),
        "weight_decay": float(best["weight_decay"]),
        "batch_size": int(best["batch_size"]),
        "objective": float(best["objective"]),
        "objective_name": objective_name,
        "n_trials_total": int(len(trials_df)),
    }
    out_dir = cfg["experiment"]["output_dir"] / cfg["experiment"]["name"] / "tuning"
    save_json(best_params, out_dir / "best_params.json")
    trials_df.to_csv(out_dir / "trials.csv", index=False)
    if logger:
        logger.info(f"tuning done: best {best_params} -> {out_dir / 'best_params.json'}")
    return best_params, trials_df
