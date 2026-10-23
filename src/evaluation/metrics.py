"""One-vs-rest metrics (Supplementary Methods S5.3-S5.4, Eqs. S20-S23).

For each diagnostic stage the target class is positive and the remaining two
classes are negative; ACC/SEN/SPE come from the binary counts and AUC from the
softmax probability of the target class.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from src.data.dataset import CLASS_NAMES


def ovr_metrics(y_true, probs):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    y_pred = probs.argmax(axis=1)
    n = len(y_true)
    result = {"overall_acc": float((y_pred == y_true).mean()) if n else float("nan")}
    for c, name in enumerate(CLASS_NAMES):
        pos = y_true == c
        pred_pos = y_pred == c
        tp = int((pos & pred_pos).sum())
        tn = int((~pos & ~pred_pos).sum())
        fp = int((~pos & pred_pos).sum())
        fn = int((pos & ~pred_pos).sum())
        entry = {
            "acc": (tp + tn) / n if n else float("nan"),
            "sen": tp / (tp + fn) if (tp + fn) else float("nan"),
            "spe": tn / (tn + fp) if (tn + fp) else float("nan"),
        }
        try:
            entry["auc"] = float(roc_auc_score(pos.astype(int), probs[:, c]))
        except ValueError:
            entry["auc"] = float("nan")
        result[name] = entry
    return result


def summarize_runs(metric_values):
    """mean +/- std across repeated hold-out experiments (Eq. S23)."""
    values = np.asarray([v for v in metric_values if not np.isnan(v)], dtype=float)
    if len(values) == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n": int(len(values))}
