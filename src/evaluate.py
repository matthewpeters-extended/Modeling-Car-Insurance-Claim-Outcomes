"""Metrics.

Defect D5 in PLAN.md: the reference reports accuracy at a fixed 0.5 threshold and
nothing else. On a target that is 31.33% positive, accuracy alone cannot distinguish
a model that has found signal from one that has learned to say "no claim" — and it
hides the fact that a missed claim and a false alarm cost an insurer different
amounts. Every metric here is reported next to the majority-class baseline.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)


def classification_metrics(y_true, y_prob, threshold: float = 0.5,
                           baseline: float | None = None) -> dict:
    """Score predicted probabilities against binary truth.

    `baseline` is the majority-class accuracy. Passing it turns raw accuracy into
    lift, which is the number that actually says whether the model did anything.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)

    out = {
        "accuracy": acc,
        # AUC is undefined if a split somehow contains one class only.
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        # Brier score measures calibration, which the Session 4 threshold work
        # depends on: a threshold is only meaningful if the probabilities are.
        "brier": brier_score_loss(y_true, y_prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "predicted_positive_rate": float(y_pred.mean()),
    }
    if baseline is not None:
        out["baseline"] = baseline
        out["lift_over_baseline"] = acc - baseline
        out["beats_baseline"] = bool(acc > baseline)
    return out


def expected_cost(y_true, y_prob, threshold: float, fn_to_fp_ratio: float) -> float:
    """Average cost per customer, with a false negative costing `fn_to_fp_ratio`
    times a false positive.

    Used in Session 4. Stated as an assumption, not a measured quantity.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    _, fp, fn, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float((fp + fn_to_fp_ratio * fn) / len(y_true))
