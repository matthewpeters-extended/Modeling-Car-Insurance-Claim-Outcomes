"""Robustness of the headline result to the synthetic-data artifact.

Notebook 01 found that all 120 customers in postal code 21217 filed a claim. That
is an artifact of how this dataset was generated, not a fact about drivers, and
the README lists it as a limitation.

A limitation stated is weaker than a limitation measured. This module removes
those 120 rows, re-runs the whole pipeline from the split onward, and reports
whether the headline survives. If the answer moves, the caveat was load-bearing.
If it does not, the caveat can be stated with a number attached.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from . import config as cfg
from .data import majority_class_baseline, split_data
from .evaluate import classification_metrics
from .features import MedianImputer, build_design

ARTIFACT_POSTAL_CODE = 21217
COMPARE_FEATURES = ["driving_experience", "age", "income", "vehicle_ownership"]


def _score_on_test(data: pd.DataFrame, features: list[str]) -> list[dict]:
    """Split, impute, fit on train, score each feature on that data's test split."""
    train, _, test = split_data(data)
    imputer = MedianImputer().fit(train)
    tr, te = imputer.transform(train), imputer.transform(test)
    baseline = majority_class_baseline(train[cfg.TARGET])
    y_test = test[cfg.TARGET].to_numpy()

    rows = []
    for feature in features:
        X_tr = sm.add_constant(build_design(tr, [feature], "dummy"))
        X_te = sm.add_constant(build_design(te, [feature], "dummy"),
                               has_constant="add")[X_tr.columns]
        probs = sm.Logit(train[cfg.TARGET], X_tr).fit(disp=0).predict(X_te).to_numpy()
        m = classification_metrics(y_test, probs, baseline=baseline)
        rows.append({"feature": feature, "n_rows": len(data), "baseline": baseline,
                     "accuracy": m["accuracy"], "lift_over_baseline": m["lift_over_baseline"],
                     "roc_auc": m["roc_auc"]})
    return rows


def artifact_sensitivity(df: pd.DataFrame, features=None) -> pd.DataFrame:
    """Headline metrics with and without the perfectly-separating postal code."""
    features = list(features or COMPARE_FEATURES)
    without = df[df["postal_code"] != ARTIFACT_POSTAL_CODE]

    rows = []
    for label, data in [("full dataset", df), ("artifact removed", without)]:
        for row in _score_on_test(data, features):
            rows.append({"dataset": label, **row})
    out = pd.DataFrame(rows)

    wide = out.pivot(index="feature", columns="dataset",
                     values=["accuracy", "lift_over_baseline", "roc_auc"])
    # Flatten the MultiIndex so downstream code can index by a plain string.
    wide.columns = [f"{metric}__{dataset}" for metric, dataset in wide.columns]
    wide["lift_change"] = (wide["lift_over_baseline__artifact removed"]
                           - wide["lift_over_baseline__full dataset"])
    wide["auc_change"] = (wide["roc_auc__artifact removed"]
                          - wide["roc_auc__full dataset"])
    return out, wide
