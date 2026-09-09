"""One logistic regression per feature — the client's actual question.

The client has no ML infrastructure and asked for the single most predictive
feature. This module answers that, with the evaluation the reference omits:
fitted on train, scored on held-out validation, parameter count reported, and
separation detected rather than silently absorbed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .. import config as cfg
from ..evaluate import classification_metrics
from ..features import build_design, n_parameters


def _fit_logit(X, y):
    """Fit, and report honestly whether the optimiser converged.

    postal_code contains a category in which every customer claimed, which makes
    the maximum-likelihood estimate diverge. The reference would absorb this
    silently; here it becomes a reported column.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = sm.Logit(y, X).fit(disp=0)
        separated = any("Maximum Likelihood" in str(w.message)
                        or "Perfect separation" in str(w.message)
                        or "Singular" in str(w.message) for w in caught)
    converged = bool(model.mle_retvals.get("converged", False))
    return model, converged, separated


def fit_single_feature(train, val, feature: str, scheme: str = "dummy",
                       imputer=None, baseline: float | None = None) -> dict:
    """Fit `outcome ~ feature` on train, score on val.

    Returns one row of the results table.
    """
    tr = imputer.transform(train) if imputer is not None else train
    va = imputer.transform(val) if imputer is not None else val

    X_tr = sm.add_constant(build_design(tr, [feature], scheme), has_constant="add")
    X_va = sm.add_constant(build_design(va, [feature], scheme), has_constant="add")
    X_va = X_va[X_tr.columns]  # guard against column drift between splits

    y_tr, y_va = tr[cfg.TARGET], va[cfg.TARGET]
    model, converged, separated = _fit_logit(X_tr, y_tr)

    row = {"feature": feature, "scheme": scheme,
           "n_params": n_parameters(tr, feature, scheme)}
    row.update(classification_metrics(y_va, model.predict(X_va), baseline=baseline))
    # AIC prices the extra degrees of freedom that dummy encoding buys. It is
    # meaningless when the likelihood did not converge, so it is withheld there
    # rather than reported as a number.
    row["aic"] = float(model.aic) if converged else np.nan
    row["converged"] = converged
    row["separation"] = separated
    return row


def sweep_features(train, val, features=None, schemes=("linear", "dummy"),
                   imputer=None, baseline: float | None = None) -> pd.DataFrame:
    """Every feature under every encoding scheme, one row each."""
    features = list(features if features is not None else cfg.FEATURES)
    rows = [fit_single_feature(train, val, f, s, imputer, baseline)
            for s in schemes for f in features]
    return pd.DataFrame(rows)
