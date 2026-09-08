"""Imputation and encoding.

Two defects from PLAN.md are fixed here.

**D3 (leakage).** `MedianImputer` is fitted on the training split and applied
unchanged to validation and test. The reference computes medians over the full
dataset, which lets test rows influence the values used to fill training rows.

**D4 (unfair feature comparison).** Encoding is driven by the explicit taxonomy in
config.py, not by pandas dtypes. The reference let storage type decide: string
columns were silently expanded into dummy variables while integer-coded columns
entered as a single linear term, so a 4-parameter model was compared against a
2-parameter one on raw accuracy. Here the choice is a named argument, applied
uniformly, and the parameter count is reported alongside every result.
"""

from __future__ import annotations

import pandas as pd

from . import config as cfg

SCHEMES = ("linear", "dummy")


class MedianImputer:
    """Fills the two columns that have missing values with training medians.

    Deliberately not sklearn's SimpleImputer: keeping this explicit means the
    leakage fix is visible in the code rather than buried in a pipeline object.
    """

    def __init__(self, columns=None):
        self.columns = list(columns if columns is not None else cfg.COLUMNS_WITH_MISSING)
        self.medians_: dict[str, float] | None = None

    def fit(self, df: pd.DataFrame) -> "MedianImputer":
        self.medians_ = {c: float(df[c].median()) for c in self.columns}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None:
            raise RuntimeError("MedianImputer.transform called before fit")
        out = df.copy()
        for c, v in self.medians_.items():
            out[c] = out[c].fillna(v)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def encode_feature(df: pd.DataFrame, feature: str, scheme: str = "dummy") -> pd.DataFrame:
    """Return the design columns for one feature, without an intercept.

    `scheme` governs ordinals only:

    - "linear": map ordered levels to 0, 1, 2, ... and enter as one column. Imposes
      a monotone, evenly-spaced effect on the log-odds — a strong assumption, but
      it costs a single parameter.
    - "dummy": one indicator per level after the first. Assumes nothing about
      shape or spacing, at the cost of (k - 1) parameters.

    Nominal features are always dummies: `postal_code` is a label, and fitting a
    slope across the integers 10238 to 92101 would be meaningless arithmetic.
    Binary, count and continuous features pass through unchanged.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    if feature not in cfg.FEATURES:
        raise ValueError(f"{feature!r} is not a modelling feature")

    s = df[feature]

    if feature in cfg.ORDINAL_LEVELS:
        levels = cfg.ORDINAL_LEVELS[feature]
        unseen = set(s.dropna().unique()) - set(levels)
        if unseen:
            raise ValueError(f"{feature}: unseen levels {sorted(unseen)}")
        if scheme == "linear":
            # Rank by the real-world ordering in config, never alphabetically.
            return s.map({lvl: i for i, lvl in enumerate(levels)}).to_frame(feature)
        cat = pd.Categorical(s, categories=levels, ordered=True)
        return pd.get_dummies(cat, prefix=feature, drop_first=True).astype(float).set_index(s.index)

    if feature in cfg.NOMINAL:
        levels = sorted(df[feature].unique())
        cat = pd.Categorical(s, categories=levels)
        d = pd.get_dummies(cat, prefix=feature, drop_first=True).astype(float).set_index(s.index)
        return d

    # Binary, counts, continuous.
    return s.astype(float).to_frame(feature)


def build_design(df: pd.DataFrame, features=None, scheme: str = "dummy") -> pd.DataFrame:
    """Design matrix for a list of features, without an intercept."""
    features = list(features if features is not None else cfg.FEATURES)
    return pd.concat([encode_feature(df, f, scheme) for f in features], axis=1)


def n_parameters(df: pd.DataFrame, feature: str, scheme: str = "dummy") -> int:
    """Columns a feature contributes, excluding the intercept.

    Reported next to every accuracy in Session 3. Without it, a 4-parameter model
    beating a 2-parameter one looks like a better feature rather than a bigger one.
    """
    return encode_feature(df, feature, scheme).shape[1]
