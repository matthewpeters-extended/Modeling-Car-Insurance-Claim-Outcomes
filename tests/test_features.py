"""Tests for imputation and encoding — defects D3 and D4.

Both defects are silent by nature: leaky imputation and dtype-driven encoding
produce numbers that look completely reasonable. These tests are the only thing
standing between a plausible result and a correct one.
"""

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src.data import load_raw, split_data
from src.features import MedianImputer, build_design, encode_feature, n_parameters


@pytest.fixture(scope="module")
def splits():
    return split_data(load_raw())


@pytest.fixture(scope="module")
def train(splits):
    return splits[0]


# --- D3: imputation must not see the test set ------------------------------

def test_imputer_uses_training_medians_only(splits):
    train, val, test = splits
    imp = MedianImputer().fit(train)
    for col, value in imp.medians_.items():
        assert value == pytest.approx(train[col].median())


def test_imputer_ignores_val_and_test_when_fitting(splits):
    """Fitting on train must give a different result from fitting on everything.

    credit_score's full-data median differs from the training median by ~0.001.
    Small, but the point is that the pipeline cannot see it at all — not that the
    difference happens to be tolerable.
    """
    train, val, test = splits
    full = pd.concat([train, val, test])
    honest = MedianImputer().fit(train).medians_["credit_score"]
    leaky = MedianImputer().fit(full).medians_["credit_score"]
    assert honest != leaky


def test_transform_fills_every_missing_value(splits):
    train, val, test = splits
    imp = MedianImputer().fit(train)
    for part in (train, val, test):
        assert imp.transform(part).isna().sum().sum() == 0


def test_transform_does_not_mutate_its_input(train):
    before = train["credit_score"].isna().sum()
    MedianImputer().fit(train).transform(train)
    assert train["credit_score"].isna().sum() == before


def test_transform_before_fit_raises(train):
    with pytest.raises(RuntimeError, match="before fit"):
        MedianImputer().transform(train)


# --- D4: encoding must be driven by semantics, not dtype -------------------

def test_ordinal_linear_encoding_follows_real_world_order(train):
    """`income` alphabetises to middle < poverty < upper < working, which is
    nonsense. The encoding must use the ordering declared in config.py."""
    enc = encode_feature(train, "income", scheme="linear")["income"]
    lookup = pd.Series(enc.to_numpy(), index=train["income"].to_numpy())
    assert lookup.loc["poverty"].iloc[0] == 0
    assert lookup.loc["working class"].iloc[0] == 1
    assert lookup.loc["middle class"].iloc[0] == 2
    assert lookup.loc["upper class"].iloc[0] == 3


def test_ordinal_linear_encoding_is_one_parameter(train):
    for feature in cfg.ORDINAL_LEVELS:
        assert n_parameters(train, feature, "linear") == 1


def test_ordinal_dummy_encoding_is_k_minus_one(train):
    for feature, levels in cfg.ORDINAL_LEVELS.items():
        assert n_parameters(train, feature, "dummy") == len(levels) - 1


def test_the_unfair_comparison_is_reproduced_then_fixed(train):
    """The reference compared these two features on raw accuracy.

    Under dtype-driven encoding driving_experience contributes 3 columns and age
    contributes 1, so one model had three extra degrees of freedom. Under either
    of our schemes, applied uniformly, they contribute the same number.
    """
    for scheme in ("linear", "dummy"):
        assert (n_parameters(train, "driving_experience", scheme)
                == n_parameters(train, "age", scheme))


def test_postal_code_is_never_treated_as_a_number(train):
    """Postal codes are labels. Fitting a slope across 10238..92101 is meaningless
    arithmetic, and it is what the reference does by leaving the column as int."""
    for scheme in ("linear", "dummy"):
        cols = encode_feature(train, "postal_code", scheme).columns
        assert len(cols) == train["postal_code"].nunique() - 1
        assert all(c.startswith("postal_code_") for c in cols)


def test_dummies_are_disjoint_indicators(train):
    d = encode_feature(train, "driving_experience", "dummy")
    assert set(np.unique(d.to_numpy())) <= {0.0, 1.0}
    assert d.sum(axis=1).max() <= 1  # reference level is all-zero


def test_passthrough_features_are_unchanged(train):
    for feature in cfg.COUNTS + cfg.BINARY:
        enc = encode_feature(train, feature, "dummy")
        assert enc.shape[1] == 1
        assert np.allclose(enc[feature], train[feature].astype(float))


def test_encoding_preserves_row_alignment(train):
    enc = build_design(train, ["age", "driving_experience", "credit_score"], "dummy")
    assert list(enc.index) == list(train.index)
    assert len(enc) == len(train)


def test_unseen_level_raises(train):
    bad = train.copy()
    bad.loc[bad.index[0], "education"] = "postgraduate"
    with pytest.raises(ValueError, match="unseen levels"):
        encode_feature(bad, "education", "dummy")


def test_invalid_scheme_raises(train):
    with pytest.raises(ValueError, match="scheme must be one of"):
        encode_feature(train, "age", scheme="onehot")


def test_non_feature_column_raises(train):
    for col in (cfg.ID_COL, cfg.TARGET):
        with pytest.raises(ValueError, match="not a modelling feature"):
            encode_feature(train, col, "dummy")


def test_full_design_matrix_has_no_missing_values(train):
    imp = MedianImputer().fit(train)
    X = build_design(imp.transform(train), scheme="dummy")
    assert X.isna().sum().sum() == 0
    assert len(X) == len(train)
