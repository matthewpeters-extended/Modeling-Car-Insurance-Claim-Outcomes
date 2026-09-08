"""Tests for the data contract and the train/validation/test split.

Written before any model was fitted. The split is the only thing in this project
that, if quietly wrong, would make every downstream number wrong while still
looking entirely reasonable.
"""

import pandas as pd
import pytest

from src import config as cfg
from src.data import load_raw, majority_class_baseline, split_data


@pytest.fixture(scope="module")
def df():
    return load_raw()


@pytest.fixture(scope="module")
def splits(df):
    return split_data(df)


# --- The data contract -----------------------------------------------------

def test_raw_shape_and_columns(df):
    assert df.shape == (cfg.EXPECTED_ROWS, len(cfg.EXPECTED_COLUMNS))
    assert list(df.columns) == cfg.EXPECTED_COLUMNS


def test_target_is_binary(df):
    assert set(df[cfg.TARGET].unique()) == {0, 1}


def test_only_expected_columns_have_missing_values(df):
    have_missing = set(df.columns[df.isna().any()])
    assert have_missing == set(cfg.COLUMNS_WITH_MISSING)


def test_feature_taxonomy_partitions_the_features():
    """Every feature is classified exactly once.

    This is what keeps defect D4 fixed: if someone adds a column and forgets to
    classify it, the encoder would fall back to dtype-driven behaviour, which is
    the bug we are correcting.
    """
    groups = [
        list(cfg.ORDINAL_LEVELS), cfg.NOMINAL, cfg.BINARY, cfg.COUNTS, cfg.CONTINUOUS
    ]
    flat = [c for g in groups for c in g]
    assert len(flat) == len(set(flat)), "a feature is classified twice"
    assert set(flat) == set(cfg.FEATURES)


def test_ordinal_levels_match_the_data(df):
    for col, levels in cfg.ORDINAL_LEVELS.items():
        assert set(df[col].unique()) == set(levels), f"{col} levels drifted"


def test_load_raw_rejects_a_missing_column(tmp_path, df):
    bad = tmp_path / "bad.csv"
    df.drop(columns=["credit_score"]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="Column mismatch"):
        load_raw(bad)


def test_load_raw_rejects_wrong_row_count(tmp_path, df):
    bad = tmp_path / "short.csv"
    df.head(100).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="Expected 10000 rows"):
        load_raw(bad)


# --- The split -------------------------------------------------------------

def test_split_sizes(splits):
    train, val, test = splits
    assert len(train) == 7000
    assert len(val) == 1500
    assert len(test) == 1500


def test_splits_are_disjoint(splits):
    train, val, test = splits
    ids = [set(s[cfg.ID_COL]) for s in (train, val, test)]
    assert ids[0] & ids[1] == set()
    assert ids[0] & ids[2] == set()
    assert ids[1] & ids[2] == set()


def test_splits_cover_every_row(df, splits):
    train, val, test = splits
    recovered = set(train[cfg.ID_COL]) | set(val[cfg.ID_COL]) | set(test[cfg.ID_COL])
    assert recovered == set(df[cfg.ID_COL])


def test_split_is_stratified(df, splits):
    """All three sets must carry the full dataset's claim rate.

    If they did not, the majority-class baseline would differ between sets and
    the lift numbers in Session 3 would not be comparable.
    """
    overall = df[cfg.TARGET].mean()
    for part in splits:
        assert part[cfg.TARGET].mean() == pytest.approx(overall, abs=0.005)


def test_split_is_deterministic(df):
    a = split_data(df, seed=cfg.SEED)
    b = split_data(df, seed=cfg.SEED)
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x, y)


def test_different_seed_gives_a_different_split(df):
    a_train, _, _ = split_data(df, seed=cfg.SEED)
    b_train, _, _ = split_data(df, seed=cfg.SEED + 1)
    assert set(a_train[cfg.ID_COL]) != set(b_train[cfg.ID_COL])


# --- The baseline ----------------------------------------------------------

def test_majority_class_baseline_on_a_known_series():
    y = pd.Series([0] * 70 + [1] * 30)
    assert majority_class_baseline(y) == pytest.approx(0.70)


def test_majority_class_baseline_is_the_project_yardstick(df):
    """68.67%. Every reported accuracy is read against this number."""
    assert majority_class_baseline(df[cfg.TARGET]) == pytest.approx(0.6867, abs=1e-4)
