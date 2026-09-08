"""Loading, schema validation, and splitting.

The split is the foundation the whole project rests on, so it is defined once,
here, and tested in tests/test_split.py before any model is fitted on it.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as cfg


def load_raw(path=None) -> pd.DataFrame:
    """Read the raw CSV and assert the data contract.

    Fails loudly on any deviation from the expected shape or column order. A
    silently reshaped upstream file is the kind of thing that produces a
    plausible but wrong number, which is worse than a crash.
    """
    df = pd.read_csv(path or cfg.DATA_RAW)

    if list(df.columns) != cfg.EXPECTED_COLUMNS:
        raise ValueError(
            f"Column mismatch.\n  expected: {cfg.EXPECTED_COLUMNS}\n  got:      {list(df.columns)}"
        )
    if len(df) != cfg.EXPECTED_ROWS:
        raise ValueError(f"Expected {cfg.EXPECTED_ROWS} rows, got {len(df)}")

    # The CSV stores several genuinely integer columns as floats (1.0 / 0.0).
    # Only credit_score and annual_mileage are allowed to be missing; casting
    # the others to int therefore both tidies the dtypes and acts as an
    # assertion that no unexpected NaN has crept in.
    int_cols = [cfg.TARGET] + cfg.BINARY
    missing_where_unexpected = df[int_cols].isna().sum()
    if missing_where_unexpected.any():
        raise ValueError(f"Unexpected missing values:\n{missing_where_unexpected[missing_where_unexpected > 0]}")
    df[int_cols] = df[int_cols].astype(int)

    unexpected_missing = set(df.columns[df.isna().any()]) - set(cfg.COLUMNS_WITH_MISSING)
    if unexpected_missing:
        raise ValueError(f"Missing values in unexpected columns: {sorted(unexpected_missing)}")

    if not df[cfg.TARGET].isin([0, 1]).all():
        raise ValueError("Target contains values outside {0, 1}")
    if df[cfg.ID_COL].duplicated().any():
        raise ValueError("Duplicate client ids")

    return df


def split_data(df: pd.DataFrame, seed: int | None = None):
    """Stratified 70/15/15 train/validation/test split.

    Stratified on the target so all three sets carry the same 31.33% claim rate;
    an unstratified split would let the baseline accuracy differ between sets and
    make the comparisons in Session 3 incoherent.

    Returns (train, val, test). The test set is not to be touched until Session 4.
    """
    seed = cfg.SEED if seed is None else seed
    y = df[cfg.TARGET]

    train_val, test = train_test_split(
        df, test_size=cfg.TEST_SIZE, random_state=seed, stratify=y, shuffle=True
    )
    # VAL_SIZE is a fraction of the full dataset, so rescale it to a fraction of
    # the remaining train_val block.
    val_fraction_of_remainder = cfg.VAL_SIZE / (1.0 - cfg.TEST_SIZE)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_remainder,
        random_state=seed,
        stratify=train_val[cfg.TARGET],
        shuffle=True,
    )
    return train, val, test


def majority_class_baseline(y: pd.Series) -> float:
    """Accuracy of always predicting the more common class.

    This is the number every result in the project is read against. The
    reference implementation reports 77.71% accuracy without ever stating it.
    """
    return float(max(y.mean(), 1.0 - y.mean()))
