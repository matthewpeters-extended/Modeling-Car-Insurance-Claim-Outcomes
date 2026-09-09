#!/usr/bin/env python
"""Regenerate every published result table from the raw CSV.

This script is the project's reproducibility guarantee. Delete data/processed/
entirely, run this, and every number in the README and the notebooks comes back
identical. Nothing here reads a cached artifact; the only input is
data/raw/car_insurance.csv.

    python scripts/run_experiment.py

Runs in well under a minute on a laptop: 10,000 rows and logistic regression.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg                                    # noqa: E402
from src.data import load_raw, majority_class_baseline, split_data  # noqa: E402
from src.evaluate import classification_metrics, expected_cost   # noqa: E402
from src.features import MedianImputer, build_design, n_parameters  # noqa: E402
from src.models import sweep_features                            # noqa: E402

N_BOOTSTRAP = 10_000
THRESHOLDS = np.arange(0.02, 0.99, 0.01)
TOP_FEATURES = ["driving_experience", "age", "income", "vehicle_ownership",
                "speeding_violations", "past_accidents"]


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def main() -> int:
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(cfg.SEED)
    out = cfg.DATA_PROCESSED
    out.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Loading and splitting")
    df = load_raw()
    train, val, test = split_data(df)
    imputer = MedianImputer().fit(train)          # D3: train-fitted, never full-data
    tr, va, te = (imputer.transform(x) for x in (train, val, test))
    baseline = majority_class_baseline(train[cfg.TARGET])
    y_val, y_test = val[cfg.TARGET].to_numpy(), test[cfg.TARGET].to_numpy()
    log(f"train {len(train)} | val {len(val)} | test {len(test)} | baseline {baseline:.4f}")

    pd.DataFrame([{
        "split": name, "n": len(part),
        "claim_rate": part[cfg.TARGET].mean(),
        "majority_class_baseline": majority_class_baseline(part[cfg.TARGET]),
    } for name, part in [("train", train), ("val", val), ("test", test)]]
    ).to_csv(out / "results_baseline.csv", index=False)

    print("\n[2/6] EDA: claim-rate spread by feature")
    cat_cols = (list(cfg.ORDINAL_LEVELS) + ["vehicle_year", "vehicle_type", "gender"]
                + cfg.BINARY)
    spread = pd.DataFrame([{
        "feature": col,
        "n_levels": train[col].nunique(),
        "min_rate": train.groupby(col, observed=True)[cfg.TARGET].mean().min(),
        "max_rate": train.groupby(col, observed=True)[cfg.TARGET].mean().max(),
        "spread": (train.groupby(col, observed=True)[cfg.TARGET].mean().max()
                   - train.groupby(col, observed=True)[cfg.TARGET].mean().min()),
    } for col in cat_cols]).sort_values("spread", ascending=False).reset_index(drop=True)
    spread.to_csv(out / "results_eda_feature_spread.csv", index=False)
    log(f"widest spread: {spread.iloc[0].feature} ({spread.iloc[0].spread:.4f})")

    print("\n[3/6] D6: is the missingness informative?")
    missingness = pd.DataFrame([{
        "column": col,
        "n_missing": int(train[col].isna().sum()),
        "pct_missing": train[col].isna().mean() * 100,
        "claim_rate_present": train.loc[~train[col].isna(), cfg.TARGET].mean(),
        "claim_rate_missing": train.loc[train[col].isna(), cfg.TARGET].mean(),
        "difference": (train.loc[train[col].isna(), cfg.TARGET].mean()
                       - train.loc[~train[col].isna(), cfg.TARGET].mean()),
        "chi2": chi2_contingency(pd.crosstab(train[col].isna(), train[cfg.TARGET]))[0],
        "p_value": chi2_contingency(pd.crosstab(train[col].isna(), train[cfg.TARGET]))[1],
    } for col in cfg.COLUMNS_WITH_MISSING])
    missingness.to_csv(out / "results_missingness.csv", index=False)

    candidates = (list(cfg.ORDINAL_LEVELS)
                  + ["gender", "vehicle_year", "vehicle_type", "postal_code"] + cfg.BINARY)
    mar = pd.DataFrame([{
        "missing_in": mc, "tested_against": f,
        "chi2": chi2_contingency(pd.crosstab(train[mc].isna(), train[f]))[0],
        "p_value": chi2_contingency(pd.crosstab(train[mc].isna(), train[f]))[1],
    } for mc in cfg.COLUMNS_WITH_MISSING for f in candidates]
    ).sort_values("p_value").reset_index(drop=True)
    mar.to_csv(out / "results_missingness_mar.csv", index=False)
    log(f"smallest p = {mar.p_value.min():.4f} vs Bonferroni {0.05 / len(mar):.5f}: "
        f"{'informative' if mar.p_value.min() < 0.05 / len(mar) else 'not informative'}")

    print("\n[4/6] D3: what does the imputation leak cost?")
    leaky = MedianImputer().fit(df)               # what the reference does
    rows = []
    for col in cfg.COLUMNS_WITH_MISSING:
        for label, imp in [("honest (train-fitted)", imputer), ("leaky (full-data)", leaky)]:
            X_tr = sm.add_constant(build_design(imp.transform(train), [col], "dummy"))
            X_va = sm.add_constant(build_design(imp.transform(val), [col], "dummy"),
                                   has_constant="add")[X_tr.columns]
            model = sm.Logit(train[cfg.TARGET], X_tr).fit(disp=0)
            rows.append({"feature": col, "imputation": label,
                         "val_accuracy": ((model.predict(X_va) >= 0.5).astype(int)
                                          == y_val).mean()})
    leak = pd.DataFrame(rows).pivot(index="feature", columns="imputation",
                                    values="val_accuracy")
    leak["delta"] = leak["leaky (full-data)"] - leak["honest (train-fitted)"]
    leak["rows_flipped"] = (leak["delta"].abs() * len(val)).round().astype(int)
    leak.to_csv(out / "results_imputation_leak.csv")
    log(f"leak flips {int(leak.rows_flipped.max())} of {len(val)} validation predictions")

    counts = pd.DataFrame({
        "linear": [n_parameters(tr, f, "linear") for f in cfg.FEATURES],
        "dummy": [n_parameters(tr, f, "dummy") for f in cfg.FEATURES],
    }, index=cfg.FEATURES)
    counts["kind"] = ["ordinal" if f in cfg.ORDINAL_LEVELS else
                      "nominal" if f in cfg.NOMINAL else
                      "binary" if f in cfg.BINARY else
                      "count" if f in cfg.COUNTS else "continuous" for f in cfg.FEATURES]
    counts[["kind", "linear", "dummy"]].to_csv(out / "results_encoding_parameters.csv")

    print("\n[5/6] Single-feature sweep, both encoding schemes")
    results = sweep_features(train, val, imputer=imputer, baseline=baseline)
    results.to_csv(out / "results_single_feature.csv", index=False)
    winner = results[results.scheme == "dummy"].nlargest(1, "accuracy").iloc[0]
    log(f"{len(results)} models fitted; best = {winner.feature} "
        f"({winner.accuracy:.4f}, lift {winner.lift_over_baseline:+.4f})")

    print("\n[6/6] Bootstrap, thresholds, ceiling, and the test set")

    def fit_predict(features, target):
        X_tr = sm.add_constant(build_design(tr, features, "dummy"))
        X = sm.add_constant(build_design(target, features, "dummy"),
                            has_constant="add")[X_tr.columns]
        return sm.Logit(tr[cfg.TARGET], X_tr).fit(disp=0).predict(X).to_numpy()

    val_probs = {f: fit_predict([f], va) for f in TOP_FEATURES}

    idx = rng.integers(0, len(y_val), size=(N_BOOTSTRAP, len(y_val)))
    boots = {f: ((p >= 0.5).astype(int) == y_val).astype(float)[idx].mean(axis=1)
             for f, p in val_probs.items()}
    ci = pd.DataFrame([{
        "feature": f,
        "accuracy": ((val_probs[f] >= 0.5).astype(int) == y_val).mean(),
        "ci_low": np.percentile(b, 2.5), "ci_high": np.percentile(b, 97.5),
    } for f, b in boots.items()]).sort_values("accuracy", ascending=False)
    ci.to_csv(out / "results_bootstrap_ci.csv", index=False)
    gap = boots["driving_experience"] - boots["age"]
    log(f"winner vs runner-up: 95% CI [{np.percentile(gap, 2.5):+.4f}, "
        f"{np.percentile(gap, 97.5):+.4f}], P(better) = {(gap > 0).mean():.4f}")

    claim_rate = y_val.mean()
    rows = [{"feature": f,
             "optimal_threshold": THRESHOLDS[np.argmin(
                 [expected_cost(y_val, p, t, cfg.FN_TO_FP_COST_RATIO) for t in THRESHOLDS])],
             "cost_at_optimum": min(expected_cost(y_val, p, t, cfg.FN_TO_FP_COST_RATIO)
                                    for t in THRESHOLDS),
             "cost_at_0.5": expected_cost(y_val, p, 0.5, cfg.FN_TO_FP_COST_RATIO),
             "accuracy_at_0.5": ((p >= 0.5).astype(int) == y_val).mean()}
            for f, p in val_probs.items()]
    rows += [
        {"feature": "[always predict claim]", "optimal_threshold": np.nan,
         "cost_at_optimum": 1 - claim_rate, "cost_at_0.5": 1 - claim_rate,
         "accuracy_at_0.5": claim_rate},
        {"feature": "[always predict no claim]", "optimal_threshold": np.nan,
         "cost_at_optimum": cfg.FN_TO_FP_COST_RATIO * claim_rate,
         "cost_at_0.5": cfg.FN_TO_FP_COST_RATIO * claim_rate,
         "accuracy_at_0.5": 1 - claim_rate},
    ]
    cost_table = pd.DataFrame(rows).sort_values("cost_at_optimum").reset_index(drop=True)
    cost_table.to_csv(out / "results_cost_thresholds.csv", index=False)

    multi_features = [f for f in cfg.FEATURES if f != "postal_code"]
    p_multi_val = fit_predict(multi_features, va)
    comparison = pd.DataFrame({
        "single feature (driving_experience)": classification_metrics(
            y_val, val_probs["driving_experience"], baseline=baseline),
        f"all {len(multi_features)} features": classification_metrics(
            y_val, p_multi_val, baseline=baseline),
    }).T[["accuracy", "lift_over_baseline", "roc_auc", "precision", "recall", "brier"]]
    comparison.to_csv(out / "results_single_vs_multi.csv")

    # The test set. Scored once, with every choice already fixed above.
    p_de_test = fit_predict(["driving_experience"], te)
    p_multi_test = fit_predict(multi_features, te)
    final = []
    for label, p, t in [
        ("driving_experience @ 0.5", p_de_test, 0.5),
        ("driving_experience @ 0.05 (cost-optimal)", p_de_test, 0.05),
        ("all features @ 0.5 (ceiling)", p_multi_test, 0.5),
        ("all features @ 0.16 (cost-optimal)", p_multi_test, 0.16),
    ]:
        m = classification_metrics(y_test, p, threshold=t, baseline=baseline)
        m["model"] = label
        m["expected_cost_5to1"] = expected_cost(y_test, p, t, cfg.FN_TO_FP_COST_RATIO)
        final.append(m)
    final_df = pd.DataFrame(final).set_index("model")[
        ["accuracy", "baseline", "lift_over_baseline", "roc_auc",
         "precision", "recall", "expected_cost_5to1"]]
    final_df.to_csv(out / "results_final.csv")

    headline = final_df.loc["driving_experience @ 0.5"]
    print("\n" + "=" * 78)
    print(f"  HEADLINE: driving_experience, {headline.accuracy:.4f} test accuracy")
    print(f"            baseline {headline.baseline:.4f}, "
          f"lift {headline.lift_over_baseline:+.4f}, AUC {headline.roc_auc:.4f}")
    print(f"  Wrote {len(list(out.glob('results_*.csv')))} tables to {out}")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
