# Setup

Clone to results in about three minutes on a laptop. No GPU, no API keys, no cloud.
The whole project is 10,000 rows and logistic regression.

## Requirements

- **Python 3.12.** Built and tested on 3.12.14 (Homebrew, Apple Silicon). On macOS the
  system `python3` is 3.9 and will not work — use the versioned binary.
- About 700 MB of disk for the virtual environment.

## Install

From the project root:

```bash
python3.12 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt
```

`requirements.txt` lists direct dependencies with lower bounds.
`requirements-lock.txt` is the exact `pip freeze` from the environment these results were
produced in — install from that instead if a future release of pandas or statsmodels
changes a number.

## Reproduce the results

```bash
.venv/bin/python scripts/run_experiment.py
```

Takes well under a minute and writes 11 tables to `data/processed/`. Deleting that
directory first is a fair test: every table returns byte-identical, because the only
input is `data/raw/car_insurance.csv`.

## Run the tests

```bash
.venv/bin/python -m pytest -q
```

42 tests covering the data contract, the split (disjoint, stratified, seeded), the
encoding schemes, the imputation leakage fix, and the metrics — the last checked against
hand-computed values.

## Run the notebooks

```bash
.venv/bin/python -m jupyterlab
```

Notebooks 01 to 05 are committed with their outputs, so they can be read without
executing anything. They run in order and depend only on `src/` and the raw CSV, except
notebook 05, which reads the tables in `data/processed/`.

To re-execute one from the command line:

```bash
cd notebooks && ../.venv/bin/python -m nbconvert --to notebook --execute --inplace 01_eda.ipynb
```

## Where the numbers live

| File | Contents |
|---|---|
| `data/processed/results_baseline.csv` | The 68.67% majority-class baseline, per split |
| `data/processed/results_single_feature.csv` | All 32 single-feature models, both schemes |
| `data/processed/results_final.csv` | The held-out test results |
| `data/processed/results_bootstrap_ci.csv` | Confidence intervals on the top features |
| `data/processed/results_cost_thresholds.csv` | Cost-optimal thresholds at 5:1 |

Seed, split ratios, feature taxonomy and the cost ratio are all in `src/config.py`.
