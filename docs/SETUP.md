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
python3.12 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements-lock.txt
```

**Install from the lock file, not `requirements.txt`.** The latter lists direct
dependencies with lower bounds, so it will resolve to whatever is current — fine for
reading the code, but it does not guarantee the published numbers.
`requirements-lock.txt` is the exact `pip freeze` from the environment these results were
produced in.

The versions that produced the committed tables are also recorded in `src/environment.py`
and in `data/processed/results_environment.csv`:

| package | version |
|---|---|
| python | 3.12.14 |
| numpy | 2.5.3 |
| pandas | 3.0.5 |
| scipy | 1.18.1 |
| statsmodels | 0.15.0 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.1 |

`scripts/run_experiment.py` checks this on every run and prints a warning if what is
installed differs. A mismatch is not an error — newer libraries are usually fine — but it
means byte-identical reproduction is no longer guaranteed, and you should know that rather
than wonder why a digit moved. To see the comparison on its own:

```bash
.venv/bin/python -m src.environment
```

## Reproduce the results

```bash
.venv/bin/python scripts/run_experiment.py
```

Takes well under a minute and writes 13 tables to `data/processed/`. Deleting that
directory first is a fair test: every table returns byte-identical, because the only
input is `data/raw/car_insurance.csv`.

## Run the tests

```bash
.venv/bin/python -m pytest -q
```

51 tests covering the data contract, the split (disjoint, stratified, seeded), the
encoding schemes, the imputation leakage fix, the metrics (checked against hand-computed
values), the pinned environment, and the sensitivity of the headline to the
synthetic-data artifact.

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
| `data/processed/results_sensitivity.csv` | The headline with and without the 21217 artifact |
| `data/processed/results_environment.csv` | Library versions the tables were produced under |

Seed, split ratios, feature taxonomy and the cost ratio are all in `src/config.py`.
