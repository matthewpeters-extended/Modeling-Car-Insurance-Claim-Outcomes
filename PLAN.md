# Plan of Execution: Car Insurance Claim Outcomes

**Repo:** `~/projects/car-insurance-claim-modeling`
**Started:** 2026-09-08
**Estimated effort:** 12-14 hours across 5 working sessions
**Status:** planning complete, no analysis code written yet

---

## 0. What this project is

On the Road car insurance wants to know which customers will file a claim during
the policy period. They have no ML infrastructure, so they asked for **the single
most predictive feature** — one variable, one simple model, something an analyst
can run in a spreadsheet if they have to.

That is the brief from the original DataCamp project. It is a real constraint that
real small insurers face, and it is a better exercise than "throw XGBoost at it,"
because it forces you to defend a comparison between models rather than just
report a leaderboard.

### The framing decision that makes this resume-worthy

The reference implementation answers the brief in about 40 lines and reports
**77.71% accuracy** for `driving_experience`. That answer is directionally correct
and the code runs. It is also **not defensible in an interview**, for reasons that
are easy to state and easy to fix.

Here is the one that matters most:

> The target is 31.33% positive. A model that predicts "no claim" for every single
> customer scores **68.67%**. The reference reports 77.71% and never mentions the
> 68.67% floor, so the reader cannot tell whether the model found signal worth
> **9.04 points** or whether 77.71% is just what the base rate looks like.

That gap — between a number and a number you can defend — is the entire project.
We are not going to "improve the model." We are going to build the evaluation the
reference is missing, report honestly what survives it, and write it up so that a
hiring manager reading the README sees someone who knows the difference between
a result and an artifact.

This mirrors the framing that worked on `netflix-stock-forecasting`: the headline
is not "my model won," it is "here is what is actually true, and here is how I
know."

---

## 1. Source synthesis: what we inherit and what we fix

Sources are catalogued in `docs/sources.md`. The reference notebook is committed
unmodified at `docs/reference_solution.ipynb` so we can diff our results against
it and cite specific cells in the write-up.

### The seven defects we inherit from the reference notebook, and fix

These are not nitpicks. Each one changes a number we would otherwise report.

**D1. No train/test split.** Every model is fitted on all 10,000 rows and scored
on the same 10,000 rows (`ref cell 6`: `model.predict()` with no argument predicts
in-sample). The 77.71% is training accuracy. For a one-feature logistic regression
the optimism is small, but "small" is a claim we have to measure, not assume.
**Fix:** stratified 70/15/15 train/validation/test split, seeded. Test set touched
exactly once, at the end.

**D2. Accuracy is reported without its baseline.** Covered above. 68.67% is the
number every result gets compared against.
**Fix:** the majority-class baseline is row one of every results table, and lift
over baseline is a reported column.

**D3. Imputation leaks across the split.** `ref cell 3` fills `credit_score` and
`annual_mileage` with the median of the **full** dataset, including rows that
would later be test rows. The median of a 10k-row column is stable enough that
this probably moves the result very little — which is exactly why it is worth
measuring rather than hand-waving.
**Fix:** compute imputation values on train only, apply to validation and test.
Report the delta between leaky and honest imputation so the write-up can say how
much it mattered instead of guessing.

**D4. The feature comparison is not apples to apples.** This is the subtle one.
`driving_experience` is a string with 4 levels, so the statsmodels formula
interface silently expands it into **3 dummy variables**. `age` is stored as an
integer, so it enters as **1 linear term**. The loop in `ref cell 5` therefore
compares a 4-parameter model against a 2-parameter model and declares the winner
on raw accuracy, with no penalty for the extra degrees of freedom.
**Fix:** encode every ordinal consistently, then run the comparison **twice** —
once treating ordinals as linear, once as full dummies — and report both. Add AIC
alongside accuracy so the parameter count is visible.

**D5. Accuracy is the wrong metric for this business problem.** A false negative
(missed claim) and a false positive (flagged a good customer) do not cost an
insurer the same amount. Accuracy at a fixed 0.5 threshold hides both.
**Fix:** report ROC-AUC (threshold-free), precision/recall, and a threshold chosen
by expected cost under a stated, explicit cost ratio. We keep accuracy too, since
the client asked for it — but it stops being the only number.

**D6. Missingness is assumed to be random.** ~10% of `credit_score` and
`annual_mileage` is missing, median-imputed with no check on whether missingness
correlates with `outcome`. If it does, the fact that a value is missing is itself
a feature, and imputing it destroys information.
**Fix:** test claim rate among missing vs. present for both columns before
imputing. If the difference is real, add a missingness indicator and report it.

**D7. No seed, no tests, no reproducibility.** Nothing pins randomness and there
is no way to verify a refactor did not change a result.
**Fix:** single seed in `src/config.py`, `scripts/run_experiment.py` regenerates
every table end to end, and `tests/` covers the split and encoding logic.

### One finding that will survive all of this

Claim rate by driving experience, computed during planning:

| `driving_experience` | n | claim rate |
|---|---|---|
| 0-9y | 3,530 | 0.628 |
| 10-19y | 3,299 | 0.239 |
| 20-29y | 2,119 | 0.051 |
| 30y+ | 1,052 | 0.019 |

That is a monotone 33x spread from the safest bucket to the riskiest. The
reference's *conclusion* — driving experience is the best single predictor — is
almost certainly right. It is the *evidence offered for it* that we are rebuilding.
Say this plainly in the README. Confirming someone's answer with better method is
a stronger story than contradicting it, and it is what actually happened.

### One thing to check early

`vehicle_type` is 95.2% `sedan`. A near-constant feature can post a high accuracy
by doing nothing at all. It should land at or below the 68.67% baseline, and if it
does not, something is wrong with our harness. Use it as a canary.

---

## 2. Environment: what we need before writing any code

- **Python 3.12** via Homebrew, same as `netflix-stock-forecasting`
  (`/opt/homebrew/opt/python@3.12/bin/python3.12`). The system Python at
  `/usr/bin/python3` is 3.9 — do not use it.
- **No GPU, no cloud, no API keys.** 10,000 rows and logistic regression. Every
  result on this project runs in seconds on the laptop.
- **`gh` CLI is not installed.** Either `brew install gh`, or create the GitHub
  repo through the web UI and add the remote by hand. Not a blocker either way.
- Dataset is already in place at `data/raw/car_insurance.csv` (10,000 rows, 1 MB),
  and unlike the Netflix project it **is committed to git** — it is small, and a
  fresh clone should reproduce the results table with no external download.

### Step 0 — build the environment (~10 min, do this first)

```bash
cd ~/projects/car-insurance-claim-modeling && /opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt && .venv/bin/pip freeze > requirements-lock.txt
```

### Step 1 — git

```bash
cd ~/projects/car-insurance-claim-modeling && git init && git add -A && git commit -m "Scaffold, dataset, plan, and source notes"
```

Remote goes to `matthewpeters-extended` to match the Netflix project. Push at the
end of Session 5, once the README carries real numbers — a half-finished README
is worse for a portfolio than no repo at all.

---

## 3. Repository layout

```
car-insurance-claim-modeling/
├── README.md                  # written LAST, in Session 5, with measured numbers
├── PLAN.md                    # this file
├── WALKTHROUGH.md             # the narrative: what I tried, what broke, what I learned
├── requirements.txt           # direct deps
├── requirements-lock.txt      # pip freeze, generated in Step 0
├── data/
│   ├── raw/car_insurance.csv  # committed: 10k rows, 1 MB
│   └── processed/             # regenerated; only results_*.csv are committed
├── docs/
│   ├── sources.md             # provenance + data dictionary  [written]
│   ├── reference_solution.ipynb  # the original, unmodified, for diffing  [saved]
│   └── SETUP.md               # clone-to-results instructions
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_missingness_and_encoding.ipynb
│   ├── 03_single_feature_models.ipynb
│   ├── 04_evaluation_and_threshold.ipynb
│   └── 05_results.ipynb       # reads data/processed/results_*.csv, makes final figures
├── reports/figures/           # exported plots the README embeds
├── scripts/run_experiment.py  # one command, regenerates every table
├── src/
│   ├── config.py              # SEED, paths, split ratios, cost ratio
│   ├── data.py                # load, validate schema, split
│   ├── features.py            # encoding, imputation (train-fitted)
│   ├── evaluate.py            # metrics, confusion matrices, baselines
│   ├── plots.py
│   └── models/                # single-feature and multi-feature fitting
└── tests/
    ├── test_split.py          # split is stratified, seeded, and disjoint
    ├── test_features.py       # imputer fitted on train never sees test rows
    └── test_evaluate.py       # metrics match hand-computed values on a toy set
```

Notebooks tell the story; `src/` holds the logic; `scripts/run_experiment.py` is
the reproducibility guarantee. Same split that worked on the Netflix project.

---

## 4. Execution phases

### Session 1 — Environment, data contract, EDA (2.5h)
- Step 0 and Step 1 above.
- `src/config.py`, `src/data.py`: load, assert the schema (18 columns, 10,000 rows,
  `outcome` in {0,1}), stratified 70/15/15 split.
- `tests/test_split.py` — write this before the EDA, so the split is trustworthy
  before anything is fitted on it.
- `01_eda.ipynb`: class balance, per-feature claim rates, the `vehicle_type`
  imbalance, correlation structure between features.
- **Deliverable:** the 68.67% baseline computed and committed, split reproducible.

### Session 2 — Missingness and encoding (2h)
- Test D6: is missingness in `credit_score` / `annual_mileage` associated with
  `outcome`? Chi-square, and report the claim rate in each group.
- Build train-fitted imputation (D3). Quantify the leaky-vs-honest delta.
- Resolve D4: consistent ordinal encoding, with both linear and dummy variants
  available behind a flag.
- `tests/test_features.py`.
- **Deliverable:** `02_missingness_and_encoding.ipynb`, one clear answer on whether
  missingness carries signal.

### Session 3 — The single-feature models, done properly (2.5h)
- Fit one logistic regression per feature on **train**, score on **validation**.
- Report per feature: accuracy, lift over 68.67%, ROC-AUC, precision, recall, AIC,
  and number of parameters (this is what exposes D4).
- Run it under both encoding schemes and compare the rankings.
- **Deliverable:** `03_single_feature_models.ipynb`, `results_single_feature.csv`.
  This is the table that answers the client's actual question.

### Session 4 — Evaluation, thresholds, and honest limits (2.5h)
- ROC and precision-recall curves for the top features.
- Cost-sensitive threshold: state a cost ratio explicitly, derive the threshold
  that minimises expected cost, show how much the "best" decision moves as the
  ratio changes. This is the section that reads as insurance work rather than
  homework.
- Bootstrap confidence interval on the winner's accuracy, so the write-up can say
  whether the gap to second place is real or noise.
- Fit **one** multi-feature model as a ceiling check: how much is the client giving
  up by restricting to a single feature? That is a genuinely useful business answer
  and it costs 20 lines.
- Score the final chosen model on **test**, once.
- **Deliverable:** `04_evaluation_and_threshold.ipynb`, `results_final.csv`.

### Session 5 — Write-up and publish (2.5h)
- `scripts/run_experiment.py` regenerates every committed table from scratch.
  Delete `data/processed/`, run it, confirm the numbers come back identical.
- `05_results.ipynb` and the figures the README embeds.
- `WALKTHROUGH.md`: the narrative, including the things that did not work.
- `README.md` last, with real numbers.
- `docs/SETUP.md`, then push.

---

## 5. Metrics we report, and why

| Metric | Why it is here |
|---|---|
| **Majority-class accuracy (68.67%)** | The floor. Every other number is read against it. |
| Accuracy | The client asked for it. Kept, but never alone. |
| Lift over baseline | The honest version of accuracy. |
| ROC-AUC | Threshold-free ranking quality; immune to the 0.5 arbitrariness. |
| Precision / Recall | An insurer cares about these separately. |
| Expected cost at chosen threshold | Turns the model into a decision. |
| AIC | Makes D4's parameter-count asymmetry visible. |
| Bootstrap CI on the winner | Says whether the winning margin is real. |

---

## 6. README plan: the part that gets the interview

Structure that worked on the Netflix project, adapted:

1. **One-line question.** "Which single customer attribute best predicts a car
   insurance claim — and is the answer actually better than guessing?"
2. **Headline result box.** The winning feature, its test accuracy, the 68.67%
   baseline, the lift, and the confidence interval. State the baseline in the same
   sentence as the result. Never report one without the other.
3. **Results table.** Every feature, every metric, one held-out test set.
4. **"What I found and what it means"** — 3-4 subsections, each a real finding:
   the baseline gap, the dummy-variable asymmetry (D4), what missingness turned
   out to carry, and what the single-feature constraint costs versus the
   multi-feature ceiling.
5. **Reproduce it** — the one command.
6. **What I would do with more time / what this does not show.** Explicit limits.
   This section signals seniority more than any model does.
7. **Attribution.** The reference repo and DataCamp, named plainly, with a clear
   statement of what we changed and why our numbers differ.

Point 7 matters. Building on a public solution is completely normal, and saying
so directly is the difference between a portfolio project and a plagiarism
question in an interview.

---

## 7. Resume bullets (fill the bracketed numbers in Session 5)

- Rebuilt a public insurance claim-prediction analysis with a held-out evaluation
  harness, finding the original's reported 77.71% accuracy sat only 9.04 points
  above the 68.67% majority-class baseline it never reported.
- Identified a silent dummy-variable expansion in the original's statsmodels
  formula loop that compared 4-parameter and 2-parameter models on raw accuracy;
  re-ran the comparison under consistent encoding with AIC, [changing/confirming]
  the feature ranking.
- Replaced fixed-threshold accuracy with cost-sensitive threshold selection and
  ROC-AUC on 10,000 policies, [result].
- Shipped a reproducible pipeline — seeded splits, train-fitted imputation, pytest
  coverage, single-command regeneration of every published table.

---

## 8. Open questions

**Q1. How hard do we lean on the "fintech" framing?** This is insurance risk
modeling, not payments or markets. The honest bridge is expected loss: a claim
probability times a claim cost is a pricing input, and that is the same
cost-asymmetry reasoning credit risk uses. Framing the threshold section in
expected-cost terms makes it read as quantitative risk work without overclaiming.
Recommend doing that; flagging it because it is a positioning call, not a
technical one.

**Q2. Do we include the multi-feature ceiling model?** It technically violates the
client's stated constraint. Recommend including it, clearly labelled as a
reference point rather than the deliverable — "here is what you are giving up" is
useful to a client and shows judgment to a reader. Cheap to build, easy to cut.

---

## 9. Scope guardrails

Things that would improve a Kaggle score and would **not** improve this project:

- No XGBoost, no random forests, no neural nets. The brief says simple, and the
  point of the project is evaluation quality, not model capacity.
- No hyperparameter search. Logistic regression with default settings is correct.
- No SMOTE or resampling. 31% positive is not severe imbalance, and resampling
  would distort the calibrated probabilities the threshold analysis needs.
- No feature engineering beyond encoding. Inventing interaction terms breaks the
  "single feature" brief.

If a session runs long, cut Session 4's bootstrap CI and the multi-feature ceiling
first. Cut the README last — it is the deliverable.
