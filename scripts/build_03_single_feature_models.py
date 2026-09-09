import nbformat as nbf, pathlib
nb = nbf.v4.new_notebook(); c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s.strip()))
co = lambda s: c.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 03 — Single-feature models

This notebook answers the client's question: **which single customer attribute best
predicts a claim?**

One logistic regression per feature, fitted on the 7,000-row training split and scored on
the 1,500-row validation split it never saw. Both encoding schemes. Parameter counts,
AIC, and ranking metrics reported alongside accuracy, because accuracy alone cannot tell
you whether a model found signal or learned to say "no."

The test split is still untouched. It is used once, in notebook 04.
""")

co("""
import sys; sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config as cfg
from src.data import load_raw, split_data, majority_class_baseline
from src.features import MedianImputer
from src.models import sweep_features
from src.plots import set_style, save_fig

set_style()
pd.set_option("display.width", 200)

df = load_raw()
train, val, test = split_data(df)
imputer = MedianImputer().fit(train)          # D3: fitted on train only
baseline = majority_class_baseline(train[cfg.TARGET])

print(f"train {len(train)} | val {len(val)} | test {len(test)} (untouched)")
print(f"majority-class baseline: {baseline:.4f}")
""")

co("""
results = sweep_features(train, val, imputer=imputer, baseline=baseline)
results.to_csv(cfg.DATA_PROCESSED / "results_single_feature.csv", index=False)
print(f"{len(results)} models fitted ({results.feature.nunique()} features x {results.scheme.nunique()} schemes)")
""")

md("""
## 1. The results

Sorted by validation accuracy, dummy encoding. The `baseline` column is 68.67% throughout
— that is the accuracy of predicting "no claim" for every customer, and it is the number
every row must be read against.
""")

co("""
cols = ["feature", "n_params", "accuracy", "lift_over_baseline", "roc_auc",
        "precision", "recall", "predicted_positive_rate", "aic", "separation"]
dummy = results[results.scheme == "dummy"].sort_values("accuracy", ascending=False)
dummy[cols].round(4).reset_index(drop=True)
""")

md("""
### The headline

**`driving_experience` wins: 78.80% validation accuracy against a 68.67% baseline — a
lift of 10.13 points — with a ROC-AUC of 0.813.**

The reference implementation's conclusion was right. Its evidence was not: 77.71%
measured in-sample, with no baseline stated, so a reader could not tell whether it
represented ten points of signal or none. Rebuilding the evaluation confirms the answer
rather than overturning it, which is the more common outcome of doing this kind of work
and worth saying plainly.

`driving_experience` also wins under **every** metric in the table and under **both**
encoding schemes. That robustness is itself the finding — the recommendation does not
depend on which yardstick the client happens to prefer.
""")

md("""
## 2. Nine of sixteen features never predict a claim at all

Look at the `predicted_positive_rate` column. Nine features score **exactly** 68.67% —
not approximately, exactly — with precision and recall of zero.

Those models predict "no claim" for all 1,500 validation customers. They score the
baseline because they *are* the baseline. Accuracy cannot distinguish them from a model
that has learned nothing, because there is nothing to distinguish.
""")

co("""
flat = dummy[dummy.predicted_positive_rate == 0]
print(f"{len(flat)} of {len(dummy)} features never predict a single claim:\\n")
print(flat[["feature", "accuracy", "roc_auc", "predicted_positive_rate"]]
      .sort_values("roc_auc", ascending=False).round(4).to_string(index=False))
""")

md("""
This happens because a feature can shift the predicted probability substantially without
ever pushing it across 0.5. Moving a group from a 31% claim rate to a 46% claim rate is a
real, useful change in risk — and every one of those customers is still classified "no
claim." Accuracy at a fixed threshold is blind to it.

Two of those nine are not weak features at all.
""")

co("""
fig, ax = plt.subplots(figsize=(9, 5.5))
d = dummy.sort_values("roc_auc")
colors = ["#C44E52" if r > 0 else "#B0B0B0" for r in d.lift_over_baseline]
ax.barh(d.feature, d.roc_auc, color=colors, alpha=0.9)
ax.axvline(0.5, color="#937860", ls="--", lw=1.5, label="AUC 0.5 = no ranking signal")
for i, (auc, lift) in enumerate(zip(d.roc_auc, d.lift_over_baseline)):
    ax.text(auc + 0.006, i, f"{auc:.3f}" + ("" if lift > 0 else "  (0 accuracy lift)"),
            va="center", fontsize=8, color="#333")
ax.set_xlim(0.45, 0.95)
ax.set_xlabel("ROC-AUC on validation")
ax.set_title("Red = beats the accuracy baseline. Grey = scores exactly 68.67%.", fontsize=11)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
save_fig(fig, "auc_vs_accuracy_lift")
plt.show()
""")

md("""
`speeding_violations` has an **AUC of 0.732** and `past_accidents` **0.726**. Both rank
customers by risk about as well as `income` does. Both deliver **zero** accuracy lift.

This is defect D5 in a single picture. The metric you choose determines the answer you
get, and the reference chose the one metric that cannot see these two features at all.
""")

co("""
by_acc = dummy.sort_values("accuracy", ascending=False).feature.tolist()
by_auc = dummy.sort_values("roc_auc", ascending=False).feature.tolist()
pd.DataFrame({"rank": range(1, 7),
              "by accuracy": by_acc[:6],
              "by ROC-AUC": by_auc[:6]})
""")

md("""
The top three agree. Ranks four and five do not: `speeding_violations` and
`past_accidents` displace `vehicle_ownership` and `credit_score` under AUC.

For this client the accuracy ranking is the one that answers their question, since they
asked for a single feature to threshold on. But an insurer pricing premiums cares about
ordering customers by risk, not about a binary flag — and for that job the AUC column is
the right one. Notebook 04 makes the cost of that choice explicit.
""")

md("""
## 3. Does the encoding scheme change the answer? (D4)

Session 2 built an explicit feature taxonomy to fix the reference's dtype-driven
encoding, in which `driving_experience` silently became 3 parameters while `age` stayed
at 1. With both schemes now applied uniformly, the question is what that fix bought.
""")

co("""
pivot = results.pivot(index="feature", columns="scheme",
                      values=["accuracy", "roc_auc", "aic"])
ordinals = pivot.loc[list(cfg.ORDINAL_LEVELS)]
ordinals["aic_gain_from_dummy"] = ordinals[("aic", "linear")] - ordinals[("aic", "dummy")]
ordinals.round(4)
""")

co("""
same_rank = (results[results.scheme == "linear"].sort_values("accuracy", ascending=False).feature.tolist()
             == results[results.scheme == "dummy"].sort_values("accuracy", ascending=False).feature.tolist())
print("linear and dummy produce the same accuracy ranking:", same_rank)
print()
print("top 5, linear:", results[results.scheme == 'linear'].nlargest(5, 'accuracy').feature.tolist())
print("top 5, dummy :", results[results.scheme == 'dummy'].nlargest(5, 'accuracy').feature.tolist())
""")

md("""
### The honest answer: the fix did not change the ranking

Both schemes give identical accuracy for every feature, and the same ordering. The defect
was real, the fix was correct, and **it did not change who wins.**

That is worth reporting as it stands rather than quietly dropping. The reason it does not
matter here is specific and checkable.
""")

co("""
# The probabilities genuinely differ between schemes. The class predictions do not.
import statsmodels.api as sm
from src.features import build_design

tr, va = imputer.transform(train), imputer.transform(val)
rows = []
for f in cfg.ORDINAL_LEVELS:
    probs = {}
    for s in ("linear", "dummy"):
        X_tr = sm.add_constant(build_design(tr, [f], s))
        X_va = sm.add_constant(build_design(va, [f], s), has_constant="add")[X_tr.columns]
        probs[s] = sm.Logit(tr[cfg.TARGET], X_tr).fit(disp=0).predict(X_va).to_numpy()
    rows.append({
        "feature": f,
        "max_prob_difference": np.abs(probs["linear"] - probs["dummy"]).max(),
        "identical_class_predictions": ((probs["linear"] >= .5) == (probs["dummy"] >= .5)).mean(),
    })
pd.DataFrame(rows).round(4)
""")

md("""
The two encodings produce measurably different probabilities — up to 0.050 apart on
`age`. But for each of these features only one level sits above a 50% claim rate, and
both schemes agree on which one. Every customer therefore lands on the same side of the
threshold, and accuracy records no difference at all.

So D4 was a genuine defect that the client's chosen metric is too blunt to detect. AIC is
not too blunt.
""")

co("""
fig, ax = plt.subplots(figsize=(7.5, 4))
gain = ordinals["aic_gain_from_dummy"].sort_values()
ax.barh(gain.index, gain.values,
        color=["#4C72B0" if v < 0 else "#C44E52" for v in gain.values], alpha=0.9)
ax.axvline(0, color="#333", lw=1)
for i, v in enumerate(gain.values):
    ax.text(v + (1.5 if v >= 0 else -1.5), i, f"{v:+.1f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=9)
ax.set_xlabel("AIC improvement from dummy encoding  (positive = dummy worth its extra parameters)")
ax.set_title("Do the extra degrees of freedom earn their keep?", fontsize=11)
fig.tight_layout()
save_fig(fig, "aic_encoding_gain")
plt.show()
""")

md("""
This confirms a prediction made in notebook 02 **before** these models were fitted.

Session 2 plotted log-odds across ordinal levels and observed that `income` and
`education` were near-straight while `age` and `driving_experience` bent. Linear encoding
assumes straightness; where it holds, the extra dummy parameters are wasted.

That is exactly what happened. `age` gains **53.5 AIC points** from dummy encoding — its
log-odds curve was the most bent. `income` and `education` *lose* 2.5 and 1.2 points,
because their straight lines never needed the extra parameters. `driving_experience`
lands at +0.7, effectively a tie.

The prediction was made in advance and the data confirmed it. That is a stronger claim
than noticing the pattern afterwards.
""")

md("""
## 4. `postal_code` — the feature that must not ship

Session 1 found that all 120 customers in postal code 21217 filed a claim, causing
perfect separation. Here is what that looks like as a model.
""")

co("""
pc = dummy[dummy.feature == "postal_code"].iloc[0]
print(f"accuracy   {pc.accuracy:.4f}   (lift +{pc.lift_over_baseline:.4f})")
print(f"precision  {pc.precision:.4f}")
print(f"recall     {pc.recall:.4f}")
print(f"roc_auc    {pc.roc_auc:.4f}")
print(f"converged  {pc.converged}")
print(f"aic        {pc.aic}   <- withheld: meaningless when the likelihood diverges")
print()
print(f"predicts a claim for {pc.predicted_positive_rate:.1%} of validation customers,")
print(f"and is right every time ({int(pc.tp)} true positives, {int(pc.fp)} false positives).")
""")

md("""
Precision of **1.000** and recall of **0.045**. The model flags only the 21217 customers,
and it is never wrong about them — because in this dataset that postal code *is* the
outcome.

It clears the baseline by 1.4 points and it is worthless. The 100% claim rate is an
artifact of how the data was generated, not a fact about drivers in Baltimore, and a
model built on it would fail the moment it met a real customer from that postal code who
did not claim. Its AIC is withheld rather than reported, because a diverged likelihood
does not have a meaningful one.

Recorded here as a warning, excluded from the recommendation.
""")

md("""
## 5. Harness check

`vehicle_type` was designated the canary in notebook 01: 95% sedan, with near-identical
claim rates across the two categories. If a feature carrying no information were to post
a real accuracy lift, the harness would be broken.
""")

co("""
canary = dummy[dummy.feature == "vehicle_type"].iloc[0]
print(f"vehicle_type: accuracy {canary.accuracy:.4f}, lift {canary.lift_over_baseline:+.4f}, AUC {canary.roc_auc:.4f}")
assert canary.lift_over_baseline <= 0, "canary beat the baseline -- the harness is wrong"
assert 0.45 < canary.roc_auc < 0.55, "canary shows ranking signal it should not have"
print("PASS: the canary carries no signal, as designed.")
""")

md("""
## 6. What carries into Session 4

- **`driving_experience` is the answer**: 78.80% validation accuracy, +10.13 points over
  baseline, AUC 0.813, winning under every metric and both encodings.
- **Accuracy is too blunt for nine of sixteen features.** `speeding_violations`
  (AUC 0.732) and `past_accidents` (AUC 0.726) rank risk well and score zero accuracy
  lift. Session 4's threshold analysis is where they get a fair hearing.
- **The encoding fix did not move the ranking** — but it moved the probabilities, and AIC
  detected what accuracy could not.
- **`postal_code` is excluded** from the recommendation and reported as a warning.
- Nothing has touched the test set yet. Session 4 scores the final choice on it **once**.
""")

nb["cells"] = c
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.12.14"}}
out = pathlib.Path.home()/"projects/car-insurance-claim-modeling/notebooks/03_single_feature_models.ipynb"
nbf.write(nb, str(out)); print("wrote", out, "cells:", len(c))
