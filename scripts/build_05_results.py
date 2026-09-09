import nbformat as nbf, pathlib
nb = nbf.v4.new_notebook(); c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s.strip()))
co = lambda s: c.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 05 — Results

Every table in this notebook is read from `data/processed/`, not recomputed. Those files
are produced by:

```bash
python scripts/run_experiment.py
```

Delete `data/processed/` entirely, run that command, and every number below returns
byte-identical. This notebook builds the figures the README embeds.
""")

co("""
import sys; sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config as cfg
from src.plots import set_style, save_fig

set_style()
pd.set_option("display.width", 200)

read = lambda name, **kw: pd.read_csv(cfg.DATA_PROCESSED / f"results_{name}.csv", **kw)

final = read("final", index_col=0)
single = read("single_feature")
ci = read("bootstrap_ci")
cost = read("cost_thresholds")
baseline_tbl = read("baseline")
vs_multi = read("single_vs_multi", index_col=0)

BASELINE = baseline_tbl.loc[baseline_tbl.split == "train", "majority_class_baseline"].iloc[0]
print(f"majority-class baseline: {BASELINE:.4f}")
""")

md("""
## The headline

One feature, one held-out test set of 1,500 customers, scored once.
""")

co("""
headline = final.loc["driving_experience @ 0.5"]
ceiling = final.loc["all features @ 0.5 (ceiling)"]

print(f"  feature              driving_experience")
print(f"  test accuracy        {headline.accuracy:.4f}")
print(f"  baseline             {headline.baseline:.4f}   (predict 'no claim' for everyone)")
print(f"  lift                 {headline.lift_over_baseline:+.4f}")
print(f"  ROC-AUC              {headline.roc_auc:.4f}")
print()
print(f"  reference reported   0.7771   (in-sample, all 10,000 rows, no baseline stated)")
print(f"  15-feature ceiling   {ceiling.accuracy:.4f}   (lift {ceiling.lift_over_baseline:+.4f})")
""")

co("""
final.round(4)
""")

md("""
## Figure 1 — what the accuracy number actually contains

The reference reports 77.71% as its result. The bar below decomposes what a number like
that is made of: most of it is the base rate, available for free by predicting "no claim"
for everyone.
""")

co("""
fig, ax = plt.subplots(figsize=(9, 3.4))

acc = headline.accuracy
ax.barh(["driving_experience\\n(this project, held-out)"], [BASELINE],
        color="#B0B0B0", label=f"free from the base rate ({BASELINE:.1%})")
ax.barh(["driving_experience\\n(this project, held-out)"], [acc - BASELINE], left=[BASELINE],
        color="#C44E52", label=f"signal from the model (+{acc - BASELINE:.1%})")
ax.barh(["reference\\n(in-sample, no baseline)"], [0.7771], color="#D8D8D8",
        edgecolor="#888", hatch="//", label="reference, reported as one number")

ax.axvline(BASELINE, color="#333", ls="--", lw=1.5)
ax.text(BASELINE - 0.005, 1.42, "baseline 68.67%", ha="right", fontsize=9, color="#333")
ax.set_xlim(0, 1.0)
ax.set_xlabel("accuracy")
ax.set_title("Nine points of a 76.5% score are the base rate, not the model", fontsize=11.5)
ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
fig.tight_layout()
save_fig(fig, "headline_accuracy_decomposition")
plt.show()
""")

md("""
## Figure 2 — the metric decides the answer

Every feature, ranked two ways. Under accuracy, nine of sixteen features are worthless.
Under expected cost at a 5:1 false-negative-to-false-positive ratio, two of those nine
are among the best available.
""")

co("""
dummy = single[single.scheme == "dummy"].set_index("feature")
merged = cost[~cost.feature.str.startswith("[")].set_index("feature").join(
    dummy[["lift_over_baseline", "roc_auc"]])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

d1 = merged.sort_values("lift_over_baseline")
axes[0].barh(d1.index, d1.lift_over_baseline,
             color=["#C44E52" if v > 0 else "#B0B0B0" for v in d1.lift_over_baseline])
axes[0].axvline(0, color="#333", lw=1)
axes[0].set_xlabel("accuracy lift over the 68.67% baseline")
axes[0].set_title("Ranked by accuracy", fontsize=10.5)

d2 = merged.sort_values("cost_at_optimum", ascending=False)
highlight = {"speeding_violations", "past_accidents"}
axes[1].barh(d2.index, d2.cost_at_optimum,
             color=["#4C72B0" if f in highlight else "#C44E52" for f in d2.index])
axes[1].set_xlabel("expected cost at the optimal threshold (5:1) — lower is better")
axes[1].set_title("Ranked by cost (blue = zero accuracy lift)", fontsize=10.5)

fig.suptitle("The same six features, two defensible metrics, two different answers",
             y=1.02, fontsize=12)
fig.tight_layout()
save_fig(fig, "metric_changes_the_answer")
plt.show()
""")

co("""
merged[["lift_over_baseline", "roc_auc", "optimal_threshold",
        "cost_at_optimum", "cost_at_0.5"]].sort_values("cost_at_optimum").round(4)
""")

md("""
## Figure 3 — the winner's margin is not real

`driving_experience` beat `age` by 0.93 points on validation. The bootstrap interval for
that gap spans zero.
""")

co("""
fig, ax = plt.subplots(figsize=(8, 3.6))
d = ci.sort_values("accuracy")
ax.errorbar(d.accuracy, d.feature,
            xerr=[d.accuracy - d.ci_low, d.ci_high - d.accuracy],
            fmt="o", color="#C44E52", capsize=4, markersize=7, lw=1.6)
ax.axvline(BASELINE, color="#937860", ls="--", lw=1.6, label=f"baseline {BASELINE:.4f}")
ax.set_xlabel("validation accuracy, 95% bootstrap interval (10,000 resamples)")
ax.set_title("Top two overlap. Every interval clears the baseline.", fontsize=11)
ax.legend(fontsize=9)
fig.tight_layout()
save_fig(fig, "results_bootstrap_ci")
plt.show()
""")

md("""
## What the single-feature constraint costs

The client asked for one feature because they cannot deploy anything larger. Here is the
price of that constraint, measured rather than asserted.
""")

co("""
vs_multi.round(4)
""")

co("""
gap_acc = final.loc["all features @ 0.5 (ceiling)", "accuracy"] - final.loc["driving_experience @ 0.5", "accuracy"]
gap_auc = final.loc["all features @ 0.5 (ceiling)", "roc_auc"] - final.loc["driving_experience @ 0.5", "roc_auc"]
c_single = final.loc["driving_experience @ 0.05 (cost-optimal)", "expected_cost_5to1"]
c_multi = final.loc["all features @ 0.16 (cost-optimal)", "expected_cost_5to1"]

print("On the test set, the one-feature constraint costs:")
print(f"  {gap_acc:.4f} accuracy")
print(f"  {gap_auc:.4f} ROC-AUC")
print(f"  {c_single / c_multi - 1:.1%} higher expected cost at 5:1")
""")

md("""
## Summary of every finding

| # | Finding | Evidence |
|---|---|---|
| 1 | The reference's 77.71% is 9 points of signal on a 68.67% floor it never states | class balance, 31.33% positive |
| 2 | Held-out accuracy is 76.47%, lift +7.80, CI [+5.66, +9.93] | test set, scored once |
| 3 | The winner is statistically tied with `age` | bootstrap gap CI [−0.010, +0.029] |
| 4 | Accuracy is blind to nine of sixteen features | zero predicted-positive rate |
| 5 | Two of those nine are top-3 under cost | `past_accidents`, `speeding_violations` |
| 6 | The 0.5 threshold is never optimal | 0.25 at 1:1, 0.05 at 5:1 |
| 7 | The baseline itself flips under asymmetric cost | always-claim 0.687 vs always-no-claim 1.567 |
| 8 | Missingness carries no signal | p = 0.67, 0.091; nothing survives Bonferroni |
| 9 | The imputation leak is worth one row in 1,500 | leaky vs honest refit |
| 10 | Encoding by dtype compared 4-parameter and 2-parameter models | `n_parameters` by scheme |
| 11 | `postal_code` perfectly separates 120 customers | 100% claim rate, coefficient 22.3 |
| 12 | The one-feature constraint costs ~6 accuracy points | 15-feature ceiling |
""")

nb["cells"] = c
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.12.14"}}
out = pathlib.Path.home()/"projects/car-insurance-claim-modeling/notebooks/05_results.ipynb"
nbf.write(nb, str(out)); print("wrote", out, "cells:", len(c))
