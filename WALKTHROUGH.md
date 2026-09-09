# Walkthrough

The narrative version of this project: what I set out to do, what surprised me, what I
got wrong along the way, and what I would do differently. The README reports the results.
This file reports the process.

---

## Why rebuild someone else's finished project?

I started from a public solution to the DataCamp guided project *Modeling Car Insurance
Claim Outcomes*, by AchrafSL. It is about forty lines of clean pandas and statsmodels, it
runs, and it reaches a defensible conclusion: `driving_experience` is the best single
predictor of a claim, at 77.71% accuracy.

Copying that and putting it on a résumé would have taught me nothing. Instead I asked a
narrower question: **is 77.71% a good number?**

That question turned out to have a whole project inside it.

---

## The first thing I checked, and why it reframed everything

Before writing any modelling code I computed the class balance. The target is 31.33%
positive, so predicting "no claim" for every customer scores **68.67%**.

The reference reports 77.71% and never mentions 68.67%. A reader has no way to tell
whether the model found ten points of signal or none at all.

That single comparison set the shape of the whole project. I was not going to build a
better model — the brief explicitly asks for a simple one. I was going to build the
**evaluation the reference is missing**, and report honestly whatever survived it.

I wrote the plan around seven specific defects before touching the data. All seven are
listed in `PLAN.md`; the interesting ones are below.

---

## What I expected to find, and what actually happened

### I expected the leakage fix to matter. It did not.

The reference computes imputation medians over all 10,000 rows, including future test
rows. Textbook leakage. I built a train-fitted imputer, then measured the difference.

The medians differ by **0.001** on `credit_score` and by exactly zero on
`annual_mileage`. Downstream, the leak flips **one validation prediction out of 1,500**.

I could have written this up as a serious flaw. It is not one, here. Ten thousand rows
and a stable median make it immaterial, and saying otherwise would have been dressing up
a principle as a finding. What I wrote instead is that you cannot know it is immaterial
without measuring it — and that the same defect on a smaller sample or a group-wise
imputation would not be so forgiving.

This was the first of several places where the honest answer was less dramatic than the
one I was hoping for.

### I expected the encoding fix to change the ranking. It did not.

The reference's biggest defect, to my eye, was that it compares features on raw accuracy
while letting pandas dtypes decide how each one enters the model. `driving_experience` is
a string, so statsmodels silently expands it into 3 dummy variables. `age` is an integer,
so it enters as 1 linear term. A 4-parameter model is compared against a 2-parameter
model and declared the winner.

I built an explicit feature taxonomy, implemented both encoding schemes uniformly, and
re-ran everything expecting the ranking to shift.

Both schemes produced **identical accuracy for every feature** and the same ordering.

Working out why was more interesting than the fix. The probabilities genuinely differ —
up to 0.050 apart on `age` — but for each of these features only one level sits above a
50% claim rate, and both encodings agree on which one. Every customer lands on the same
side of the threshold, so accuracy records nothing.

So D4 was a real defect that the client's chosen metric is structurally incapable of
detecting. AIC detects it fine: `age` gains 53.5 AIC points from dummy encoding.

### The one prediction I made in advance that came true

In notebook 02, before fitting any single-feature models, I plotted log-odds across each
ordinal's levels and noted that `income` and `education` were near-straight while `age`
and `driving_experience` bent. Linear encoding assumes straightness, so I predicted dummy
encoding would pay for itself on the curved features and waste parameters on the straight
ones.

Notebook 03 confirmed it: `age` +53.5 AIC, `driving_experience` +0.7, `income` −2.5,
`education` −1.2. Exactly the predicted pattern.

I mention it because the alternative — noticing the pattern in the results and
retrofitting an explanation — is much easier and worth much less.

---

## The two things I did not see coming

### A postal code that is not a feature

While computing claim rates by category in notebook 01, one row stopped me. Postal code
21217: **120 customers, 120 claims. A 100% claim rate.** In the training, validation and
test splits alike.

I checked whether those customers were simply high-risk. They are not — average credit
score 0.54, average mileage 12,098, an age distribution close to everyone else's. Nothing
about them explains it. It is an artifact of how this synthetic dataset was generated.

Mechanically it causes perfect separation: the maximum-likelihood coefficient diverges to
22.3 on the log-odds scale — an odds ratio of about 4.7 billion — and statsmodels emits a
`ConvergenceWarning` that the reference's loop would swallow silently.

As a model it looks superb and is worthless: precision **1.000**, recall **0.045**. It
flags only the 21217 customers and is never wrong about them, because in this dataset
that postal code *is* the outcome. It clears the baseline by 1.4 points and would fail
the instant it met a real customer from Baltimore who did not claim.

There is a second, quieter bug hiding underneath. `postal_code` is stored as an integer,
so the reference's dtype-driven pipeline fits a single linear slope across the values
10238, 21217, 32765 and 92101. Those are labels. That arithmetic is meaningless. It is
the same defect as the dummy-expansion asymmetry, wearing a different hat — and my
feature taxonomy closes it off by making the nominal encoding the only one expressible.

### The winner does not actually win

This is the finding I was least prepared for.

`driving_experience` scored 78.80% on validation and `age` scored 77.87%. I was ready to
report the first as the answer. Then I ran a paired bootstrap on the gap.

**95% CI: [−0.010, +0.029].** It spans zero. `driving_experience` came out ahead in 82%
of 10,000 resamples, which sounds comfortable until you notice that a fair coin clears
80% about as often as this data separates these two features.

The lift over baseline, by contrast, is unambiguous: CI [+0.081, +0.121], ahead in 10,000
resamples out of 10,000.

So the defensible claim is not "driving experience is the best predictor." It is
**"driving experience and age are statistically indistinguishable, and both clearly beat
the baseline."**

In hindsight this was foreshadowed. Notebook 01 found that `age` spreads the outcome
slightly *wider* than `driving_experience` (0.627 vs 0.612), that their Cramér's V is
0.668, and that the crosstab between them is triangular — a 16-25 year old cannot have 30
years of driving experience. Two features encoding largely the same underlying fact
should be hard to separate. I noticed the association early and still expected a clean
winner.

---

## The result that justifies the whole exercise

The reference optimises accuracy at a threshold of 0.5. I replaced that with expected
cost under a stated false-negative-to-false-positive ratio, and the picture inverted.

Nine of sixteen features score **exactly** 68.67% accuracy with a predicted-positive rate
of zero — they classify every customer as "no claim." Two of those nine,
`speeding_violations` (AUC 0.732) and `past_accidents` (AUC 0.726), rank customers by
risk better than two features that *do* beat the accuracy baseline. They were penalised
purely for never crossing an arbitrary threshold.

Score them on cost at 5:1 and they place **second and third**, ahead of `age` and
`income`.

Three further things fell out of that analysis:

- **The optimal threshold is never 0.5** — it is 0.25 even at symmetric cost, because the
  classes are imbalanced, and 0.05 at 5:1.
- **At the cost-optimal threshold, accuracy drops to 61.3%**, below the baseline. A model
  tuned for business cost looks worse on the client's chosen metric. Both numbers are
  correct; they answer different questions.
- **The baseline itself flips.** Under 5:1, "always predict claim" costs 0.687 while
  "always predict no claim" costs 1.567. The majority-class yardstick I had been anchored
  to for four sessions is not a fixed property of the data.

Tuning the threshold turns out to be worth more than choosing the feature.

---

## Mistakes I made building this

**I wrote a claim before checking it.** In the first draft of notebook 01 I wrote that
`driving_experience` "separates the outcome more sharply than anything else." Then the
spread table came back with `age` marginally ahead, 0.627 to 0.612. I had written the
sentence I expected rather than the one the data supported. Corrected, and the corrected
version is a better story — it is why the Session 4 bootstrap was worth running.

**I shipped a figure that silently dropped a feature.** The claim-rate grid was 3×3 for
ten features, and `zip` truncated without complaint, so `children` never appeared. The
irony of a silent truncation bug in a project about silent methodological bugs was not
lost on me. Fixed by sizing the grid from the feature list, with an assert.

**My reproducibility script did not reproduce.** Wiping `data/processed/` and re-running
`scripts/run_experiment.py` produced one table that differed from the notebook's version
— same values, different column order. Trivial, and exactly the kind of drift that
becomes untrustworthy when it is not caught. All 11 tables are now byte-identical after a
full wipe.

**Two result tables were not in the repository at all.** The `.gitignore` allowlist
matched `results_*.csv`, but notebook 01 wrote `baseline.csv` and
`eda_feature_spread.csv`. Both were silently untracked, so a fresh clone would have been
missing the baseline — the single most important number in the project. Renamed to fit
the convention.

---

## What I would do with more time

- **Firth's penalised logistic regression for `postal_code`.** I detect the separation and
  withhold the AIC, which is honest but conservative. Firth's correction would produce a
  usable coefficient estimate instead of a divergent one.
- **Calibration curves.** The Brier score is in the results table but I never plotted
  reliability. The threshold analysis assumes the probabilities mean what they say, and I
  did not verify that as carefully as I would like.
- **Repeated splits.** Every number here comes from one seeded 70/15/15 split. Repeating
  across seeds would separate genuine effects from split luck — a lesson I took from an
  earlier project where a promising result turned out to be seed variance.
- **A real cost ratio.** The 5:1 figure is a stated assumption, not a measured one. With
  actual claim severities and acquisition costs, the threshold analysis would produce a
  number an insurer could act on rather than a demonstration of method.

---

## What I would tell the client

Use `driving_experience`. It will be right about 76% of the time against a 69% floor, so
it is genuinely earning roughly seven points. `age` would work about as well and you
should pick between them on which is cheaper to collect, not on the accuracy figures.

Do not use 0.5 as your cut-off. Work out what a missed claim costs you relative to a
false alarm, and set the threshold from that. It will matter more than the feature choice.

Do not use `postal_code`, however good it looks.

And if you can ever run more than one variable, doing so is worth about six accuracy
points and 20% lower expected cost on held-out data. That is the price of the constraint — whether it is
worth paying is your call, but now it has a number attached.
