"""Tests for the metrics layer, against hand-computed values.

A metrics bug is the most dangerous kind in this project: it would not crash, it
would just produce a results table that is wrong in a plausible direction.
"""

import numpy as np
import pytest

from src.evaluate import classification_metrics, expected_cost


@pytest.fixture
def toy():
    """10 cases, worked out by hand.

    truth: 6 negatives, 4 positives.
    at threshold 0.5 -> tp=3, fn=1, tn=5, fp=1
    """
    y_true = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    y_prob = [0.1, 0.2, 0.3, 0.4, 0.45, 0.7, 0.6, 0.8, 0.9, 0.2]
    return y_true, y_prob


def test_confusion_counts(toy):
    m = classification_metrics(*toy)
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (3, 1, 5, 1)


def test_accuracy_precision_recall(toy):
    m = classification_metrics(*toy)
    assert m["accuracy"] == pytest.approx(0.8)      # (3 + 5) / 10
    assert m["precision"] == pytest.approx(0.75)    # 3 / (3 + 1)
    assert m["recall"] == pytest.approx(0.75)       # 3 / (3 + 1)
    assert m["f1"] == pytest.approx(0.75)


def test_lift_is_accuracy_minus_baseline(toy):
    m = classification_metrics(*toy, baseline=0.6)
    assert m["lift_over_baseline"] == pytest.approx(0.2)
    assert m["beats_baseline"] is True


def test_a_model_that_never_predicts_positive(toy):
    """The nine-feature case from notebook 03: predicts one class, scores the
    baseline, and has zero precision and recall."""
    y_true, _ = toy
    y_prob = [0.1] * 10
    m = classification_metrics(y_true, y_prob, baseline=0.6)
    assert m["predicted_positive_rate"] == 0.0
    assert m["precision"] == 0.0 and m["recall"] == 0.0
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["lift_over_baseline"] == pytest.approx(0.0)
    assert m["beats_baseline"] is False


def test_auc_is_threshold_independent(toy):
    """AUC must not change with the classification threshold. This is the whole
    reason it is in the table."""
    y_true, y_prob = toy
    aucs = {classification_metrics(y_true, y_prob, threshold=t)["roc_auc"]
            for t in (0.1, 0.3, 0.5, 0.7, 0.9)}
    assert len(aucs) == 1


def test_perfect_and_inverted_rankings():
    y_true = [0, 0, 1, 1]
    assert classification_metrics(y_true, [0.1, 0.2, 0.8, 0.9])["roc_auc"] == 1.0
    assert classification_metrics(y_true, [0.9, 0.8, 0.2, 0.1])["roc_auc"] == 0.0


def test_brier_score_on_known_values():
    y_true = [1, 0]
    y_prob = [0.8, 0.3]
    # ((1-0.8)^2 + (0-0.3)^2) / 2 = (0.04 + 0.09) / 2
    assert classification_metrics(y_true, y_prob)["brier"] == pytest.approx(0.065)


def test_threshold_changes_predictions(toy):
    y_true, y_prob = toy
    low = classification_metrics(y_true, y_prob, threshold=0.15)
    high = classification_metrics(y_true, y_prob, threshold=0.85)
    assert low["predicted_positive_rate"] > high["predicted_positive_rate"]
    assert low["recall"] >= high["recall"]


# --- Cost, used in Session 4 ----------------------------------------------

def test_expected_cost_weights_false_negatives(toy):
    y_true, y_prob = toy
    # fp = 1, fn = 1 at threshold 0.5; with a 5x ratio that is (1 + 5) / 10
    assert expected_cost(y_true, y_prob, 0.5, fn_to_fp_ratio=5.0) == pytest.approx(0.6)
    # with symmetric costs it is (1 + 1) / 10
    assert expected_cost(y_true, y_prob, 0.5, fn_to_fp_ratio=1.0) == pytest.approx(0.2)


def test_higher_fn_cost_favours_a_lower_threshold(toy):
    """The premise of the Session 4 analysis: when missing a claim is expensive,
    the cost-minimising threshold moves down."""
    y_true, y_prob = toy
    ratio = 10.0
    costs = {t: expected_cost(y_true, y_prob, t, ratio) for t in np.arange(0.1, 0.95, 0.05)}
    best = min(costs, key=costs.get)
    assert best < 0.5
