"""Tests for the two claims the README makes about its own trustworthiness:
that the environment is pinned, and that the headline does not rest on the
synthetic-data artifact.
"""

import pandas as pd
import pytest

from src import config as cfg
from src.data import load_raw
from src.environment import REFERENCE, as_frame, current
from src.sensitivity import ARTIFACT_POSTAL_CODE, artifact_sensitivity


# --- Environment -----------------------------------------------------------

def test_reference_environment_covers_every_result_critical_package():
    """Anything that can move a number must be pinned. matplotlib is included
    because figures are committed; pytest is not, because it cannot change a
    result."""
    for pkg in ("python", "numpy", "pandas", "scipy", "statsmodels", "scikit-learn"):
        assert pkg in REFERENCE


def test_no_result_critical_package_is_missing_from_the_install():
    installed = current()
    missing = [p for p, v in installed.items() if v == "MISSING"]
    assert not missing, f"not installed: {missing}"


def test_environment_frame_flags_mismatches():
    frame = as_frame()
    assert set(frame.columns) == {"package", "recorded", "installed", "matches"}
    assert len(frame) == len(REFERENCE)
    # matches must be a real comparison, not a constant
    assert frame["matches"].dtype == bool


def test_lock_file_pins_every_direct_dependency():
    """requirements.txt carries lower bounds; requirements-lock.txt is what makes
    the byte-identical claim true. Every direct dependency must appear in it."""
    import re
    direct = [re.split(r"[><=]", line)[0].strip().lower()
              for line in (cfg.ROOT / "requirements.txt").read_text().splitlines()
              if line.strip() and not line.startswith("#")]
    locked = {re.split(r"[><=]", line)[0].strip().lower()
              for line in (cfg.ROOT / "requirements-lock.txt").read_text().splitlines()
              if line.strip() and not line.startswith("#")}
    assert direct, "requirements.txt parsed as empty"
    assert not [d for d in direct if d not in locked]


# --- Sensitivity to the artifact -------------------------------------------

@pytest.fixture(scope="module")
def sensitivity():
    return artifact_sensitivity(load_raw())


def test_the_artifact_rows_are_what_we_think_they_are():
    """All 120 customers in this postal code claimed. If that ever stops being
    true, the sensitivity analysis is answering a different question."""
    df = load_raw()
    sub = df[df["postal_code"] == ARTIFACT_POSTAL_CODE]
    assert len(sub) == 120
    assert sub[cfg.TARGET].mean() == 1.0


def test_removing_the_artifact_removes_exactly_those_rows():
    df = load_raw()
    without = df[df["postal_code"] != ARTIFACT_POSTAL_CODE]
    assert len(without) == len(df) - 120
    assert ARTIFACT_POSTAL_CODE not in set(without["postal_code"])


def test_headline_survives_removing_the_artifact(sensitivity):
    """The README states the result does not depend on the artifact. This is that
    claim as an assertion: the lift must stay clearly positive and must not move
    by more than a point."""
    _, wide = sensitivity
    de = wide.loc["driving_experience"]
    assert de["lift_over_baseline__artifact removed"] > 0.05
    assert abs(de["lift_change"]) < 0.01


def test_removing_the_artifact_does_not_hurt_ranking_quality(sensitivity):
    """AUC should not fall when a degenerate group is removed -- if it did, the
    artifact was propping up the result."""
    _, wide = sensitivity
    assert wide.loc["driving_experience", "auc_change"] > 0


def test_sensitivity_reports_both_datasets(sensitivity):
    long, _ = sensitivity
    assert set(long["dataset"]) == {"full dataset", "artifact removed"}
    # Removing 120 all-claim rows must lower the claim rate, raising the baseline.
    full = long[long.dataset == "full dataset"]["baseline"].iloc[0]
    without = long[long.dataset == "artifact removed"]["baseline"].iloc[0]
    assert without > full
