"""Project-wide constants.

Every number that could silently change a result lives here, so that a reader
can audit the experimental setup without reading the pipeline.
"""

from pathlib import Path

# --- Reproducibility -------------------------------------------------------

SEED = 42

# --- Paths -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "car_insurance.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"

# --- Data contract ---------------------------------------------------------

TARGET = "outcome"
ID_COL = "id"

EXPECTED_ROWS = 10_000

# Column order as it appears in the raw CSV. load_raw() asserts an exact match,
# so a silently reshaped upstream file fails loudly instead of shifting a column.
EXPECTED_COLUMNS = [
    "id", "age", "gender", "driving_experience", "education", "income",
    "credit_score", "vehicle_ownership", "vehicle_year", "married", "children",
    "postal_code", "annual_mileage", "vehicle_type", "speeding_violations",
    "duis", "past_accidents", "outcome",
]

FEATURES = [c for c in EXPECTED_COLUMNS if c not in (ID_COL, TARGET)]

# --- Feature taxonomy ------------------------------------------------------
#
# This grouping is the fix for defect D4 in PLAN.md. The reference notebook let
# pandas dtypes decide how each feature entered the model: string columns were
# silently expanded into dummy variables while integer-coded columns entered as
# a single linear term. Naming the semantics here, rather than inferring them
# from storage type, is what makes the feature comparison fair.

# Ordered categories. The ordering is the real-world one, not alphabetical.
ORDINAL_LEVELS = {
    "age": [0, 1, 2, 3],  # 16-25, 26-39, 40-64, 65+
    "driving_experience": ["0-9y", "10-19y", "20-29y", "30y+"],
    "education": ["none", "high school", "university"],
    "income": ["poverty", "working class", "middle class", "upper class"],
}

# Unordered categories.
NOMINAL = ["gender", "vehicle_year", "vehicle_type", "postal_code"]

# Already 0/1.
BINARY = ["vehicle_ownership", "married", "children"]

# Non-negative integer counts.
COUNTS = ["speeding_violations", "duis", "past_accidents"]

# Continuous, and the only two columns with missing values.
CONTINUOUS = ["credit_score", "annual_mileage"]
COLUMNS_WITH_MISSING = ["credit_score", "annual_mileage"]

# --- Splitting -------------------------------------------------------------

TEST_SIZE = 0.15
VAL_SIZE = 0.15  # as a fraction of the full dataset, not of the remainder

# --- Evaluation ------------------------------------------------------------

# Cost of a false negative (a missed claim) relative to a false positive
# (a good customer wrongly flagged). Session 4 reports how the chosen threshold
# moves as this ratio changes; it is a stated assumption, not a measured fact.
FN_TO_FP_COST_RATIO = 5.0
