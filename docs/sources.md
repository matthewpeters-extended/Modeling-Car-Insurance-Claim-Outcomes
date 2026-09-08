# Sources

Every external input to this project, what it contributed, and what we did not take from it.

## 1. Reference implementation (the starting point)

- **URL:** https://github.com/AchrafSL/Modeling-Car-Insurance-Claim-Outcomes-DataCamp
- **Author:** AchrafSL
- **Local copy:** `docs/reference_solution.ipynb` (unmodified, for diffing against our work)
- **What we took:** the problem statement, the dataset, and the data dictionary.
- **What we did not take:** the evaluation methodology. See `PLAN.md` §1 for the
  seven defects we inherit and fix. Our numbers are not expected to match theirs,
  and the README explains why.

## 2. Dataset

- **File:** `data/raw/car_insurance.csv`
- **Provenance:** distributed with the DataCamp guided project
  "Modeling Car Insurance Claim Outcomes"; mirrored in the reference repo above.
- **Shape:** 10,000 rows x 18 columns.
- **Target:** `outcome` (1 = filed a claim during the policy period, 0 = did not).
- **Class balance:** 31.33% positive, so the majority-class baseline accuracy is
  **68.67%**. This number is the yardstick for every result we report.
- **Missingness:** `credit_score` 9.82%, `annual_mileage` 9.57%. No other column
  has missing values.
- **Licence:** DataCamp project data, redistributed by the reference repo. Used
  here for non-commercial portfolio work with attribution.

## 3. Domain framing

- Accenture, "Machine Learning in Insurance" — cited in the original DataCamp
  brief as background on why insurers model claim likelihood. Used only for the
  framing paragraph in the README; no data or method comes from it.

## Data dictionary

| Column | Description |
|---|---|
| `id` | Unique client identifier. Dropped before modeling. |
| `age` | Ordinal bucket: 0 = 16-25, 1 = 26-39, 2 = 40-64, 3 = 65+ |
| `gender` | 0 = Female, 1 = Male |
| `driving_experience` | String bucket: `0-9y`, `10-19y`, `20-29y`, `30y+` |
| `education` | String: `none`, `high school`, `university` |
| `income` | String: `poverty`, `working class`, `middle class`, `upper class` |
| `credit_score` | Float in [0, 1]. 9.82% missing. |
| `vehicle_ownership` | 0 = financing, 1 = owns outright |
| `vehicle_year` | String: `before 2015`, `after 2015` |
| `married` | 0 = no, 1 = yes |
| `children` | 0 = no, 1 = yes |
| `postal_code` | Integer code. High cardinality; treated as categorical. |
| `annual_mileage` | Miles driven per year. 9.57% missing. |
| `vehicle_type` | String: `sedan` (95.2%), `sports car` (4.8%) |
| `speeding_violations` | Count |
| `duis` | Count of driving-under-influence incidents |
| `past_accidents` | Count |
| `outcome` | **Target.** 1 = made a claim, 0 = did not. |
