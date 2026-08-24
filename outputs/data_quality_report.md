# Data Quality & Preparation Report

## 1. Raw data audit (before cleaning)

- Rows in raw export: **4080**
- Duplicate records found (by content, not id): **80**
- Rows with `active_counters` <= 0 (impossible): **4**
- Rows with `num_people_waiting` < 0 (impossible): **4**
- Rows with `num_people_waiting` > 80 (almost certainly a data-entry error, not a real queue): **4**
- Rows with `actual_waiting_time` < 0 (impossible): **4**
- Missing values by column:
  - `avg_service_time`: 115 missing (2.8%)
  - `priority_customers`: 89 missing (2.2%)
  - `historical_queue_load`: 118 missing (2.9%)
  - `previous_waiting_time`: 87 missing (2.1%)

## 2. Cleaning actions taken

- Removed **80** duplicate rows (matched on all feature columns, ignoring `queue_id` since ids are re-issued on re-submission).
- Removed **16** rows with impossible values that could not be safely repaired (0 counters, negative counts, a >80-people entry, negative wait time). These look like logging/entry errors rather than genuine rare events, so they were dropped instead of imputed - imputing a value for something that's *impossible* would fabricate data.
- Imputed missing values with the column **median** (robust to the right-skew that wait-time-adjacent columns naturally have). Median values used:
  - `avg_service_time` -> 7.5
  - `historical_queue_load` -> 8.7
  - `previous_waiting_time` -> 14.8
  - `priority_customers` -> 0.0
- Rows remaining after cleaning: **3984**

## 3. Feature engineering

- `hour` extracted from `time_of_day` (0-23).
- `time_period` bucketed from `hour` (Night / Morning / Afternoon / Evening).
- `is_weekend` derived from `day_of_week` (kept for EDA/discussion; the model uses the fuller `day_of_week` one-hot encoding instead, since it's a strict superset of the weekend/weekday signal).
- `day_of_week` and `time_period` are one-hot encoded **inside the modeling pipeline** (train.py), not here, so the cleaned CSV stays human-readable and the same pipeline can be reused directly on raw-looking prediction input.

## 4. Target leakage investigation

- Correlation of `previous_waiting_time` with `actual_waiting_time`: **0.544**
- Correlation of `historical_queue_load` with `actual_waiting_time`: **0.245**
- Neither is a copy of the target: both represent information that would genuinely be **known before** the current person's wait is over (the previous customer's already-completed wait, and a recent rolling average). A moderate correlation is *expected and desirable* here - that's what makes them useful predictors. What would signal true leakage is a near-perfect correlation (> ~0.97) or a feature that could only be computed *after* the target is known; neither is the case here. `src/train.py` also runs a quick ablation (with vs without these two columns) to confirm the model still performs reasonably without them, so the pipeline isn't silently over-relying on a single feature.

## 5. Outlier check on the target (`actual_waiting_time`)

- IQR bounds: [-15.8, 65.0] minutes -> **127** rows fall outside this range (3.2% of cleaned data).
- These were **kept**, not removed. Unlike the impossible values above, a wait of 60-140 minutes is a plausible 'system slowdown / short-staffed' event, not a data error. Removing them would teach the model that long waits never happen, which is precisely the case operators most want a warning about. This is discussed further in `REPORT.md`.

## 6. Train / test split

- 80/20 split, `random_state=42` -> **3187** training rows, **797** test rows.
- A plain random split (rather than a time-based split) is appropriate here because each row is an independent queue snapshot, not a strictly ordered time series.
