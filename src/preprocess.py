"""
preprocess.py
--------------
Data preparation stage of the pipeline (Assignment Section 6).

Responsibilities:
  1. Load the raw export and AUDIT it (missing values, duplicates,
     impossible values) - printed and saved as a markdown report.
  2. CLEAN it:
       - drop duplicate records (checked on the meaningful feature columns,
         not the row id - a logging system will happily hand a re-submitted
         record a brand new id, so de-duplicating on id would miss them)
       - drop rows with impossible/unrealistic values that cannot be safely
         corrected (negative counts, 0 active counters, negative target,
         an absurd "500 people waiting" typo-style outlier)
       - impute genuinely missing values (median for numeric fields)
  3. ENGINEER a couple of interpretable features from the raw time value
     (hour, time_period bucket) - Section 6: "Extract useful information
     from time values, such as hour or time period."
  4. Check for TARGET LEAKAGE in previous_waiting_time / historical_queue_load
     and explain the reasoning (Section 6, last bullet).
  5. Split into train/test sets (Section 6, last bullet) and persist both,
     plus the full cleaned dataset for the EDA step.

Run:
    python src/preprocess.py
Inputs:
    data/raw_queue_data.csv
Outputs:
    data/cleaned_queue_data.csv
    data/train.csv
    data/test.csv
    outputs/data_quality_report.md
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw_queue_data.csv"
CLEAN_PATH = "data/cleaned_queue_data.csv"
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
REPORT_PATH = "outputs/data_quality_report.md"

FEATURE_COLS_NO_ID = [
    "num_people_waiting", "queue_length", "active_counters", "avg_service_time",
    "priority_customers", "day_of_week", "time_of_day", "historical_queue_load",
    "previous_waiting_time", "actual_waiting_time",
]


def bucket_time_period(hour: pd.Series) -> pd.Series:
    """Derive a coarse time-of-day bucket straight from the parsed hour.
    Kept independent from generate_data.py on purpose - a real pipeline
    only ever sees the raw hour, not the simulator's internals."""
    return pd.cut(
        hour,
        bins=[-1, 5, 11, 16, 20, 23],
        labels=["Night", "Morning", "Afternoon", "Evening", "Night"],
        ordered=False,
    ).astype(str)


def audit(df: pd.DataFrame) -> dict:
    dup_count = df.duplicated(subset=FEATURE_COLS_NO_ID).sum()
    missing = df.isna().sum()
    missing = missing[missing > 0]
    bad_counters = (df["active_counters"] <= 0).sum()
    bad_people = (df["num_people_waiting"] < 0).sum()
    extreme_people = (df["num_people_waiting"] > 80).sum()
    bad_target = (df["actual_waiting_time"] < 0).sum()
    return {
        "n_rows": len(df),
        "duplicates": int(dup_count),
        "missing": missing,
        "bad_counters": int(bad_counters),
        "bad_people": int(bad_people),
        "extreme_people": int(extreme_people),
        "bad_target": int(bad_target),
    }


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    before = len(df)
    log = {}

    # 1) de-duplicate on content, not on the (possibly re-issued) id
    df = df.drop_duplicates(subset=FEATURE_COLS_NO_ID, keep="first").copy()
    log["removed_duplicates"] = before - len(df)

    # 2) drop rows with impossible values we cannot safely repair
    n0 = len(df)
    df = df[df["active_counters"] > 0]
    df = df[df["num_people_waiting"] >= 0]
    df = df[df["num_people_waiting"] <= 80]  # domain cap - beyond this is a logging error, not a real queue
    df = df[df["actual_waiting_time"] >= 0]
    log["removed_impossible_values"] = n0 - len(df)

    # 3) impute genuinely missing values (median - robust to the skew/outliers
    #    naturally present in wait-time-adjacent columns)
    numeric_missing_cols = ["avg_service_time", "historical_queue_load", "previous_waiting_time", "priority_customers"]
    impute_values = {}
    for col in numeric_missing_cols:
        med = df[col].median()
        impute_values[col] = round(float(med), 2)
        df[col] = df[col].fillna(med)
    # priority_customers is a count -> round back to a whole number after imputation
    df["priority_customers"] = df["priority_customers"].round().astype(int)
    log["impute_values"] = impute_values

    # 4) feature engineering from the raw time value
    df["hour"] = df["time_of_day"].str.slice(0, 2).astype(int)
    df["time_period"] = bucket_time_period(df["hour"])
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])

    df = df.reset_index(drop=True)
    df["queue_id"] = np.arange(1, len(df) + 1)
    return df, log


def leakage_check(df: pd.DataFrame) -> dict:
    corr = df[["previous_waiting_time", "historical_queue_load", "actual_waiting_time"]].corr()["actual_waiting_time"]
    return {
        "corr_previous_waiting_time": round(float(corr["previous_waiting_time"]), 3),
        "corr_historical_queue_load": round(float(corr["historical_queue_load"]), 3),
    }


def iqr_outliers(series: pd.Series) -> tuple[int, float, float]:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((series < lo) | (series > hi)).sum())
    return n_out, round(float(lo), 2), round(float(hi), 2)


def main():
    raw = pd.read_csv(RAW_PATH)
    before_audit = audit(raw)

    cleaned, clean_log = clean(raw)
    leak = leakage_check(cleaned)
    n_out, lo, hi = iqr_outliers(cleaned["actual_waiting_time"])

    cleaned.to_csv(CLEAN_PATH, index=False)

    train_df, test_df = train_test_split(cleaned, test_size=0.2, random_state=42)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    # ---------------- report ----------------
    lines = []
    lines.append("# Data Quality & Preparation Report\n")
    lines.append("## 1. Raw data audit (before cleaning)\n")
    lines.append(f"- Rows in raw export: **{before_audit['n_rows']}**")
    lines.append(f"- Duplicate records found (by content, not id): **{before_audit['duplicates']}**")
    lines.append(f"- Rows with `active_counters` <= 0 (impossible): **{before_audit['bad_counters']}**")
    lines.append(f"- Rows with `num_people_waiting` < 0 (impossible): **{before_audit['bad_people']}**")
    lines.append(f"- Rows with `num_people_waiting` > 80 (almost certainly a data-entry error, not a real queue): **{before_audit['extreme_people']}**")
    lines.append(f"- Rows with `actual_waiting_time` < 0 (impossible): **{before_audit['bad_target']}**")
    lines.append("- Missing values by column:")
    for col, cnt in before_audit["missing"].items():
        lines.append(f"  - `{col}`: {cnt} missing ({cnt/before_audit['n_rows']*100:.1f}%)")

    lines.append("\n## 2. Cleaning actions taken\n")
    lines.append(f"- Removed **{clean_log['removed_duplicates']}** duplicate rows (matched on all feature columns, ignoring `queue_id` since ids are re-issued on re-submission).")
    lines.append(f"- Removed **{clean_log['removed_impossible_values']}** rows with impossible values that could not be safely repaired (0 counters, negative counts, a >80-people entry, negative wait time). These look like logging/entry errors rather than genuine rare events, so they were dropped instead of imputed - imputing a value for something that's *impossible* would fabricate data.")
    lines.append("- Imputed missing values with the column **median** (robust to the right-skew that wait-time-adjacent columns naturally have). Median values used:")
    for col, val in clean_log["impute_values"].items():
        lines.append(f"  - `{col}` -> {val}")
    lines.append(f"- Rows remaining after cleaning: **{len(cleaned)}**")

    lines.append("\n## 3. Feature engineering\n")
    lines.append("- `hour` extracted from `time_of_day` (0-23).")
    lines.append("- `time_period` bucketed from `hour` (Night / Morning / Afternoon / Evening).")
    lines.append("- `is_weekend` derived from `day_of_week` (kept for EDA/discussion; the model uses the fuller `day_of_week` one-hot encoding instead, since it's a strict superset of the weekend/weekday signal).")
    lines.append("- `day_of_week` and `time_period` are one-hot encoded **inside the modeling pipeline** (train.py), not here, so the cleaned CSV stays human-readable and the same pipeline can be reused directly on raw-looking prediction input.")

    lines.append("\n## 4. Target leakage investigation\n")
    lines.append(f"- Correlation of `previous_waiting_time` with `actual_waiting_time`: **{leak['corr_previous_waiting_time']}**")
    lines.append(f"- Correlation of `historical_queue_load` with `actual_waiting_time`: **{leak['corr_historical_queue_load']}**")
    lines.append(
        "- Neither is a copy of the target: both represent information that would genuinely be "
        "**known before** the current person's wait is over (the previous customer's already-completed "
        "wait, and a recent rolling average). A moderate correlation is *expected and desirable* here - "
        "that's what makes them useful predictors. What would signal true leakage is a near-perfect "
        "correlation (> ~0.97) or a feature that could only be computed *after* the target is known; "
        "neither is the case here. `src/train.py` also runs a quick ablation (with vs without these two "
        "columns) to confirm the model still performs reasonably without them, so the pipeline isn't "
        "silently over-relying on a single feature."
    )

    lines.append("\n## 5. Outlier check on the target (`actual_waiting_time`)\n")
    lines.append(f"- IQR bounds: [{lo}, {hi}] minutes -> **{n_out}** rows fall outside this range ({n_out/len(cleaned)*100:.1f}% of cleaned data).")
    lines.append(
        "- These were **kept**, not removed. Unlike the impossible values above, a wait of 60-140 minutes "
        "is a plausible 'system slowdown / short-staffed' event, not a data error. Removing them would teach "
        "the model that long waits never happen, which is precisely the case operators most want a warning "
        "about. This is discussed further in `REPORT.md`."
    )

    lines.append("\n## 6. Train / test split\n")
    lines.append(f"- 80/20 split, `random_state=42` -> **{len(train_df)}** training rows, **{len(test_df)}** test rows.")
    lines.append("- A plain random split (rather than a time-based split) is appropriate here because each row is an independent queue snapshot, not a strictly ordered time series.")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nSaved: {CLEAN_PATH}, {TRAIN_PATH}, {TEST_PATH}, {REPORT_PATH}")


if __name__ == "__main__":
    main()
