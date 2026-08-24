"""
generate_data.py
-----------------
Creates a realistic SYNTHETIC dataset for the Smart Queue Waiting-Time
Predictor assignment.

Why synthetic data?
The assignment explicitly allows a "realistic synthetic dataset" as long as
it has "enough variation to represent busy and quiet periods rather than
generating the target from a simple fixed formula" (see Section 7 of the
brief). A public, ready-made "queue waiting time" dataset with the exact
feature set requested (counters, priority customers, day/time, historical
load, previous wait, etc.) does not really exist off the shelf, so a
carefully designed simulator is the more honest choice than forcing a
mismatched public dataset into this problem.

Design approach (documented for the report / candidate questions):
  * A hidden "business_level" (how busy the branch is right now) drives
    several features at once, the way a real queue would behave. This
    avoids a "simple fixed formula" -> target relationship because many
    features are only INDIRECTLY related to the target through this shared
    latent factor, plus every row also gets its own independent noise.
  * The target (actual_waiting_time) is produced from a multi-factor
    formula: base queueing math (people ahead x service time / counters),
    a priority-customer penalty, peak-hour and day-of-week multipliers, a
    small effect from recent historical load, a small "momentum" effect
    from the previous customer's wait time, random per-row noise, and a
    rare random "slowdown" event (~2% of rows) that produces realistic
    outliers for the EDA/outlier-handling step.
  * previous_waiting_time and historical_queue_load are generated from the
    SAME underlying business_level but with their OWN independent noise
    draw - they are correlated with the target (as a real predictive
    feature should be) without being a copy of it. This is what lets the
    "target leakage" investigation in preprocess.py be a real analysis
    rather than a formality.
  * Data-quality problems are injected on purpose (missing values,
    duplicate rows, impossible values such as 0 counters or negative
    waiting time) so that the "Data Preparation" section of the assignment
    has real, non-trivial work to do.

Run:
    python src/generate_data.py
Output:
    data/raw_queue_data.csv
"""

import argparse
import numpy as np
import pandas as pd

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Mild day-of-week demand multiplier (kept close to 1.0 - a queue system is
# generic across hospitals/banks/restaurants/govt offices, so day effects
# are modest rather than dramatic; documented as an assumption in REPORT.md)
DAY_MULTIPLIER = {
    "Monday": 1.10,
    "Tuesday": 1.00,
    "Wednesday": 0.97,
    "Thursday": 1.00,
    "Friday": 1.08,
    "Saturday": 1.15,
    "Sunday": 0.90,
}


def peak_hour_multiplier(hour: np.ndarray) -> np.ndarray:
    """Smooth-ish peak effect around late morning and early evening."""
    lunch_peak = np.exp(-0.5 * ((hour - 12.5) / 1.6) ** 2)
    evening_peak = np.exp(-0.5 * ((hour - 18) / 1.8) ** 2)
    night_lull = np.where((hour < 7) | (hour >= 22), -0.25, 0.0)
    return 1.0 + 0.35 * lunch_peak + 0.30 * evening_peak + night_lull


def time_of_day_bucket(hour: np.ndarray) -> np.ndarray:
    bins = [-1, 5, 11, 16, 20, 23]
    labels = ["Night", "Morning", "Afternoon", "Evening", "Night"]
    # two "Night" labels (wrap-around) -> handle with np.select instead of pd.cut labels reuse
    return np.select(
        [
            (hour <= 5),
            (hour >= 6) & (hour <= 11),
            (hour >= 12) & (hour <= 16),
            (hour >= 17) & (hour <= 20),
            (hour >= 21),
        ],
        ["Night", "Morning", "Afternoon", "Evening", "Night"],
    )


def generate(n_rows: int, seed: int, missing_frac: float, dup_frac: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---- time context -----------------------------------------------
    day_of_week = rng.choice(DAYS, size=n_rows)
    # bias hours toward operating hours (7-21) with some 24/7 activity
    hour_choices = np.arange(24)
    hour_weights = np.where((hour_choices >= 7) & (hour_choices <= 21), 3.0, 0.6)
    hour_weights = hour_weights / hour_weights.sum()
    hour = rng.choice(hour_choices, size=n_rows, p=hour_weights)
    minute = rng.integers(0, 60, size=n_rows)
    time_of_day = [f"{h:02d}:{m:02d}" for h, m in zip(hour, minute)]

    day_mult = np.array([DAY_MULTIPLIER[d] for d in day_of_week])
    peak_mult = peak_hour_multiplier(hour.astype(float))

    # ---- hidden "how busy is it right now" latent factor -------------
    business_level = np.clip(
        day_mult * peak_mult * rng.normal(1.0, 0.18, size=n_rows), 0.3, None
    )

    # ---- queue-state features driven (partly) by business_level ------
    base_lambda = 7.5
    num_people_waiting = rng.poisson(np.clip(base_lambda * business_level, 0.2, None))
    num_people_waiting = np.clip(num_people_waiting, 0, 45)

    active_counters = np.clip(
        np.round(rng.normal(2.0 + 1.6 * business_level, 0.9, size=n_rows)), 1, 6
    ).astype(int)

    avg_service_time = np.clip(rng.normal(7.5, 2.6, size=n_rows), 2, 20)

    priority_customers = rng.poisson(np.clip(0.55 * business_level, 0.05, None))
    priority_customers = np.clip(priority_customers, 0, 6)

    queue_length = num_people_waiting + active_counters + rng.poisson(1.0, size=n_rows)

    historical_queue_load = np.clip(
        base_lambda * business_level + rng.normal(0, 2.2, size=n_rows), 0, None
    )

    # ---- target: actual waiting time ----------------------------------
    base_wait = (num_people_waiting * avg_service_time) / active_counters
    priority_extra = priority_customers * 0.5 * avg_service_time
    hist_effect = 0.15 * (historical_queue_load - base_lambda)

    # previous_waiting_time: represents the PREVIOUS customer's already-
    # completed wait. It shares the same day/hour context (day_mult,
    # peak_mult) and this row's service-time/staffing setup (those change
    # slowly - same shift, same staff) but gets its OWN independent draw of
    # "how busy exactly" (business_level_recent) rather than reusing this
    # row's exact num_people_waiting/priority realization, which changes
    # customer to customer. That keeps it a genuinely separate observation
    # (moderately correlated with the target, not a restatement of it).
    business_level_recent = np.clip(
        day_mult * peak_mult * rng.normal(1.0, 0.22, size=n_rows), 0.3, None
    )
    previous_base_wait = (base_lambda * business_level_recent * avg_service_time) / active_counters
    prev_noise = rng.normal(0, 6.0, size=n_rows)
    previous_waiting_time = np.clip(previous_base_wait * 0.85 + prev_noise, 0, None)

    momentum = 0.12 * previous_waiting_time
    noise = rng.normal(0, 2.3, size=n_rows)

    # rare slowdown events -> realistic outliers for the EDA section
    slowdown_mask = rng.random(n_rows) < 0.02
    slowdown = np.where(slowdown_mask, rng.uniform(15, 42, size=n_rows), 0.0)

    actual_waiting_time = (
        base_wait * peak_mult * day_mult + priority_extra + hist_effect + momentum + noise + slowdown
    )
    actual_waiting_time = np.clip(actual_waiting_time, 0, None)

    df = pd.DataFrame(
        {
            "queue_id": np.arange(1, n_rows + 1),
            "num_people_waiting": num_people_waiting,
            "queue_length": queue_length,
            "active_counters": active_counters,
            "avg_service_time": np.round(avg_service_time, 1),
            "priority_customers": priority_customers,
            "day_of_week": day_of_week,
            "time_of_day": time_of_day,
            "historical_queue_load": np.round(historical_queue_load, 1),
            "previous_waiting_time": np.round(previous_waiting_time, 1),
            "actual_waiting_time": np.round(actual_waiting_time, 1),
        }
    )

    # ------------------------------------------------------------------
    # Intentionally inject data-quality problems so preprocess.py has
    # real cleaning work to do (mirrors what a real operational export
    # would look like).
    # ------------------------------------------------------------------
    rng2 = np.random.default_rng(seed + 1)

    # 1) Missing values scattered across a few plausible columns
    missing_cols = ["avg_service_time", "historical_queue_load", "previous_waiting_time", "priority_customers"]
    for col in missing_cols:
        mask = rng2.random(n_rows) < missing_frac
        df.loc[mask, col] = np.nan

    # 2) Duplicate rows (simulate double-logging at source)
    n_dupes = int(n_rows * dup_frac)
    if n_dupes > 0:
        dupe_rows = df.sample(n=n_dupes, random_state=seed).copy()
        df = pd.concat([df, dupe_rows], ignore_index=True)

    # 3) A handful of impossible / unrealistic values
    n_bad = max(6, int(n_rows * 0.004))
    bad_idx = rng2.choice(df.index, size=n_bad, replace=False)
    for i, idx in enumerate(bad_idx):
        kind = i % 4
        if kind == 0:
            df.loc[idx, "active_counters"] = 0  # impossible: no counters open
        elif kind == 1:
            df.loc[idx, "num_people_waiting"] = -rng2.integers(1, 5)  # negative count
        elif kind == 2:
            df.loc[idx, "actual_waiting_time"] = -round(float(rng2.uniform(1, 10)), 1)  # negative wait
        else:
            df.loc[idx, "num_people_waiting"] = int(rng2.integers(300, 600))  # absurd outlier

    # shuffle rows so duplicates aren't all at the bottom (more realistic)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["queue_id"] = np.arange(1, len(df) + 1)

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic queue waiting-time dataset")
    parser.add_argument("--n_rows", type=int, default=4000, help="number of clean rows to simulate before issues are injected")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing_frac", type=float, default=0.025, help="fraction of rows with a missing value per affected column")
    parser.add_argument("--dup_frac", type=float, default=0.02, help="fraction of extra duplicate rows to add")
    parser.add_argument("--out", type=str, default="data/raw_queue_data.csv")
    args = parser.parse_args()

    df = generate(args.n_rows, args.seed, args.missing_frac, args.dup_frac)
    df.to_csv(args.out, index=False)

    print(f"Generated {len(df)} rows -> {args.out}")
    print(f"  (base rows: {args.n_rows}, includes injected duplicates + data-quality issues)")
    print("\nColumn overview:")
    print(df.dtypes)
    print("\nSample rows:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
