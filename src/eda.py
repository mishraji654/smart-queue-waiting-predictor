"""
eda.py
------
Exploratory Data Analysis (Assignment Section 11).

Generates plots tied directly to the questions the brief asks about:
  1. Distribution of waiting times
  2. Waiting times across days of the week
  3. Waiting times across time-of-day
  4. Queue size vs waiting time
  5. Active counters vs waiting time
  6. A correlation heatmap (bonus - ties the whole feature set together)
  7. A short outlier note (detailed numbers already live in
     outputs/data_quality_report.md; this just visualizes it)

Run:
    python src/eda.py
Input:
    data/cleaned_queue_data.csv
Output:
    outputs/plots/*.png
    outputs/eda_summary.md
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")

DATA_PATH = "data/cleaned_queue_data.csv"
PLOTS_DIR = "outputs/plots"
SUMMARY_PATH = "outputs/eda_summary.md"

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PERIOD_ORDER = ["Morning", "Afternoon", "Evening", "Night"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/{name}.png", dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_PATH)
    notes = []
    notes.append("# Exploratory Data Analysis Summary\n")
    notes.append(f"Cleaned dataset: **{len(df)} rows**, **{df.shape[1]} columns**.\n")

    # 1. Distribution of waiting times ---------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(df["actual_waiting_time"], bins=40, kde=True, ax=ax, color="#4C72B0")
    ax.set_title("Distribution of Actual Waiting Time")
    ax.set_xlabel("Waiting time (minutes)")
    ax.set_ylabel("Number of queue records")
    save(fig, "01_waiting_time_distribution")

    skew = df["actual_waiting_time"].skew()
    notes.append("## 1. Distribution of waiting times")
    notes.append(
        f"- Mean **{df['actual_waiting_time'].mean():.1f} min**, median **{df['actual_waiting_time'].median():.1f} min**, "
        f"std **{df['actual_waiting_time'].std():.1f} min**. Skew = {skew:.2f} -> right-skewed, as expected for a wait-time "
        "metric (most waits are short-to-moderate, with a long tail of slow/busy episodes). This is why MAE/RMSE are "
        "reported alongside the median rather than assuming a symmetric, normal-like target.\n"
    )

    # 2. Waiting time by day of week ------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=df, x="day_of_week", y="actual_waiting_time", order=DAY_ORDER, ax=ax)
    ax.set_title("Waiting Time by Day of Week")
    ax.set_xlabel("")
    ax.set_ylabel("Waiting time (minutes)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save(fig, "02_waiting_time_by_day")

    day_means = df.groupby("day_of_week")["actual_waiting_time"].mean().reindex(DAY_ORDER)
    notes.append("## 2. Waiting time by day of week")
    notes.append(
        f"- Highest average wait: **{day_means.idxmax()}** ({day_means.max():.1f} min). "
        f"Lowest: **{day_means.idxmin()}** ({day_means.min():.1f} min). "
        "The spread is modest rather than dramatic, which matches how the synthetic data was generated "
        "(a mild day-of-week demand effect on top of a much stronger queue-load/staffing effect) - "
        "day-of-week alone is a weak predictor and should not be over-weighted.\n"
    )

    # 3. Waiting time by time period / hour ------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(data=df, x="time_period", y="actual_waiting_time", order=PERIOD_ORDER, ax=axes[0])
    axes[0].set_title("Waiting Time by Time-of-Day Bucket")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Waiting time (minutes)")

    hourly = df.groupby("hour")["actual_waiting_time"].mean()
    sns.lineplot(x=hourly.index, y=hourly.values, marker="o", ax=axes[1], color="#DD8452")
    axes[1].set_title("Average Waiting Time by Hour")
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("Average waiting time (minutes)")
    axes[1].set_xticks(range(0, 24, 2))
    save(fig, "03_waiting_time_by_time")

    peak_hour = hourly.idxmax()
    notes.append("## 3. Waiting time by time of day")
    notes.append(
        f"- Clear peaks are visible around midday and early evening, with hour **{peak_hour}:00** showing the highest "
        "average wait in this sample. This lines up with typical lunch-hour and after-work congestion in real queueing "
        "systems (hospitals, banks, service counters), which is what the peak-hour multiplier in `generate_data.py` "
        "was designed to mimic.\n"
    )

    # 4. Queue size vs waiting time ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.scatterplot(data=df.sample(min(1200, len(df)), random_state=1), x="num_people_waiting", y="actual_waiting_time",
                     alpha=0.35, ax=axes[0], s=18, color="#4C72B0")
    sns.regplot(data=df, x="num_people_waiting", y="actual_waiting_time", scatter=False, ax=axes[0], color="black", line_kws={"linewidth": 1.5})
    axes[0].set_title("People Waiting vs Waiting Time")
    axes[0].set_xlabel("Number of people waiting")
    axes[0].set_ylabel("Waiting time (minutes)")

    sns.scatterplot(data=df.sample(min(1200, len(df)), random_state=1), x="queue_length", y="actual_waiting_time",
                     alpha=0.35, ax=axes[1], s=18, color="#55A868")
    sns.regplot(data=df, x="queue_length", y="actual_waiting_time", scatter=False, ax=axes[1], color="black", line_kws={"linewidth": 1.5})
    axes[1].set_title("Queue Length vs Waiting Time")
    axes[1].set_xlabel("Total queue length")
    axes[1].set_ylabel("Waiting time (minutes)")
    save(fig, "04_queue_size_vs_waiting_time")

    corr_people = df["num_people_waiting"].corr(df["actual_waiting_time"])
    corr_qlen = df["queue_length"].corr(df["actual_waiting_time"])
    notes.append("## 4. Queue size vs waiting time")
    notes.append(
        f"- Correlation with `num_people_waiting`: **{corr_people:.2f}**. Correlation with `queue_length`: **{corr_qlen:.2f}**. "
        "Both show a clear positive, roughly linear-ish relationship, confirming queue size is one of the strongest "
        "single predictors - as expected for a queueing problem.\n"
    )

    # 5. Active counters vs waiting time -----------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=df, x="active_counters", y="actual_waiting_time", ax=ax, color="#8172B2")
    ax.set_title("Waiting Time by Number of Active Counters")
    ax.set_xlabel("Active counters")
    ax.set_ylabel("Waiting time (minutes)")
    save(fig, "05_counters_vs_waiting_time")

    counter_means = df.groupby("active_counters")["actual_waiting_time"].mean()
    notes.append("## 5. Active counters vs waiting time")
    notes.append(
        f"- Average wait drops from **{counter_means.iloc[0]:.1f} min** with {counter_means.index[0]} counter(s) open to "
        f"**{counter_means.iloc[-1]:.1f} min** with {counter_means.index[-1]} open, a clear and intuitive inverse "
        "relationship - more staffed counters -> shorter waits.\n"
    )

    # 6. Correlation heatmap ------------------------------------------------
    num_cols = ["num_people_waiting", "queue_length", "active_counters", "avg_service_time",
                "priority_customers", "historical_queue_load", "previous_waiting_time", "hour", "actual_waiting_time"]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap (Numeric Features)")
    save(fig, "06_correlation_heatmap")
    notes.append("## 6. Correlation heatmap")
    notes.append("- See `06_correlation_heatmap.png`. Queue size and counters dominate; time/priority/history contribute smaller, complementary signal.\n")

    # 7. Outliers -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.boxplot(y=df["actual_waiting_time"], ax=ax, color="#C44E52")
    ax.set_title("Outlier View: Actual Waiting Time")
    ax.set_ylabel("Waiting time (minutes)")
    save(fig, "07_waiting_time_outliers")
    notes.append("## 7. Outliers")
    notes.append(
        "- The long upper tail visible here corresponds to the rare 'slowdown' events described in "
        "`outputs/data_quality_report.md` (IQR-based outlier count and the decision to keep them are documented there).\n"
    )

    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(notes) + "\n")

    print("\n".join(notes))
    print(f"\nPlots saved to {PLOTS_DIR}/, summary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
