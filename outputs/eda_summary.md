# Exploratory Data Analysis Summary

Cleaned dataset: **3984 rows**, **14 columns**.

## 1. Distribution of waiting times
- Mean **26.5 min**, median **22.9 min**, std **16.9 min**. Skew = 1.47 -> right-skewed, as expected for a wait-time metric (most waits are short-to-moderate, with a long tail of slow/busy episodes). This is why MAE/RMSE are reported alongside the median rather than assuming a symmetric, normal-like target.

## 2. Waiting time by day of week
- Highest average wait: **Saturday** (30.2 min). Lowest: **Sunday** (21.7 min). The spread is modest rather than dramatic, which matches how the synthetic data was generated (a mild day-of-week demand effect on top of a much stronger queue-load/staffing effect) - day-of-week alone is a weak predictor and should not be over-weighted.

## 3. Waiting time by time of day
- Clear peaks are visible around midday and early evening, with hour **13:00** showing the highest average wait in this sample. This lines up with typical lunch-hour and after-work congestion in real queueing systems (hospitals, banks, service counters), which is what the peak-hour multiplier in `generate_data.py` was designed to mimic.

## 4. Queue size vs waiting time
- Correlation with `num_people_waiting`: **0.57**. Correlation with `queue_length`: **0.45**. Both show a clear positive, roughly linear-ish relationship, confirming queue size is one of the strongest single predictors - as expected for a queueing problem.

## 5. Active counters vs waiting time
- Average wait drops from **54.9 min** with 1 counter(s) open to **23.7 min** with 6 open, a clear and intuitive inverse relationship - more staffed counters -> shorter waits.

## 6. Correlation heatmap
- See `06_correlation_heatmap.png`. Queue size and counters dominate; time/priority/history contribute smaller, complementary signal.

## 7. Outliers
- The long upper tail visible here corresponds to the rare 'slowdown' events described in `outputs/data_quality_report.md` (IQR-based outlier count and the decision to keep them are documented there).

