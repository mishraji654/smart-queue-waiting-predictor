# Report: Smart Queue Waiting-Time Predictor

This document is the write-up for the assignment: what was built, why it was built that way, the results, and answers to the assignment's "Candidate Questions to Address" (Section 15). The raw, generated evidence behind every claim here lives in `outputs/` (`data_quality_report.md`, `eda_summary.md`, `training_report.md`, `model_comparison.csv`, and `plots/`) and can be regenerated at any time by re-running the pipeline (see `README.md`).

---

## 1. The dataset - source and assumptions

**Source:** a purpose-built **synthetic dataset** (`src/generate_data.py`), not a public dataset. No public dataset combines this project's exact requested feature set (active counters currently open, priority customers, a recent-history load signal, a previous-customer wait signal, etc.) for a generic multi-domain queue, so adapting a mismatched real dataset would have meant either faking half the columns anyway or dropping most of the assignment's suggested features. The assignment explicitly allows this route (Section 7): *"The candidate may use a publicly available dataset, create a realistic synthetic dataset, or combine both approaches."*

**How it avoids being "a simple fixed formula":**
- A hidden **"business level"** (how busy the branch is right now, driven by day-of-week and hour-of-day patterns plus random noise) drives several features at once, the way a real queue would - `num_people_waiting`, `active_counters`, `priority_customers`, and `historical_queue_load` are all generated from this shared but noisy latent factor, not from each other directly.
- The target (`actual_waiting_time`) combines base queueing math (people ahead × service time ÷ counters), a priority-customer penalty, peak-hour and day-of-week multipliers, a small historical-load effect, a small "momentum" effect from the previous customer's wait, per-row random noise, **and** a rare (~2%) random "slowdown" event that produces realistic outliers.
- `previous_waiting_time` is generated from the *same* business-level context but its **own independent noise draw** - correlated with the target (as a genuinely useful feature should be) without being a disguised copy of it. See Section 4 below for the empirical check.

**Assumptions made (stated explicitly, as the assignment asks):**
1. The system is generic across hospitals/banks/restaurants/government offices/service centers rather than modeling one specific domain - so day-of-week and hour effects are kept moderate rather than domain-specific (e.g., no assumption that the "business" is closed on Sundays).
2. Queue snapshots are treated as **independent observations**, not a strict time series - each row is "a person entering the queue at some moment," not a sequential log. This is why a random 80/20 split was used instead of a time-based split.
3. `queue_length` (total active queue size) is assumed to be slightly larger than `num_people_waiting` (people specifically ahead of this person) - it also includes people currently being served, so `queue_length ≈ num_people_waiting + active_counters + a little noise`.
4. Roughly 4,000 base records were generated, then ~2% duplicate rows and a handful of impossible values were deliberately mixed in to mimic what a real operational export looks like (see Section 2). After cleaning, **3,984 rows** remain - comfortably enough for the models used here to learn real patterns rather than memorize noise.

---

## 2. Data preparation

Full numbers are in `outputs/data_quality_report.md`; summary:

| Issue | Found | Action |
|---|---|---|
| Duplicate records (by content, not row id) | 80 | Dropped |
| `active_counters` ≤ 0 (impossible) | 4 | Dropped |
| `num_people_waiting` < 0 (impossible) | 4 | Dropped |
| `num_people_waiting` > 80 (data-entry error, e.g. a stray extra digit) | 4 | Dropped |
| `actual_waiting_time` < 0 (impossible) | 4 | Dropped |
| Missing values (`avg_service_time`, `historical_queue_load`, `previous_waiting_time`, `priority_customers`) | 2-3% per column | Median-imputed |

Impossible values were **dropped rather than corrected**, because there's no reliable way to "fix" a negative wait time or 0 counters into a true value - imputing one would fabricate data. Genuinely rare-but-real outliers (long waits from the simulated "slowdown" events, ~3% of rows, IQR-flagged) were **kept**, because they represent real operational events a manager would want the model to have seen, not data errors.

`day_of_week` and the engineered `time_period` bucket are one-hot encoded **inside the model pipeline** (not in the saved CSV), so the cleaned data stays human-readable and the exact same pipeline object can take a raw-looking prediction request straight from the CLI/web form.

---

## 3. Exploratory analysis - key findings

Full write-up in `outputs/eda_summary.md`; charts in `outputs/plots/01`-`07`. Highlights:

- **Distribution:** right-skewed (skew ≈ 1.47), mean ≈ 26.5 min, median ≈ 22.9 min - most waits are short-to-moderate with a long tail of busy/slow episodes. This is why MAE/RMSE (not an assumption of normally-distributed error) drive the evaluation.
- **Day of week:** a real but modest effect - Saturday highest (≈30 min average), Sunday lowest (≈22 min average). Not a strong predictor on its own.
- **Time of day:** clear peaks around midday and early evening (lunch-hour/after-work congestion), visible in both the hour-by-hour line chart and the time-period boxplot.
- **Queue size vs. wait:** `num_people_waiting` correlates with the target at **r ≈ 0.57**, `queue_length` at **r ≈ 0.45** - the strongest simple relationships in the data, as expected for a queueing problem.
- **Counters vs. wait:** a clear inverse relationship - average wait drops from ≈55 min with 1 counter open to ≈24 min with 6 open.
- **Outliers:** ~3% of records sit outside the IQR fence on the target, corresponding to the simulated slowdown events - visualized in the boxplot and discussed above.

---

## 4. Machine learning approach

**Baselines established first** (Section 8's "simple reference"):
- A **median predictor** (always guesses the training median, 22.7 min) - MAE 12.87, RMSE 17.80, R² -0.07 (worse than guessing nothing, since it ignores queue size entirely).
- A **domain formula** with no ML at all - `people_waiting × service_time ÷ counters` - MAE 9.18, RMSE 12.18, R² 0.50. This is what a manager might already estimate by hand; any trained model needs to clearly beat it to justify the added complexity.

**Three regression models compared** (Section 8), all inside one reusable `ColumnTransformer → model` pipeline, evaluated with **5-fold cross-validation** on the training set and then confirmed on a held-out test set:

| Model | CV MAE | CV RMSE | CV R² | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|---|
| Linear Regression | 4.89 | 7.42 | 0.805 | 5.00 | 7.30 | 0.820 |
| Random Forest | 4.58 | 7.24 | 0.814 | 4.51 | 7.17 | 0.827 |
| **Gradient Boosting** | **3.66** | **6.01** | **0.872** | **3.56** | **5.85** | **0.885** |

All three comfortably beat both baselines (see `outputs/plots/08_model_comparison.png`). **Gradient Boosting was selected** - lowest cross-validated MAE, and the test-set result (not used for model selection) confirms the same ranking, which is a good sign the model generalizes rather than having gotten lucky on one split.

**Why Gradient Boosting fits this problem:** the target is driven by several features interacting non-linearly (e.g., the effect of one more counter matters a lot more when the queue is long than when it's short; priority customers matter more during peak hours) - boosted trees capture that kind of interaction naturally, where plain Linear Regression can only add each feature's effect independently. Random Forest also captures interactions but tends to average across many similar trees, which slightly smooths out the sharper effects Gradient Boosting's sequential error-correction picks up here.

---

## 5. Feature interpretation

Computed via **permutation importance** on the test set (shuffle one feature, see how much worse MAE gets, repeat 15×, average) - model-agnostic, so it works the same way regardless of which model won. Full chart: `outputs/plots/09_feature_importance.png`.

**Ranking:** `num_people_waiting` > `avg_service_time` > `active_counters` > `priority_customers` ≈ `previous_waiting_time` > `hour` > `day_of_week` > `time_period` > `historical_queue_load` > `queue_length`.

This matches domain intuition well: the three biggest levers are exactly the three terms in the basic queueing formula (people ahead, how long each takes, how many servers) - which is reassuring, since it means the model learned genuine queueing dynamics rather than some spurious correlation. `queue_length` ranks lowest despite being correlated with the target in the raw EDA - because it's highly collinear with `num_people_waiting` (r ≈ 0.94, see the correlation heatmap), the model gets almost all of that signal from `num_people_waiting` already, so shuffling `queue_length` alone barely hurts it.

**Target leakage check** (Section 6's last requirement): `previous_waiting_time` correlates with the target at r ≈ 0.55, `historical_queue_load` at r ≈ 0.25 - both moderate, not the ~0.97+ that would suggest one is secretly a copy of the other. To go further than just eyeballing the correlation, `train.py` re-trains the winning model **without** those two columns: MAE moves from 3.56 to 3.45 minutes - i.e., removing them doesn't hurt (if anything it's a hair better, well within run-to-run noise). That's the strongest evidence against leakage: a genuinely leaky feature would cause a *sharp* drop in performance when removed, not nothing. Full detail in `outputs/training_report.md`, Section 6.

---

## 6. Edge cases (Section 13) - how each is handled

All of these are exercised automatically by `python src/predict_cli.py --demo`.

| Edge case | Handling |
|---|---|
| Zero people waiting | Valid input - predicts a small (~1-6 min) but non-zero wait, reflecting that being served still takes a moment. |
| One counter during a busy period | Valid input, no special-casing needed - the model was trained on this combination and predicts a correctly large wait (e.g. ≈82 min for 18 people / 1 counter at Friday evening peak). |
| Multiple priority customers | Valid input - increases the prediction, consistent with the training data's priority-penalty effect. |
| Waiting count far above training data | **Warning shown, prediction still returned**, explicitly flagged as an extrapolation (e.g. "500 people waiting is above the highest value seen in training (24)..."). Not blocked, because refusing to answer is often worse than an honest, flagged estimate - but the user is told not to trust it blindly. |
| Missing/incomplete input | Required fields (`num_people_waiting`, `active_counters`, `avg_service_time`) must be present and valid; everything else defaults sensibly (0 priority customers, today/now for day/time, the typical training value for historical load / previous wait) with each default reported back to the user. |
| Impossible input (0 counters, negative counts, bad day name, malformed time) | **Rejected with a specific error message**, no prediction returned - re-prompted in the CLI, shown inline in the web form. |
| Rare time period (e.g. 3 AM) | Valid input - the model still predicts (using whatever it learned about the `Night` bucket / low-traffic hours), though naturally with less training support than peak hours; this is noted as a limitation in Section 8 below. |

---

## 7. Optional extensions implemented (Section 16)

Three of the five suggested extensions were built in, since they were low-effort and materially improve the demo:

1. **Prediction range alongside the estimate** - every prediction comes with an 80% empirical range from the held-out residual distribution (e.g. "18 min, likely 13-22 min"), not just a bare number.
2. **Visualize queue trends over the day** - `outputs/plots/03_waiting_time_by_time.png` (hour-by-hour average line chart + time-period boxplot).
3. **Alert when predicted wait exceeds a threshold** - both interfaces flag predictions ≥ 30 minutes with a visible warning.

Also included as a small bonus (not in the original list): `predict_cli.py --demo` finishes by showing how the same busy scenario's prediction changes as active counters go from 1 → 2 → 3, which is effectively a lightweight version of "compare predictions before/after increasing counters."

Not implemented: **per-service-category prediction**, since the current feature set and dataset are deliberately generic across business types rather than modeling distinct service categories within one business - adding this properly would mean redesigning the data generation around categories rather than a quick bolt-on.

---

## 8. Candidate Questions to Address (Section 15)

**Which factors have the strongest relationship with waiting time?**
`num_people_waiting`, `avg_service_time`, and `active_counters` - the three terms of the basic queueing formula - dominate, both in raw correlation (Section 3) and in the trained model's permutation importance (Section 5). Priority customers and the previous customer's wait time are secondary but real contributors; day-of-week and the specific hour are weaker still.

**Why is the selected model suitable for this problem?**
Gradient Boosting won on cross-validated MAE and confirmed it on the untouched test set, and the *reason* it wins fits the problem: waiting time is a non-linear, interacting function of its inputs (an extra counter matters far more when the queue is already long; a priority customer matters more during a peak hour than a quiet one), and boosted trees are built to capture exactly that kind of conditional effect, whereas Linear Regression can only sum each feature's effect independently of the others.

**How accurate are the predictions in practical terms?**
Test-set MAE is **3.56 minutes** (RMSE 5.85, R² 0.885) - on a typical wait of ~20-30 minutes, that's usually within about 10-15% of the true value, which is tight enough to be operationally useful (e.g., telling a customer "about 20 minutes, could be 15-25" is a genuinely different, more useful message than not estimating at all, and is a large improvement over the 9.18-minute MAE of the no-ML domain-formula baseline).

**Which evaluation metric is most useful for this use case, and why?**
**MAE** is the most directly useful for communicating to an end user, since it's in the same units as the answer ("off by about 3.6 minutes on average") and isn't dominated by the rare slowdown outliers. **RMSE** is still worth tracking alongside it precisely *because* it penalizes large errors more - for an operational tool, being wildly wrong occasionally (e.g., telling someone "10 minutes" when it's actually 60) is more costly than being consistently off by a couple of minutes, and RMSE surfaces that risk in a way MAE alone would understate.

**What assumptions were made while collecting or generating the dataset?**
Listed in full in Section 1 above - most importantly: the system is generic across business types rather than modeling one domain specifically, each row is an independent snapshot rather than a time-ordered sequence, and `queue_length` is assumed to be `num_people_waiting` plus roughly the number of people currently being served.

**What types of situations might cause the model to make inaccurate predictions?**
Three concrete cases observed in testing: (1) **extrapolation** - inputs far outside the training range (e.g. 150+ people waiting, see the demo output) get a flagged-but-uncertain prediction, since tree-based models can't reliably extrapolate past what they've seen, they effectively plateau; (2) **rare hour/day combinations** with little training support (e.g. 3 AM) get a technically-valid prediction with less evidence behind it than a well-represented peak hour; (3) **structural shocks not represented in training**, like a system outage, a public holiday, or a sudden policy change (e.g. a counter permanently closing) - the model has no way to know about a real-world change it never saw examples of.

**How could the model be improved if more real-world data became available?**
Swap in real historical logs once available (the pipeline's column names/interfaces would stay the same - only `generate_data.py` would be replaced). With more data: (a) replace the residual-quantile prediction range with proper **quantile regression or conformal prediction** for statistically calibrated intervals; (b) add genuine **time-series structure** (actual rolling averages, real previous-customer chains) instead of the current independent-snapshot simulation; (c) consider **per-service-category models** if the real data has distinguishable service types; (d) revisit whether `queue_length` is worth keeping at all given how collinear it is with `num_people_waiting` in this data - real data might reveal it carries more independent signal than the synthetic version does.

---

## 9. One thing worth flagging explicitly

The assignment's own worked example (Section 4/10: 12 people waiting, 3 counters, 7-minute service, 2 priority customers → "18 minutes") is a **format illustration**, not a target this project's model was built to reproduce - it's this project's own synthetic dataset and its own learned formula, not the assignment's. Running that exact scenario through this model gives a different number (see `predict_cli.py --demo`, first scenario) because the underlying data-generating assumptions (how much a priority customer adds, how strong the peak-hour effect is, etc.) were defined independently here, as the assignment invites candidates to do. The *mechanics* - a numeric wait-time estimate, in minutes, from a set of queue conditions - match exactly; the specific number for that specific example naturally won't, since no shared ground-truth dataset was specified.
