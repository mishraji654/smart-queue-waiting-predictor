# Model Training & Evaluation Report

Training rows: **3187**, Test rows: **797**

## 1. Baselines

- **Median predictor** (always predicts 22.7 min): MAE=12.87, RMSE=17.8, R2=-0.067
- **Domain formula** (people_waiting x service_time / counters, no ML): MAE=9.18, RMSE=12.18, R2=0.5

Both are meant to be beaten by a properly trained model - if a trained model can't outperform a manager's back-of-envelope formula, it isn't earning its complexity.

## 2. Candidate models - 5-fold cross-validation on the training set

| Model | CV MAE (mean +/- std) | CV RMSE (mean +/- std) | CV R2 |
|---|---|---|---|
| Linear Regression | 4.89 +/- 0.14 | 7.42 +/- 0.30 | 0.805 |
| Random Forest | 4.58 +/- 0.15 | 7.24 +/- 0.50 | 0.814 |
| Gradient Boosting | 3.66 +/- 0.13 | 6.01 +/- 0.38 | 0.872 |

## 3. Same models - held-out test set (unseen data)

| Model | Test MAE | Test RMSE | Test R2 |
|---|---|---|---|
| Linear Regression | 5.0 | 7.3 | 0.82 |
| Random Forest | 4.51 | 7.17 | 0.827 |
| Gradient Boosting | 3.56 | 5.85 | 0.885 |

## 4. Selected model: **Gradient Boosting**

Chosen by lowest cross-validated MAE (3.66 min), which is more reliable than a single train/test split since it's averaged over 5 different held-out folds. Its test-set performance (MAE=3.56, RMSE=5.85, R2=0.885) confirms the CV ranking holds on genuinely unseen data.

**Practical interpretation:** an MAE of 3.56 minutes means predictions are, on average, off by about 3.56 minutes from the true waiting time on the test data - close enough to be genuinely useful for setting customer expectations, e.g. showing '~4 min give or take' alongside the estimate.

## 5. Feature importance (permutation importance on the test set)

- Computed by measuring how much MAE gets worse when a single feature's values are randomly shuffled (repeated 15x per feature, averaged). Larger increase = model relies on it more.
- Top 3 most influential features: **num_people_waiting, avg_service_time, active_counters**.
- Full ranking saved in `outputs/plots/09_feature_importance.png`.

## 5b. Prediction range (Optional Extension - Section 16)

- From the held-out test residuals, the middle 80% of prediction errors fall between **-5.2** and **+4.3** minutes of the point prediction. `predictor.py` adds this range to every prediction (e.g. '18 min, likely between 12-25 min') rather than a bare single number - closer to how a real operational tool should communicate uncertainty. This is an empirical residual-quantile interval, not a full conformal-prediction or quantile-regression model; that would be the natural next step with more data (see REPORT.md).

## 6. Leakage ablation: with vs without previous_waiting_time / historical_queue_load

- **With** the two features: MAE=3.56, RMSE=5.85, R2=0.885
- **Without** the two features: MAE=3.45, RMSE=5.63, R2=0.893
- Removing them changes MAE by only **+0.11 minutes** - essentially no difference (within run-to-run noise). This is actually the strongest evidence against leakage: if `previous_waiting_time` were a disguised copy of the target, dropping it would cripple the model. Instead the other queue-state features (people waiting, counters, service time) already explain most of the same variance, so this model keeps the two columns mainly for robustness/interpretability rather than because it depends on them.

