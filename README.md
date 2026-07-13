# B2B Subscriber Churn Prediction — Telecom ML Pipeline

> End-to-end production ML system for predicting B2B subscriber churn in a Central Asian telecom operator. From raw monthly data in ClickHouse to a scored retention list ready for the call center.

---

## Results

| Metric | CV (Fold 5) | OOT (held-out) |
|---|---|---|
| ROC-AUC | 0.8078 | 0.8062 |
| PR-AUC | 0.1269 | 0.1253 |
| Optimal threshold | — | **0.18** (vs default 0.5) |
| Class imbalance | ~27:1 | ~27:1 |

> **Note:** With a 27:1 class imbalance, PR-AUC is the primary business metric — ROC-AUC is reported for benchmarking. The gap between CV and OOT is <0.2%, confirming the temporal fold structure is leak-free.

---

## Architecture

```
ClickHouse (raw monthly ABT)
        │
        ▼
01_data_prep_eda.ipynb       — M2M filtering, device registry repair, null cleanup
        │
        ▼
02_feature_engineering.ipynb — Ratios, Velocity (V1/V3), Acceleration, Revival flags
        │
        ├──► df_logreg.parquet    (OHE, no multicollinear features)
        └──► df_catboost.parquet  (raw categoricals, no OHE)
        │
        ▼
03_model_training.ipynb      — Hybrid CV (Sliding + Expanding Window), LogReg baseline + CatBoost
        │
        ▼
run_optuna.py                — Multi-GPU Optuna search (4× NVIDIA A16, ~2.3× speedup)
        │
        ▼
05_inference_pipeline.ipynb  — Monthly batch scoring → Excel retention list for call center
```

---

## Key Engineering Decisions

### Temporal validation (no data leakage)
Churn is defined as: active in month **t**, no activity in month **t+1**. Features are computed from month **t-1** (lag-1 design), so the model never sees future data. Validation uses a hybrid scheme:

- **Sliding window** (Folds 1–2): 3-month rolling train, 1-month val — tests short-term stability
- **Expanding window** (Folds 3–5): growing train set, 1-month val — tests generalization over time

### Feature engineering: behavioral dynamics
Beyond static features, the pipeline captures *change signals* — how a subscriber's behavior is evolving:

- **Velocity V1**: `metric[t] / metric[t-1]` — month-over-month ratio
- **Velocity V3**: `metric[t] / rolling_mean(t-3:t-1)` — smoothed trend
- **Acceleration**: `V1[t] - V1[t-1]` — rate of change of the ratio
- **Revival flags**: binary indicator when a previously-zero metric becomes non-zero (resurrection signal)
- **Loyalty ratios**: LTE share, on-net voice ratio, network isolation index

All velocity features are clipped using empirically derived percentile bounds per feature to prevent extreme outliers from dominating.

### Two-branch pipeline
The feature matrix is split at engineering time into two branches:

| Branch | Purpose | Key differences |
|---|---|---|
| `df_logreg` | Linear baseline | OHE categoricals, V3/Accel dropped (overfits in linear models) |
| `df_catboost` | Production model | Raw string categoricals, full feature set |

### Class imbalance handling
With ~27:1 imbalance, standard 0.5 threshold is useless. Approach:
- `auto_class_weights='SqrtBalanced'` in CatBoost (outperformed `scale_pos_weight`)
- Optimal classification threshold tuned to **0.18** via precision-recall curve
- Business filter: score ≥ 0.40 AND `DATE_LAD_days ≤ 1` (only actively-using subscribers, avoids wasting retention budget on already-churned accounts)

### Multi-GPU hyperparameter search
Optuna study distributed across 4 GPUs using `multiprocessing` + `JournalFileBackend` (file-based storage avoids Jupyter pickling issues). GPU availability checked dynamically before launch. Achieved ~2.3× wall-clock speedup vs single-GPU.

Memory optimization: `gpu_cat_features_storage='CpuPinned'` + `borders_count=64` to prevent CUDA OOM across parallel workers.

---

## Stack

- **Data warehouse**: ClickHouse
- **Processing**: pandas, NumPy
- **Models**: CatBoost, scikit-learn (LogisticRegression baseline)
- **Hyperparameter tuning**: Optuna (multi-GPU, JournalStorage)
- **Validation**: custom temporal CV (sliding + expanding window)
- **Output**: Excel retention list with churn probability scores

---

## Project Structure

```
churn_b2b/
├── data/
│   ├── raw/                  # Monthly ABT parquets from ClickHouse
│   ├── interim/              # Cleaned + feature-engineered checkpoints
│   └── processed/            # Final model-ready datasets
├── models/
│   └── catboost_b2b_churn_sniper_v1.cbm
├── notebooks/
│   ├── 01_data_prep_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 05_inference_pipeline.ipynb
├── reports/
│   └── churn_predict_YYYY-Month_top_sniper.xlsx
├── run_optuna.py             # Multi-GPU Optuna worker
├── feature_lists.json        # Feature contract (locked at training time)
└── README.md
```

---

## Inference Pipeline

The production scoring script (`05_inference_pipeline.ipynb`) runs monthly:

1. Loads raw data for month **t** (requires 4–5 month buffer for lag features)
2. Applies full feature engineering pipeline (same transforms as training)
3. Scores all active subscribers
4. Applies business filters (score threshold + recency filter)
5. Outputs ranked Excel file auto-named for month **t+1**

Output columns: `CTN`, `Score`, `DATE_LAD_days`, `TOTAL_MOU`, `BALANCE_END`, `PRICE_PLAN_FIXED`

---

## What I'd improve next

- **Switch primary metric to PR-AUC** in Optuna objective (currently optimizes ROC-AUC — a known limitation)
- **MLflow tracking** for experiment reproducibility
- **FastAPI wrapper** around the inference pipeline for on-demand scoring
- **SHAP explanations** per subscriber in the output list (explainability for retention agents)
- Containerize with Docker for deployment-agnostic runs
