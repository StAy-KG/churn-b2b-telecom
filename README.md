# B2B Subscriber Churn Prediction — Telecom ML Pipeline

> End-to-end ML pipeline for predicting telecom subscriber churn. From raw customer data to a scored retention list, with a dual-branch architecture (Logistic Regression baseline + CatBoost production model).

**Dataset:** IBM Telco Customer Churn (7,043 customers, 20 features) — used as a public substitute to demonstrate the pipeline architecture originally built on proprietary B2B ClickHouse data.

---

## Results

| Metric | LogReg Baseline | CatBoost |
|---|---|---|
| ROC-AUC | 0.840 ± 0.014 | **0.849 ± 0.009** |
| PR-AUC | 0.654 ± 0.022 | **0.665 ± 0.016** |
| Optimal threshold | — | **0.40** (vs default 0.5) |
| At threshold: Precision / Recall / F1 | — | 0.531 / 0.799 / 0.638 |
| Class imbalance | ~2.8:1 | ~2.8:1 |
| Validation | Stratified K-Fold (5 folds) | Stratified K-Fold (5 folds) |

> CatBoost outperforms LogReg on both metrics with lower variance across folds (±0.009 vs ±0.014 ROC-AUC). Threshold tuned to maximize F1 — prioritizing Recall (0.80) since missing a churner costs more than a false alarm in a retention context.

---

## Architecture

```
IBM Telco CSV (raw)
        │
        ▼
01_data_prep_eda.ipynb       — null cleanup, binary encoding, EDA visualizations
        │
        ▼
02_feature_engineering.ipynb — service ratios, bill shock index, tenure bands, loyalty flags
        │
        ├──► df_logreg.parquet    (OHE, multicollinear features dropped)
        └──► df_catboost.parquet  (raw string categoricals, full feature set)
        │
        ▼
03_model_training.ipynb      — Stratified K-Fold CV, LogReg baseline + CatBoost + threshold tuning
        │
        ▼
run_optuna.py                — Optuna hyperparameter search (multi-GPU ready)
```

---

## Key Engineering Decisions

### Dual-branch pipeline

The feature matrix splits at engineering time into two branches:

| Branch | Purpose | Key differences |
|---|---|---|
| `df_logreg` | Linear baseline | OHE categoricals, multicollinear features dropped (TotalCharges = MonthlyCharges × tenure) |
| `df_catboost` | Production model | Raw string categoricals (CatBoost handles natively), full feature set |

### Feature engineering: behavioral proxies

- **Bill Shock Index**: `MonthlyCharges / Avg_Lifetime_Charge` — detects charge spikes vs historical average
- **Service intensity ratios**: `Streaming_Ratio`, `Protection_Ratio`, `Charge_Per_Service` — measure engagement depth and switching cost
- **Lifecycle flags**: `Is_NewCustomer` (≤6 months), `Is_Established`, `Is_Loyal` (>24 months)
- **Contract risk score**: ordinal encoding (Month-to-month=2, One year=1, Two year=0) — churn rate 42.7% vs 2.8% across extremes
- **Digital risk combo**: `Is_ElectronicCheck × PaperlessBilling` — interaction flag for highest-churn payment segment

### Class imbalance handling

With 2.8:1 imbalance, default 0.5 threshold recovers only ~50% of churners:
- `auto_class_weights='SqrtBalanced'` in CatBoost
- Optimal threshold tuned to **0.40** via precision-recall curve
- Result: Recall 0.80 — catches 4 out of 5 churners before they leave

### Feature contract (JSON)

Feature lists locked at engineering time and loaded by all downstream notebooks — prevents train/inference skew.

### Hyperparameter search (Optuna)

`run_optuna.py` runs a distributed Optuna study using `multiprocessing` + `JournalFileBackend`. Detects free GPUs via `nvidia-smi` dynamically, falls back to CPU automatically.

---

## Top Feature Importances (CatBoost)

| Rank | Feature | Note |
|---|---|---|
| 1 | Contract | Month-to-month 15× higher churn than two-year |
| 2 | InternetService | Fiber optic: 42% churn rate |
| 3 | Contract_Risk | Engineered ordinal encoding ✓ |
| 4 | Tenure_Band | Engineered lifecycle bucket ✓ |
| 5 | tenure | Raw tenure in months |
| 6 | PaymentMethod | Electronic check: highest-risk segment |
| 9 | Bill_Shock_Index | Engineered behavioral proxy ✓ |
| 10 | Streaming_Ratio | Engineered engagement ratio ✓ |

Engineered features appear in top 10 — confirming feature construction adds signal beyond raw attributes.

---

## EDA Highlights

| Finding | Value |
|---|---|
| Overall churn rate | 26.5% |
| Month-to-month churn | 42.7% |
| Two-year contract churn | 2.8% |
| Fiber optic churn | 41.9% |
| Median tenure (churned) | 10 months |
| Median tenure (retained) | 38 months |

---

## Stack

- **Models**: CatBoost, scikit-learn (LogisticRegression)
- **Hyperparameter tuning**: Optuna (multi-GPU, JournalStorage)
- **Processing**: pandas, NumPy
- **Validation**: Stratified K-Fold (5 folds)
- **Visualization**: matplotlib, seaborn

---

## Project Structure

```
churn-b2b-telecom/
├── data/raw/
│   └── telco_churn_raw.csv            # IBM Telco public dataset
├── notebooks/
│   ├── 01_data_prep_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── reports/
│   ├── 01_eda_churn_patterns.png
│   └── 03_model_results.png
├── run_optuna.py                       # Multi-GPU Optuna worker
├── requirements.txt
└── README.md
```

---

## What I'd Improve Next

- **MLflow tracking** — experiment reproducibility across Optuna trials
- **FastAPI wrapper** — on-demand scoring endpoint instead of batch notebook
- **SHAP explanations** — per-customer feature contribution for retention agents
- **Docker containerization** — deployment-agnostic inference pipeline
- **Switch Optuna objective to PR-AUC** — more informative than ROC-AUC under class imbalance
