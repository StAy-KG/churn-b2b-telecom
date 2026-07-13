"""
Multi-GPU Optuna hyperparameter search for CatBoost churn model.

In production this script ran on 4× NVIDIA A16 GPUs using multiprocessing
with JournalFileBackend for crash-safe distributed storage.
On a single machine, it runs sequentially.

Usage:
    python run_optuna.py
"""

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from multiprocessing import Process
import pandas as pd
import numpy as np
import json
import gc
import subprocess

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH     = './data/interim/df_catboost.parquet'
CONTRACT_PATH = './notebooks/feature_lists.json'
LOG_PATH      = './optuna_churn.log'
STUDY_NAME    = 'telco_churn_catboost'
TOTAL_TRIALS  = 60

N_FOLDS       = 5   # StratifiedKFold
RANDOM_SEED   = 42
# ──────────────────────────────────────────────────────────────────────────────


def get_free_gpus(min_free_mb: int = 8000, max_util_pct: int = 20) -> list[int]:
    """Return indices of GPUs with enough free VRAM and low utilization."""
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=index,memory.total,memory.used,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        free = []
        for line in result.stdout.strip().split('\n'):
            idx, mem_total, mem_used, util = [x.strip() for x in line.split(',')]
            if (int(mem_total) - int(mem_used)) >= min_free_mb and int(util) <= max_util_pct:
                free.append(int(idx))
        return free
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []   # CPU fallback


def objective(trial: optuna.Trial) -> float:
    with open(CONTRACT_PATH) as f:
        contract = json.load(f)

    df = pd.read_parquet(DATA_PATH)
    cat_features = contract['cat_features']
    X = df[contract['X_catboost_cols']].copy()
    y = df['Churn']

    for col in cat_features:
        X[col] = X[col].astype(str)

    params = {
        'iterations':          trial.suggest_int('iterations', 500, 1500, step=100),
        'learning_rate':       trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth':               trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg':         trial.suggest_int('l2_leaf_reg', 1, 10),
        'random_strength':     trial.suggest_float('random_strength', 0.1, 5.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'auto_class_weights':  trial.suggest_categorical('auto_class_weights', ['SqrtBalanced', 'Balanced']),
        'eval_metric':         'AUC',
        'random_seed':         RANDOM_SEED,
        'verbose':             0,
    }

    # GPU memory optimizations (active when GPU is available)
    gpu_ids = get_free_gpus()
    if gpu_ids:
        params.update({
            'task_type':               'GPU',
            'gpu_cat_features_storage': 'CpuPinned',  # 4× VRAM saving
            'borders_count':            64,             # OOM protection
        })

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    pr_aucs = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            Pool(X_train, y_train, cat_features=cat_features),
            eval_set=Pool(X_val, y_val, cat_features=cat_features),
            use_best_model=True,
        )
        preds = model.predict_proba(X_val)[:, 1]
        pr_aucs.append(average_precision_score(y_val, preds))

        del model
        gc.collect()

    return float(np.mean(pr_aucs))


def run_worker(gpu_id: int, study_name: str, n_trials: int) -> None:
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    storage = JournalStorage(JournalFileBackend(LOG_PATH))
    study = optuna.load_study(study_name=study_name, storage=storage)
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))


if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = JournalStorage(JournalFileBackend(LOG_PATH))
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction='maximize',
        load_if_exists=True,
    )

    gpu_ids = get_free_gpus()

    if len(gpu_ids) > 1:
        print(f'Multi-GPU mode: {len(gpu_ids)} GPUs detected → {gpu_ids}')
        base, remainder = divmod(TOTAL_TRIALS, len(gpu_ids))
        processes = []
        for i, gpu_id in enumerate(gpu_ids):
            n = base + (1 if i < remainder else 0)
            p = Process(target=run_worker, args=(gpu_id, STUDY_NAME, n))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        mode = f'single GPU {gpu_ids[0]}' if gpu_ids else 'CPU'
        print(f'Running {TOTAL_TRIALS} trials on {mode}...')
        study.optimize(objective, n_trials=TOTAL_TRIALS)

    print(f'\nBest PR-AUC: {study.best_value:.4f}')
    print(f'Best params:')
    for k, v in study.best_params.items():
        print(f'  {k}: {v}')
