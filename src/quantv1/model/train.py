"""Train the signal model with walk-forward, time-series cross-validation.

The label is binary: did the disclosed purchase beat SPY over LABEL_HORIZON
trading days after its FILING date. We evaluate with an expanding-window
walk-forward split (train on the past, test on the next block, roll forward) so
the reported AUC/IC reflects genuine out-of-sample performance — never random
K-fold, which would leak future information into the past.

Two models:
  * LogisticRegression  — interpretable baseline
  * LightGBM            — the workhorse; SHAP via native pred_contrib

The final production model is refit on ALL labeled data and saved for scoring.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import DATA_DIR
from ..db import connect
from ..research.returns import PriceStore
from . import features as F

MODEL_PATH = DATA_DIR / "signal_model.pkl"


def _walk_forward_splits(dates: pd.Series, n_folds: int = 5):
    """Expanding-window splits ordered by filing date."""
    order = np.argsort(dates.values)
    n = len(order)
    fold = n // (n_folds + 1)
    for k in range(1, n_folds + 1):
        train_idx = order[: fold * k]
        test_idx = order[fold * k: fold * (k + 1)]
        if len(test_idx) == 0:
            continue
        yield train_idx, test_idx


def _spearman_ic(y_true_excess: np.ndarray, y_score: np.ndarray) -> float:
    """Rank correlation between model score and realized excess return."""
    from scipy.stats import spearmanr
    if len(y_score) < 5:
        return np.nan
    rho, _ = spearmanr(y_score, y_true_excess)
    return float(rho)


def evaluate(data: pd.DataFrame) -> dict:
    X = data[F.FEATURE_COLS].astype(float)
    y = data["label"].to_numpy()
    excess = data["fwd_excess"].to_numpy()
    dates = data["filing_date"]

    results = {"logistic": [], "lightgbm": []}
    for train_idx, test_idx in _walk_forward_splits(dates):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue

        logit = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        # Median-impute, then 0-fill any column that was entirely NaN in train
        # (e.g. market_cap before sector data is populated).
        med = Xtr.median()
        Xtr_f = Xtr.fillna(med).fillna(0.0)
        Xte_f = Xte.fillna(med).fillna(0.0)
        logit.fit(Xtr_f, ytr)
        p_log = logit.predict_proba(Xte_f)[:, 1]

        gbm = LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=15,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, verbose=-1,
        )
        gbm.fit(Xtr, ytr)  # LightGBM handles NaNs natively
        p_gbm = gbm.predict_proba(Xte)[:, 1]

        for name, p in (("logistic", p_log), ("lightgbm", p_gbm)):
            results[name].append({
                "n_test": int(len(test_idx)),
                "auc": float(roc_auc_score(yte, p)),
                "ic": _spearman_ic(excess[test_idx], p),
                "test_start": str(dates.iloc[test_idx].min().date()),
                "test_end": str(dates.iloc[test_idx].max().date()),
            })

    summary = {}
    for name, folds in results.items():
        if folds:
            summary[name] = {
                "folds": folds,
                "mean_auc": float(np.mean([f["auc"] for f in folds])),
                "mean_ic": float(np.nanmean([f["ic"] for f in folds])),
            }
    return summary


def fit_final(data: pd.DataFrame) -> LGBMClassifier:
    X = data[F.FEATURE_COLS].astype(float)
    y = data["label"].to_numpy()
    gbm = LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=15,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, verbose=-1,
    )
    gbm.fit(X, y)
    return gbm


def run(exclude_estimated: bool = True) -> dict:
    store = PriceStore()
    data = F.build(store=store, with_label=True, purchases_only=True)
    if exclude_estimated:
        data = data[~data["filing_estimated"]].reset_index(drop=True)

    metrics = evaluate(data)
    model = fit_final(data)

    payload = {
        "model": model,
        "feature_cols": F.FEATURE_COLS,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": int(len(data)),
        "base_rate": float(data["label"].mean()),
        "metrics": metrics,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    # persist a compact metrics record next to the model for the dashboard
    with open(DATA_DIR / "model_metrics.json", "w") as f:
        json.dump({"trained_at": payload["trained_at"], "n_train": payload["n_train"],
                   "base_rate": payload["base_rate"], "metrics": metrics}, f, indent=2)
    return payload


if __name__ == "__main__":
    out = run()
    print(f"n_train={out['n_train']} base_rate={out['base_rate']:.3f}")
    for name, m in out["metrics"].items():
        print(f"{name:9s} mean_AUC={m['mean_auc']:.3f} mean_IC={m['mean_ic']:.3f} "
              f"({len(m['folds'])} folds)")
