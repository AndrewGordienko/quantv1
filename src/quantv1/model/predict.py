"""Score disclosed trades with the trained model and explain each with SHAP.

SHAP contributions come straight from LightGBM (`pred_contrib=True`) — no extra
dependency. For each scored trade we keep the top contributing features so the
dashboard can say *why* a name is recommended ("high-skill member + 3-member
cluster + committee match") instead of showing an opaque probability.
"""

from __future__ import annotations

import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..db import connect
from ..research.returns import PriceStore
from . import features as F
from .train import MODEL_PATH

# Human-readable labels for SHAP explanations on the dashboard.
FEATURE_LABELS = {
    "member_skill": "member skill",
    "committee_match": "committee jurisdiction match",
    "amount_mid_log": "trade size",
    "size_vs_member": "large vs member's norm",
    "cluster_count": "cluster buying",
    "disclosure_lag": "disclosure speed",
    "owner_self": "traded in own account",
    "asset_option": "options trade",
    "party_dem": "party",
    "momentum_63": "3-month momentum",
    "momentum_252": "12-month momentum",
    "volatility_63": "volatility",
    "market_cap_log": "market cap",
}


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _explain(model, X: pd.DataFrame, cols: list[str], top_k: int = 4) -> list[list[dict]]:
    """Per-row top SHAP contributions using LightGBM native pred_contrib."""
    booster = model.booster_
    contrib = booster.predict(X.to_numpy(), pred_contrib=True)  # (n, n_feat+1)
    out = []
    for row in contrib:
        vals = row[:-1]  # last col is the base/expected value
        order = np.argsort(-np.abs(vals))[:top_k]
        out.append([
            {"feature": cols[j], "label": FEATURE_LABELS.get(cols[j], cols[j]),
             "contribution": float(vals[j])}
            for j in order if abs(vals[j]) > 1e-6
        ])
    return out


def score(con=None, store: PriceStore | None = None,
          as_of: str | None = None, lookback_days: int = 90) -> pd.DataFrame:
    """Score trades filed within `lookback_days` of `as_of` (default: latest)."""
    payload = load_model()
    model, cols = payload["model"], payload["feature_cols"]

    own = con is None
    con = con or connect(read_only=True)
    store = store or PriceStore(con)
    feats = F.build(con=con, store=store, with_label=False, purchases_only=True)
    if own:
        con.close()

    feats["filing_date"] = pd.to_datetime(feats["filing_date"])
    asof = pd.Timestamp(as_of) if as_of else feats["filing_date"].max()
    lo = asof - pd.Timedelta(days=lookback_days)
    recent = feats[(feats["filing_date"] > lo) & (feats["filing_date"] <= asof)].copy()
    if recent.empty:
        return recent

    X = recent[cols].astype(float)
    recent["score"] = model.predict_proba(X)[:, 1]
    recent["contribs"] = _explain(model, X, cols)
    return recent.sort_values("score", ascending=False).reset_index(drop=True)


def persist(scored: pd.DataFrame) -> None:
    import json
    con = connect()
    now = datetime.now(timezone.utc)
    con.executemany(
        "INSERT OR REPLACE INTO signals VALUES (?,?,?,?,?,?,?)",
        [[r.trade_id, r.filing_date.date() if hasattr(r.filing_date, "date")
          else r.filing_date, r.ticker, r.member, float(r.score),
          json.dumps(r.contribs), now]
         for r in scored.itertuples(index=False)],
    )
    con.close()


if __name__ == "__main__":
    s = score()
    print(f"scored {len(s)} recent trades")
    if not s.empty:
        cols = ["filing_date", "member", "ticker", "score"]
        print(s[cols].head(15).to_string(index=False))
        persist(s)
