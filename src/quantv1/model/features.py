"""Feature engineering for the signal model.

Given a set of disclosed trades, build a point-in-time feature matrix. Every
feature must be knowable at the trade's filing date — no look-ahead. The label
(did the stock beat SPY over LABEL_HORIZON trading days after filing) is only
attached for training rows old enough to have realized.

Key features (the "why is this trade interesting" vector):
  * member_skill        empirical-Bayes shrunk CAR of the trading member
  * committee_match      stock's sector is in the member's committee jurisdiction
  * amount_mid_log       log dollar size (range midpoint)
  * size_vs_member       this trade's size relative to the member's own history
  * cluster_count        # distinct members buying same ticker within 30d prior
  * disclosure_lag       days from transaction to filing (fast filing = fresher)
  * is_purchase / owner_self / asset_option flags
  * momentum_63/252, volatility_63 of the stock at filing
  * market_cap_log, sector one-hot
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import CLUSTER_WINDOW_DAYS, LABEL_HORIZON
from ..db import connect
from ..research.returns import PriceStore


def _jur_sectors(c) -> set[str]:
    if not c:
        return set()
    try:
        return set(json.loads(c).get("jurisdiction_sectors", []))
    except (TypeError, ValueError):
        return set()


def build(con=None, store: PriceStore | None = None, *,
          with_label: bool = True, purchases_only: bool = True,
          skill_map: dict | None = None) -> pd.DataFrame:
    own = con is None
    con = con or connect(read_only=True)
    store = store or PriceStore(con)

    where = "t.ticker IS NOT NULL"
    if purchases_only:
        where += " AND t.tx_type = 'purchase'"
    trades = con.execute(f"""
        SELECT t.trade_id, t.member, t.member_key, t.ticker, t.tx_type,
               t.tx_date, t.filing_date, t.filing_estimated, t.disclosure_lag,
               t.amount_mid, t.owner, t.asset_type,
               m.committees, m.party,
               s.sector, s.market_cap
        FROM trades t
        LEFT JOIN members m USING (member_key)
        LEFT JOIN ticker_sectors s USING (ticker)
        WHERE {where}
        ORDER BY t.filing_date
    """).df()

    # Skill scores (may be pre-passed to avoid recompute during backtests).
    if skill_map is None:
        try:
            rows = con.execute(
                "SELECT member_key, shrunk_car FROM skill_scores"
            ).fetchall()
            skill_map = dict(rows)
        except Exception:  # noqa: BLE001 - table may not exist yet
            skill_map = {}
    if own:
        con.close()

    trades["filing_date"] = pd.to_datetime(trades["filing_date"])
    trades["tx_date"] = pd.to_datetime(trades["tx_date"])

    # --- cluster_count: distinct members buying same ticker in prior window ---
    trades = trades.sort_values("filing_date").reset_index(drop=True)
    cluster = np.zeros(len(trades), dtype=int)
    by_ticker: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    win = pd.Timedelta(days=CLUSTER_WINDOW_DAYS)
    for i, r in enumerate(trades.itertuples(index=False)):
        hist = by_ticker.setdefault(r.ticker, [])
        members = {mk for (d, mk) in hist if r.filing_date - d <= win}
        cluster[i] = len(members - {r.member_key})
        hist.append((r.filing_date, r.member_key))
    trades["cluster_count"] = cluster

    # --- size relative to the member's own median trade size (expanding) ---
    trades["amount_mid"] = trades["amount_mid"].fillna(trades["amount_mid"].median())
    med_so_far = (trades.groupby("member_key")["amount_mid"]
                  .apply(lambda s: s.shift().expanding().median()).reset_index(level=0, drop=True))
    trades["size_vs_member"] = (trades["amount_mid"] /
                                med_so_far.replace(0, np.nan)).fillna(1.0).clip(0, 20)

    # --- per-row features requiring price context ---
    feats = []
    for r in trades.itertuples(index=False):
        jur = _jur_sectors(r.committees)
        sector = r.sector if isinstance(r.sector, str) else "Unknown"
        row = {
            "trade_id": r.trade_id,
            "filing_date": r.filing_date,
            "member": r.member,
            "member_key": r.member_key,
            "ticker": r.ticker,
            "sector": sector,
            "member_skill": float(skill_map.get(r.member_key, 0.0)),
            "committee_match": int(sector in jur and sector != "Unknown"),
            "amount_mid_log": float(np.log10(max(r.amount_mid, 1_000))),
            "size_vs_member": float(r.size_vs_member),
            "cluster_count": int(r.cluster_count),
            "disclosure_lag": float(r.disclosure_lag if r.disclosure_lag is not None else 30),
            "owner_self": int(r.owner in ("self", "unknown")),
            "asset_option": int(isinstance(r.asset_type, str)
                                and "option" in r.asset_type.lower()),
            "party_dem": int(isinstance(r.party, str) and "democrat" in r.party.lower()),
            "momentum_63": _trailing_return(store, r.ticker, r.filing_date, 63),
            "momentum_252": _trailing_return(store, r.ticker, r.filing_date, 252),
            "volatility_63": _trailing_vol(store, r.ticker, r.filing_date, 63),
            "market_cap_log": float(np.log10(r.market_cap))
                              if (r.market_cap and r.market_cap > 0) else np.nan,
            "filing_estimated": bool(r.filing_estimated),
        }
        if with_label:
            lbl = store.beats_market(r.ticker, r.filing_date, LABEL_HORIZON)
            row["label"] = lbl
            row["fwd_excess"] = store.abnormal_return(r.ticker, r.filing_date, LABEL_HORIZON)
        feats.append(row)

    out = pd.DataFrame(feats)
    if with_label:
        out = out.dropna(subset=["label"])
        out["label"] = out["label"].astype(int)
    return out


def _trailing_return(store: PriceStore, ticker: str, asof, window: int) -> float:
    """Return over the `window` trading days ending at asof (momentum)."""
    if not store.has(ticker):
        return np.nan
    i = store._pos_on_or_after(asof)
    if i is None:
        return np.nan
    col = store.close[ticker].to_numpy()
    e = i if i < len(col) else len(col) - 1
    lo = e - window
    if lo < 0 or not (np.isfinite(col[e]) and np.isfinite(col[lo])) or col[lo] <= 0:
        return np.nan
    return float(col[e] / col[lo] - 1.0)


def _trailing_vol(store: PriceStore, ticker: str, asof, window: int) -> float:
    if not store.has(ticker):
        return np.nan
    i = store._pos_on_or_after(asof)
    if i is None:
        return np.nan
    s = store.close[ticker].iloc[max(0, i - window):i].pct_change().dropna()
    return float(s.std()) if len(s) > 5 else np.nan


# Feature columns fed to the model (order matters for SHAP display).
FEATURE_COLS = [
    "member_skill", "committee_match", "amount_mid_log", "size_vs_member",
    "cluster_count", "disclosure_lag", "owner_self", "asset_option",
    "party_dem", "momentum_63", "momentum_252", "volatility_63", "market_cap_log",
]
