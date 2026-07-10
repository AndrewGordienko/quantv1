"""Point-in-time backtest of the congressional-copy strategy.

This is where the project lives or dies, so it is deliberately strict:

* Signal is generated only from data knowable at each rebalance date. The model
  is REFIT every rebalance on trades whose outcome had already realized, and the
  member-skill feature is recomputed point-in-time from those same realized
  trades — no full-history leakage.
* Entry is at the filing date (never the transaction date).
* Turnover is charged COST_BPS per side.
* Benchmarks: SPY buy-and-hold and a naive "copy every purchase, equal weight"
  book. The model has to beat the naive copy, or it is adding nothing.

Returns are compounded across monthly holding intervals using adjusted closes.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from ..config import (BENCHMARK_TICKER, COST_BPS, LABEL_HORIZON, LOOKBACK_DAYS,
                      TOP_K)
from ..db import connect
from ..model import features as F
from ..research.returns import PriceStore
from ..research.skill import eb_shrink
from . import construct as C

# ~calendar days for LABEL_HORIZON trading days to have elapsed (realization gate)
REALIZE_CALENDAR_DAYS = int(LABEL_HORIZON * 1.45)


def _fit_model(train: pd.DataFrame) -> LGBMClassifier | None:
    if len(train) < 200 or train["label"].nunique() < 2:
        return None
    gbm = LGBMClassifier(
        n_estimators=250, learning_rate=0.03, num_leaves=15,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, verbose=-1,
    )
    gbm.fit(train[F.FEATURE_COLS].astype(float), train["label"].to_numpy())
    return gbm


def _pit_skill(realized: pd.DataFrame) -> dict:
    """Point-in-time member skill from already-realized purchases."""
    if realized.empty:
        return {}
    df = realized.rename(columns={"fwd_excess": "ar"})[["member_key", "member", "ar"]].dropna()
    if df.empty:
        return {}
    s = eb_shrink(df)
    return dict(zip(s["member_key"], s["shrunk_car"]))


def _interval_return(store: PriceStore, ticker: str, d0, d1) -> float | None:
    """Close-to-close return of ticker between two calendar dates."""
    e0 = store.entry_price(ticker, d0)
    e1 = store.entry_price(ticker, d1)
    if e0 is None or e1 is None or e0[1] <= 0:
        return None
    return e1[1] / e0[1] - 1.0


def run(start_after: str = "2016-01-01", freq_days: int = 21,
        persist_results: bool = True, verbose: bool = True) -> dict:
    store = PriceStore()
    # Full labeled feature frame (House-only for point-in-time rigor).
    data = F.build(store=store, with_label=True, purchases_only=True)
    data = data[~data["filing_estimated"]].reset_index(drop=True)
    data["filing_date"] = pd.to_datetime(data["filing_date"])

    start = pd.Timestamp(start_after)
    end = data["filing_date"].max()
    rebal_dates = pd.date_range(start, end, freq=f"{freq_days}D")

    equity = {"model": [1.0], "spy": [1.0], "naive": [1.0]}
    dates_out = [rebal_dates[0]]
    prev_book: pd.DataFrame = pd.DataFrame(columns=["ticker", "weight"])
    prev_naive: dict = {}
    books_log = []

    for i in range(len(rebal_dates) - 1):
        D, Dn = rebal_dates[i], rebal_dates[i + 1]

        # --- point-in-time training set: outcomes realized before D ---------
        realized = data[data["filing_date"] <= D - pd.Timedelta(days=REALIZE_CALENDAR_DAYS)]
        skill_map = _pit_skill(realized)
        train = realized.copy()
        train["member_skill"] = train["member_key"].map(skill_map).fillna(0.0)
        model = _fit_model(train)

        # --- scoring window: trades filed in the trailing lookback ----------
        window = data[(data["filing_date"] > D - pd.Timedelta(days=LOOKBACK_DAYS)) &
                      (data["filing_date"] <= D)].copy()
        book = pd.DataFrame(columns=["ticker", "weight"])
        if model is not None and not window.empty:
            window["member_skill"] = window["member_key"].map(skill_map).fillna(0.0)
            window["score"] = model.predict_proba(
                window[F.FEATURE_COLS].astype(float))[:, 1]
            book = C.construct(window, top_k=TOP_K, score_threshold=0.5)

        # --- realize model portfolio return over [D, Dn] --------------------
        r_model = _book_return(store, book, D, Dn)
        turn = _turnover(prev_book, book)
        r_model -= turn * 2 * COST_BPS / 1e4
        equity["model"].append(equity["model"][-1] * (1 + r_model))
        prev_book = book

        # --- naive copy-all-purchases (equal weight) ------------------------
        naive_tickers = sorted(set(window["ticker"])) if not window.empty else []
        naive_book = pd.DataFrame({"ticker": naive_tickers,
                                   "weight": [1.0 / len(naive_tickers)] * len(naive_tickers)}) \
            if naive_tickers else pd.DataFrame(columns=["ticker", "weight"])
        r_naive = _book_return(store, naive_book, D, Dn)
        turn_n = _turnover(_dict_to_book(prev_naive), naive_book)
        r_naive -= turn_n * 2 * COST_BPS / 1e4
        equity["naive"].append(equity["naive"][-1] * (1 + r_naive))
        prev_naive = dict(zip(naive_book["ticker"], naive_book["weight"]))

        # --- SPY buy & hold -------------------------------------------------
        r_spy = _interval_return(store, BENCHMARK_TICKER, D, Dn) or 0.0
        equity["spy"].append(equity["spy"][-1] * (1 + r_spy))

        dates_out.append(Dn)
        books_log.append({"date": D, "n_positions": len(book),
                          "tickers": list(book["ticker"]) if not book.empty else []})
        if verbose and i % 12 == 0:
            print(f"  {D.date()} model={equity['model'][-1]:.3f} "
                  f"spy={equity['spy'][-1]:.3f} naive={equity['naive'][-1]:.3f} "
                  f"(book={len(book)})")

    result = _metrics(dates_out, equity, freq_days)
    result["books_log"] = books_log
    if persist_results:
        _persist(dates_out, equity)
    if verbose:
        for k, m in result["stats"].items():
            print(f"{k:6s} CAGR={m['cagr']:.2%} Sharpe={m['sharpe']:.2f} "
                  f"maxDD={m['max_dd']:.2%} finalx={m['final']:.2f}")
    return result


# --- helpers ---------------------------------------------------------------
def _dict_to_book(d: dict) -> pd.DataFrame:
    return (pd.DataFrame({"ticker": list(d), "weight": list(d.values())})
            if d else pd.DataFrame(columns=["ticker", "weight"]))


def _book_return(store: PriceStore, book: pd.DataFrame, d0, d1) -> float:
    if book.empty:
        return 0.0
    tot, wsum = 0.0, 0.0
    for r in book.itertuples(index=False):
        ret = _interval_return(store, r.ticker, d0, d1)
        if ret is None:
            continue
        tot += r.weight * ret
        wsum += r.weight
    return tot / wsum if wsum > 0 else 0.0


def _turnover(old: pd.DataFrame, new: pd.DataFrame) -> float:
    ow = dict(zip(old["ticker"], old["weight"])) if not old.empty else {}
    nw = dict(zip(new["ticker"], new["weight"])) if not new.empty else {}
    tickers = set(ow) | set(nw)
    return 0.5 * sum(abs(nw.get(t, 0) - ow.get(t, 0)) for t in tickers)


def _metrics(dates, equity, freq_days) -> dict:
    per_year = 252 / freq_days
    stats = {}
    for k, eq in equity.items():
        eq = np.array(eq)
        rets = np.diff(eq) / eq[:-1]
        years = (dates[-1] - dates[0]).days / 365.25
        cagr = eq[-1] ** (1 / years) - 1 if years > 0 and eq[-1] > 0 else np.nan
        sharpe = (np.mean(rets) / np.std(rets) * np.sqrt(per_year)
                  if np.std(rets) > 0 else np.nan)
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.min(eq / peak - 1))
        stats[k] = {"cagr": float(cagr), "sharpe": float(sharpe),
                    "max_dd": max_dd, "final": float(eq[-1])}
    curve = [{"date": str(d.date()), "model": float(equity["model"][i]),
              "spy": float(equity["spy"][i]), "naive": float(equity["naive"][i])}
             for i, d in enumerate(dates)]
    return {"stats": stats, "curve": curve}


def _persist(dates, equity) -> None:
    con = connect()
    con.execute("DELETE FROM backtest_equity")
    rows = []
    for strat in ("model", "spy", "naive"):
        for i, d in enumerate(dates):
            rows.append([strat, d.date(), float(equity[strat][i])])
    con.executemany("INSERT INTO backtest_equity VALUES (?,?,?)", rows)
    con.close()


if __name__ == "__main__":
    run()
