"""Event study v2 — leak-free, factor-adjusted, cluster-robust.

Fixes the v1 event study on every axis the methodology review flagged:

* ENTRY at the next session's OPEN after the disclosure date (the first price we
  could actually transact at), not the filing-day close.
* ALPHA = Carhart 4-factor abnormal return. For each trade we estimate factor
  betas (MKT-RF, SMB, HML, MOM) on a trailing estimation window ending before
  entry, then the event-window CAR is the sum of daily returns in excess of the
  factor-model expectation. This strips out market/size/value/momentum tilt so a
  growth/tech lean is not miscounted as political alpha. SPY- and sector-adjusted
  CARs are kept alongside for comparison.
* SIGNIFICANCE via block bootstrap CLUSTERED by member and by ticker — because
  repeated trades by one member in one name are not independent, so a naive
  t-stat overstates significance.

Then it re-tests the hypotheses that actually matter (overall, large trades,
fast filings, repeat conviction, large+repeat+momentum) and reports each slice's
factor-adjusted mean CAR with cluster-robust CIs. It also splits a locked
2024-2026 holdout so we can see whether any effect survives out of sample.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import BENCHMARK_TICKER, DATA_DIR
from ..db import connect

HORIZONS = [5, 21, 63]
EST_WINDOW = 252     # estimation window length (trading days)
EST_GAP = 10         # gap between estimation window and entry
MIN_EST = 100        # minimum estimation observations
HOLDOUT_START = "2024-01-01"
FACTOR_COLS = ["mkt_rf", "smb", "hml", "mom"]


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def _load(con):
    px = con.execute("SELECT ticker, date, open, close FROM prices ORDER BY date").df()
    px["date"] = pd.to_datetime(px["date"])
    op = px.pivot_table(index="date", columns="ticker", values="open")
    cl = px.pivot_table(index="date", columns="ticker", values="close")
    ret = cl.pct_change()                       # close-to-close daily returns
    fac = con.execute("SELECT * FROM factors ORDER BY date").df()
    fac["date"] = pd.to_datetime(fac["date"])
    fac = fac.set_index("date").reindex(cl.index)   # align to trading calendar
    return op, cl, ret, fac


def _sector_etf_map(con) -> dict:
    """ticker -> its sector's SPDR ETF, for sector-adjusted CAR."""
    yahoo_to_etf = {
        "Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV",
        "Energy": "XLE", "Industrials": "XLI", "Consumer Cyclical": "XLY",
        "Consumer Defensive": "XLP", "Utilities": "XLU", "Basic Materials": "XLB",
        "Real Estate": "XLRE", "Communication Services": "XLC",
    }
    rows = con.execute("SELECT ticker, sector FROM ticker_sectors").fetchall()
    return {t: yahoo_to_etf.get(s) for t, s in rows}


# ---------------------------------------------------------------------------
# Per-trade CARs
# ---------------------------------------------------------------------------
def compute(con=None) -> pd.DataFrame:
    own = con is None
    con = con or connect(read_only=True)
    op, cl, ret, fac = _load(con)
    sector_etf = _sector_etf_map(con)
    trades = con.execute("""
        SELECT trade_id, member_key, member, ticker, tx_date, filing_date,
               disclosure_lag, amount_mid
        FROM trades
        WHERE tx_type='purchase' AND ticker IS NOT NULL AND NOT filing_estimated
        ORDER BY filing_date
    """).df()
    if own:
        con.close()

    trades["filing_date"] = pd.to_datetime(trades["filing_date"])
    trades["tx_date"] = pd.to_datetime(trades["tx_date"])
    cal = cl.index
    fac_arr = fac[FACTOR_COLS].to_numpy()
    rf_arr = fac["rf"].to_numpy()

    # prior-purchase lookup for repeat detection
    hist: dict[tuple, list] = {}
    tk_hist: dict[str, list] = {}

    recs = []
    for r in trades.itertuples(index=False):
        col = cl.columns.get_loc(r.ticker) if r.ticker in cl.columns else None
        # next session strictly after the disclosure date
        i0 = cal.searchsorted(r.filing_date, side="right")
        # repeat / cluster features (point-in-time: only prior filings)
        key = (r.member_key, r.ticker)
        prior_same = [d for d in hist.get(key, []) if d < r.filing_date]
        is_repeat = any(r.filing_date - d <= pd.Timedelta(days=365) for d in prior_same)
        clus = len({m for (m, d) in tk_hist.get(r.ticker, [])
                    if 0 <= (r.filing_date - d).days <= 30 and m != r.member_key})
        hist.setdefault(key, []).append(r.filing_date)
        tk_hist.setdefault(r.ticker, []).append((r.member_key, r.filing_date))

        rec = {"trade_id": r.trade_id, "member_key": r.member_key, "ticker": r.ticker,
               "filing_date": r.filing_date, "disclosure_lag": r.disclosure_lag,
               "amount_mid": r.amount_mid, "is_repeat": bool(is_repeat),
               "cluster_count": int(clus), "year": int(r.filing_date.year)}

        if col is None or i0 >= len(cal):
            recs.append({**rec, **{f"car_ff_{h}": np.nan for h in HORIZONS}})
            continue
        cl_col = cl.iloc[:, col].to_numpy()
        op_col = op.iloc[:, col].to_numpy()
        ret_col = ret.iloc[:, col].to_numpy()

        # entry: first finite open at/after i0
        e = next((j for j in range(i0, len(cal)) if np.isfinite(op_col[j])), None)
        if e is None:
            recs.append({**rec, **{f"car_ff_{h}": np.nan for h in HORIZONS}})
            continue

        # factor betas on the estimation window [e-GAP-EST, e-GAP]
        lo = max(0, e - EST_GAP - EST_WINDOW)
        hi = max(0, e - EST_GAP)
        y = ret_col[lo:hi] - rf_arr[lo:hi]
        X = fac_arr[lo:hi]
        betas = _fit_betas(y, X)

        # post-disclosure momentum (20d return before entry) for the combo slice
        mom20 = (cl_col[e - 1] / cl_col[e - 21] - 1.0
                 if e >= 21 and np.isfinite(cl_col[e - 1]) and np.isfinite(cl_col[e - 21])
                 and cl_col[e - 21] > 0 else np.nan)
        rec["mom20"] = float(mom20) if np.isfinite(mom20) else np.nan

        for h in HORIZONS:
            rec[f"car_ff_{h}"] = _car_ff(op_col, cl_col, ret_col, rf_arr, fac_arr, betas, e, h)
            rec[f"car_spy_{h}"] = _car_bench(op_col, cl_col, e, h, cl, BENCHMARK_TICKER)
            etf = sector_etf.get(r.ticker)
            rec[f"car_sec_{h}"] = (_car_bench(op_col, cl_col, e, h, cl, etf)
                                   if etf and etf in cl.columns else np.nan)
        recs.append(rec)

    return pd.DataFrame(recs)


def _fit_betas(y: np.ndarray, X: np.ndarray) -> np.ndarray | None:
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if m.sum() < MIN_EST:
        return None
    yv, Xv = y[m], X[m]
    A = np.column_stack([np.ones(len(Xv)), Xv])       # intercept + 4 factors
    try:
        coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return coef[1:]                                    # drop intercept (alpha)


def _car_ff(op_col, cl_col, ret_col, rf, fac, betas, e, h) -> float:
    """Cumulative factor-adjusted return: buy next open, hold h trading days."""
    if betas is None or e + h >= len(cl_col):
        return np.nan
    if not (np.isfinite(op_col[e]) and np.isfinite(cl_col[e]) and op_col[e] > 0):
        return np.nan
    # day-1 return from entry open to entry close, then close-to-close
    day1 = cl_col[e] / op_col[e] - 1.0
    abn = 0.0
    # expected day-1 ~ factor model on entry day
    if np.all(np.isfinite(fac[e])):
        abn += (day1 - rf[e]) - betas @ fac[e]
    else:
        abn += day1
    for j in range(e + 1, e + h):
        rj = ret_col[j]
        if not np.isfinite(rj):
            continue
        if np.all(np.isfinite(fac[j])):
            abn += (rj - rf[j]) - betas @ fac[j]
        else:
            abn += rj
    return float(abn)


def _car_bench(op_col, cl_col, e, h, cl, bench) -> float:
    """Buy-and-hold abnormal return vs a benchmark (next-open entry)."""
    if bench is None or bench not in cl.columns or e + h >= len(cl_col):
        return np.nan
    b = cl[bench].to_numpy()
    if not (np.isfinite(op_col[e]) and np.isfinite(cl_col[e + h]) and op_col[e] > 0):
        return np.nan
    stock = cl_col[e + h] / op_col[e] - 1.0
    if not (np.isfinite(b[e]) and np.isfinite(b[e + h]) and b[e] > 0):
        return np.nan
    return float(stock - (b[e + h] / b[e] - 1.0))


# ---------------------------------------------------------------------------
# Clustered block bootstrap
# ---------------------------------------------------------------------------
def _cluster_bootstrap(df: pd.DataFrame, col: str, cluster_key: str,
                       n_boot: int = 2000, seed: int = 7) -> dict:
    v = df[[col, cluster_key]].dropna()
    if len(v) < 20:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": int(len(v))}
    rng = np.random.default_rng(seed)
    groups = [g[col].to_numpy() for _, g in v.groupby(cluster_key)]
    gmeans = np.array([g.mean() for g in groups])
    gsizes = np.array([len(g) for g in groups], dtype=float)
    ng = len(groups)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, ng, ng)
        boot[b] = np.average(gmeans[idx], weights=gsizes[idx])
    return {"mean": float(v[col].mean()),
            "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)),
            "n": int(len(v)), "n_clusters": int(ng)}


def _amount_bucket(m) -> str:
    if not np.isfinite(m):
        return "unknown"
    if m < 15_000:
        return "$1k-15k"
    if m < 50_000:
        return "$15k-50k"
    if m < 250_000:
        return "$50k-250k"
    if m < 1_000_000:
        return "$250k-1M"
    return "$1M+"


def _slice_report(df: pd.DataFrame, col: str) -> dict:
    """For a metric column, report cluster-robust CIs (member & ticker) for the
    overall set and each hypothesis slice."""
    df = df.copy()
    df["bucket"] = df["amount_mid"].map(_amount_bucket)
    df["is_large"] = df["amount_mid"] >= 50_000
    df["is_fast"] = df["disclosure_lag"] <= 14
    df["combo"] = df["is_large"] & df["is_repeat"] & (df["mom20"] > 0)

    def both(sub):
        return {"by_member": _cluster_bootstrap(sub, col, "member_key"),
                "by_ticker": _cluster_bootstrap(sub, col, "ticker")}

    out = {"overall": both(df)}
    out["by_amount"] = {b: both(df[df["bucket"] == b]) for b in df["bucket"].unique()}
    out["large_trades"] = both(df[df["is_large"]])
    out["fast_filings"] = both(df[df["is_fast"]])
    out["repeat_purchases"] = both(df[df["is_repeat"]])
    out["large_repeat_momentum"] = both(df[df["combo"]])
    return out


def run(verbose: bool = True) -> dict:
    per_trade = compute()
    # persist per-trade CARs for reuse by the backtest / dashboard
    con = connect()
    con.register("per_trade_df", per_trade)
    con.execute("CREATE OR REPLACE TABLE event_v2 AS SELECT * FROM per_trade_df")
    con.close()

    train = per_trade[per_trade["filing_date"] < HOLDOUT_START]
    holdo = per_trade[per_trade["filing_date"] >= HOLDOUT_START]

    report = {}
    for h in HORIZONS:
        report[f"ff_{h}"] = {
            "train": _slice_report(train, f"car_ff_{h}"),
            "holdout_2024plus": _slice_report(holdo, f"car_ff_{h}"),
        }
    payload = {"horizons": HORIZONS, "holdout_start": HOLDOUT_START,
               "adjustment": "carhart_4factor", "entry": "next_session_open",
               "report": report, "generated_at": datetime.now(timezone.utc).isoformat()}
    with open(DATA_DIR / "event_study_v2.json", "w") as f:
        json.dump(payload, f, indent=2)

    if verbose:
        _print(report)
    return payload


def _print(report):
    print("=== Event study v2 — Carhart-adjusted CAR, cluster-robust 95% CI ===")
    for h in HORIZONS:
        print(f"\n--- horizon {h}d (TRAIN <2024) ---")
        r = report[f"ff_{h}"]["train"]
        for name in ["overall", "large_trades", "fast_filings",
                     "repeat_purchases", "large_repeat_momentum"]:
            s = r[name]["by_member"]
            st = r[name]["by_ticker"]
            if s["mean"] is None:
                continue
            sig = "  *" if (s["ci_low"] > 0 or s["ci_high"] < 0) else ""
            sigt = "*" if (st["ci_low"] is not None and (st["ci_low"] > 0 or st["ci_high"] < 0)) else " "
            print(f"  {name:22s} mean={s['mean']*100:+.2f}%  "
                  f"member-CI[{s['ci_low']*100:+.2f},{s['ci_high']*100:+.2f}]{sig}  "
                  f"ticker-sig={sigt}  n={s['n']} ({s['n_clusters']} members)")
        # holdout for the headline large-trade slice
        hh = report[f"ff_{h}"]["holdout_2024plus"]["large_trades"]["by_member"]
        if hh["mean"] is not None:
            print(f"  {'large_trades [HOLDOUT]':22s} mean={hh['mean']*100:+.2f}%  "
                  f"CI[{hh['ci_low']*100:+.2f},{hh['ci_high']*100:+.2f}]  n={hh['n']}")


if __name__ == "__main__":
    run()
