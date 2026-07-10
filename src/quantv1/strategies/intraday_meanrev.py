"""Sector-relative intraday mean-reversion (fast-trigger proof-of-concept).

The generic "provide trades when no political events occur" strategy from the
plan, at hourly resolution (all the free intraday history yfinance gives). It is
a standard cross-sectional reversal:

  1. sector-relative return  r_rel = r_stock - r_sector_etf   (removes sector move)
  2. signal = -(cumulative r_rel over the last K bars)         (recent losers score high)
  3. cross-sectionally demean the signal each bar, cap, gross-normalize to 1
     -> dollar-neutral long-losers / short-winners book
  4. execute at the NEXT bar's open; charge spread+slippage on turnover

This is deliberately honest about costs, because hourly reversal trades a lot and
transaction costs usually decide whether it survives. We report gross AND net, a
2025+ holdout, and a turnover-cost sweep so the cost sensitivity is explicit.

Caveats: ~2 years of hourly bars is thin; overnight gaps are inside the return
series; and this is a general strategy with no political-information edge — it is
scaffolding for the fast layer, not a validated alpha.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import DATA_DIR, SECTOR_ETFS, BENCHMARK_TICKER
from ..db import connect

# Yahoo sector -> SPDR sector ETF (same mapping used in the event study).
YAHOO_TO_ETF = {
    "Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV",
    "Energy": "XLE", "Industrials": "XLI", "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP", "Utilities": "XLU", "Basic Materials": "XLB",
    "Real Estate": "XLRE", "Communication Services": "XLC",
}
HOLDOUT_START = "2025-01-01"
BARS_PER_YEAR = 252 * 6.5


@dataclass
class Params:
    lookback: int = 6         # K bars for the reversal signal (~1 day)
    max_w: float = 0.05       # per-name weight cap
    cost_bps_side: float = 7  # spread + slippage per side (bps)


def _load(con):
    bars = con.execute("SELECT ticker, ts, open, close FROM bars_hourly ORDER BY ts").df()
    bars["ts"] = pd.to_datetime(bars["ts"])
    close = bars.pivot_table(index="ts", columns="ticker", values="close")
    open_ = bars.pivot_table(index="ts", columns="ticker", values="open")
    sec = dict(con.execute("SELECT ticker, sector FROM ticker_sectors").fetchall())
    return close, open_, sec


def run(params: Params | None = None, verbose: bool = True) -> dict:
    p = params or Params()
    con = connect(read_only=True)
    close, open_, sec = _load(con)
    con.close()

    etfs = set(SECTOR_ETFS) | {BENCHMARK_TICKER, "QQQ"}
    names = [t for t in close.columns
             if t not in etfs and YAHOO_TO_ETF.get(sec.get(t)) in close.columns]
    if len(names) < 20:
        raise RuntimeError(f"only {len(names)} usable names — ingest hourly bars first")

    ret = close.pct_change()
    # sector-relative return per name
    rel = pd.DataFrame(index=close.index)
    for t in names:
        etf = YAHOO_TO_ETF[sec[t]]
        rel[t] = ret[t] - ret[etf]
    # signal: negative cumulative sector-relative return over K bars (losers high)
    signal = -rel.rolling(p.lookback).sum()

    # execution return: decide at close[i], buy at open[i+1], exit at open[i+2].
    # fwd_open_ret.iloc[i] = open[i+2] / open[i+1] - 1  (no overlap with signal bar)
    oret = open_[names].shift(-2) / open_[names].shift(-1) - 1.0

    idx = close.index
    equity, dates = [1.0], [idx[p.lookback]]
    w_prev = pd.Series(0.0, index=names)
    turn_series = []
    for i in range(p.lookback, len(idx) - 2):
        s = signal.iloc[i][names]
        s = s[np.isfinite(s)]
        if len(s) < 10:
            equity.append(equity[-1]); dates.append(idx[i + 1])
            turn_series.append(0.0); continue
        s = s - s.mean()                     # cross-sectional demean -> dollar neutral
        w = s / s.abs().sum()                # gross ~ 1
        w = w.clip(-p.max_w, p.max_w)
        w = w / w.abs().sum()                # renormalize gross to 1 after cap
        w = w.reindex(names).fillna(0.0)

        r_next = oret.iloc[i][names].fillna(0.0)
        gross_ret = float((w * r_next).sum())
        turn = float((w - w_prev).abs().sum())
        turn_series.append(turn)
        net_ret = gross_ret - turn * p.cost_bps_side / 1e4
        equity.append(equity[-1] * (1 + net_ret))
        dates.append(idx[i + 1])
        w_prev = w

    res = _metrics(dates, equity, np.mean(turn_series) if turn_series else 0)
    res["params"] = {"lookback": p.lookback, "max_w": p.max_w,
                     "cost_bps_side": p.cost_bps_side}
    res["n_names"] = len(names)

    # cost sensitivity: rerun net metrics under a few cost levels (reuse gross path)
    res["cost_sweep"] = _cost_sweep(dates, equity, turn_series, p)

    with open(DATA_DIR / "intraday_meanrev.json", "w") as f:
        json.dump(res, f, indent=2)
    if verbose:
        _print(res)
    return res


def _seg(dates, eq, mask):
    eq = np.asarray(eq)[mask]
    if len(eq) < 50:
        return None
    eq = eq / eq[0]
    r = np.diff(eq) / eq[:-1]
    sharpe = float(np.mean(r) / np.std(r) * np.sqrt(BARS_PER_YEAR)) if np.std(r) > 0 else None
    yrs = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25
    return {"cagr": float(eq[-1] ** (1 / yrs) - 1) if yrs > 0 else None,
            "sharpe": sharpe,
            "max_dd": float(np.min(eq / np.maximum.accumulate(eq) - 1)),
            "final": float(eq[-1])}


def _metrics(dates, equity, avg_turn):
    idx = pd.to_datetime(dates)
    ho = pd.Timestamp(HOLDOUT_START)
    return {"full": _seg(dates, equity, np.ones(len(equity), bool)),
            "train": _seg(dates, equity, np.asarray(idx < ho)),
            "holdout_2025plus": _seg(dates, equity, np.asarray(idx >= ho)),
            "avg_turnover_per_bar": float(avg_turn),
            "curve": [{"ts": str(d), "equity": float(e)}
                      for d, e in zip(dates[::5], equity[::5])]}


def _cost_sweep(dates, equity, turn_series, p):
    # back out gross per-bar return from net, then re-apply different costs
    eq = np.asarray(equity)
    net_r = np.diff(eq) / eq[:-1]
    turns = np.asarray(turn_series)
    gross_r = net_r + turns * p.cost_bps_side / 1e4
    out = {}
    for c in [0, 3, 5, 7, 10]:
        e = np.cumprod(1 + (gross_r - turns * c / 1e4))
        r = gross_r - turns * c / 1e4
        sharpe = float(np.mean(r) / np.std(r) * np.sqrt(BARS_PER_YEAR)) if np.std(r) > 0 else None
        out[f"{c}bps"] = {"sharpe": sharpe, "final": float(e[-1])}
    return out


def _print(res):
    print(f"=== Sector-relative intraday (hourly) mean reversion — {res['n_names']} names ===")
    for k in ("full", "train", "holdout_2025plus"):
        m = res[k]
        if m:
            print(f"  {k:16s} Sharpe={m['sharpe']:+.2f} CAGR={m['cagr']*100:+.1f}% "
                  f"maxDD={m['max_dd']*100:.1f}% finalx={m['final']:.2f}")
    print(f"  avg turnover/bar={res['avg_turnover_per_bar']:.2f}")
    print("  cost sweep (Sharpe): " +
          " ".join(f"{c}={v['sharpe']:+.2f}" for c, v in res["cost_sweep"].items()))


if __name__ == "__main__":
    run()
