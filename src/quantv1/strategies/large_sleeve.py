"""Deployable LARGE sleeve v1 — volatility-targeted, original rule unchanged.

The audit showed LARGE is the strongest candidate (positive Carhart alpha, low
market beta, ~62% avg gross) but NOT fully proven (alpha t=1.90). This wraps the
UNCHANGED LARGE rule (amount_mid >= $50k) in a volatility-targeting overlay so it
can be paper-deployed at a chosen risk level, without adding any of the
audit-discovered filters (spouse / lag / new-position / $250k-1M) — those carry
multiple-comparison risk and live only as shadow portfolios.

Volatility target (per the spec):
    m_t = clip( sigma_target / max(sigma_20d, sigma_60d), 0.5, m_max )
m_t is smoothed (EWMA) to avoid noisy resizing and excess turnover, and the
resulting gross is capped; leverage above 100% is charged a financing rate.

Two configs:
    default    16% vol, gross <= 100% (no borrowing needed given 62% base)
    aggressive 22% vol, gross <= 140%, financing on the >100% portion
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import DATA_DIR, BENCHMARK_TICKER
from ..db import connect
from ..research.large_audit import _run_large

# Forward-holdout freeze: everything from this date is the GENUINE out-of-sample
# record (2024+ was consumed by the audit and is no longer untouched).
FORWARD_FREEZE = "2026-07-10"

CONFIGS = {
    "default":    {"target_vol": 0.16, "max_gross": 1.00, "m_max": 2.5, "financing": 0.00},
    "aggressive": {"target_vol": 0.22, "max_gross": 1.40, "m_max": 3.0, "financing": 0.06},
}
SMOOTH_SPAN = 5     # EWMA span for the multiplier (days)


def _vol_target(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    r = daily["ret"].to_numpy()
    base_gross = daily["gross"].to_numpy()
    s20 = pd.Series(r).rolling(20).std() * np.sqrt(252)
    s60 = pd.Series(r).rolling(60).std() * np.sqrt(252)
    vol = np.maximum(s20, s60)
    m = cfg["target_vol"] / vol.replace(0, np.nan)
    m = m.clip(0.5, cfg["m_max"])
    m = m.shift(1)                                   # use vol known through t-1 (no look-ahead)
    m = m.ewm(span=SMOOTH_SPAN).mean().fillna(1.0)   # smooth to cut turnover

    eff_gross = np.minimum(base_gross * m.to_numpy(), cfg["max_gross"])
    eff_mult = np.divide(eff_gross, base_gross,
                         out=np.zeros_like(eff_gross), where=base_gross > 0)
    lev_ret = eff_mult * r
    borrow = np.maximum(eff_gross - 1.0, 0.0)
    lev_ret = lev_ret - borrow * cfg["financing"] / 252.0
    out = daily.copy()
    out["mult"] = eff_mult
    out["eff_gross"] = eff_gross
    out["lev_ret"] = lev_ret
    return out


def _metrics(dates, rets) -> dict:
    rets = np.asarray(rets)
    rets = rets[np.isfinite(rets)]
    if len(rets) < 30:
        return {}
    eq = np.cumprod(1 + rets)
    yrs = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25
    return {"cagr": float(eq[-1] ** (1 / yrs) - 1) if yrs > 0 else None,
            "vol": float(rets.std() * np.sqrt(252)),
            "sharpe": float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else None,
            "max_dd": float(np.min(eq / np.maximum.accumulate(eq) - 1)),
            "final": float(eq[-1])}


def run(verbose: bool = True) -> dict:
    con = connect(read_only=True)
    daily = _run_large(con)
    con.close()
    daily["date"] = pd.to_datetime(daily["date"])

    out = {"forward_freeze": FORWARD_FREEZE, "smooth_span": SMOOTH_SPAN, "configs": {}}
    base = _metrics(daily["date"].tolist(), daily["ret"])
    out["unlevered_base"] = {**base, "avg_gross": float(daily["gross"].mean())}

    for name, cfg in CONFIGS.items():
        lev = _vol_target(daily, cfg)
        full = _metrics(lev["date"].tolist(), lev["lev_ret"])
        curve = [{"date": str(d.date()), "equity": float(e)}
                 for d, e in zip(lev["date"][::5],
                                 np.cumprod(1 + lev["lev_ret"].fillna(0))[::5])]
        out["configs"][name] = {
            **cfg, **full,
            "avg_gross": float(lev["eff_gross"].mean()),
            "max_gross_used": float(lev["eff_gross"].max()),
            "pct_time_levered": float((lev["eff_gross"] > 1.0).mean()),
            "avg_multiplier": float(lev["mult"].mean()),
            "curve": curve,
        }

    with open(DATA_DIR / "large_sleeve.json", "w") as f:
        json.dump(out, f, indent=2)
    if verbose:
        b = out["unlevered_base"]
        print(f"unlevered LARGE: CAGR={b['cagr']:.1%} vol={b['vol']:.1%} "
              f"Sharpe={b['sharpe']:.2f} DD={b['max_dd']:.1%} gross={b['avg_gross']:.0%}")
        for name, c in out["configs"].items():
            print(f"{name:11s} vol_target={c['target_vol']:.0%} -> "
                  f"CAGR={c['cagr']:.1%} vol={c['vol']:.1%} Sharpe={c['sharpe']:.2f} "
                  f"DD={c['max_dd']:.1%} avg_gross={c['avg_gross']:.0%} "
                  f"levered {c['pct_time_levered']:.0%} of days")
        print(f"forward holdout freeze: {FORWARD_FREEZE} (2024+ was consumed by the audit)")
    return out


if __name__ == "__main__":
    run()
