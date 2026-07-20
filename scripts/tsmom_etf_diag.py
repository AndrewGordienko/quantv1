"""Diversified time-series (trend) momentum on liquid ETF proxies — DIAGNOSTIC.

Backlog #5 (Moskowitz-Ooi-Pedersen). Uniquely attractive among the backlog:
  * UNBLOCKED and SURVIVORSHIP-IMMUNE — we trade liquid index/asset ETF proxies
    themselves (all present, full 2012-2026 history), not a delisting cross-
    section. The pit_panel_audit survivorship problem does not apply here.
  * LOW TURNOVER (monthly rebalance) -> cost-robust, unlike the daily-reversal
    cost-trap that killed backlog #2's simple form.
  * Strong published prior, but TSMOM has decayed since ~2011 -> genuinely
    uncertain over 2012-2026. This is a real test, not a foregone null.

Frozen, point-in-time rules (no look-ahead): signal decided at month-end t from
data through t; position held over month t+1; vol scaling from trailing vol
through t. Costs on realized month-end turnover. Diagnostic, not a promotion:
Sharpe>1 + delayed-entry + doubled-cost gates still apply before any claim.

Output: data/tsmom_etf_diag.json.
"""

from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd

from quantv1.config import DB_PATH, DATA_DIR

OUT = DATA_DIR / "tsmom_etf_diag.json"

# --- frozen diversified basket (de-duplicated across asset classes) --------
BASKET = {
    "equity": ["SPY", "IWM", "QQQ", "EFA", "EEM"],
    "rates":  ["TLT", "IEF"],
    "credit": ["LQD", "HYG"],
    "commodity": ["GLD", "USO", "DBA"],
    "real_estate": ["VNQ"],
}
INSTRUMENTS = [t for v in BASKET.values() for t in v]      # 13
LOOKBACKS = [63, 126, 252]     # ~3, 6, 12 months
TARGET_VOL = 0.10              # per-instrument annualized vol target
MAX_LEV = 3.0                  # per-instrument leverage cap (vol-floor)
VOL_WIN = 60                   # trailing daily-vol window
COST_BPS = [2.0, 5.0]          # per-side ETF cost (liquid); doubled stress via 5
ANN = 252


def load():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    px = con.execute(
        "SELECT ticker,date,close FROM prices WHERE ticker IN (SELECT UNNEST(?)) "
        "ORDER BY date", [INSTRUMENTS]).df()
    con.close()
    close = px.pivot(index="date", columns="ticker", values="close").sort_index()
    return close[INSTRUMENTS]


def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return s.groupby([idx.year, idx.month]).last().values


def run(close, lookbacks, bps):
    rets = close.pct_change()
    vol = rets.rolling(VOL_WIN, min_periods=VOL_WIN // 2).std() * np.sqrt(ANN)
    me = pd.DatetimeIndex(month_ends(close.index))
    # combined signal = average sign across lookbacks (each in {-1,0,+1})
    sig_components = []
    for L in lookbacks:
        cr = close / close.shift(L) - 1.0
        sig_components.append(np.sign(cr))
    signal = sum(sig_components) / len(sig_components)   # in [-1,1]
    scale = (TARGET_VOL / vol).clip(upper=MAX_LEV)
    w_me = (signal * scale).reindex(me)                  # weights set at month-ends
    w_me = w_me.where(scale.reindex(me).notna())
    # daily weights: hold month-end weight through the next month
    w_daily = w_me.reindex(close.index).ffill()
    # position effective the day AFTER decision; pnl per instrument averaged
    gross = (w_daily.shift(1) * rets).sum(axis=1) / len(INSTRUMENTS)
    # turnover only at rebalance days
    turn = (w_me.fillna(0.0).diff().abs().sum(axis=1) / len(INSTRUMENTS))
    cost_series = pd.Series(0.0, index=close.index)
    cost_series.loc[turn.index] = turn.values * (bps / 1e4)
    net = gross - cost_series.reindex(close.index).fillna(0.0)
    # trim warmup (need max lookback + vol window)
    start = close.index[max(LOOKBACKS) + VOL_WIN]
    gross, net = gross[gross.index >= start], net[net.index >= start]
    return gross.dropna(), net.dropna(), turn[turn.index >= start]


def stats(x, spy=None):
    mu, sd = x.mean(), x.std()
    eq = (1 + x).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    out = {"ann_return": round(float(mu * ANN), 4),
           "ann_vol": round(float(sd * np.sqrt(ANN)), 4),
           "sharpe": round(float(mu / sd * np.sqrt(ANN)), 3) if sd else None,
           "max_drawdown": round(dd, 4), "n_days": int(len(x))}
    if spy is not None:
        j = pd.concat([x, spy], axis=1, join="inner").dropna()
        out["corr_to_spy"] = round(float(j.iloc[:, 0].corr(j.iloc[:, 1])), 3)
    return out


def main():
    close = load()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    spy = con.execute("SELECT date,close FROM prices WHERE ticker='SPY' ORDER BY date").df()
    con.close()
    spy_ret = spy.set_index("date")["close"].pct_change()

    results, n_trials = {}, 0
    variants = [("combined", LOOKBACKS)] + [(f"L{L}", [L]) for L in LOOKBACKS]
    for name, lbs in variants:
        for bps in COST_BPS:
            gross, net, turn = run(close, lbs, bps)
            if len(net) < ANN * 2:
                continue
            n_trials += 1
            results[f"{name}|{int(bps)}bps"] = {
                "gross": stats(gross),
                "net": stats(net, spy_ret),
                "avg_monthly_turnover": round(float(turn[turn > 0].mean()), 3),
                "net_sharpe_gt_1": bool((stats(net)["sharpe"] or 0) > 1.0),
            }

    base = {k: v for k, v in results.items() if k.endswith("2bps")}
    best = max(base.items(), key=lambda kv: kv[1]["net"]["sharpe"] or -9, default=(None, None))
    any_sharpe_gt_1 = any(v["net_sharpe_gt_1"] for v in results.values())
    report = {
        "label": "DIAGNOSTIC_SURVIVORSHIP_IMMUNE_ETF_PROXIES",
        "not_a_promotion_test": True,
        "basket": BASKET, "n_instruments": len(INSTRUMENTS),
        "rules": {"lookbacks_days": LOOKBACKS, "target_vol": TARGET_VOL,
                  "max_leverage": MAX_LEV, "vol_window": VOL_WIN,
                  "rebalance": "monthly", "cost_bps_per_side": COST_BPS},
        "n_variations_tried": n_trials,
        "date_span": [str(close.index.min().date()), str(close.index.max().date())],
        "results": results,
        "best_net_at_2bps": {"variant": best[0],
                             "net_sharpe": best[1]["net"]["sharpe"] if best[1] else None},
        "verdict": ("TREND_SIGNAL_PRESENT_WORTH_GATING" if any_sharpe_gt_1
                    else "WEAK_OR_NO_TREND_SIGNAL_2012_2026"),
        "interpretation": (
            "Even net Sharpe well below 1 can be portfolio-useful if corr_to_spy "
            "is low (diversification). But the promotion gate needs net Sharpe>1 "
            "with positive delayed-entry and doubled-cost; a monthly-rebalanced "
            "TSMOM here is cost-robust, so the Sharpe is the honest read. TSMOM's "
            "post-2011 decay means a weak result is expected, not a bug."),
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"instruments={len(INSTRUMENTS)} variations={n_trials} span={report['date_span']}")
    print(f"{'variant':16s} {'g_Sh':>6} {'n_Sh':>6} {'n_ann':>7} {'n_vol':>6} "
          f"{'maxDD':>7} {'corrSPY':>7} {'turn':>5}")
    for k, v in results.items():
        g, n = v["gross"], v["net"]
        print(f"{k:16s} {g['sharpe']:>6} {n['sharpe']:>6} {n['ann_return']:>7} "
              f"{n['ann_vol']:>6} {n['max_drawdown']:>7} {n.get('corr_to_spy'):>7} "
              f"{v['avg_monthly_turnover']:>5}")
    print(f"VERDICT: {report['verdict']}  best={report['best_net_at_2bps']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
