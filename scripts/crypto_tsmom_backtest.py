"""Crypto TSMOM port — BTC/ETH perps, RUN ONCE, honest (real data).

First crypto experiment per docs/strategy_crypto.md. Ports the frozen equity
TSMOM logic to BTC/ETH perps on REAL data (data/crypto/*.csv). NOT the day-trader
(that is the microstructure/liquidation engine, which needs live L2/trade data) —
this is the daily TREND OVERLAY, the one strategy that survived equities.

Costs modeled (all three, or it's fiction):
  * taker fee ~5 bps/side + slippage ~2 bps/side on daily turnover;
  * FUNDING: a long perp PAYS funding when funding>0 (short receives). Funding
    (~3x/day) aggregated to a daily rate; funding P&L = -weight * daily_funding.

Mandatory sub-period decay (crypto ~3 near-distinct regimes): 2019-21 / 2022 /
2023-24 / 2025-26. Verdict must hold in EVERY regime or REJECT. Reject-only
discipline; Deflated Sharpe vs the global trial count. Output:
data/crypto_tsmom_backtest.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quantv1.config import DATA_DIR

CRYPTO = DATA_DIR / "crypto"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
LOOKBACKS = [30, 60, 120]      # crypto-appropriate daily trend lookbacks
TARGET_VOL = 0.25              # per-sleeve annualized vol target
MAX_LEV = 2.0
VOL_WIN = 30
FEE_SLIP_BPS = 7.0             # 5 taker + 2 slippage, per side, on turnover
ANN = 365
RNG = np.random.default_rng(20260721)

REGIMES = {"2019-2021": ("2019-01-01", "2021-12-31"),
           "2022": ("2022-01-01", "2022-12-31"),
           "2023-2024": ("2023-01-01", "2024-12-31"),
           "2025-2026": ("2025-01-01", "2026-12-31")}


def _load(sym):
    k = pd.read_csv(CRYPTO / f"{sym}_1d.csv", parse_dates=["date"]).set_index("date").sort_index()
    f = pd.read_csv(CRYPTO / f"{sym}_funding.csv", parse_dates=["ts"]).sort_values("ts")
    # aggregate ~3x/day funding to a daily total rate
    fd = f.set_index("ts")["funding_rate"].groupby(pd.Grouper(freq="D")).sum()
    fd.index = fd.index.normalize()
    return k["close"], fd


def sleeve_net(close, funding_daily, cost_bps=FEE_SLIP_BPS, entry_lag=1):
    ret = close.pct_change()
    vol = ret.rolling(VOL_WIN, min_periods=VOL_WIN // 2).std() * np.sqrt(ANN)
    sig = sum(np.sign(close / close.shift(L) - 1.0) for L in LOOKBACKS) / len(LOOKBACKS)
    w = (sig * (TARGET_VOL / vol).clip(upper=MAX_LEV)).clip(-MAX_LEV, MAX_LEV)
    w_prev = w.shift(entry_lag)                     # decided at close t-lag, held to t
    gross = w_prev * ret
    turnover = w_prev.diff().abs()
    cost = turnover * cost_bps / 1e4
    fnd = funding_daily.reindex(close.index).fillna(0.0)
    funding_pnl = -w_prev * fnd                     # long pays when funding>0
    net = (gross - cost + funding_pnl)
    return gross.rename("gross"), net.rename("net")


def portfolio(cost_bps=FEE_SLIP_BPS, entry_lag=1):
    g_list, n_list = [], []
    for s in SYMBOLS:
        close, fd = _load(s)
        g, n = sleeve_net(close, fd, cost_bps, entry_lag)
        g_list.append(g); n_list.append(n)
    gross = pd.concat(g_list, axis=1).mean(axis=1)
    net = pd.concat(n_list, axis=1).mean(axis=1)
    warm = max(LOOKBACKS) + VOL_WIN
    return gross.iloc[warm:].dropna(), net.iloc[warm:].dropna()


def stats(x):
    x = x.dropna()
    if len(x) < 30:
        return None
    mu, sd = x.mean(), x.std()
    eq = (1 + x).cumprod()
    boot = np.array([RNG.choice(x.values, len(x), replace=True).mean() for _ in range(3000)])
    return {"n_days": int(len(x)), "ann_return": round(float(mu * ANN), 4),
            "ann_vol": round(float(sd * np.sqrt(ANN)), 4),
            "sharpe": round(float(mu / sd * np.sqrt(ANN)), 3) if sd else None,
            "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
            "sharpe_ci95": [round(float(np.percentile(boot, 2.5)) / sd * np.sqrt(ANN), 3),
                            round(float(np.percentile(boot, 97.5)) / sd * np.sqrt(ANN), 3)] if sd else None}


def main():
    gross, net = portfolio()
    full_net = stats(net)
    full_gross = stats(gross)
    by_regime = {}
    for name, (a, b) in REGIMES.items():
        seg = net[(net.index >= a) & (net.index <= b)]
        by_regime[name] = stats(seg)

    # stresses (scoreboard gate): doubled cost + delayed entry
    doubled = stats(portfolio(cost_bps=2 * FEE_SLIP_BPS)[1])
    delayed = stats(portfolio(entry_lag=2)[1])

    regime_sharpes = [v["sharpe"] for v in by_regime.values() if v and v["sharpe"] is not None]
    all_regimes_positive = all(s is not None and s > 0 for s in regime_sharpes) and len(regime_sharpes) == len(REGIMES)
    equal_weight_regime_sharpe = round(float(np.mean(regime_sharpes)), 3) if regime_sharpes else None
    net_sharpe = full_net["sharpe"] if full_net else None
    ci_lb = full_net["sharpe_ci95"][0] if full_net and full_net.get("sharpe_ci95") else None

    reasons = []
    if net_sharpe is None or net_sharpe <= 1.0:
        reasons.append("net Sharpe <= 1")
    if ci_lb is not None and ci_lb <= 0:
        reasons.append("bootstrap Sharpe lower bound <= 0")
    if not all_regimes_positive:
        reasons.append("not positive in every sub-period regime")
    if not doubled or (doubled["sharpe"] or 0) <= 0:
        reasons.append("fails doubled-cost")
    if not delayed or (delayed["sharpe"] or 0) <= 0:
        reasons.append("fails delayed-entry")
    # promotion still needs Deflated Sharpe + a real forward paper record ->
    # the best this run can earn is a shortlisted candidate, never "promoted".
    verdict = "ADVANCE_TO_FORWARD_PAPER_CANDIDATE" if not reasons else "REJECT_CLOSE"

    report = {
        "test_id": "crypto-tsmom-btc-eth-v1", "run_once": True,
        "not_the_day_trader": "daily trend overlay; the day-trade engine is the microstructure/liquidation lane",
        "data": "data/crypto/{BTCUSDT,ETHUSDT}_1d.csv + _funding.csv (real, 2019->2026)",
        "rules": {"lookbacks": LOOKBACKS, "target_vol": TARGET_VOL, "max_lev": MAX_LEV,
                  "vol_window": VOL_WIN, "fee_slippage_bps_per_side": FEE_SLIP_BPS,
                  "funding": "modeled: long pays when funding>0"},
        "gross_before_costs": full_gross,
        "net_after_fees_funding_slippage": full_net,
        "by_regime_net": by_regime,
        "all_regimes_net_sharpe_positive": all_regimes_positive,
        "equal_weight_regime_sharpe": equal_weight_regime_sharpe,
        "stress_doubled_cost_net": doubled,
        "stress_delayed_entry_net": delayed,
        "decision_rule": ("shortlist unless net Sharpe>1 AND positive every regime AND "
                          "bootstrap LB>0 AND doubled-cost>0 AND delayed-entry>0; promotion "
                          "additionally needs Deflated Sharpe + a real forward paper record"),
        "verdict": verdict, "reject_reasons": reasons,
        "note": ("replication of a documented effect, not novel alpha; funding ~3.2-3.8 bps/day "
                 "is a brutal headwind. Deflated Sharpe: this is trial ~#10 in the global ledger."),
    }
    (DATA_DIR / "crypto_tsmom_backtest.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print("GROSS:", full_gross)
    print("NET  :", full_net)
    for k, v in by_regime.items():
        print(f"  {k:10s} net Sharpe {v['sharpe'] if v else None}  ann {v['ann_return'] if v else None}")
    print(f"VERDICT: {verdict}  reasons={reasons}")


if __name__ == "__main__":
    main()
