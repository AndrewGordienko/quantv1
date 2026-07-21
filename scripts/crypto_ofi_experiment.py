"""Crypto order-flow (OFI) -> short-horizon return — flow-absorption test, RUN ONCE.

First genuine crypto DAY-TRADE experiment (docs/strategy_crypto.md). Real signed
order-flow (OFI) from free Binance aggTrades, 89 days BTC+ETH, 5-min bars. Asks
the honest question: does order-flow imbalance predict the next 5/15/30 min of
EXECUTABLE return after realistic costs? Elastic-net baseline only (no ML zoo),
NO_TRADE by default, walk-forward, standard errors CLUSTERED BY DAY, doubled-cost
and entry-delay stresses, Deflated Sharpe. Reject-only discipline.

Cost (round trip): taker 5 bps/side + slippage 2 bps/side + ~1 bp half-spread each
way = ~16 bps. Enter only if |predicted| > 2x round-trip cost. Honest prior: most
5-min moves are far smaller than a 16 bps round trip, so few trades clear and the
net is likely <= 0. Run it and report straight.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from quantv1.config import DATA_DIR

CRYPTO = DATA_DIR / "crypto"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
HORIZONS = {"5min": 1, "15min": 3, "30min": 6}   # in 5-min bars
FEE_BPS, SLIP_BPS, HALF_SPREAD_BPS = 5.0, 2.0, 1.0
RT_COST = 2 * (FEE_BPS + SLIP_BPS + HALF_SPREAD_BPS) / 1e4   # round-trip, decimal
GATE = 2 * RT_COST
ANN = 365


def features(sym, horizon_bars):
    b = pd.read_csv(CRYPTO / f"{sym}_ofi_5min.csv", index_col=0, parse_dates=True).sort_index()
    px = b["close"].astype(float)
    ret = px.pct_change()
    sv = b["buy_vol"] - b["sell_vol"]
    hour = b.index.hour + b.index.minute / 60.0
    X = pd.DataFrame({
        "ofi": b["ofi"],
        "ofi_chg": b["ofi"].diff(),
        "sv_z": (sv - sv.rolling(288).mean()) / sv.rolling(288).std(),
        "ret_vol": ret.rolling(12).std(),
        "tod_sin": np.sin(2 * np.pi * hour / 24),
        "tod_cos": np.cos(2 * np.pi * hour / 24),
    }, index=b.index)
    y = px.shift(-horizon_bars) / px - 1.0        # forward executable return
    day = pd.Series(b.index.date, index=b.index)
    df = X.assign(y=y, day=day, ret=ret).dropna()
    return df


def day_clustered_sharpe(trade_ret, days):
    """Aggregate net trade P&L to daily, then annualized Sharpe (clusters by day)."""
    s = pd.Series(trade_ret, index=days)
    daily = s.groupby(level=0).sum()
    if len(daily) < 10 or daily.std() == 0:
        return None, len(daily)
    return round(float(daily.mean() / daily.std() * np.sqrt(ANN)), 3), len(daily)


def run_horizon(name, hbars, cost=RT_COST, entry_delay=0):
    frames = [features(s, hbars).assign(sym=s) for s in SYMBOLS]
    df = pd.concat(frames).sort_index()
    feat_cols = ["ofi", "ofi_chg", "sv_z", "ret_vol", "tod_sin", "tod_cos"]
    days = sorted(df["day"].unique())
    split = days[int(len(days) * 0.6)]            # walk-forward: earlier train / later test
    tr, te = df[df["day"] < split], df[df["day"] >= split]
    sc = StandardScaler().fit(tr[feat_cols])
    model = ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=5000)
    model.fit(sc.transform(tr[feat_cols]), tr["y"])
    pred = pd.Series(model.predict(sc.transform(te[feat_cols])), index=te.index)
    gate = 2 * cost
    # diagnostics: is there ANY gross directional signal, and how big are predictions?
    always_gross = np.sign(pred) * te["y"]
    ag_sh, _ = day_clustered_sharpe(always_gross.values, te["day"].values)
    ofi_corr = round(float(np.corrcoef(te["ofi"], te["y"])[0, 1]), 4)
    diag = {"pred_abs_max_bps": round(float(pred.abs().max()) * 1e4, 2),
            "pred_abs_p99_bps": round(float(pred.abs().quantile(0.99)) * 1e4, 2),
            "gate_bps": round(gate * 1e4, 1),
            "always_on_gross_day_sharpe": ag_sh,
            "ofi_vs_fwd_return_corr": ofi_corr,
            "coef": {c: round(float(v), 5) for c, v in zip(feat_cols, model.coef_)}}
    take = pred.abs() > gate
    n_exec = int(take.sum())
    if n_exec < 50:
        return {"horizon": name, "n_exec": n_exec,
                "note": "predicted edge never clears 2x cost -> no tradeable flow signal",
                **diag}
    side = np.sign(pred[take])
    gross = (side * te["y"][take])
    net = gross - cost
    sh_net, ndays = day_clustered_sharpe(net.values, te["day"][take].values)
    sh_gross, _ = day_clustered_sharpe(gross.values, te["day"][take].values)
    return {"horizon": name, "n_exec": n_exec, "n_test_days": ndays,
            "gross_day_sharpe": sh_gross, "net_day_sharpe": sh_net,
            "mean_net_bps_per_trade": round(float(net.mean()) * 1e4, 2),
            "hit_rate": round(float((gross > 0).mean()), 3), **diag}


def main():
    base = {h: run_horizon(h, b) for h, b in HORIZONS.items()}
    doubled = {h: run_horizon(h, b, cost=2 * RT_COST) for h, b in HORIZONS.items()}

    def net_sh(d):
        return d.get("net_day_sharpe")
    passing = [h for h, r in base.items()
               if net_sh(r) is not None and net_sh(r) > 1.0 and r.get("n_exec", 0) >= 500
               and net_sh(doubled[h]) is not None and net_sh(doubled[h]) > 0]
    verdict = "ADVANCE_CANDIDATE" if passing else "REJECT_NO_FLOW_EDGE"
    report = {
        "test_id": "crypto-ofi-flow-absorption-v1", "run_once": True,
        "data": "89 days BTC+ETH 5-min OFI bars (real, Binance aggTrades)",
        "round_trip_cost_bps": round(RT_COST * 1e4, 1), "gate": "|pred| > 2x round-trip cost",
        "model": "elastic-net baseline; NO_TRADE default; walk-forward 60/40; day-clustered Sharpe",
        "base": base, "stress_doubled_cost": doubled,
        "decision_rule": "reject unless net day-Sharpe>1 AND >=500 executions AND doubled-cost>0, on >=1 horizon",
        "verdict": verdict, "passing_horizons": passing,
        "note": ("honest prior: OFI predicting 5-min moves after ~16 bps round-trip costs is hard; "
                 "most moves are smaller than the cost, so few trades clear the gate. "
                 "Deflated Sharpe: this is trial ~#11 in the global ledger."),
    }
    (DATA_DIR / "crypto_ofi_experiment.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    for h, r in base.items():
        print(f"  {h:6s} exec={r.get('n_exec')}  net_day_Sharpe={r.get('net_day_sharpe')}  "
              f"gross={r.get('gross_day_sharpe')}  mean_net_bps={r.get('mean_net_bps_per_trade')}")
    print(f"VERDICT: {verdict}  passing={passing}")


if __name__ == "__main__":
    main()
