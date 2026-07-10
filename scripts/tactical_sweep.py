"""Characterize how much each layer of the tactical overlay helps or hurts.

Precomputes the expensive OOS scores ONCE, then runs several configurations so
we can see the effect of the aggressive stops without cherry-picking a winner.
This is sensitivity analysis, not tuning — every config is reported.
"""

from __future__ import annotations

import json

from quantv1.config import DATA_DIR
from quantv1.portfolio import tactical as T

CONFIGS = {
    # name -> overrides on TacticalParams
    "full (all stops)": {},
    "no kill-switch (X6 off)": {"kill_day": -1.0},
    "patient (X6 off, trail 5xATR)": {"kill_day": -1.0, "atr_trail": 5.0},
    "disaster-only (X1/X2/X5)": {"kill_day": -1.0, "atr_trail": 99.0,
                                 "use_x4_trendbreak": False},
    "no trend gate E4": {"kill_day": -1.0, "atr_trail": 5.0,
                         "use_e4_trend": False, "use_x4_trendbreak": False},
}


def main():
    print("preparing (one-time OOS score precompute)…")
    ctx = T.prepare()
    rows = []
    for name, ov in CONFIGS.items():
        p = T.TacticalParams()
        for k, v in ov.items():
            setattr(p, k, v)
        res = T.run_backtest(params=p, verbose=False, ctx=ctx)
        s = res["stats"]["tactical"]
        rows.append({
            "config": name,
            "cagr": s["cagr"], "sharpe": s["sharpe"], "max_dd": s["max_dd"],
            "final": s["final"], "n_trades": res["n_trades"],
            "win_rate": res["win_rate"], "avg_win": res["avg_win"],
            "avg_loss": res["avg_loss"], "avg_hold_td": res["avg_hold_td"],
        })
        print(f"{name:34s} CAGR={s['cagr']:+.2%} Sharpe={s['sharpe']:.2f} "
              f"DD={s['max_dd']:.1%} x={s['final']:.2f} "
              f"trades={res['n_trades']} hold={res['avg_hold_td']:.0f}td")

    spy = res["stats"]["spy"]
    print(f"{'SPY buy&hold':34s} CAGR={spy['cagr']:+.2%} Sharpe={spy['sharpe']:.2f} "
          f"DD={spy['max_dd']:.1%} x={spy['final']:.2f}")

    with open(DATA_DIR / "tactical_sweep.json", "w") as f:
        json.dump({"configs": rows, "spy": spy}, f, indent=2)
    print(f"\nwrote {DATA_DIR / 'tactical_sweep.json'}")


if __name__ == "__main__":
    main()
