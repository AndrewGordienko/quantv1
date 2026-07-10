"""PoC that validates the replay engine on data we already have.

Flagship strategy A (event-shock continuation) at the only resolution currently
available without Alpaca keys: Federal Register significant rules -> the affected
sector's SPDR ETF, using the hourly bars we ingested. Public_time = the rule's
publication date; the signal observes the ETF's first post-publication hour and
bets continuation of that move; returns are SPY-adjusted and net of costs.

This is a stepping stone, NOT the real system — daily-resolution events + hourly
bars are coarse and the result is expected to be weak. The point is to prove the
leak-free replay machinery end-to-end before minute data + real-time news land.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import DATA_DIR
from ..db import connect
from .replay import BarPanel, ReplayParams, replay

YAHOO_TO_ETF = {
    "Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV",
    "Energy": "XLE", "Industrials": "XLI", "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP", "Utilities": "XLU", "Basic Materials": "XLB",
    "Real Estate": "XLRE", "Communication Services": "XLC",
}
TEST_START = "2025-07-01"     # untouched time-based holdout


def _fr_events(con) -> pd.DataFrame:
    rows = con.execute("""
        SELECT source_time, payload FROM events WHERE event_type='reg_rule'
    """).df()
    out = []
    for r in rows.itertuples(index=False):
        try:
            secs = json.loads(r.payload).get("sectors", [])
        except (TypeError, ValueError):
            secs = []
        for s in secs:
            etf = YAHOO_TO_ETF.get(s)
            if etf:
                out.append({"public_time": r.source_time, "ticker": etf})
    return pd.DataFrame(out)


def continuation_signal(event, panel, i_pub, i_dec) -> dict:
    """Bet the direction of the ETF's first post-publication hour (continuation)."""
    d = panel.data[event.ticker]
    if i_dec >= len(d["close"]) or i_pub == 0:
        return {"side": 0}
    r0 = d["close"][i_dec - 1] / d["open"][i_pub] - 1.0    # move over the observation window
    if not np.isfinite(r0) or abs(r0) < 0.003:             # need a real initial move
        return {"side": 0}
    return {"side": int(np.sign(r0)), "reason": "continuation"}


def run(verbose: bool = True) -> dict:
    con = connect(read_only=True)
    events = _fr_events(con)
    panel = BarPanel(con)
    con.close()
    events = events[events["ticker"].isin(panel.data.keys())]

    params = ReplayParams(obs_bars=1, max_hold=6, tp=0.02, sl=0.01, n_trials=5)
    res = replay(events, panel, continuation_signal, params, test_start=TEST_START)
    res["events_considered"] = int(len(events))
    with open(DATA_DIR / "v4_event_reaction_poc.json", "w") as f:
        json.dump(res, f, indent=2, default=str)

    if verbose:
        print("=== V4 replay PoC: FR rules -> sector ETF, event-shock continuation ===")
        print(f"events considered: {res['events_considered']}  trades: {res.get('n_trades')}  "
              f"rejects: {res.get('rejects')}")
        if res.get("n_trades", 0) > 0:
            o = res["overall"]
            print(f"net Sharpe (daily): {res['net_sharpe_daily']:.2f}  "
                  f"deflated Sharpe: {res['deflated_sharpe']:.2f}")
            print(f"mean net/trade: {o['mean_net']*100:+.2f}%  hit rate: {o['hit_rate']:.0%}  "
                  f"total net: {res['total_net_return']*100:+.1f}%")
            print(f"exit reasons: {res['exit_reasons']}")
            for seg in ("train", "holdout"):
                s = res.get(seg)
                if s:
                    print(f"  {seg:8s} n={s['n']} mean_net={s['mean_net']*100:+.2f}% "
                          f"hit={s['hit_rate']:.0%}")
        print("\nEngine validated. Real system needs minute bars + timestamped news "
              "(Alpaca keys) — this coarse PoC is expected to be weak/null.")
    return res


if __name__ == "__main__":
    run()
