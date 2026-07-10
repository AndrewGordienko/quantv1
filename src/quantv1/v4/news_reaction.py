"""Strategy A on REAL data: news-event shock continuation, minute resolution.

For each news article (event bus layer 'N', public_time = published_utc), observe
the stock's first post-publication minutes and bet continuation of the immediate
move, through the leak-free replay engine on Polygon minute bars. Returns are
SPY-adjusted and net of costs; a 2024+ style time holdout is locked.

This is the first genuine V4 test — real timestamped news against real minute
reactions. It's still a baseline (no LLM sentiment/novelty yet — just the sign of
the immediate reaction); the point is to measure whether news-shock continuation
has any edge before layering the richer event-extraction features.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import DATA_DIR
from ..db import connect
from .replay import BarPanel, ReplayParams, replay


def news_events(con) -> pd.DataFrame:
    """Catalyst-level events: ONE row per (catalyst, ticker) at the catalyst's
    earliest public time — deduplicated so intraday updates of the same story
    don't generate multiple trades. Falls back to raw events if not yet built."""
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    has_cat = "news_catalysts" in tables and con.execute(
        "SELECT COUNT(*) FROM events WHERE layer='N' AND catalyst_id IS NOT NULL"
    ).fetchone()[0] > 0
    if has_cat:
        return con.execute("""
            SELECT catalyst_id, ticker, MIN(source_time) AS public_time
            FROM events
            WHERE layer='N' AND ticker IS NOT NULL AND catalyst_id IS NOT NULL
              AND source_time IS NOT NULL
            GROUP BY catalyst_id, ticker
        """).df()
    r = con.execute("""
        SELECT source_time AS public_time, ticker FROM events
        WHERE layer='N' AND event_type='news' AND ticker IS NOT NULL AND source_time IS NOT NULL
    """).df()
    r["catalyst_id"] = None
    return r


# PRE-REGISTERED thresholds (fixed a priori, NOT tuned to the result). The AAPL+
# MSFT pilot showed continuation loses (~34% hit) while FADING large, high-volume
# news spikes wins (~62% hit) — consistent with intraday overreaction→reversal.
# These are deliberately mid-range, not the pilot's best-fit values.
MOVE_THR = 0.006     # only fade moves bigger than this over the observation window
VOL_THR = 1.8        # ...and only when volume is elevated (real reaction)


def fade_spike_signal(event, panel, i_pub, i_dec) -> dict:
    """FADE (reversal): a large, high-volume news spike tends to overshoot and
    mean-revert intraday, so bet AGAINST the initial move."""
    d = panel.data[event.ticker]
    if i_dec >= len(d["close"]) or i_pub == 0:
        return {"side": 0}
    r0 = d["close"][i_dec - 1] / d["open"][i_pub] - 1.0
    rel_vol = (d["vol"][i_pub:i_dec].mean() /
               (np.nanmean(d["vol"][max(0, i_pub - 30):i_pub]) + 1e-9))
    if not np.isfinite(r0) or abs(r0) < MOVE_THR or rel_vol < VOL_THR:
        return {"side": 0}
    return {"side": -int(np.sign(r0)), "reason": "fade_news_spike"}


def run(hold_min: int = 30, verbose: bool = True) -> dict:
    con = connect(read_only=True)
    events = news_events(con)
    panel = BarPanel(con, table="bars_minute")
    con.close()
    if not panel.data:
        return {"note": "no minute bars — run scripts/v4_ingest.py (needs POLYGON_API_KEY)"}
    events = events[events["ticker"].isin(panel.data.keys())]

    # observe 5 minutes, hold up to hold_min, symmetric barriers; FADE the spike
    params = ReplayParams(obs_bars=5, max_hold=hold_min, tp=0.008, sl=0.008,
                          spread_bps=2, slippage_bps=2, n_trials=8, cooldown_bars=30)
    res = replay(events, panel, fade_spike_signal, params, test_start="2026-05-01")
    res["news_events_in_universe"] = int(len(events))
    with open(DATA_DIR / "v4_news_reaction.json", "w") as f:
        json.dump(res, f, indent=2, default=str)

    if verbose:
        print("=== V4 Strategy A (REAL): news-shock continuation, minute bars ===")
        print(f"news events in universe: {res['news_events_in_universe']}  "
              f"trades: {res.get('n_trades')}  rejects: {res.get('rejects')}")
        if res.get("n_trades", 0) > 0:
            o = res["overall"]
            print(f"net Sharpe (daily): {res['net_sharpe_daily']:.2f}  "
                  f"deflated Sharpe: {res['deflated_sharpe']:.2f}")
            print(f"mean net/trade: {o['mean_net']*100:+.3f}%  hit: {o['hit_rate']:.0%}  "
                  f"total net: {res['total_net_return']*100:+.1f}%")
            print(f"exit reasons: {res['exit_reasons']}")
    return res


if __name__ == "__main__":
    run()
