"""Decisive validation of the fade-news-spike hypothesis.

Addresses the review head-on:
  * REAL portfolio curve (position sizing + concurrency cap + daily equity),
    not the arithmetic sum of per-trade returns.
  * CONTROLS — the load-bearing test: is this a NEWS signal, or just "large 5-min
    mega-cap moves reverse"?
      1. shuffled-news: real news timestamps randomly reassigned within each
         ticker's trading days. If the edge survives, timing/news is irrelevant.
      2. no-news spikes: fade triggered on large moves that are NOT near any news.
    News adds value only if the real-news fade beats BOTH controls.
  * ROBUSTNESS: 2x transaction costs and a delayed-entry (later executable bar)
    stress test.
  * RESIDUAL move option: fade (stock - beta*SPY), not the raw move, so we aren't
    just fading market-wide moves.

Runs on whatever minute bars are loaded (currently AAPL/MSFT + benchmarks). Small
n — this is a hypothesis test, not a promotion decision.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import DATA_DIR, BENCHMARK_TICKER
from ..db import connect
from .replay import BarPanel, ReplayParams, replay
from .news_reaction import news_events, MOVE_THR, VOL_THR

MAX_CONCURRENT = 5
SEED = 17


def _fade_signal(residual=False, spy=None):
    def sig(event, panel, i_pub, i_dec):
        d = panel.data[event.ticker]
        if i_dec >= len(d["close"]) or i_pub == 0:
            return {"side": 0}
        r0 = d["close"][i_dec - 1] / d["open"][i_pub] - 1.0
        if residual and spy is not None:
            m = _spy_ret_between(spy, d["ts"][i_pub], d["ts"][i_dec - 1])
            r0 = r0 - m                       # market-adjusted move (beta≈1)
        rel = d["vol"][i_pub:i_dec].mean() / (np.nanmean(d["vol"][max(0, i_pub - 30):i_pub]) + 1e-9)
        if not np.isfinite(r0) or abs(r0) < MOVE_THR or rel < VOL_THR:
            return {"side": 0}
        return {"side": -int(np.sign(r0))}
    return sig


def _spy_ret_between(spy, t0, t1):
    a = spy["ts"]
    i0, i1 = int(np.searchsorted(a, t0)), int(np.searchsorted(a, t1))
    c = spy["close"]
    if i0 >= len(c) or i1 >= len(c) or c[i0] <= 0:
        return 0.0
    return float(c[i1] / c[i0] - 1)


def _portfolio_sim(trades: pd.DataFrame, max_concurrent=MAX_CONCURRENT) -> dict:
    """Real portfolio: <=max_concurrent positions, each sized 1/max_concurrent;
    trades that arrive with no free slot are SKIPPED (capacity). Daily equity."""
    filled = trades[trades["status"] == "filled"].sort_values("entry_ns") if not trades.empty else trades
    if filled.empty:
        return {"n": 0}
    w = 1.0 / max_concurrent
    open_until = []            # exit_ns of currently-open positions
    day_pnl: dict = {}
    taken = 0
    for t in filled.itertuples(index=False):
        open_until = [x for x in open_until if x > t.entry_ns]
        if len(open_until) >= max_concurrent:
            continue           # no capital slot — skip
        open_until.append(t.exit_ns)
        taken += 1
        day = pd.Timestamp(t.entry_ns, unit="ns").date()
        day_pnl[day] = day_pnl.get(day, 0.0) + w * t.net
    s = pd.Series(day_pnl).sort_index()
    eq = (1 + s).cumprod()
    sharpe = float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else np.nan
    yrs = max((s.index[-1] - s.index[0]).days / 365.25, 1e-6)
    return {"n_signals": int(len(filled)), "n_taken": int(taken),
            "final_equity": float(eq.iloc[-1]), "total_return": float(eq.iloc[-1] - 1),
            "portfolio_sharpe": sharpe,
            "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1),
            "max_dd": float((eq / eq.cummax() - 1).min()),
            "active_days": int(len(s))}


def _shuffle_news(events: pd.DataFrame, panel, rng) -> pd.DataFrame:
    """Reassign each ticker's news to random bar-timestamps of the SAME ticker
    (preserves count + trading-hours distribution, destroys the actual timing)."""
    out = []
    for tk, g in events.groupby("ticker"):
        if tk not in panel.data:
            continue
        ts_pool = panel.data[tk]["ts"]
        picks = rng.choice(ts_pool[:-40], size=len(g), replace=True)
        for p in picks:
            out.append({"ticker": tk, "public_time": pd.Timestamp(p, unit="ns")})
    return pd.DataFrame(out)


def _no_news_spikes(events: pd.DataFrame, panel, rng, per_ticker=400) -> pd.DataFrame:
    """Random bar-timestamps NOT within 30 min of any real news for that ticker."""
    out = []
    for tk in panel.data:
        news_ns = np.sort(events[events["ticker"] == tk]["public_time"]
                          .astype("int64").to_numpy()) if not events.empty else np.array([])
        ts_pool = panel.data[tk]["ts"][:-40]
        cand = rng.choice(ts_pool, size=min(per_ticker * 4, len(ts_pool)), replace=False)
        kept = []
        for c in cand:
            if len(news_ns) == 0:
                kept.append(c)
            else:
                j = np.searchsorted(news_ns, c)
                near = (j < len(news_ns) and abs(news_ns[j] - c) < 1.8e12) or \
                       (j > 0 and abs(news_ns[j - 1] - c) < 1.8e12)   # 30 min in ns
                if not near:
                    kept.append(c)
            if len(kept) >= per_ticker:
                break
        for c in kept:
            out.append({"ticker": tk, "public_time": pd.Timestamp(c, unit="ns")})
    return pd.DataFrame(out)


def run(verbose=True) -> dict:
    con = connect(read_only=True)
    events = news_events(con)
    panel = BarPanel(con, table="bars_minute")
    con.close()
    events = events[events["ticker"].isin(panel.data.keys())].reset_index(drop=True)
    spy = panel.data.get(BENCHMARK_TICKER)
    rng = np.random.default_rng(SEED)

    base = ReplayParams(obs_bars=5, max_hold=30, tp=0.008, sl=0.008,
                        spread_bps=2, slippage_bps=2, n_trials=8, cooldown_bars=30)

    results = {}

    def trades_for(ev, params, residual=False):
        return _replay_trades(ev, panel, _fade_signal(residual, spy), params)

    real_tr = trades_for(events, base)
    results["real_news"] = _portfolio_sim(real_tr)
    results["real_news_residual"] = _portfolio_sim(trades_for(events, base, residual=True))

    shuf = _shuffle_news(events, panel, rng)
    results["shuffled_news"] = _portfolio_sim(trades_for(shuf, base))

    nonews = _no_news_spikes(events, panel, rng)
    results["no_news_spikes"] = _portfolio_sim(trades_for(nonews, base))

    # robustness on real news
    p2 = ReplayParams(**{**base.__dict__, "spread_bps": 4, "slippage_bps": 4})
    results["real_news_2x_cost"] = _portfolio_sim(trades_for(events, p2))
    p_delay = ReplayParams(**{**base.__dict__, "obs_bars": 8})   # enter later
    results["real_news_delayed_entry"] = _portfolio_sim(trades_for(events, p_delay))

    out = {"universe": sorted(panel.data.keys()), "news_events": int(len(events)),
           "max_concurrent": MAX_CONCURRENT, "results": results,
           "move_thr": MOVE_THR, "vol_thr": VOL_THR}
    with open(DATA_DIR / "v4_fade_validation.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    if verbose:
        print("=== Fade validation — REAL portfolio, controls, robustness ===")
        print(f"universe: {out['universe']}  news events: {out['news_events']}\n")
        hdr = f"{'scenario':26s} {'taken':>6s} {'ret%':>7s} {'pSharpe':>8s} {'maxDD%':>7s} {'days':>5s}"
        print(hdr)
        for k, r in results.items():
            if r.get("n") == 0 or "n_taken" not in r:
                print(f"{k:26s}   no trades"); continue
            print(f"{k:26s} {r['n_taken']:6d} {r['total_return']*100:7.1f} "
                  f"{r['portfolio_sharpe']:8.2f} {r['max_dd']*100:7.1f} {r['active_days']:5d}")
        rn, sn, nn = results["real_news"], results["shuffled_news"], results["no_news_spikes"]
        print("\nVERDICT:")
        if rn.get("n_taken", 0) < 30:
            print("  too few real-news trades to conclude anything.")
        else:
            edge = (rn["total_return"] > sn.get("total_return", 0) and
                    rn["total_return"] > nn.get("total_return", 0))
            print(f"  real-news beats shuffled AND no-news controls: {edge}")
            print("  -> if False, the fade is generic mega-cap reversal, NOT a news signal.")
    return out


def _replay_trades(ev, panel, signal_fn, params):
    """Run replay's per-event loop and return the raw trades DataFrame."""
    import quantv1.v4.replay as R
    cost = (params.spread_bps + params.slippage_bps + params.fees_bps) / 1e4
    e = ev.sort_values("public_time").reset_index(drop=True)
    e["public_time"] = pd.to_datetime(e["public_time"]).astype("int64")
    trades, last_exit = [], {}
    for row in e.itertuples(index=False):
        tk = row.ticker
        if not tk or not panel.has(tk):
            continue
        if last_exit.get(tk, 0) >= row.public_time:
            continue
        i_pub = panel.next_idx_after(tk, row.public_time)
        if i_pub is None:
            trades.append({"ticker": tk, "status": "reject_no_bar"}); continue
        i_dec = i_pub + params.obs_bars
        d = panel.data[tk]
        if i_dec + 1 >= len(d["ts"]):
            trades.append({"ticker": tk, "status": "reject_eod"}); continue
        sig = signal_fn(row, panel, i_pub, i_dec)
        side = sig.get("side", 0)
        if side == 0:
            continue
        entry = d["open"][i_dec] * (1 + side * cost)
        exit_px, exit_i = None, None
        for j in range(i_dec, min(i_dec + params.max_hold, len(d["ts"]))):
            up = entry * (1 + params.tp) if side > 0 else entry * (1 - params.tp)
            dn = entry * (1 - params.sl) if side > 0 else entry * (1 + params.sl)
            if side > 0 and d["high"][j] >= up: exit_px, exit_i = up, j; break
            if side > 0 and d["low"][j] <= dn: exit_px, exit_i = dn, j; break
            if side < 0 and d["low"][j] <= up: exit_px, exit_i = up, j; break
            if side < 0 and d["high"][j] >= dn: exit_px, exit_i = dn, j; break
        if exit_px is None:
            exit_i = min(i_dec + params.max_hold - 1, len(d["ts"]) - 1)
            exit_px = d["close"][exit_i]
        gross = side * (exit_px / entry - 1)
        bench = R._bench_return_timealigned(panel, BENCHMARK_TICKER, d["ts"][i_dec], d["ts"][exit_i])
        net = gross - side * bench - cost
        trades.append({"ticker": tk, "status": "filled", "net": float(net),
                       "entry_ns": int(d["ts"][i_dec]), "exit_ns": int(d["ts"][exit_i])})
        bar_ns = int(d["ts"][exit_i] - d["ts"][exit_i - 1]) if exit_i > 0 else 60_000_000_000
        last_exit[tk] = int(d["ts"][exit_i]) + params.cooldown_bars * bar_ns
    return pd.DataFrame(trades)


if __name__ == "__main__":
    run()
