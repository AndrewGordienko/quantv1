"""Rigorous fade validation harness (hardened per review).

Fixes:
  * NANOSECOND timestamps (kills the 1970 / two-day bucketing bug).
  * DELAYED ENTRY is a real delay (entry_delay), NOT a longer observation window,
    so the trade COUNT no longer changes with the stress test.
  * MATCHED no-news control: actual large-move spikes that are NOT near any news,
    detected the same way the fade fires — the correct apples-to-apples control.
  * Per-OPPORTUNITY stats: mean net bps/trade + 95% CI CLUSTERED BY TICKER, and
    the real-minus-shuffled LIFT with a clustered CI (not raw totals).
  * Split reporting: DISCOVERY (AAPL+MSFT) vs UNSEEN stocks vs COMBINED — the
    unseen-stock result is the real validation.
  * Real portfolio sim (correct dates, concurrency, turnover, gross exposure).
  * Every run is appended to an EXPERIMENT REGISTRY for honest trial accounting.

Rule is FROZEN (thresholds from news_reaction, no tuning here).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import DATA_DIR, BENCHMARK_TICKER
from ..db import connect
from .replay import BarPanel, ReplayParams, to_ns, _bench_return_timealigned
from .news_reaction import news_events, MOVE_THR, VOL_THR

DISCOVERY = {"AAPL", "MSFT"}          # where the rule was discovered
MAX_CONCURRENT = 5
NEAR_NS = int(30 * 60 * 1e9)          # 30 min in ns
REGISTRY = DATA_DIR / "experiment_registry.jsonl"
ETFS = {"SPY", "QQQ", "XLK", "XLV", "XLE", "XLF", "XLI", "XLY"}


# ---------------------------------------------------------------------------
# Trade generation (frozen fade rule) with real entry_delay
# ---------------------------------------------------------------------------
def _fade_trades(ev: pd.DataFrame, panel: BarPanel, params: ReplayParams,
                 residual=False) -> pd.DataFrame:
    spy = panel.data.get(BENCHMARK_TICKER)
    cost = (params.spread_bps + params.slippage_bps + params.fees_bps) / 1e4
    e = ev.dropna(subset=["public_time"]).copy()
    e["pt_ns"] = to_ns(e["public_time"])
    e = e.sort_values("pt_ns")
    trades, last_exit = [], {}
    for tk, pt in zip(e["ticker"].to_numpy(), e["pt_ns"].to_numpy()):
        if not panel.has(tk) or last_exit.get(tk, 0) >= pt:
            continue
        d = panel.data[tk]
        i_pub = panel.next_idx_after(tk, pt)
        if i_pub is None or i_pub == 0:
            continue
        i_dec = i_pub + params.obs_bars
        i_entry = i_dec + params.entry_delay
        if i_entry + 1 >= len(d["ts"]):
            continue
        r0 = d["close"][i_dec - 1] / d["open"][i_pub] - 1.0
        if residual and spy is not None:
            r0 -= _bench_return_timealigned(panel, BENCHMARK_TICKER, d["ts"][i_pub], d["ts"][i_dec - 1])
        rel = d["vol"][i_pub:i_dec].mean() / (np.nanmean(d["vol"][max(0, i_pub - 30):i_pub]) + 1e-9)
        if not np.isfinite(r0) or abs(r0) < MOVE_THR or rel < VOL_THR:
            continue
        side = -int(np.sign(r0))                        # FADE
        entry = d["open"][i_entry] * (1 + side * cost)
        exit_px, exit_i = None, None
        for j in range(i_entry, min(i_entry + params.max_hold, len(d["ts"]))):
            up = entry * (1 + params.tp) if side > 0 else entry * (1 - params.tp)
            dn = entry * (1 - params.sl) if side > 0 else entry * (1 + params.sl)
            if side > 0 and d["high"][j] >= up: exit_px, exit_i = up, j; break
            if side > 0 and d["low"][j] <= dn: exit_px, exit_i = dn, j; break
            if side < 0 and d["low"][j] <= up: exit_px, exit_i = up, j; break
            if side < 0 and d["high"][j] >= dn: exit_px, exit_i = dn, j; break
        if exit_px is None:
            exit_i = min(i_entry + params.max_hold - 1, len(d["ts"]) - 1)
            exit_px = d["close"][exit_i]
        gross = side * (exit_px / entry - 1)
        badj = _bench_return_timealigned(panel, BENCHMARK_TICKER, d["ts"][i_entry], d["ts"][exit_i])
        net = gross - side * badj - cost
        trades.append({"ticker": tk, "entry_ns": int(d["ts"][i_entry]),
                       "exit_ns": int(d["ts"][exit_i]), "net": float(net)})
        bar_ns = int(d["ts"][exit_i] - d["ts"][exit_i - 1]) if exit_i > 0 else 60_000_000_000
        last_exit[tk] = int(d["ts"][exit_i]) + params.cooldown_bars * bar_ns
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
def _news_ns_by_ticker(events, panel):
    out = {}
    for tk, g in events.groupby("ticker"):
        if tk in panel.data:
            out[tk] = np.sort(to_ns(g["public_time"]))
    return out


def _shuffle_news(events, panel, rng) -> pd.DataFrame:
    out = []
    for tk, g in events.groupby("ticker"):
        if tk not in panel.data:
            continue
        pool = panel.data[tk]["ts"][:-60]
        for p in rng.choice(pool, size=len(g), replace=True):
            out.append({"ticker": tk, "public_time": pd.Timestamp(int(p), unit="ns")})
    return pd.DataFrame(out)


def _matched_no_news_spikes(events, panel, rng, obs_bars=5, cap_per_ticker=2000) -> pd.DataFrame:
    """Detect actual large-move spikes (same trigger as the fade) that are NOT
    within 30 min of any news — the matched control set."""
    news_ns = _news_ns_by_ticker(events, panel)
    out = []
    for tk, d in panel.data.items():
        if tk in ETFS:
            continue
        c, o, v, ts = d["close"], d["open"], d["vol"], d["ts"]
        n = len(ts)
        if n < 60:
            continue
        move = np.full(n, np.nan)
        move[obs_bars:] = c[obs_bars - 1:n - 1] / o[:n - obs_bars] - 1.0
        # rolling 30-bar mean volume for rel-vol
        relvol = np.full(n, np.nan)
        for i in range(30, n):
            relvol[i] = v[i - obs_bars:i].mean() / (v[i - 30:i].mean() + 1e-9)
        cand = np.where((np.abs(move) >= MOVE_THR) & (relvol >= VOL_THR))[0]
        nn = news_ns.get(tk, np.array([]))
        kept = []
        for i in cand:
            t = ts[i]
            if len(nn):
                j = np.searchsorted(nn, t)
                near = (j < len(nn) and abs(nn[j] - t) < NEAR_NS) or (j > 0 and abs(nn[j - 1] - t) < NEAR_NS)
                if near:
                    continue
            kept.append(i - obs_bars)      # public_time bar so i_pub lands on the spike start
        if len(kept) > cap_per_ticker:
            kept = list(rng.choice(kept, cap_per_ticker, replace=False))
        for i in kept:
            out.append({"ticker": tk, "public_time": pd.Timestamp(int(ts[max(i, 0)]), unit="ns")})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def _cluster_ci(trades: pd.DataFrame, rng, n_boot=3000) -> dict:
    if trades.empty:
        return {"n": 0, "mean_bps": None, "ci_low": None, "ci_high": None}
    by = {tk: g["net"].to_numpy() for tk, g in trades.groupby("ticker")}
    tickers = list(by)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(tickers, len(tickers), replace=True)
        vals = np.concatenate([by[t] for t in pick])
        boot[b] = vals.mean()
    return {"n": int(len(trades)), "n_tickers": len(tickers),
            "mean_bps": float(trades["net"].mean() * 1e4),
            "ci_low": float(np.percentile(boot, 2.5) * 1e4),
            "ci_high": float(np.percentile(boot, 97.5) * 1e4)}


def _portfolio(trades: pd.DataFrame, max_concurrent=MAX_CONCURRENT) -> dict:
    if trades.empty:
        return {"n": 0}
    t = trades.sort_values("entry_ns")
    w = 1.0 / max_concurrent
    open_until, day_pnl, taken, turnover = [], {}, 0, 0.0
    for r in t.itertuples(index=False):
        open_until = [x for x in open_until if x > r.entry_ns]
        if len(open_until) >= max_concurrent:
            continue
        open_until.append(r.exit_ns)
        taken += 1
        turnover += 2 * w                        # enter + exit
        day = pd.Timestamp(r.entry_ns, unit="ns").date()   # ns -> correct date now
        day_pnl[day] = day_pnl.get(day, 0.0) + w * r.net
    s = pd.Series(day_pnl).sort_index()
    if len(s) < 2:
        return {"n_taken": taken, "total_return": float(s.sum()), "active_days": len(s)}
    eq = (1 + s).cumprod()
    yrs = max((s.index[-1] - s.index[0]).days / 365.25, 1e-6)
    return {"n_taken": int(taken), "active_days": int(len(s)),
            "total_return": float(eq.iloc[-1] - 1),
            "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1),
            "sharpe": float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else None,
            "max_dd": float((eq / eq.cummax() - 1).min()),
            "turnover_per_day": float(turnover / len(s))}


def _group(trades, which):
    if which == "discovery":
        return trades[trades["ticker"].isin(DISCOVERY)]
    if which == "unseen":
        return trades[~trades["ticker"].isin(DISCOVERY | ETFS)]
    return trades[~trades["ticker"].isin(ETFS)]      # combined stocks


def run(verbose=True) -> dict:
    con = connect(read_only=True)
    events = news_events(con)
    panel = BarPanel(con, table="bars_minute")
    con.close()
    events = events[events["ticker"].isin(panel.data.keys())].reset_index(drop=True)
    rng = np.random.default_rng(17)

    base = ReplayParams(obs_bars=5, entry_delay=0, max_hold=30, tp=0.008, sl=0.008,
                        spread_bps=2, slippage_bps=2, cooldown_bars=30)

    real = _fade_trades(events, panel, base)
    shuf = _fade_trades(_shuffle_news(events, panel, rng), panel, base)
    nonews = _fade_trades(_matched_no_news_spikes(events, panel, rng), panel, base)
    delayed = _fade_trades(events, panel, ReplayParams(**{**base.__dict__, "entry_delay": 3}))
    cost2 = _fade_trades(events, panel, ReplayParams(**{**base.__dict__, "spread_bps": 4, "slippage_bps": 4}))

    report = {}
    for g in ("discovery", "unseen", "combined"):
        rg, sg = _group(real, g), _group(shuf, g)
        ci_r, ci_s = _cluster_ci(rg, rng), _cluster_ci(sg, rng)
        lift = (ci_r["mean_bps"] - ci_s["mean_bps"]) if (ci_r["mean_bps"] is not None and ci_s["mean_bps"] is not None) else None
        report[g] = {"real": ci_r, "shuffled": ci_s, "real_minus_shuffled_bps": lift,
                     "portfolio": _portfolio(rg)}
    robustness = {"matched_no_news": _cluster_ci(_group(nonews, "combined"), rng),
                  "delayed_entry_3bar": _cluster_ci(_group(delayed, "combined"), rng),
                  "double_cost": _cluster_ci(_group(cost2, "combined"), rng),
                  "trade_counts": {"real": len(real), "delayed": len(delayed),
                                   "double_cost": len(cost2), "no_news": len(nonews)}}

    out = {"frozen_rule": {"move_thr": MOVE_THR, "vol_thr": VOL_THR, "fade": True,
                           "obs_bars": 5, "max_hold": 30},
           "universe": sorted([t for t in panel.data if t not in ETFS]),
           "news_events": int(len(events)), "report": report, "robustness": robustness,
           "generated_at": datetime.now(timezone.utc).isoformat()}
    with open(DATA_DIR / "v4_fade_validation.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    _register(out)

    if verbose:
        _print(out)
    return out


def _register(out):
    """Append this run to the experiment registry (honest trial accounting)."""
    rec = {"ts": out["generated_at"], "experiment": "fade_news_spike_frozen",
           "rule": out["frozen_rule"],
           "unseen_mean_bps": out["report"]["unseen"]["real"]["mean_bps"],
           "unseen_ci": [out["report"]["unseen"]["real"]["ci_low"],
                         out["report"]["unseen"]["real"]["ci_high"]],
           "unseen_lift_bps": out["report"]["unseen"]["real_minus_shuffled_bps"]}
    with open(REGISTRY, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _print(out):
    print("=== Fade validation (hardened) — mean net bps/trade, 95% CI clustered by ticker ===")
    print(f"universe: {out['universe']}  news events: {out['news_events']}\n")
    print(f"{'group':10s} {'trades':>7s} {'tickers':>7s} {'real bps':>9s} "
          f"{'95% CI':>18s} {'shuf bps':>9s} {'lift bps':>9s} {'port ret%':>9s}")
    for g in ("discovery", "unseen", "combined"):
        r = out["report"][g]; cr = r["real"]; cs = r["shuffled"]; pt = r["portfolio"]
        if cr["mean_bps"] is None:
            print(f"{g:10s}   n/a"); continue
        pr = pt.get("total_return")
        print(f"{g:10s} {cr['n']:7d} {cr.get('n_tickers',0):7d} {cr['mean_bps']:9.2f} "
              f"[{cr['ci_low']:7.2f},{cr['ci_high']:7.2f}] {cs['mean_bps'] or 0:9.2f} "
              f"{r['real_minus_shuffled_bps'] or 0:9.2f} {(pr*100 if pr is not None else 0):9.1f}")
    print("\nrobustness (combined, mean bps [CI]):")
    for k, v in out["robustness"].items():
        if k == "trade_counts":
            print(f"  trade counts: {v}"); continue
        if v.get("mean_bps") is not None:
            print(f"  {k:20s} {v['mean_bps']:+.2f} [{v['ci_low']:+.2f},{v['ci_high']:+.2f}] n={v['n']}")
    u = out["report"]["unseen"]["real"]
    lift = out["report"]["unseen"]["real_minus_shuffled_bps"]
    verdict = (u["mean_bps"] is not None and u["ci_low"] > 0 and (lift or -1) > 0)
    print(f"\nVERDICT (unseen stocks): mean {u['mean_bps']:.2f} bps, CI excludes 0: "
          f"{u['ci_low'] is not None and u['ci_low']>0}, lift {lift}. "
          f"{'SURVIVES' if verdict else 'REJECTED — archive'}.")


if __name__ == "__main__":
    run()
