"""Rigorous fade validation harness (hardened v2, per second review).

Correctness fixes over v1:
  * ONE shared trigger function used by BOTH real events and the matched control
    (no divergent signal code; control timestamps align so next_idx_after lands
    on the intended spike bar).
  * Matched no-news control uses the SAME rel-volume definition (30 bars BEFORE
    the spike window, no overlap).
  * Shuffled control preserves TIME-OF-DAY and per-day CLUSTERING: it permutes
    which calendar day a ticker's news-day maps to, keeping intraday structure.
  * Triple barrier is PESSIMISTIC: if both TP and SL are touched in one bar, take
    the stop (until historical quotes exist).
  * Portfolio includes ZERO-TRADE days (full trading calendar) for Sharpe, and
    reports average gross exposure + turnover.
  * Real-minus-shuffled LIFT has a bootstrapped CI (resampled jointly by ticker);
    the main mean CI uses a day-BLOCK bootstrap (same-day/catalyst correlation).
  * Reporting: CI-excludes-zero checks BOTH sides; stock-only counts are consistent.
  * Experiment registry records code hash, dataset hash, full params, experiment id.

Rule is FROZEN. Fade is ARCHIVED_NEGATIVE; this harness is the reusable substrate
for the earnings model.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import DATA_DIR, BENCHMARK_TICKER, ROOT
from ..db import connect
from .replay import BarPanel, ReplayParams, to_ns, _bench_return_timealigned
from .news_reaction import news_events, MOVE_THR, VOL_THR

DISCOVERY = {"AAPL", "MSFT"}
ETFS = {"SPY", "QQQ", "XLK", "XLV", "XLE", "XLF", "XLI", "XLY"}
MAX_CONCURRENT = 5
NEAR_NS = int(30 * 60 * 1e9)
REGISTRY = DATA_DIR / "experiment_registry.jsonl"
_MIN_NS = 60_000_000_000


_SESSION_NS = int(6.5 * 3600 * 1e9)
_DAY_NS = 86_400_000_000_000


def _day_id(ns):
    return ns // _DAY_NS                     # integer calendar day (UTC)


def attach_tod_volume(panel):
    """Per-ticker TIME-OF-DAY volume profile: mean volume by minute-of-day, so
    relative volume compares a bar to the same clock-minute historically (not to
    premarket / the preceding 30 min, which behaves badly around the open)."""
    panel.tod_vol = {}
    for tk, d in panel.data.items():
        minute = (d["ts"] // _MIN_NS) % 1440
        vol = np.nan_to_num(d["vol"])
        prof = np.zeros(1440)
        cnt = np.zeros(1440)
        np.add.at(prof, minute, vol)
        np.add.at(cnt, minute, 1)
        prof = prof / np.maximum(cnt, 1)
        panel.tod_vol[tk] = (minute, prof)


def _trigger(panel, tk, i_pub, obs_bars, residual):
    """Return (fires, side). Move over [i_pub, i_dec); relative volume normalized
    by the TIME-OF-DAY profile. One implementation for real events + control."""
    d = panel.data[tk]
    i_dec = i_pub + obs_bars
    if i_pub == 0 or i_dec > len(d["close"]):
        return False, 0
    r0 = d["close"][i_dec - 1] / d["open"][i_pub] - 1.0
    if residual and BENCHMARK_TICKER in panel.data:
        r0 -= _bench_return_timealigned(panel, BENCHMARK_TICKER, d["ts"][i_pub], d["ts"][i_dec - 1])
    minute, prof = panel.tod_vol[tk]
    expected = prof[minute[i_pub:i_dec]].mean()
    rel = d["vol"][i_pub:i_dec].mean() / (expected + 1e-9) if expected else 0.0
    if not np.isfinite(r0) or abs(r0) < MOVE_THR or rel < VOL_THR:
        return False, 0
    return True, -int(np.sign(r0))          # FADE the move


def _fade_trades(ev: pd.DataFrame, panel: BarPanel, params: ReplayParams,
                 residual=False) -> pd.DataFrame:
    cost = (params.spread_bps + params.slippage_bps + params.fees_bps) / 1e4
    e = ev.dropna(subset=["public_time"]).copy()
    e["pt_ns"] = to_ns(e["public_time"])
    if "catalyst_id" not in e.columns:
        e["catalyst_id"] = None
    e = e.sort_values("pt_ns")
    trades, last_exit = [], {}
    for tk, pt, cat in zip(e["ticker"].to_numpy(), e["pt_ns"].to_numpy(),
                           e["catalyst_id"].to_numpy()):
        if not panel.has(tk) or last_exit.get(tk, 0) >= pt:
            continue
        d = panel.data[tk]
        i_pub = panel.next_idx_after(tk, pt)
        if i_pub is None:
            continue
        fires, side = _trigger(panel, tk, i_pub, params.obs_bars, residual)
        if not fires:
            continue
        i_entry = i_pub + params.obs_bars + params.entry_delay
        if i_entry + 1 >= len(d["ts"]):
            continue
        entry = d["open"][i_entry] * (1 + side * cost)
        entry_day = _day_id(d["ts"][i_entry])
        exit_px, exit_i = None, None
        for j in range(i_entry, min(i_entry + params.max_hold, len(d["ts"]))):
            if not params.allow_overnight and _day_id(d["ts"][j]) != entry_day:
                exit_px, exit_i = d["close"][j - 1], j - 1; break   # EOD liquidation
            up = entry * (1 + params.tp) if side > 0 else entry * (1 - params.tp)
            dn = entry * (1 - params.sl) if side > 0 else entry * (1 + params.sl)
            hit_tp = (side > 0 and d["high"][j] >= up) or (side < 0 and d["low"][j] <= up)
            hit_sl = (side > 0 and d["low"][j] <= dn) or (side < 0 and d["high"][j] >= dn)
            if hit_sl:                      # PESSIMISTIC: stop wins ties
                exit_px, exit_i = dn, j; break
            if hit_tp:
                exit_px, exit_i = up, j; break
        if exit_px is None:
            last = min(i_entry + params.max_hold - 1, len(d["ts"]) - 1)
            # don't let the timeout exit cross the session either
            while last > i_entry and _day_id(d["ts"][last]) != entry_day:
                last -= 1
            exit_i, exit_px = last, d["close"][last]
        raw = side * (exit_px / entry - 1) - cost                  # executable long/short
        badj = _bench_return_timealigned(panel, BENCHMARK_TICKER, d["ts"][i_entry], d["ts"][exit_i])
        fadj = raw - side * badj                                   # market/factor-adjusted
        hedged = raw - side * badj - cost                          # incl. SPY-hedge round trip
        trades.append({"ticker": tk, "catalyst_id": cat, "entry_ns": int(d["ts"][i_entry]),
                       "exit_ns": int(d["ts"][exit_i]), "net": float(fadj),
                       "raw_net": float(raw), "fadj_net": float(fadj), "hedged_net": float(hedged),
                       "day": str(pd.Timestamp(int(d["ts"][i_entry]), unit="ns").date())})
        bar_ns = int(d["ts"][exit_i] - d["ts"][exit_i - 1]) if exit_i > 0 else _MIN_NS
        last_exit[tk] = int(d["ts"][exit_i]) + params.cooldown_bars * bar_ns
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
def _news_ns_by_ticker(events, panel):
    return {tk: np.sort(to_ns(g["public_time"]))
            for tk, g in events.groupby("ticker") if tk in panel.data}


def _matched_no_news_spikes(events, panel, obs_bars, residual, rng, cap=2000) -> pd.DataFrame:
    """Bars where the SAME trigger fires and are NOT within 30 min of news.
    Emits public_time = ts[i_pub-1] so next_idx_after lands exactly on i_pub."""
    news_ns = _news_ns_by_ticker(events, panel)
    out = []
    for tk, d in panel.data.items():
        if tk in ETFS:
            continue
        ts = d["ts"]
        n = len(ts)
        nn = news_ns.get(tk, np.array([]))
        hits = []
        for i in range(31, n - obs_bars - 2):
            fires, _ = _trigger(panel, tk, i, obs_bars, residual)
            if not fires:
                continue
            t = ts[i]
            if len(nn):
                j = np.searchsorted(nn, t)
                if (j < len(nn) and abs(nn[j] - t) < NEAR_NS) or (j > 0 and abs(nn[j - 1] - t) < NEAR_NS):
                    continue
            hits.append(i)
        if len(hits) > cap:
            hits = sorted(rng.choice(hits, cap, replace=False))
        for i in hits:
            out.append({"ticker": tk, "public_time": pd.Timestamp(int(ts[i - 1]), unit="ns")})
    return pd.DataFrame(out)


def _shuffle_news(events, panel, rng) -> pd.DataFrame:
    """Shuffle the calendar DAY of each CATALYST (moving all its tickers together,
    preserving cross-ticker membership + time-of-day + per-day clustering). Only
    the date identity is destroyed. Falls back to per-ticker-day if no catalyst."""
    ev = events.copy()
    ev["public_time"] = pd.to_datetime(ev["public_time"])
    trade_dates = np.array(sorted({pd.Timestamp(int(x), unit="ns").date()
                                   for x in panel.data[BENCHMARK_TICKER]["ts"][::390]})) \
        if BENCHMARK_TICKER in panel.data else None
    out = []
    if "catalyst_id" in ev.columns and ev["catalyst_id"].notna().any() and trade_dates is not None:
        for cid, g in ev.groupby("catalyst_id"):
            nd = rng.choice(trade_dates)                 # one new day for the whole catalyst
            for r in g.itertuples(index=False):
                t = r.public_time
                out.append({"ticker": r.ticker, "catalyst_id": cid,
                            "public_time": pd.Timestamp(datetime(nd.year, nd.month, nd.day, t.hour, t.minute))})
        return pd.DataFrame(out)
    # fallback: per-ticker day shuffle
    for tk, g in ev.groupby("ticker"):
        if tk not in panel.data:
            continue
        bar_dates = np.array(sorted({pd.Timestamp(int(x), unit="ns").date()
                                     for x in panel.data[tk]["ts"][::390]}))
        if len(bar_dates) < 2:
            continue
        mp = {sd: rng.choice(bar_dates) for sd in g["public_time"].dt.date.unique()}
        for t in g["public_time"]:
            nd = mp[t.date()]
            out.append({"ticker": tk, "public_time": pd.Timestamp(datetime(nd.year, nd.month, nd.day, t.hour, t.minute))})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _day_block_ci(trades, rng, n_boot=3000) -> dict:
    """Mean net (bps) with a CI that resamples trading DAYS (same-day/catalyst
    correlation), so trades within a day are kept together."""
    if trades.empty:
        return {"n": 0, "mean_bps": None, "ci_low": None, "ci_high": None}
    by_day = {d: g["net"].to_numpy() for d, g in trades.groupby("day")}
    days = list(by_day)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(days, len(days), replace=True)
        boot[b] = np.concatenate([by_day[d] for d in pick]).mean()
    return {"n": int(len(trades)), "n_days": len(days),
            "n_tickers": int(trades["ticker"].nunique()),
            "mean_bps": float(trades["net"].mean() * 1e4),
            "ci_low": float(np.percentile(boot, 2.5) * 1e4),
            "ci_high": float(np.percentile(boot, 97.5) * 1e4)}


def _catalyst_block_ci(trades, rng, n_boot=3000) -> dict:
    """Mean net (bps) resampling CATALYSTS (a catalyst's trades — across tickers
    and its intraday updates — stay together). Falls back to day-block if no
    catalyst_id. This is the two-way clustering the review asked for."""
    if trades.empty:
        return {"n": 0, "mean_bps": None, "ci_low": None, "ci_high": None, "cluster": "none"}
    key = "catalyst_id" if trades["catalyst_id"].notna().any() else "day"
    by = {c: g["net"].to_numpy() for c, g in trades.groupby(key)}
    keys = list(by)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(keys, len(keys), replace=True)
        boot[b] = np.concatenate([by[k] for k in pick]).mean()
    return {"n": int(len(trades)), "n_clusters": len(keys), "cluster": key,
            "n_tickers": int(trades["ticker"].nunique()),
            "mean_bps": float(trades["net"].mean() * 1e4),
            "ci_low": float(np.percentile(boot, 2.5) * 1e4),
            "ci_high": float(np.percentile(boot, 97.5) * 1e4)}


def _lift_ci(real, shuffled, rng, n_boot=3000) -> dict:
    """Bootstrapped CI on (real_mean - shuffled_mean), resampled JOINTLY by
    ticker (the shared cluster)."""
    if real.empty or shuffled.empty:
        return {"lift_bps": None, "ci_low": None, "ci_high": None}
    rt = {tk: g["net"].to_numpy() for tk, g in real.groupby("ticker")}
    st = {tk: g["net"].to_numpy() for tk, g in shuffled.groupby("ticker")}
    common = [t for t in rt if t in st]
    if len(common) < 3:
        return {"lift_bps": float((real["net"].mean() - shuffled["net"].mean()) * 1e4),
                "ci_low": None, "ci_high": None}
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(common, len(common), replace=True)
        rm = np.concatenate([rt[t] for t in pick]).mean()
        sm = np.concatenate([st[t] for t in pick]).mean()
        boot[b] = (rm - sm) * 1e4
    return {"lift_bps": float((real["net"].mean() - shuffled["net"].mean()) * 1e4),
            "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5))}


def _portfolio(trades, panel, max_concurrent=MAX_CONCURRENT, pnl_col="raw_net") -> dict:
    """Concurrency-capped portfolio on a chosen P&L column, over the FULL trading
    calendar (zeros on no-trade days). Gross exposure is TIME-WEIGHTED from actual
    entry/exit intervals (not count-of-trades-started)."""
    if trades.empty:
        return {"n": 0}
    cal = sorted({int(_day_id(x)) for x in panel.data[BENCHMARK_TICKER]["ts"]}) \
        if BENCHMARK_TICKER in panel.data else None
    t = trades.sort_values("entry_ns")
    w = 1.0 / max_concurrent
    open_intervals, day_pnl, taken, turnover, open_time_ns = [], {}, 0, 0.0, 0
    for r in t.itertuples(index=False):
        open_intervals = [x for x in open_intervals if x > r.entry_ns]
        if len(open_intervals) >= max_concurrent:
            continue
        open_intervals.append(r.exit_ns)
        taken += 1
        turnover += 2 * w
        open_time_ns += (r.exit_ns - r.entry_ns)          # position-time for exposure
        day = int(_day_id(r.entry_ns))
        day_pnl[day] = day_pnl.get(day, 0.0) + w * getattr(r, pnl_col)
    idx = [d for d in (cal or sorted(day_pnl)) if min(day_pnl) <= d <= max(day_pnl)]
    s = pd.Series({d: day_pnl.get(d, 0.0) for d in idx}).sort_index()
    if len(s) < 2:
        return {"n_taken": taken, "total_return": float(s.sum()), "active_days": len(day_pnl)}
    eq = (1 + s).cumprod()
    yrs = max((s.index[-1] - s.index[0]) / 365.25, 1e-6)
    # time-weighted avg gross = (avg concurrent positions) * w, over session time
    total_session_ns = len(s) * _SESSION_NS
    avg_gross = (open_time_ns / total_session_ns) * w if total_session_ns else 0.0
    return {"n_taken": int(taken), "trading_days": int(len(s)), "active_days": int(len(day_pnl)),
            "pnl_basis": pnl_col, "total_return": float(eq.iloc[-1] - 1),
            "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1),
            "sharpe": float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else None,
            "max_dd": float((eq / eq.cummax() - 1).min()),
            "avg_gross_exposure_time_weighted": float(avg_gross),
            "turnover_per_active_day": float(turnover / max(len(day_pnl), 1))}


def _group(trades, which):
    if which == "discovery":
        return trades[trades["ticker"].isin(DISCOVERY)]
    if which == "unseen":
        return trades[~trades["ticker"].isin(DISCOVERY | ETFS)]
    return trades[~trades["ticker"].isin(ETFS)]


def _excludes_zero(ci):
    return ci["ci_low"] is not None and (ci["ci_low"] > 0 or ci["ci_high"] < 0)


# ---------------------------------------------------------------------------
# Registry (proper trial ledger)
# ---------------------------------------------------------------------------
_DEP_FILES = [
    "src/quantv1/v4/fade_validation.py", "src/quantv1/v4/replay.py",
    "src/quantv1/v4/news_reaction.py", "src/quantv1/config.py",
    "src/quantv1/v4/data_pipeline.py",
]


def _git_commit():
    import subprocess
    try:
        g = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return (g.stdout.strip() or "nogit") + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "nogit"


def _code_hash():
    h = hashlib.sha1()
    for f in _DEP_FILES:                              # every strategy dependency
        try:
            h.update((ROOT / f).read_bytes())
        except OSError:
            h.update(b"missing")
    return h.hexdigest()[:12]


def _dataset_hash(con):
    from .data_pipeline import PARQUET
    manifest = sorted((p.name, p.stat().st_size) for p in PARQUET.glob("*.parquet"))
    r = con.execute("SELECT COUNT(*), COUNT(DISTINCT ticker), MIN(ts), MAX(ts) FROM bars_minute").fetchone()
    n = con.execute("SELECT COUNT(*) FROM events WHERE layer='N'").fetchone()[0]
    return hashlib.sha1(f"{r}|{n}|{manifest}".encode()).hexdigest()[:12]


def run(verbose=True) -> dict:
    con = connect(read_only=True)
    events = news_events(con)
    panel = BarPanel(con, table="bars_minute")
    ds_hash = _dataset_hash(con)
    con.close()
    events = events[events["ticker"].isin(panel.data.keys())].reset_index(drop=True)
    attach_tod_volume(panel)                          # time-of-day volume profiles
    rng = np.random.default_rng(17)

    base = ReplayParams(obs_bars=5, entry_delay=0, max_hold=30, tp=0.008, sl=0.008,
                        spread_bps=2, slippage_bps=2, cooldown_bars=30, allow_overnight=False)

    real = _fade_trades(events, panel, base)
    shuf = _fade_trades(_shuffle_news(events, panel, rng), panel, base)
    nonews = _fade_trades(_matched_no_news_spikes(events, panel, 5, False, rng), panel, base)
    delayed = _fade_trades(events, panel, ReplayParams(**{**base.__dict__, "entry_delay": 3}))
    cost2 = _fade_trades(events, panel, ReplayParams(**{**base.__dict__, "spread_bps": 4, "slippage_bps": 4}))

    report = {}
    for g in ("discovery", "unseen", "combined"):
        rg, sg = _group(real, g), _group(shuf, g)
        report[g] = {"real": _catalyst_block_ci(rg, rng), "shuffled": _catalyst_block_ci(sg, rng),
                     "lift": _lift_ci(rg, sg, rng),
                     "portfolio_raw": _portfolio(rg, panel, pnl_col="raw_net"),
                     "portfolio_factor_adj": _portfolio(rg, panel, pnl_col="fadj_net")}
    robustness = {"matched_no_news": _catalyst_block_ci(_group(nonews, "combined"), rng),
                  "delayed_entry_3bar": _catalyst_block_ci(_group(delayed, "combined"), rng),
                  "double_cost": _catalyst_block_ci(_group(cost2, "combined"), rng),
                  "stock_trade_counts": {"real": int(len(_group(real, "combined"))),
                                         "delayed": int(len(_group(delayed, "combined"))),
                                         "double_cost": int(len(_group(cost2, "combined"))),
                                         "no_news": int(len(_group(nonews, "combined")))}}

    code_hash = _code_hash()
    universe = sorted([t for t in panel.data if t not in ETFS])
    experiment_id = hashlib.sha1(
        f"fade_news_spike|{code_hash}|{ds_hash}|{base.__dict__}|{universe}|"
        f"discovery={sorted(DISCOVERY)}".encode()).hexdigest()[:16]
    out = {"experiment": "fade_news_spike", "experiment_id": experiment_id,
           "status": "ARCHIVED_NEGATIVE",
           "git_commit": _git_commit(), "code_hash": code_hash, "dataset_hash": ds_hash,
           "params": base.__dict__, "frozen_rule": {"move_thr": MOVE_THR, "vol_thr": VOL_THR},
           "universe": universe, "discovery_holdout": {"discovery": sorted(DISCOVERY),
                                                       "validation": "unseen stocks"},
           "news_events": int(len(events)), "report": report, "robustness": robustness,
           "generated_at": datetime.now(timezone.utc).isoformat()}
    with open(DATA_DIR / "v4_fade_validation.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    _register(out)
    if verbose:
        _print(out)
    return out


def _register(out):
    u = out["report"]["unseen"]
    with open(REGISTRY, "a") as f:
        f.write(json.dumps({
            "ts": out["generated_at"], "experiment_id": out["experiment_id"],
            "experiment": out["experiment"], "status": out["status"],
            "git_commit": out["git_commit"], "code_hash": out["code_hash"],
            "dataset_hash": out["dataset_hash"], "params": out["params"],
            "universe": out["universe"], "discovery_holdout": out["discovery_holdout"],
            "unseen_mean_bps": u["real"]["mean_bps"],
            "unseen_ci": [u["real"]["ci_low"], u["real"]["ci_high"]],
            "unseen_lift": u["lift"],
        }) + "\n")


def _print(out):
    print(f"=== Fade validation v2 ({out['status']}) — mean net bps/trade, day-block 95% CI ===")
    print(f"code={out['code_hash']} data={out['dataset_hash']}  news={out['news_events']}\n")
    print(f"{'group':10s} {'trades':>6s} {'days':>5s} {'tk':>3s} {'real bps':>9s} "
          f"{'95% CI':>18s} {'lift[CI]':>20s} {'raw%':>7s} {'fadj%':>7s} {'gross':>6s}")
    for g in ("discovery", "unseen", "combined"):
        r = out["report"][g]; cr, lf = r["real"], r["lift"]
        praw, pfa = r["portfolio_raw"], r["portfolio_factor_adj"]
        if cr["mean_bps"] is None:
            print(f"{g:10s}  n/a"); continue
        lci = f"[{lf['ci_low']:.1f},{lf['ci_high']:.1f}]" if lf.get("ci_low") is not None else "[--]"
        rr = (praw.get("total_return") or 0) * 100
        fr = (pfa.get("total_return") or 0) * 100
        gx = praw.get("avg_gross_exposure_time_weighted") or 0
        print(f"{g:10s} {cr['n']:6d} {cr.get('n_clusters',0):5d} {cr['n_tickers']:3d} {cr['mean_bps']:9.2f} "
              f"[{cr['ci_low']:7.2f},{cr['ci_high']:7.2f}] {(lf['lift_bps'] or 0):6.1f}{lci:>13s} "
              f"{rr:7.1f} {fr:7.1f} {gx:6.2f}")
    print("\nrobustness (stock-only, mean bps [day-block CI]):")
    for k, v in out["robustness"].items():
        if k == "stock_trade_counts":
            print(f"  counts: {v}"); continue
        if v.get("mean_bps") is not None:
            print(f"  {k:20s} {v['mean_bps']:+.2f} [{v['ci_low']:+.2f},{v['ci_high']:+.2f}] n={v['n']}")
    u = out["report"]["unseen"]
    survives = _excludes_zero(u["real"]) and u["real"]["mean_bps"] > 0 and \
        (u["lift"].get("ci_low") or -1) > 0
    print(f"\nVERDICT (unseen): mean {u['real']['mean_bps']:.2f} bps, CI excludes 0 "
          f"({'yes' if _excludes_zero(u['real']) else 'no'}), lift {u['lift']['lift_bps']:.2f}. "
          f"{'SURVIVES' if survives else 'ARCHIVED_NEGATIVE'}.")


if __name__ == "__main__":
    run()
