"""Backtest v2 — leak-free, event-driven, next-open execution.

Built to test the honest hypotheses (not to dress up the ML model, which the
skill-persistence result undercut). It runs transparent RULE strategies so there
is no CV to leak through, plus SPY/QQQ/naive benchmarks, and reports a locked
2024-2026 holdout separately from the pre-2024 training period.

Fixes vs v1:
* Execution at the next session's OPEN (enter and exit), never the filing close.
* Event-driven lifecycle: a disclosure creates ONE signal — enter once, hold
  HOLD_TD trading days, exit; an old filing never becomes a fresh buy again.
* CASH allowed: if fewer than the target number of names qualify, the book is
  under-invested rather than force-concentrated.
* Position caps enforced by water-filling (no clip-then-renormalize breach).
* Delisted/ended tickers are realized at their last traded price and keep their
  weight — surviving names are NOT renormalized up (no survivorship inflation).
* Strategies rank on TRADE-LEVEL features (size, filing speed, repeat, momentum),
  not on politician identity.

Strategies (the four locked experiments + naive):
  naive         every purchase
  large         amount_mid >= $50k
  fast          disclosure_lag <= 14 days
  repeat        member repeat-buys the same ticker within a year
  combo         large AND repeat AND positive 20d pre-entry momentum
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import BENCHMARK_TICKER, COST_BPS, DATA_DIR
from ..db import connect
from .construct import _cap_weights

HOLD_TD = 63
FRESH_TD = 10          # only enter within 10 trading days of the filing
TARGET_N = 15          # target number of names; fewer -> hold cash
MAX_W = 0.08
HOLDOUT_START = "2024-01-01"
QQQ = "QQQ"


@dataclass
class Strategy:
    name: str
    eligible: callable      # row -> bool
    priority: callable      # row -> float (higher = preferred when capacity-limited)


STRATEGIES = [
    Strategy("naive", lambda r: True, lambda r: r["score_amt"]),
    Strategy("large", lambda r: r["amount_mid"] >= 50_000, lambda r: r["amount_mid"]),
    Strategy("fast", lambda r: r["disclosure_lag"] <= 14, lambda r: -r["disclosure_lag"]),
    Strategy("repeat", lambda r: bool(r["is_repeat"]), lambda r: r["amount_mid"]),
    Strategy("combo",
             lambda r: r["amount_mid"] >= 50_000 and bool(r["is_repeat"]) and r["mom20"] > 0,
             lambda r: r["amount_mid"]),
]


def _load_candidates(con) -> pd.DataFrame:
    """Prefer event_v2 (has repeat/mom features); else derive from trades."""
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    if "event_v2" in tables:
        c = con.execute("""
            SELECT trade_id, member_key, ticker, filing_date, disclosure_lag, amount_mid,
                   is_repeat, cluster_count, mom20
            FROM event_v2
        """).df()
    else:
        c = con.execute("""
            SELECT trade_id, member_key, ticker, filing_date, disclosure_lag, amount_mid,
                   FALSE AS is_repeat, 0 AS cluster_count, 0.0 AS mom20
            FROM trades
            WHERE tx_type='purchase' AND ticker IS NOT NULL AND NOT filing_estimated
        """).df()
    c["filing_date"] = pd.to_datetime(c["filing_date"])
    c["amount_mid"] = c["amount_mid"].fillna(8000.0)
    c["disclosure_lag"] = c["disclosure_lag"].fillna(45)
    c["mom20"] = c["mom20"].fillna(0.0)
    c["score_amt"] = np.log10(c["amount_mid"].clip(lower=1000))
    return c


def _panels(con):
    px = con.execute("SELECT ticker, date, open FROM prices ORDER BY date").df()
    px["date"] = pd.to_datetime(px["date"])
    op = px.pivot_table(index="date", columns="ticker", values="open")
    return op


def _run_strategy(strat: Strategy, cand: pd.DataFrame, op: pd.DataFrame,
                  start_i: int) -> dict:
    cal = op.index
    cols = {t: i for i, t in enumerate(op.columns)}
    O = op.to_numpy()
    end_i = len(cal) - 1

    # bucket eligible candidates by filing position
    elig = cand[cand.apply(strat.eligible, axis=1)]
    by_fpos: dict[int, list] = {}
    for r in elig.to_dict("records"):
        fp = cal.searchsorted(r["filing_date"], side="right")  # next session
        if fp < len(cal):
            r["_prio"] = strat.priority(r)
            by_fpos.setdefault(fp, []).append(r)

    open_pos: dict[str, dict] = {}
    weights_prev: dict[str, float] = {}
    equity, dates = [1.0], [cal[start_i]]
    last_px: dict[str, float] = {}

    for t in range(start_i + 1, end_i + 1):
        # mark-to-market open[t-1] -> open[t]
        ret = 0.0
        for tk, w in weights_prev.items():
            ci = cols.get(tk)
            if ci is None:
                continue
            p0 = _px(O, ci, t - 1, last_px, tk)
            p1 = _px(O, ci, t, last_px, tk)
            if p0 and p1 and p0 > 0:
                ret += w * (p1 / p0 - 1.0)
        equity.append(equity[-1] * (1 + ret))

        # exits: held to term, or delisted (no future price)
        for tk, s in list(open_pos.items()):
            ci = cols.get(tk)
            delisted = ci is not None and not np.isfinite(O[t:, ci]).any()
            if t >= s["exit_i"] or delisted:
                del open_pos[tk]

        # entries at today's open
        if len(open_pos) < TARGET_N:
            pool = {}
            for fp in range(max(0, t - FRESH_TD), t + 1):
                for r in by_fpos.get(fp, []):
                    tk = r["ticker"]
                    if tk in open_pos or cols.get(tk) is None:
                        continue
                    if not np.isfinite(O[t, cols[tk]]):
                        continue
                    if tk not in pool or r["_prio"] > pool[tk]["_prio"]:
                        pool[tk] = r
            for r in sorted(pool.values(), key=lambda x: x["_prio"], reverse=True):
                if len(open_pos) >= TARGET_N:
                    break
                tk = r["ticker"]
                open_pos[tk] = {"entry_i": t, "exit_i": t + HOLD_TD,
                                "entry_open": float(O[t, cols[tk]])}

        # weights: equal target with cash when under TARGET_N; cap-enforced
        held = list(open_pos)
        if held:
            raw = np.ones(len(held))
            w = _cap_weights(raw, MAX_W)
            # scale so each name gets 1/TARGET_N (cash if fewer than target)
            invested = min(len(held) / TARGET_N, 1.0)
            w = w / w.sum() * invested if w.sum() > 0 else w
            weights_now = dict(zip(held, w))
        else:
            weights_now = {}
        turn = 0.5 * sum(abs(weights_now.get(tk, 0) - weights_prev.get(tk, 0))
                         for tk in set(weights_now) | set(weights_prev))
        equity[-1] *= (1 - turn * 2 * COST_BPS / 1e4)
        weights_prev = weights_now
        dates.append(cal[t])

    return {"dates": dates, "equity": equity}


def _px(O, ci, t, last_px, tk):
    v = O[t, ci]
    if np.isfinite(v):
        last_px[tk] = v
        return v
    return last_px.get(tk)      # carry last for delisted/missing (no renormalize)


def _bench(op, dates, ticker) -> list:
    if ticker not in op.columns:
        return [1.0] * len(dates)
    cal = op.index
    col = op[ticker].to_numpy()
    ci = list(op.columns).index(ticker)
    idxs = [cal.searchsorted(d) for d in dates]
    eq = [1.0]
    for k in range(1, len(idxs)):
        a, b = idxs[k - 1], idxs[k]
        r = 0.0
        if a < len(cal) and b < len(cal):
            p0, p1 = op.iloc[a, ci], op.iloc[b, ci]
            if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                r = p1 / p0 - 1.0
        eq.append(eq[-1] * (1 + r))
    return eq


def _metrics(dates, eq) -> dict:
    eq = np.array(eq)
    idx = pd.to_datetime(dates)
    def seg(mask):
        e = eq[mask]
        if len(e) < 20:
            return None
        e = e / e[0]
        r = np.diff(e) / e[:-1]
        yrs = (idx[mask][-1] - idx[mask][0]).days / 365.25
        return {"cagr": float(e[-1] ** (1 / yrs) - 1) if yrs > 0 else None,
                "sharpe": float(np.mean(r) / np.std(r) * np.sqrt(252)) if np.std(r) > 0 else None,
                "max_dd": float(np.min(e / np.maximum.accumulate(e) - 1)),
                "final": float(e[-1])}
    ho = pd.Timestamp(HOLDOUT_START)
    return {"full": seg(np.ones(len(eq), bool)),
            "train": seg(np.asarray(idx < ho)),
            "holdout": seg(np.asarray(idx >= ho))}


def run(start_after: str = "2015-01-01", verbose: bool = True) -> dict:
    con = connect(read_only=True)
    cand = _load_candidates(con)
    op = _panels(con)
    con.close()

    start_i = max(int(op.index.searchsorted(pd.Timestamp(start_after))), 1)
    results, curves = {}, {}
    ref_dates = None
    for strat in STRATEGIES:
        out = _run_strategy(strat, cand, op, start_i)
        results[strat.name] = _metrics(out["dates"], out["equity"])
        curves[strat.name] = out["equity"]
        ref_dates = out["dates"]

    for b in (BENCHMARK_TICKER, QQQ):
        eq = _bench(op, ref_dates, b)
        results[b.lower()] = _metrics(ref_dates, eq)
        curves[b.lower()] = eq

    curve_out = [{"date": str(d.date()),
                  **{k: float(curves[k][i]) for k in curves}}
                 for i, d in enumerate(ref_dates)]
    payload = {"hold_td": HOLD_TD, "target_n": TARGET_N, "max_w": MAX_W,
               "holdout_start": HOLDOUT_START, "results": results,
               "curve": curve_out, "generated_at": datetime.now(timezone.utc).isoformat()}
    with open(DATA_DIR / "backtest_v2.json", "w") as f:
        json.dump(payload, f, indent=2)

    if verbose:
        _print(results)
    return payload


def _print(results):
    print("=== Backtest v2 (next-open, cash, delistings) — FULL / TRAIN / HOLDOUT-2024+ ===")
    hdr = f"{'strategy':10s} | {'FULL cagr/sharpe/DD':>26s} | {'HOLDOUT cagr/sharpe/DD':>26s}"
    print(hdr)
    for k, m in results.items():
        f, h = m["full"], m["holdout"]
        def fmt(x):
            return (f"{x['cagr']*100:+5.1f}% {x['sharpe']:+.2f} {x['max_dd']*100:5.1f}%"
                    if x else "        n/a")
        print(f"{k:10s} | {fmt(f):>26s} | {fmt(h):>26s}")


if __name__ == "__main__":
    run()
