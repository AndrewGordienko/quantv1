"""Stage-1 DESCRIPTIVE actor-impact study (B1 vs B2, before any model).

For each actor-mention event, measure the market-adjusted (stock - SPY) return of
the mentioned ticker after the next-open entry, at 30-min and 2-hour horizons:
  * IMPACT magnitude   = mean |residual|  (does the mention move the stock at all?)
  * DIRECTION          = mean signed residual (any systematic push?)
Compared to a generic-news baseline on the same tickers. CIs are clustered by
CATALYST. This does NOT fit a model — it just asks whether conditioning on WHO
carries information beyond generic news. If impact doesn't vary by actor, the
identity layer (B2) adds nothing and we stop.

Caveat: news MENTION is a proxy for the actor acting; transcript-grade sources
(Fed, earnings calls) come next. Small per-actor n — descriptive only.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import DATA_DIR, BENCHMARK_TICKER
from ..db import connect
from .replay import BarPanel, to_ns, _bench_return_timealigned

H = {"30m": 30, "2h": 120}


def _residual(panel, tk, pt_ns, bars):
    d = panel.data[tk]
    i0 = panel.next_idx_after(tk, pt_ns)
    if i0 is None or i0 + bars >= len(d["ts"]):
        return None
    if not (np.isfinite(d["open"][i0]) and np.isfinite(d["close"][i0 + bars]) and d["open"][i0] > 0):
        return None
    stock = d["close"][i0 + bars] / d["open"][i0] - 1.0
    bench = _bench_return_timealigned(panel, BENCHMARK_TICKER, d["ts"][i0], d["ts"][i0 + bars])
    return stock - bench


def _cluster_ci(vals, clusters, rng, n_boot=2000):
    v = np.asarray(vals)
    if len(v) < 10:
        return {"mean_bps": float(v.mean() * 1e4) if len(v) else None, "ci_low": None, "ci_high": None, "n": len(v)}
    by = {}
    for x, c in zip(v, clusters):
        by.setdefault(c, []).append(x)
    keys = list(by)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(keys, len(keys), replace=True)
        boot[b] = np.concatenate([by[k] for k in pick]).mean()
    return {"mean_bps": float(v.mean() * 1e4), "ci_low": float(np.percentile(boot, 2.5) * 1e4),
            "ci_high": float(np.percentile(boot, 97.5) * 1e4), "n": len(v)}


def run(verbose=True) -> dict:
    con = connect(read_only=True)
    ae = con.execute("""
        SELECT actor_id, ticker, public_time, catalyst_id FROM actor_events
        WHERE ticker IS NOT NULL AND public_time IS NOT NULL
    """).df()
    panel = BarPanel(con, table="bars_minute")
    con.close()
    ae = ae[ae["ticker"].isin(panel.data.keys())].reset_index(drop=True)
    if ae.empty:
        return {"note": "no actor-events with bars"}
    ae["pt_ns"] = to_ns(ae["public_time"])
    rng = np.random.default_rng(11)

    recs = []
    for r in ae.itertuples(index=False):
        row = {"actor": r.actor_id, "ticker": r.ticker, "catalyst": r.catalyst_id}
        for name, bars in H.items():
            row[name] = _residual(panel, r.ticker, r.pt_ns, bars)
        recs.append(row)
    df = pd.DataFrame(recs)

    out = {"horizons": list(H), "by_actor": {}, "baseline_impact_2h_bps": None}
    # per-actor impact (|resid|) and direction (signed) at 2h, catalyst-clustered
    for actor, g in df.groupby("actor"):
        v = g["2h"].dropna()
        if len(v) < 10:
            continue
        gg = g.dropna(subset=["2h"])
        out["by_actor"][actor] = {
            "n": int(len(v)),
            "impact_2h": _cluster_ci(gg["2h"].abs(), gg["catalyst"], rng),
            "direction_2h": _cluster_ci(gg["2h"], gg["catalyst"], rng),
        }
    # generic-news baseline impact on the same tickers (|resid 2h| of ALL news)
    con = connect(read_only=True)
    news = con.execute("""
        SELECT ticker, MIN(source_time) pt, catalyst_id FROM events
        WHERE layer='N' AND ticker IS NOT NULL AND catalyst_id IS NOT NULL
        GROUP BY ticker, catalyst_id
    """).df()
    con.close()
    news = news[news["ticker"].isin(panel.data.keys())]
    news["pt_ns"] = to_ns(news["pt"])
    base = [abs(x) for x in (_residual(panel, r.ticker, r.pt_ns, 120)
                             for r in news.sample(min(4000, len(news)), random_state=1).itertuples(index=False))
            if x is not None]
    out["baseline_impact_2h_bps"] = float(np.mean(base) * 1e4) if base else None

    with open(DATA_DIR / "v4_actor_impact.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    if verbose:
        print("=== Actor impact (2h market-adjusted, catalyst-clustered) ===")
        print(f"generic-news baseline |resid| 2h: {out['baseline_impact_2h_bps']:.1f} bps\n")
        print(f"{'actor':12s} {'n':>4s} {'|impact| bps [CI]':>26s} {'direction bps [CI]':>26s}")
        for a, s in sorted(out["by_actor"].items(), key=lambda x: -x[1]["impact_2h"]["mean_bps"]):
            im, di = s["impact_2h"], s["direction_2h"]
            imci = f"[{im['ci_low']:.0f},{im['ci_high']:.0f}]" if im["ci_low"] is not None else "[--]"
            dici = f"[{di['ci_low']:.0f},{di['ci_high']:.0f}]" if di["ci_low"] is not None else "[--]"
            print(f"{a:12s} {s['n']:4d} {im['mean_bps']:8.1f}{imci:>17s} {di['mean_bps']:8.1f}{dici:>17s}")
        print("\nRead: |impact| vs baseline tests whether a mention moves the stock more than "
              "generic news; direction CI excluding 0 would be a (rare) tradeable push.")
    return out


if __name__ == "__main__":
    run()
