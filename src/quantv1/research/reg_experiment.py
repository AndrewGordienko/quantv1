"""Roadmap step 3: do government (Federal Register) events improve LARGE?

Federal Register rules are sector-level (no per-company text extraction yet), so
we test the sharpest thing the data supports: does a LARGE congressional purchase
in a sector with ELEVATED recent regulatory activity outperform one in a quiet
sector? For each large buy we count significant rules affecting the stock's
sector in the 90 days before the filing, split large buys into high vs low
regulatory-activity terciles, and compare factor-adjusted 63d CAR (cluster-robust,
train + 2024+ holdout).

Honest caveat: significant rules are frequent and only sector-mapped, so this is
a coarse test. A null result mostly means "sector-level regulatory volume isn't
the signal"; per-company rule extraction (LLM) is the real next step.
"""

from __future__ import annotations

import bisect
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import DATA_DIR
from ..db import connect
from .event_study_v2 import _cluster_bootstrap, HOLDOUT_START


def run(verbose: bool = True) -> dict:
    con = connect(read_only=True)
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    if "event_v2" not in tables:
        con.close()
        raise RuntimeError("run event_study_v2 first")
    large = con.execute("""
        SELECT e.ticker, e.member_key, e.filing_date, e.car_ff_63 AS car, s.sector
        FROM event_v2 e JOIN ticker_sectors s ON e.ticker = s.ticker
        WHERE e.car_ff_63 IS NOT NULL AND e.amount_mid >= 50000 AND s.sector <> 'Unknown'
    """).df()
    rules = con.execute("""
        SELECT source_time, payload FROM events
        WHERE event_type='reg_rule'
    """).df()
    con.close()

    # sector -> sorted list of significant-rule dates
    sector_dates: dict[str, list] = {}
    for r in rules.itertuples(index=False):
        try:
            secs = json.loads(r.payload).get("sectors", [])
        except (TypeError, ValueError):
            secs = []
        d = pd.Timestamp(r.source_time)
        for s in secs:
            sector_dates.setdefault(s, []).append(d)
    for s in sector_dates:
        sector_dates[s].sort()

    def count_prior(sector, when, days=90):
        lst = sector_dates.get(sector)
        if not lst:
            return 0
        lo = when - pd.Timedelta(days=days)
        return bisect.bisect_right(lst, when) - bisect.bisect_left(lst, lo)

    large["filing_date"] = pd.to_datetime(large["filing_date"])
    large["n_rules_90d"] = [count_prior(r.sector, r.filing_date)
                            for r in large.itertuples(index=False)]

    train = large[large["filing_date"] < HOLDOUT_START]
    hold = large[large["filing_date"] >= HOLDOUT_START]

    def terciles(df):
        if len(df) < 30:
            return {}
        lo_q, hi_q = df["n_rules_90d"].quantile([0.33, 0.67])
        low = df[df["n_rules_90d"] <= lo_q]
        high = df[df["n_rules_90d"] >= hi_q]
        return {"low_reg_activity": _cluster_bootstrap(low, "car", "ticker"),
                "high_reg_activity": _cluster_bootstrap(high, "car", "ticker"),
                "thresholds": {"low<=": float(lo_q), "high>=": float(hi_q)}}

    out = {"n_large_with_sector": int(len(large)),
           "n_reg_rules": int(len(rules)),
           "train": terciles(train), "holdout": terciles(hold),
           "generated_at": datetime.now(timezone.utc).isoformat()}
    with open(DATA_DIR / "reg_experiment.json", "w") as f:
        json.dump(out, f, indent=2)

    if verbose:
        print(f"=== FR rules x LARGE ({out['n_reg_rules']} rules, "
              f"{out['n_large_with_sector']} sector-known large buys) ===")
        for seg in ("train", "holdout"):
            t = out[seg]
            if not t:
                continue
            lo, hi = t["low_reg_activity"], t["high_reg_activity"]
            if lo.get("mean") is not None and hi.get("mean") is not None:
                print(f"  {seg}: high-activity CAR={hi['mean']*100:+.2f}% n={hi['n']}  "
                      f"vs low={lo['mean']*100:+.2f}% n={lo['n']}  "
                      f"lift={(hi['mean']-lo['mean'])*100:+.2f}pp")
    return out


if __name__ == "__main__":
    run()
