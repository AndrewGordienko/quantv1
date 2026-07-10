"""Entry-timing study: how fast does post-disclosure alpha decay, and does
post-filing price action tell us whether we missed the trade?

The live portfolio buys disclosures filed up to 90 days ago as if they were
fresh. That is only defensible if the drift after a filing persists. This study
answers two questions empirically (House purchases, filing dates only):

1. DECAY — if you enter d trading days AFTER the filing (d = 0, 5, 10, 21, 42),
   what is the mean forward 63-day abnormal return from that late entry?
   If it hits ~0 by d=21, holding 90-day-old disclosures is pure noise.

2. CONDITION ON PRICE ACTION — entering 21 days late, split by
   (a) the stock's return since the filing (did it already run / already bleed?)
   (b) whether the stock sits above or below its 50-day moving average.
   This tells us whether "down since filing" names mean-revert (buy the dip)
   or keep bleeding (falling knife) — i.e. whether a trend gate is justified.

Output: data/entry_timing.json for the dashboard + the tactical overlay notes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import DATA_DIR
from ..db import connect
from .returns import PriceStore

DELAYS = [0, 5, 10, 21, 42]
FWD = 63           # forward abnormal-return window from each delayed entry
COND_DELAY = 21    # delay at which we condition on price action


def _stat(vals: list[float]) -> dict:
    v = pd.Series(vals).dropna()
    if len(v) < 30:
        return {"mean": None, "t": None, "n": int(len(v))}
    t = float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if v.std(ddof=1) > 0 else None
    return {"mean": float(v.mean()), "t": t, "n": int(len(v))}


def run(verbose: bool = True) -> dict:
    con = connect(read_only=True)
    store = PriceStore(con)
    trades = con.execute("""
        SELECT ticker, filing_date FROM trades
        WHERE tx_type = 'purchase' AND ticker IS NOT NULL AND NOT filing_estimated
        ORDER BY filing_date
    """).df()
    con.close()

    cal = store.cal
    by_delay: dict[int, list[float]] = {d: [] for d in DELAYS}
    cond_rows = []  # per-trade diagnostics at COND_DELAY

    for r in trades.itertuples(index=False):
        if not store.has(r.ticker):
            continue
        pos = store._pos_on_or_after(r.filing_date)
        if pos is None:
            continue
        col = store.close[r.ticker]
        base = col.iloc[pos] if pos < len(cal) else np.nan

        for d in DELAYS:
            e = pos + d
            if e >= len(cal):
                continue
            entry_date = cal[e]
            ar = store.abnormal_return(r.ticker, entry_date, FWD)
            if ar is not None:
                by_delay[d].append(ar)

            if d == COND_DELAY and ar is not None:
                entry_px = col.iloc[e]
                ret_since = (float(entry_px / base - 1.0)
                             if np.isfinite(entry_px) and np.isfinite(base) and base > 0
                             else np.nan)
                ma50 = col.iloc[max(0, e - 50):e].mean()
                above_ma = (bool(entry_px > ma50)
                            if np.isfinite(entry_px) and np.isfinite(ma50) else None)
                cond_rows.append({"ar": ar, "ret_since": ret_since, "above_ma": above_ma})

    decay = [{"delay": d, **_stat(by_delay[d])} for d in DELAYS]

    cond = pd.DataFrame(cond_rows)
    conditional = {}
    if not cond.empty:
        conditional["above_ma50"] = _stat(cond.loc[cond["above_ma"] == True, "ar"].tolist())   # noqa: E712
        conditional["below_ma50"] = _stat(cond.loc[cond["above_ma"] == False, "ar"].tolist())  # noqa: E712
        q = cond.dropna(subset=["ret_since"])
        if len(q) > 100:
            edges = q["ret_since"].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
            labels = ["worst q (fell most)", "q2", "q3", "best q (ran most)"]
            q = q.assign(bucket=pd.cut(q["ret_since"], bins=edges, labels=labels,
                                       include_lowest=True, duplicates="drop"))
            conditional["by_ret_since_filing"] = [
                {"bucket": str(b), **_stat(sub["ar"].tolist()),
                 "median_ret_since": float(sub["ret_since"].median())}
                for b, sub in q.groupby("bucket", observed=True)
            ]

    out = {
        "decay": decay,
        "conditional_at_delay": COND_DELAY,
        "conditional": conditional,
        "fwd_window": FWD,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(DATA_DIR / "entry_timing.json", "w") as f:
        json.dump(out, f, indent=2)

    if verbose:
        print("=== Alpha decay: mean 63d AR entering d trading days after filing ===")
        for row in decay:
            m = f"{row['mean']*100:+.2f}%" if row["mean"] is not None else "  n/a"
            t = f"t={row['t']:+.2f}" if row["t"] else ""
            print(f"  d={row['delay']:>3}  {m}  {t}  (n={row['n']})")
        print(f"\n=== Conditioned at d={COND_DELAY} ===")
        for k in ("above_ma50", "below_ma50"):
            s = conditional.get(k, {})
            if s.get("mean") is not None:
                print(f"  {k:12s} {s['mean']*100:+.2f}%  t={s['t']:+.2f}  (n={s['n']})")
        for b in conditional.get("by_ret_since_filing", []):
            if b.get("mean") is not None:
                print(f"  {b['bucket']:22s} {b['mean']*100:+.2f}%  t={b['t']:+.2f}  "
                      f"(n={b['n']}, med ret since filing {b['median_ret_since']*100:+.1f}%)")
    return out


if __name__ == "__main__":
    run()
