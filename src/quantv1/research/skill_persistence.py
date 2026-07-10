"""Does politician trading skill PERSIST year-to-year?

The whole "follow the best politicians" thesis rests on skill being a stable
trait, not luck. Test it directly: rank each member by their purchase
performance in year t, then measure how that ranking holds up in year t+1.

* Entry is the NEXT SESSION'S OPEN after the disclosure date (the first price
  we could actually transact at — not the filing-day close).
* Performance = mean 63-trading-day return minus SPY over the same window.
* Members need >= MIN_TRADES purchases in BOTH years to enter a year-pair.
* Persistence = Spearman rank correlation of member skill between adjacent years,
  plus a transition check: how do year-t top-quartile members do in year t+1?

If the pooled Spearman is ~0, "top politician" is not a tradeable trait and the
member-skill feature should be dropped (or replaced by trade-level features).

Caveat: mean-minus-SPY is still a beta=1 approximation; event_study_v2 will add
factor adjustment. This test is about persistence, for which SPY-relative is a
fair first cut.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import BENCHMARK_TICKER, DATA_DIR
from ..db import connect

MIN_TRADES = 5
FWD = 63


def _load_panels(con):
    px = con.execute(
        "SELECT ticker, date, open, close FROM prices ORDER BY date"
    ).df()
    px["date"] = pd.to_datetime(px["date"])
    op = px.pivot_table(index="date", columns="ticker", values="open")
    cl = px.pivot_table(index="date", columns="ticker", values="close")
    return op, cl


def _next_open_fwd_ar(op, cl, cal, ticker, filing_date) -> float | None:
    """63d return entering at the first OPEN strictly after filing_date, minus SPY."""
    if ticker not in op.columns:
        return None
    # strictly next session (side="right" => first index > filing_date)
    i = cal.searchsorted(pd.Timestamp(filing_date), side="right")
    if i >= len(cal):
        return None
    o = op[ticker].to_numpy()
    c = cl[ticker].to_numpy()
    spy = cl[BENCHMARK_TICKER].to_numpy() if BENCHMARK_TICKER in cl.columns else None
    # find first finite open at/after i
    e = next((j for j in range(i, len(cal)) if np.isfinite(o[j])), None)
    if e is None or e + FWD >= len(cal):
        return None
    entry, exit_ = o[e], c[e + FWD]
    if not (np.isfinite(entry) and np.isfinite(exit_)) or entry <= 0:
        return None
    stock_ret = exit_ / entry - 1.0
    if spy is None or not (np.isfinite(spy[e]) and np.isfinite(spy[e + FWD])) or spy[e] <= 0:
        return stock_ret
    return stock_ret - (spy[e + FWD] / spy[e] - 1.0)


def run(verbose: bool = True) -> dict:
    con = connect(read_only=True)
    op, cl = _load_panels(con)
    trades = con.execute("""
        SELECT member_key, member, ticker, filing_date
        FROM trades
        WHERE tx_type='purchase' AND ticker IS NOT NULL AND NOT filing_estimated
        ORDER BY filing_date
    """).df()
    con.close()

    cal = cl.index
    trades["filing_date"] = pd.to_datetime(trades["filing_date"])
    trades["year"] = trades["filing_date"].dt.year
    trades["ar"] = [
        _next_open_fwd_ar(op, cl, cal, r.ticker, r.filing_date)
        for r in trades.itertuples(index=False)
    ]
    trades = trades.dropna(subset=["ar"])

    # per member-year mean AR (require MIN_TRADES)
    grp = (trades.groupby(["member_key", "year"])
           .agg(ar=("ar", "mean"), n=("ar", "size")).reset_index())
    grp = grp[grp["n"] >= MIN_TRADES]

    years = sorted(grp["year"].unique())
    pairs = []
    pooled_x, pooled_y = [], []
    for y in years:
        a = grp[grp["year"] == y].set_index("member_key")["ar"]
        b = grp[grp["year"] == y + 1].set_index("member_key")["ar"]
        common = a.index.intersection(b.index)
        if len(common) < 8:
            continue
        rho, p = spearmanr(a[common], b[common])
        pairs.append({"year": int(y), "next": int(y + 1), "n_members": int(len(common)),
                      "spearman": float(rho), "p": float(p)})
        pooled_x += list(a[common].values)
        pooled_y += list(b[common].values)

    pooled_rho, pooled_p = (spearmanr(pooled_x, pooled_y) if len(pooled_x) > 10 else (np.nan, np.nan))

    # transition: year-t top-quartile members -> mean AR in year t+1
    trans = {"top_q_next": [], "bottom_q_next": []}
    for y in years:
        a = grp[grp["year"] == y]
        b = grp[grp["year"] == y + 1].set_index("member_key")["ar"]
        if len(a) < 8 or b.empty:
            continue
        hi = a[a["ar"] >= a["ar"].quantile(0.75)]["member_key"]
        lo = a[a["ar"] <= a["ar"].quantile(0.25)]["member_key"]
        trans["top_q_next"] += [b[m] for m in hi if m in b.index]
        trans["bottom_q_next"] += [b[m] for m in lo if m in b.index]

    out = {
        "min_trades": MIN_TRADES, "fwd_days": FWD, "entry": "next_session_open",
        "year_pairs": pairs,
        "pooled_spearman": float(pooled_rho), "pooled_p": float(pooled_p),
        "pooled_n": len(pooled_x),
        "transition": {
            "top_quartile_next_year_mean_ar":
                float(np.mean(trans["top_q_next"])) if trans["top_q_next"] else None,
            "bottom_quartile_next_year_mean_ar":
                float(np.mean(trans["bottom_q_next"])) if trans["bottom_q_next"] else None,
            "n_top": len(trans["top_q_next"]), "n_bottom": len(trans["bottom_q_next"]),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(DATA_DIR / "skill_persistence.json", "w") as f:
        json.dump(out, f, indent=2)

    if verbose:
        print("=== Politician skill persistence (year t -> t+1, next-open entry) ===")
        for pr in pairs:
            print(f"  {pr['year']}->{pr['next']}  Spearman={pr['spearman']:+.3f} "
                  f"(p={pr['p']:.2f}, n={pr['n_members']})")
        print(f"\n  POOLED Spearman = {pooled_rho:+.3f}  (p={pooled_p:.3f}, n={len(pooled_x)})")
        tt = out["transition"]
        print(f"\n  Year-t TOP quartile    -> next-year mean AR "
              f"{_pct(tt['top_quartile_next_year_mean_ar'])}  (n={tt['n_top']})")
        print(f"  Year-t BOTTOM quartile -> next-year mean AR "
              f"{_pct(tt['bottom_quartile_next_year_mean_ar'])}  (n={tt['n_bottom']})")
        print("\n  Interpretation: Spearman ~0 => skill does NOT persist; "
              "'top politician' is not a tradeable trait.")
    return out


def _pct(x):
    return f"{x*100:+.2f}%" if x is not None else "n/a"


if __name__ == "__main__":
    run()
