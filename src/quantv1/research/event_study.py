"""Event study: do congressional disclosures predict abnormal returns?

For every disclosed trade we measure the cumulative abnormal return (CAR) after
the FILING date over several horizons, then aggregate — overall and sliced by
chamber, transaction type, amount bucket, owner, and committee match. If the
all-trades CAR is ~0 (the honest prior post-STOCK-Act), the slices reveal where
signal concentrates, and those slices become the model's features.

Purchases and sales are reported separately: a purchase's "abnormal return" is
the stock's excess return (you'd have wanted to buy); for a sale we report the
same excess return but it is read inversely (a good sale precedes weakness).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import HORIZONS
from ..db import connect
from .returns import PriceStore


def _amount_bucket(mid: float | None) -> str:
    if mid is None or not np.isfinite(mid):
        return "unknown"
    if mid < 15_000:
        return "$1k-15k"
    if mid < 50_000:
        return "$15k-50k"
    if mid < 250_000:
        return "$50k-250k"
    if mid < 1_000_000:
        return "$250k-1M"
    return "$1M+"


def compute(con=None, store: PriceStore | None = None,
            purchases_only: bool = False) -> pd.DataFrame:
    """Return a per-trade frame with abnormal returns at each horizon + slice keys."""
    own = con is None
    con = con or connect(read_only=True)
    store = store or PriceStore(con)

    where = "WHERE ticker IS NOT NULL"
    if purchases_only:
        where += " AND tx_type = 'purchase'"
    trades = con.execute(f"""
        SELECT t.trade_id, t.chamber, t.member, t.member_key, t.ticker,
               t.tx_type, t.filing_date, t.filing_estimated, t.amount_mid, t.owner,
               m.committees
        FROM trades t
        LEFT JOIN members m USING (member_key)
        {where}
        ORDER BY t.filing_date
    """).df()
    if own:
        con.close()

    # Attach committee jurisdiction sectors (may be null for historical members).
    import json

    def jur_sectors(c):
        if not c:
            return []
        try:
            return json.loads(c).get("jurisdiction_sectors", [])
        except (TypeError, ValueError):
            return []

    trades["jur_sectors"] = trades["committees"].map(jur_sectors)

    recs = []
    for r in trades.itertuples(index=False):
        row = {
            "trade_id": r.trade_id, "chamber": r.chamber, "member": r.member,
            "member_key": r.member_key, "ticker": r.ticker, "tx_type": r.tx_type,
            "filing_date": r.filing_date, "filing_estimated": r.filing_estimated,
            "amount_bucket": _amount_bucket(r.amount_mid), "owner": r.owner,
        }
        for h in HORIZONS:
            row[f"ar_{h}"] = store.abnormal_return(r.ticker, r.filing_date, h, beta=1.0)
        recs.append(row)

    return pd.DataFrame(recs)


def summarize(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Mean CAR + t-stat + n at each horizon, optionally grouped by a column."""
    hcols = [c for c in df.columns if c.startswith("ar_")]

    def agg(sub: pd.DataFrame) -> dict:
        out = {"n": len(sub)}
        for c in hcols:
            v = sub[c].dropna()
            out[f"{c}_mean"] = float(v.mean()) if len(v) else np.nan
            # one-sample t-stat vs 0
            out[f"{c}_t"] = (float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))
                             if len(v) > 2 and v.std(ddof=1) > 0 else np.nan)
            out[f"{c}_n"] = int(len(v))
        return out

    if by is None:
        return pd.DataFrame([agg(df)])
    rows = []
    for key, sub in df.groupby(by):
        rows.append({by: key, **agg(sub)})
    return pd.DataFrame(rows).sort_values(f"{hcols[-1]}_mean", ascending=False)


def run_report(purchases_only: bool = True) -> dict:
    store = PriceStore()
    df = compute(store=store, purchases_only=purchases_only)
    # Restrict headline stats to non-estimated (House) filings for point-in-time rigor.
    rigorous = df[~df["filing_estimated"]]
    report = {
        "overall": summarize(rigorous),
        "by_chamber": summarize(df, "chamber"),
        "by_amount": summarize(rigorous, "amount_bucket"),
        "by_owner": summarize(rigorous, "owner"),
    }
    return {"per_trade": df, "summaries": report}


if __name__ == "__main__":
    out = run_report(purchases_only=True)
    print("=== Overall (House purchases, filing-date CAR) ===")
    print(out["summaries"]["overall"].to_string(index=False))
    print("\n=== By amount bucket ===")
    print(out["summaries"]["by_amount"].to_string(index=False))
