"""Point-in-time event store: write/read helpers + layer populators.

Every layer of the P+G+F+M+E model writes normalized events here through
`upsert_events`, always carrying a `source_time` (the public timestamp) so a
backtest at time T can select only `source_time <= T` — the single gate that
keeps the whole multi-source engine leak-free.

`populate_congress` seeds the P layer from the disclosures we already have, so
the store is live and testable before the government/fundamentals feeds exist.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from ..config import AMOUNT_RANGES
from ..db import connect

EVENT_COLS = ["event_id", "layer", "event_type", "ticker", "entity", "direction",
              "magnitude", "novelty", "effective_date", "source_time",
              "source_url", "payload"]


def event_id(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:20]


def upsert_events(rows: list[dict]) -> int:
    """Insert events by event_id, skipping ones already present.

    Uses ON CONFLICT DO NOTHING rather than a per-call DELETE scan against the
    (growing) events table — the same fix applied to bar ingestion. Events are
    effectively immutable, so skip-on-conflict is the correct semantics and keeps
    high-volume news ingestion fast.
    """
    if not rows:
        return 0
    con = connect()
    con.executemany(
        f"INSERT INTO events VALUES ({','.join(['?'] * len(EVENT_COLS))}) "
        f"ON CONFLICT DO NOTHING",
        [[r.get(c) for c in EVENT_COLS] for r in rows],
    )
    n = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    con.close()
    return n


# ---------------------------------------------------------------------------
# P layer — congressional purchases as events
# ---------------------------------------------------------------------------
_MAX_AMT = max(hi for _, hi in AMOUNT_RANGES.values())


def populate_congress(verbose: bool = True) -> dict:
    con = connect(read_only=True)
    trades = con.execute("""
        SELECT trade_id, member, member_key, ticker, tx_type, tx_date,
               filing_date, disclosure_lag, amount_mid, owner
        FROM trades
        WHERE ticker IS NOT NULL AND NOT filing_estimated
    """).df()
    con.close()

    trades["filing_date"] = pd.to_datetime(trades["filing_date"])
    import numpy as np
    rows = []
    for r in trades.itertuples(index=False):
        if r.tx_type not in ("purchase", "sale"):
            continue
        direction = 1.0 if r.tx_type == "purchase" else -1.0
        amt = r.amount_mid if (r.amount_mid and r.amount_mid > 0) else 8000.0
        magnitude = float(np.clip(np.log10(amt) / np.log10(_MAX_AMT), 0, 1))
        # novelty proxy: faster disclosure = fresher/more informative
        lag = r.disclosure_lag if r.disclosure_lag is not None else 45
        novelty = float(np.clip(1 - lag / 90, 0, 1))
        rows.append({
            "event_id": event_id("P", r.trade_id),
            "layer": "P",
            "event_type": f"congress_{r.tx_type}",
            "ticker": r.ticker,
            "entity": r.member,
            "direction": direction,
            "magnitude": magnitude,
            "novelty": novelty,
            "effective_date": r.filing_date.date(),
            # public timestamp = filing date (best available; intraday unknown)
            "source_time": r.filing_date.to_pydatetime(),
            "source_url": None,
            "payload": json.dumps({"member_key": r.member_key, "owner": r.owner,
                                   "tx_date": str(r.tx_date)[:10],
                                   "amount_mid": amt, "disclosure_lag": lag}),
        })
    upsert_events(rows)
    if verbose:
        print(f"P layer: wrote {len(rows)} congress events to store")
    return {"p_events": len(rows)}


def layer_counts() -> dict:
    con = connect(read_only=True)
    try:
        rows = con.execute(
            "SELECT layer, COUNT(*) FROM events GROUP BY layer ORDER BY 1"
        ).fetchall()
    finally:
        con.close()
    return dict(rows)


if __name__ == "__main__":
    populate_congress()
    print("layer counts:", layer_counts())
