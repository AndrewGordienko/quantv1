"""Small survivorship-safe price-panel feasibility audit; never fits returns."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from quantv1.db import connect


def run(events_path="goldset/sec_event_atlas_phaseA_pilot.jsonl", output="goldset/sec_event_atlas_price_feasibility.json"):
    events = [json.loads(x) for x in Path(events_path).read_text().splitlines() if x.strip()]
    difficult = {"merger_announced", "secondary_offering", "restatement", "restructuring", "auditor_change", "ceo_departure"}
    selected = []
    for e in sorted((e for e in events if e.get("event_type") in difficult), key=lambda e: hashlib.sha256(e["atlas_event_id"].encode()).hexdigest()):
        if e.get("cik") not in {x.get("cik") for x in selected}:
            selected.append(e)
        if len(selected) >= 20:
            break
    con = connect(read_only=True)
    available = {r[0] for r in con.execute("SELECT DISTINCT ticker FROM prices").fetchall()}
    columns = {r[0] for r in con.execute("DESCRIBE prices").fetchall()}
    con.close()
    rows = []
    for e in selected:
        rows.append({"atlas_event_id": e["atlas_event_id"], "cik": e["cik"], "accession_number": e["accession_number"], "ticker": e.get("ticker"), "price_rows": e.get("ticker") in available, "raw_adjusted_ohlcv": {"raw": False, "adjusted": False, "columns": sorted(columns)}, "corporate_actions": False, "listing_interval": False, "delisting_evidence": False, "terminal_status": "MISSING_PANEL_CONTRACT"})
    report = {"status": "BLOCKED_PANEL_INCOMPLETE", "sample_size": len(rows), "requirements": ["raw_ohlcv", "adjusted_ohlcv", "split_dividend_factors", "listing_intervals", "delisting_evidence", "last_trade", "terminal_status", "filing_era_symbol"], "rows": rows, "price_table_columns": sorted(columns)}
    Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "sample_size": len(rows), "price_rows": sum(r["price_rows"] for r in rows), "columns": sorted(columns)}))


if __name__ == "__main__":
    run()
