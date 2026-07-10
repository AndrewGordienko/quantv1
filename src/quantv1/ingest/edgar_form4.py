"""F-layer ingest: SEC Form 4 OPEN-MARKET insider purchases as events.

Uses SEC's structured Form 345 quarterly data sets (bulk TSV of every insider
filing) rather than per-company crawling — one ~12MB zip per quarter, filtered
to our traded universe. We keep only high-signal transactions:

    TRANS_CODE = 'P'  (open-market purchase)  AND  acquired ('A')

Grants, option exercises and 10b5-1 dispositions are noise for our purpose; a
director/officer spending their own cash on the open market is the confirming
signal we want to pair with a congressional purchase.

Each qualifying filing becomes a layer='F', event_type='insider_buy' event with
the FILING_DATE as its public `source_time` (point-in-time) — the earliest a
follower could have seen it.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from datetime import datetime

import numpy as np

from ..db import connect
from ..events.store import event_id, upsert_events

_UA = {"User-Agent": "quantv1 research (andrew.gordienko05@gmail.com)"}
_BASE = ("https://www.sec.gov/files/structureddata/data/"
         "insider-transactions-data-sets/{q}_form345.zip")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def _parse_date(s: str):
    s = (s or "").strip()
    if len(s) < 9:
        return None
    try:
        d, mon, y = s.split("-")
        return datetime(int(y), _MONTHS[mon.upper()], int(d)).date()
    except (ValueError, KeyError):
        return None


def _quarters(start_year: int, end_year: int, end_q: int) -> list[str]:
    out = []
    for y in range(start_year, end_year + 1):
        for q in range(1, 5):
            if y == end_year and q > end_q:
                break
            out.append(f"{y}q{q}")
    return out


def _read_tsv(zf, name):
    with zf.open(name) as f:
        header = f.readline().decode("latin-1").rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        for line in f:
            yield idx, line.decode("latin-1").rstrip("\n").split("\t")


def _resolved_ciks(con) -> dict:
    """CIK(int) -> ticker, for tickers congress actually traded."""
    rows = con.execute("""
        SELECT DISTINCT s.cik, s.ticker
        FROM sec_entities s JOIN trades t ON t.ticker = s.ticker
    """).fetchall()
    return {int(cik): tk for cik, tk in rows}


def _process_quarter(q: str, ciks: dict) -> list[dict]:
    url = _BASE.format(q=q)
    try:
        blob = urllib.request.urlopen(urllib.request.Request(url, headers=_UA),
                                      timeout=120).read()
    except Exception:  # noqa: BLE001 - quarter may not be published yet
        return []
    zf = zipfile.ZipFile(io.BytesIO(blob))

    # SUBMISSION: keep Form 4 for our CIKs -> accession -> (filing_date, ticker)
    sub = {}
    for idx, row in _read_tsv(zf, "SUBMISSION.tsv"):
        if row[idx["DOCUMENT_TYPE"]] != "4":
            continue
        try:
            cik = int(row[idx["ISSUERCIK"]])
        except (ValueError, IndexError):
            continue
        tk = ciks.get(cik)
        if tk is None:
            continue
        fd = _parse_date(row[idx["FILING_DATE"]])
        if fd:
            sub[row[idx["ACCESSION_NUMBER"]]] = (fd, tk, cik)
    if not sub:
        return []

    # NONDERIV_TRANS: open-market purchases (code P, acquired) for those accessions
    agg = {}  # accession -> {shares, value, trans_date}
    for idx, row in _read_tsv(zf, "NONDERIV_TRANS.tsv"):
        acc = row[idx["ACCESSION_NUMBER"]]
        if acc not in sub:
            continue
        if row[idx["TRANS_CODE"]] != "P":
            continue
        if row[idx["TRANS_ACQUIRED_DISP_CD"]] != "A":
            continue
        try:
            sh = float(row[idx["TRANS_SHARES"]] or 0)
            px = float(row[idx["TRANS_PRICEPERSHARE"]] or 0)
        except ValueError:
            continue
        td = _parse_date(row[idx["TRANS_DATE"]])
        a = agg.setdefault(acc, {"shares": 0.0, "value": 0.0, "trans_date": td})
        a["shares"] += sh
        a["value"] += sh * px
        if td and (a["trans_date"] is None or td < a["trans_date"]):
            a["trans_date"] = td

    # REPORTINGOWNER: name + relationship (first owner per accession)
    owner = {}
    for idx, row in _read_tsv(zf, "REPORTINGOWNER.tsv"):
        acc = row[idx["ACCESSION_NUMBER"]]
        if acc in agg and acc not in owner:
            owner[acc] = (row[idx["RPTOWNERNAME"]], row[idx["RPTOWNER_RELATIONSHIP"]])

    events = []
    for acc, a in agg.items():
        if a["value"] <= 0:
            continue
        fd, tk, cik = sub[acc]
        nm, rel = owner.get(acc, ("", ""))
        magnitude = float(np.clip(np.log10(max(a["value"], 1e3)) / np.log10(5e7), 0, 1))
        events.append({
            "event_id": event_id("F", acc),
            "layer": "F",
            "event_type": "insider_buy",
            "ticker": tk,
            "entity": nm,
            "direction": 1.0,
            "magnitude": magnitude,
            "novelty": 1.0,
            "effective_date": a["trans_date"] or fd,
            "source_time": datetime(fd.year, fd.month, fd.day),
            "source_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}",
            "payload": json.dumps({"accession": acc, "owner": nm, "relationship": rel,
                                   "shares": a["shares"], "value": round(a["value"], 2),
                                   "cik": cik}),
        })
    return events


def ingest(start_year: int = 2016, end_year: int = 2026, end_q: int = 2,
           verbose: bool = True) -> dict:
    con = connect(read_only=True)
    ciks = _resolved_ciks(con)
    con.close()
    if verbose:
        print(f"insider ingest: {len(ciks)} CIKs in traded universe")

    total = 0
    for q in _quarters(start_year, end_year, end_q):
        evs = _process_quarter(q, ciks)
        if evs:
            upsert_events(evs)
            total += len(evs)
        if verbose:
            print(f"  {q}: {len(evs)} open-market insider buys")

    con = connect(read_only=True)
    n = con.execute("SELECT COUNT(*) FROM events WHERE layer='F'").fetchone()[0]
    con.close()
    if verbose:
        print(f"F layer: {n} insider_buy events total (added {total} this run)")
    return {"added": total, "f_events": n}


if __name__ == "__main__":
    import sys
    sy = int(sys.argv[1]) if len(sys.argv) > 1 else 2016
    ingest(start_year=sy)
