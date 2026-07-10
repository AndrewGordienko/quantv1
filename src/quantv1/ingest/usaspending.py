"""G-layer ingest: USAspending federal CONTRACT awards as events.

Uses the transaction-level endpoint (each obligation action has a real
`action_date` and incremental amount) rather than award aggregates, so events
are point-in-time datable. We pull large contracts ($>=THRESHOLD) month by month
and resolve the recipient to a public-company ticker.

Point-in-time caveat: USAspending posts transactions with a lag. We restrict to
LARGE ($50M+) awards, which for DoD are also announced same-day on defense.gov's
daily contract digest, so `action_date` is a defensible public date for this
subset — but it is mildly optimistic and flagged as such.

Entity resolution: normalize the recipient legal name and match it against
`sec_entities`; a curated subsidiary->parent map covers the big contractors whose
operating-company name differs from the listed parent (Sikorsky->LMT, etc.).
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from datetime import date, timedelta

import numpy as np

from .. import net  # noqa: F401 - installs OS trust store for VERIFIED TLS
from ..config import CACHE_DIR
from ..db import connect
from ..events.store import event_id, upsert_events

_CACHE = CACHE_DIR / "usaspending"
_CACHE.mkdir(exist_ok=True)

# USAspending posts transactions with a lag, so action_date is NOT the public
# availability time. Large DoD contracts (>$7.5M) are announced same day on
# Defense.gov's contract digest, so for DoD we treat action_date as public; for
# other agencies we add a conservative posting lag. (A Defense.gov timestamp
# cross-check is the future refinement.)
NON_DOD_LAG_DAYS = 30


def _first_seen(agency: str, action_date: date) -> date:
    if agency and "defense" in agency.lower():
        return action_date               # same-day Defense.gov announcement
    return action_date + timedelta(days=NON_DOD_LAG_DAYS)

_UA = {"User-Agent": "quantv1 research (andrew.gordienko05@gmail.com)",
       "Content-Type": "application/json"}
_URL = "https://api.usaspending.gov/api/v2/search/spending_by_transaction/"
THRESHOLD = 100_000_000     # market-moving awards; keeps pages/volume manageable
_SUFFIX = re.compile(r"\b(corporation|corp|incorporated|inc|company|co|llc|lp|"
                     r"ltd|limited|holdings|group|the|and|of|systems|technologies|"
                     r"international|industries|plc|sa|nv|ag)\b")

# Operating-company / subsidiary name (normalized substring) -> parent ticker.
SUBSIDIARY_MAP = {
    "sikorsky": "LMT", "lockheed martin": "LMT", "raytheon": "RTX",
    "pratt whitney": "RTX", "collins aerospace": "RTX", "boeing": "BA",
    "huntington ingalls": "HII", "general dynamics": "GD", "northrop grumman": "NOC",
    "l3harris": "LHX", "l3 technologies": "LHX", "bae systems": "BAESY",
    "leidos": "LDOS", "science applications": "SAIC", "booz allen": "BAH",
    "general electric": "GE", "honeywell": "HON", "textron": "TXT",
    "humana government": "HUM", "mckesson": "MCK", "united technologies": "RTX",
    "palantir": "PLTR", "amazon web services": "AMZN", "microsoft": "MSFT",
    "oracle america": "ORCL", "international business machines": "IBM",
    "accenture federal": "ACN", "deloitte": None, "pfizer": "PFE",
    "merck sharp": "MRK", "caterpillar": "CAT", "deere": "DE",
}


def _normalize(name: str) -> str:
    n = (name or "").lower()
    n = re.sub(r"[.,&/-]", " ", n)
    n = _SUFFIX.sub(" ", n)
    return re.sub(r"\s+", " ", n).strip()


def _sec_index(con) -> dict:
    idx = {}
    for tk, title in con.execute("SELECT ticker, title FROM sec_entities").fetchall():
        norm = _normalize(title)
        if norm and norm not in idx:
            idx[norm] = tk
    return idx


def _resolve(recipient: str, sec_idx: dict) -> str | None:
    norm = _normalize(recipient)
    if not norm:
        return None
    for key, tk in SUBSIDIARY_MAP.items():           # curated subsidiaries first
        if key in norm:
            return tk
    if norm in sec_idx:                              # exact normalized match
        return sec_idx[norm]
    # first two tokens match (e.g. "raytheon company" vs "raytheon")
    toks = norm.split()
    if toks:
        for k in (2, 1):
            cand = " ".join(toks[:k])
            if cand in sec_idx:
                return sec_idx[cand]
    return None


def _months(start_year: int, end_year: int, end_m: int):
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            if y == end_year and m > end_m:
                break
            lo = f"{y}-{m:02d}-01"
            hi = f"{y}-{m:02d}-{[31,29,31,30,31,30,31,31,30,31,30,31][m-1]:02d}"
            yield lo, hi


def _fetch_month(lo: str, hi: str) -> tuple[list[dict], bool]:
    """Return (transactions, ok). ok is False only if the endpoint never
    responded (so the month is NOT cached and will be retried next run)."""
    out, page, any_ok = [], 1, False
    while True:
        body = {
            "filters": {"award_type_codes": ["A", "B", "C", "D"],
                        "time_period": [{"start_date": lo, "end_date": hi}],
                        "award_amounts": [{"lower_bound": THRESHOLD}]},
            "fields": ["Award ID", "Recipient Name", "Action Date",
                       "Transaction Amount", "Awarding Agency", "Award Type"],
            "sort": "Transaction Amount", "order": "desc", "limit": 100, "page": page,
        }
        req = urllib.request.Request(_URL, data=json.dumps(body).encode(),
                                     headers=_UA, method="POST")
        r = None
        for attempt in range(6):                     # endpoint is flaky/slow; retry
            try:
                r = json.load(urllib.request.urlopen(req, timeout=30))
                break
            except Exception:  # noqa: BLE001
                time.sleep(1.0 * (attempt + 1))
        if r is None:
            return out, any_ok                       # gave up this page
        any_ok = True
        res = r.get("results", [])
        out.extend(res)
        if len(res) < 100 or page >= 10:
            break
        page += 1
        time.sleep(0.4)
    return out, True


def _month_cached(month: str) -> list[dict] | None:
    f = _CACHE / f"{month}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except json.JSONDecodeError:
            return None
    return None


def _cache_month(month: str, txns: list[dict]) -> None:
    (_CACHE / f"{month}.json").write_text(json.dumps(txns))


def ingest(start_year: int = 2016, end_year: int = 2026, end_m: int = 6,
           verbose: bool = True) -> dict:
    con = connect(read_only=True)
    sec_idx = _sec_index(con)
    con.close()

    from datetime import datetime as _dt
    total, resolved, fetched, cached = 0, 0, 0, 0
    for lo, hi in _months(start_year, end_year, end_m):
        month = lo[:7]
        txns = _month_cached(month)
        if txns is None:                             # not cached -> fetch (retryable)
            txns, ok = _fetch_month(lo, hi)
            if ok:
                _cache_month(month, txns)            # persist so reruns resume
                fetched += 1
            elif verbose:
                print(f"  {month}: endpoint unavailable, will retry next run")
                continue
        else:
            cached += 1

        rows = []
        for tx in txns:
            amt = tx.get("Transaction Amount") or 0
            if amt < THRESHOLD:
                continue
            tk = _resolve(tx.get("Recipient Name", ""), sec_idx)
            total += 1
            if not tk:
                continue
            resolved += 1
            ad = tx.get("Action Date")
            try:
                y, m, d = map(int, ad.split("-"))
                adate = date(y, m, d)
            except (ValueError, AttributeError):
                continue
            agency = tx.get("Awarding Agency", "")
            seen = _first_seen(agency, adate)         # conservative public availability
            magnitude = float(np.clip(np.log10(max(amt, 1e6)) / np.log10(5e10), 0, 1))
            rows.append({
                "event_id": event_id("G", tx.get("Award ID"), ad, tk),
                "layer": "G", "event_type": "gov_contract", "ticker": tk,
                "entity": agency, "direction": 1.0,
                "magnitude": magnitude, "novelty": 1.0, "effective_date": adate,
                # source_time = first_seen_at (point-in-time gate), NOT action_date
                "source_time": _dt(seen.year, seen.month, seen.day),
                "source_url": "https://www.usaspending.gov/award/",
                "payload": json.dumps({"recipient": tx.get("Recipient Name"),
                                       "amount": amt, "agency": agency,
                                       "award_type": tx.get("Award Type"),
                                       "action_date": ad, "first_seen_at": str(seen)}),
            })
        if rows:
            upsert_events(rows)
        if verbose:
            print(f"  {month}: {len(txns)} txns, {len(rows)} resolved")

    con = connect(read_only=True)
    n = con.execute("SELECT COUNT(*) FROM events WHERE layer='G'").fetchone()[0]
    con.close()
    if verbose:
        print(f"G layer: {n} gov_contract events; resolved {resolved}/{total} "
              f"large awards to public tickers")
    return {"g_events": n, "resolved": resolved, "seen": total}


def tick(start_year: int = 2016, end_year: int = 2026, end_m: int = 6,
         time_budget_s: int = 300, verbose: bool = True) -> dict:
    """Low-priority resumable step: fetch pending (uncached) months within a time
    budget, caching each success (verified TLS, no CERT_NONE). Failed months are
    left uncached so they retry next invocation. Safe to run daily on a cron."""
    import time as _t
    con = connect(read_only=True)
    sec_idx = _sec_index(con)
    con.close()
    t0 = _t.time()
    done, failed = 0, 0
    for lo, hi in _months(start_year, end_year, end_m):
        if _t.time() - t0 > time_budget_s:
            break
        month = lo[:7]
        if _month_cached(month) is not None:
            continue                                 # already have it
        txns, ok = _fetch_month(lo, hi)
        if ok:
            _cache_month(month, txns)
            _write_events(txns, sec_idx)
            done += 1
            if verbose:
                print(f"  {month}: cached {len(txns)} txns")
        else:
            failed += 1
            _t.sleep(min(2 ** failed, 20))           # exponential backoff on failure
            if verbose:
                print(f"  {month}: endpoint down, backing off")
    con = connect(read_only=True)
    n = con.execute("SELECT COUNT(*) FROM events WHERE layer='G' AND event_type='gov_contract'").fetchone()[0]
    con.close()
    if verbose:
        print(f"tick: cached {done} new months, {failed} failed; G contract events={n}")
    return {"cached": done, "failed": failed, "g_contract_events": n}


def _write_events(txns: list[dict], sec_idx: dict) -> None:
    from datetime import datetime as _dt
    rows = []
    for tx in txns:
        amt = tx.get("Transaction Amount") or 0
        if amt < THRESHOLD:
            continue
        tk = _resolve(tx.get("Recipient Name", ""), sec_idx)
        if not tk:
            continue
        ad = tx.get("Action Date")
        try:
            y, m, d = map(int, ad.split("-"))
            adate = date(y, m, d)
        except (ValueError, AttributeError):
            continue
        agency = tx.get("Awarding Agency", "")
        seen = _first_seen(agency, adate)
        magnitude = float(np.clip(np.log10(max(amt, 1e6)) / np.log10(5e10), 0, 1))
        rows.append({
            "event_id": event_id("G", tx.get("Award ID"), ad, tk),
            "layer": "G", "event_type": "gov_contract", "ticker": tk,
            "entity": agency, "direction": 1.0, "magnitude": magnitude, "novelty": 1.0,
            "effective_date": adate, "source_time": _dt(seen.year, seen.month, seen.day),
            "source_url": "https://www.usaspending.gov/award/",
            "payload": json.dumps({"recipient": tx.get("Recipient Name"), "amount": amt,
                                   "agency": agency, "award_type": tx.get("Award Type"),
                                   "action_date": ad, "first_seen_at": str(seen)}),
        })
    if rows:
        upsert_events(rows)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "tick":
        tick()
    else:
        ingest(start_year=int(sys.argv[1]) if len(sys.argv) > 1 else 2016)
