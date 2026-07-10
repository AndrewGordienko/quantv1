"""Ingest Senate + House Stock Watcher disclosure feeds into DuckDB `trades`.

Both feeds are lightly-structured dumps parsed from official PDFs, so the values
are messy: null bytes in descriptions, half-formed amount ranges, placeholder
tickers, and inconsistent transaction-type labels. Everything is normalized to
the `trades` schema here so the rest of the pipeline never touches raw feeds.

Point-in-time note: the House feed carries `disclosure_date` (the date the trade
became public and thus tradeable). The Senate aggregate does NOT, and is stale
past ~2020, so we estimate its filing date as tx_date + SENATE_ESTIMATED_LAG_DAYS
and flag `filing_estimated = TRUE`. The rigorous backtest filters those out.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from datetime import date, datetime, timedelta

from .. import config
from ..db import connect

_UA = {"User-Agent": "quantv1-ingest"}
_TICKER_RE = re.compile(r"^[A-Z][A-Z.\-]{0,5}$")


# ---------------------------------------------------------------------------
# Field parsers
# ---------------------------------------------------------------------------
def _clean_str(s) -> str:
    if s is None:
        return ""
    # Feed descriptions contain literal NUL bytes and stray whitespace.
    return s.replace("\x00", "").strip()


def parse_date(s: str) -> date | None:
    s = _clean_str(s)
    if not s or s in {"--", "N/A"}:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def parse_amount(s: str) -> tuple[float | None, float | None, float | None]:
    """Return (low, high, midpoint) dollar bounds for a reported range."""
    s = _clean_str(s)
    if not s or s in {"Unknown", "--", "N/A"}:
        return None, None, None
    if s in config.AMOUNT_RANGES:
        lo, hi = config.AMOUNT_RANGES[s]
        return lo, hi, (lo + hi) / 2
    # Malformed single-bound values like "$15,001": treat as the lower edge of
    # its bracket by matching the leading number into the range table.
    nums = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", s)]
    if len(nums) >= 2:
        lo, hi = float(nums[0]), float(nums[1])
        return lo, hi, (lo + hi) / 2
    if len(nums) == 1:
        v = float(nums[0])
        for lo, hi in config.AMOUNT_RANGES.values():
            if lo <= v <= hi:
                return lo, hi, (lo + hi) / 2
        return v, v, v
    return None, None, None


def parse_type(s: str) -> str | None:
    """Collapse the varied labels into purchase / sale / exchange."""
    s = _clean_str(s).lower()
    if "purchase" in s or s == "buy":
        return "purchase"
    if "sale" in s or "sell" in s:
        return "sale"
    if "exchange" in s:
        return "exchange"
    return None


def parse_owner(s: str) -> str:
    s = _clean_str(s).lower()
    if not s or s in {"--", "n/a"}:
        return "unknown"
    if "spouse" in s:
        return "spouse"
    if "joint" in s:
        return "joint"
    if "child" in s or "dependent" in s:
        return "child"
    if "self" in s:
        return "self"
    return s


def clean_ticker(s: str) -> str | None:
    s = _clean_str(s).upper()
    if not s or s in {"--", "N/A", "NONE"}:
        return None
    if not _TICKER_RE.match(s):
        return None
    return s


def member_key(name: str) -> str:
    """Stable join key: lowercase, drop punctuation and middle initials/suffixes."""
    n = _clean_str(name).lower()
    n = re.sub(r"[.,]", " ", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv|mr|mrs|ms|dr|hon)\b", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _trade_id(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Feed fetch + normalization
# ---------------------------------------------------------------------------
def fetch_json(url: str):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def _row(chamber, member, ticker, rec, tx_type, tx_date, filing_date, estimated):
    amt_lo, amt_hi, amt_mid = parse_amount(rec.get("amount", ""))
    # Prefer a feed-provided midpoint when our parse yields nothing.
    if amt_mid is None and rec.get("amount_mid"):
        try:
            amt_mid = float(rec["amount_mid"])
        except (TypeError, ValueError):
            pass
    lag = (filing_date - tx_date).days if (tx_date and filing_date) else None
    mk = member_key(member)
    return {
        "trade_id": _trade_id(chamber, mk, ticker, tx_date, filing_date,
                              tx_type, rec.get("amount", ""), rec.get("owner", "")),
        "chamber": chamber,
        "member": _clean_str(member),
        "member_key": mk,
        "ticker": ticker,
        "asset_desc": _clean_str(rec.get("asset_description", ""))[:200],
        "asset_type": _clean_str(rec.get("asset_type", "")),
        "tx_type": tx_type,
        "tx_date": tx_date,
        "filing_date": filing_date,
        "filing_estimated": estimated,
        "disclosure_lag": lag,
        "amount_lo": amt_lo,
        "amount_hi": amt_hi,
        "amount_mid": amt_mid,
        "owner": parse_owner(rec.get("owner", "")),
        "raw": json.dumps({k: _clean_str(v) if isinstance(v, str) else v
                           for k, v in rec.items()}),
    }


def normalize_house(records) -> list[dict]:
    out = []
    for rec in records:
        ticker = clean_ticker(rec.get("ticker", ""))
        tx_type = parse_type(rec.get("type", ""))
        tx_date = parse_date(rec.get("transaction_date", ""))
        filing_date = parse_date(rec.get("disclosure_date", ""))
        if not (ticker and tx_type and tx_date and filing_date):
            continue
        # Guard against feed typos where tx_date is after filing or wildly old.
        if filing_date < tx_date or tx_date.year < 2012:
            continue
        out.append(_row("house", rec.get("representative", ""), ticker, rec,
                        tx_type, tx_date, filing_date, False))
    return out


def normalize_senate(records) -> list[dict]:
    lag = timedelta(days=config.SENATE_ESTIMATED_LAG_DAYS)
    out = []
    for rec in records:
        ticker = clean_ticker(rec.get("ticker", ""))
        tx_type = parse_type(rec.get("type", ""))
        tx_date = parse_date(rec.get("transaction_date", ""))
        if not (ticker and tx_type and tx_date) or tx_date.year < 2012:
            continue
        filing_date = tx_date + lag  # estimated — senate feed lacks disclosure date
        out.append(_row("senate", rec.get("senator", ""), ticker, rec,
                        tx_type, tx_date, filing_date, True))
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def ingest(verbose: bool = True) -> dict:
    house = normalize_house(fetch_json(config.HOUSE_WATCHER_URL))
    senate = normalize_senate(fetch_json(config.SENATE_WATCHER_URL))
    # The feeds contain exact-duplicate rows; collapse by deterministic id.
    rows = list({r["trade_id"]: r for r in house + senate}.values())

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    con = connect()
    con.execute("CREATE TEMP TABLE _stage AS SELECT * FROM trades WHERE 0=1")
    con.executemany(
        f"INSERT INTO _stage VALUES ({','.join(['?'] * len(_COLS))})",
        [[(now if c == "first_seen_at" else r[c]) for c in _COLS] for r in rows],
    )
    # Insert ONLY genuinely new trade_ids, stamping first_seen_at = now. Existing
    # rows are left untouched so their real first-observation time is preserved
    # (a disclosure is immutable once we've seen it). This is what makes the
    # forward record able to enter only signals first seen after launch.
    before = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    con.execute("""
        INSERT INTO trades
        SELECT * FROM _stage WHERE trade_id NOT IN (SELECT trade_id FROM trades)
    """)
    con.execute("DROP TABLE _stage")
    n_total = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    con.close()

    stats = {"house": len(house), "senate": len(senate),
             "new": n_total - before, "db_total": n_total}
    if verbose:
        print(f"Ingested house={len(house)} senate={len(senate)} -> "
              f"{n_total - before} new trades (first_seen stamped), total={n_total}")
    return stats


# Column order must match the `trades` schema in db.py exactly.
_COLS = [
    "trade_id", "chamber", "member", "member_key", "ticker", "asset_desc",
    "asset_type", "tx_type", "tx_date", "filing_date", "filing_estimated",
    "disclosure_lag", "amount_lo", "amount_hi", "amount_mid", "owner",
    "first_seen_at", "raw",
]


if __name__ == "__main__":
    ingest()
