"""Point-in-time SEC security identity mapping and coverage gates.

Ticker is an observation, never the permanent key.  Mappings are sourced from
the filing available at ``source_public_time`` and conflicting intervals fail
closed rather than silently selecting a current ticker.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import re
from html import unescape
from pathlib import Path
from typing import Iterable

from ..config import DATA_DIR
from ..events.atlas import EVENT_TYPES

SHARE_CLASS_TRADING_RULE = "PRIMARY_LIQUID_COMMON_CLASS_ISSUER_CAP_V1"


@dataclass(frozen=True)
class SecurityMapping:
    security_id: str
    cik: str
    instrument_class: str
    ticker: str
    exchange: str
    valid_from: str
    valid_to: str | None
    source_accession: str
    source_public_time: str
    mapping_method: str
    confidence: str


def _time(value: str) -> datetime:
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return dt.astimezone(timezone.utc)


def extract_cover_mapping(document: str) -> dict | None:
    """Extract trading symbol/exchange using inline-XBRL then cover text."""
    # Inline XBRL is intentionally parsed with conservative tags; malformed or
    # ambiguous documents return None and can be reviewed manually.
    def ix_value(concept: str) -> list[str]:
        chunks = re.findall(r"<ix:[^>]*\bname\s*=\s*['\"](?:dei:)?" + concept + r"['\"][^>]*>(.*?)</ix:[^>]+>", document, re.I | re.S)
        return [re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", c))).strip() for c in chunks]
    xbrl = ix_value("TradingSymbol")
    exch = ix_value("SecurityExchangeName")
    xbrl_found = bool(xbrl)
    if not xbrl:
        xbrl = re.findall(r"(?:Trading Symbol|TradingSymbol)\s*</[^>]+>\s*<[^>]+>\s*([A-Z][A-Z0-9.\-]{0,9})", document, re.I)
    if not exch:
        exch = re.findall(r"(?:Exchange|Security Exchange)\s*[:<]\s*[^>]*>?\s*([A-Z][A-Z0-9 .\-]{1,19})", document, re.I)
    tickers = sorted({x.strip().upper() for x in xbrl if x.strip() and len(x.strip()) <= 12})
    exchanges = sorted({x.strip().upper() for x in exch if x.strip()})
    if not tickers:
        return None
    # Multiple classes are retained in metadata; the interval's primary
    # ticker remains deterministic and is never confused with a permanent key.
    return {"ticker": tickers[0], "listed_tickers": tickers,
            "exchange": exchanges[0] if exchanges else "",
            "method": "INLINE_XBRL" if xbrl_found else "COVER_TEXT",
            "confidence": "HIGH" if xbrl_found and len(tickers) == 1 else "MEDIUM"}


def build_intervals(filings: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Build effective intervals; return (mappings, conflict ledger)."""
    observations = []
    conflicts = []
    for row in filings:
        mapping = row.get("mapping") or extract_cover_mapping(str(row.get("document", "")))
        source_time = row.get("source_public_time") or row.get("public_time") or row.get("acceptance_time")
        if not mapping or not row.get("cik") or not row.get("accession_number") or not source_time:
            continue
        try:
            public = _time(source_time)
        except (TypeError, ValueError):
            conflicts.append({"reason": "BAD_PUBLIC_TIME", "row": row.get("accession_number")})
            continue
        cik = str(row["cik"]).zfill(10)
        instrument_class = str(mapping.get("instrument_class") or row.get("instrument_class") or "COMMON").upper()
        security_id = str(mapping.get("security_id") or f"CIK:{cik}:CLASS:{instrument_class}")
        observations.append((cik, security_id, instrument_class, public, row, mapping))
    observations.sort(key=lambda x: (x[0], x[1], x[3], str(x[4].get("accession_number", ""))))
    by_security: dict[str, list] = {}
    for item in observations:
        by_security.setdefault(item[1], []).append(item)
    out = []
    for security_id, rows in by_security.items():
        cik = rows[0][0]
        instrument_class = rows[0][2]
        for i, (_, _, _, start, row, mapping) in enumerate(rows):
            end = rows[i + 1][3].isoformat() if i + 1 < len(rows) else None
            # An interval end caused by a later filing is not a delisting.
            # Delisting/deregistration is only recorded when the source filing
            # explicitly supplies a delisted_at date.
            explicit_end = row.get("delisted_at")
            if explicit_end:
                try:
                    end = _time(explicit_end).isoformat()
                except (TypeError, ValueError):
                    conflicts.append({"reason": "BAD_DELISTED_TIME", "row": row.get("accession_number")})
                    continue
            rec = SecurityMapping(security_id, cik, instrument_class, mapping["ticker"], mapping.get("exchange", ""), start.isoformat(), end, row["accession_number"], start.isoformat(), mapping.get("method", "UNKNOWN"), mapping.get("confidence", "LOW"))
            # Same effective instant with different identity is ambiguous.
            if out and out[-1]["cik"] == cik and out[-1]["valid_from"] == rec.valid_from and (out[-1]["ticker"], out[-1]["exchange"]) != (rec.ticker, rec.exchange):
                conflicts.append({"reason": "OVERLAPPING_CONFLICT", "cik": cik, "accessions": [out[-1]["source_accession"], rec.source_accession]})
                out[-1]["status"] = "CONFLICT"
                continue
            value = asdict(rec)
            value["listed_tickers"] = list(mapping.get("listed_tickers", [mapping["ticker"]]))
            value["status"] = "DELISTED" if explicit_end else "ACTIVE"
            out.append(value)
    return out, conflicts


def coverage_audit(events: Iterable[dict], mappings: Iterable[dict], price_windows: dict | None = None, *, filings: Iterable[dict] | None = None, min_rate: float = .80, family_floor: float = .60) -> dict:
    """Report mapping/window coverage and promotion readiness without outcomes."""
    events = list(events); mappings = list(mappings); price_windows = price_windows or {}
    def mapped(e):
        t = _time(e["public_time"]); cik = str(e["cik"]).zfill(10)
        return any(m.get("cik") == cik and m.get("status", "") != "CONFLICT" and _time(m["valid_from"]) <= t and (not m.get("valid_to") or t < _time(m["valid_to"])) for m in mappings)
    map_flags = [mapped(e) for e in events]
    win_flags = [bool(price_windows.get(e.get("atlas_event_id"), price_windows.get(e.get("accession_number")))) for e in events]
    by_family = {}
    by_year = {}
    by_issuer = {}
    for e, m, w in zip(events, map_flags, win_flags):
        f = e.get("event_family", "unknown"); x = by_family.setdefault(f, {"events": 0, "mapped": 0, "price_windows": 0}); x["events"] += 1; x["mapped"] += int(m); x["price_windows"] += int(w)
        year = str(_time(e["public_time"]).year); y = by_year.setdefault(year, {"events": 0, "mapped": 0, "price_windows": 0}); y["events"] += 1; y["mapped"] += int(m); y["price_windows"] += int(w)
        issuer = str(e.get("cik", "")); q = by_issuer.setdefault(issuer, {"events": 0, "mapped": 0, "price_windows": 0}); q["events"] += 1; q["mapped"] += int(m); q["price_windows"] += int(w)
    # The event corpus has fewer accessions than tags. Report both denominators
    # so multi-label filings cannot inflate apparent linkage or power.
    event_accessions = {str(e.get("accession_number") or e.get("atlas_event_id")) for e in events}
    mapped_accessions = {str(e.get("accession_number") or e.get("atlas_event_id")) for e, flag in zip(events, map_flags) if flag}
    windowed_accessions = {str(e.get("accession_number") or e.get("atlas_event_id")) for e, flag in zip(events, win_flags) if flag}
    event_issuers = {str(e.get("cik")) for e in events if e.get("cik")}
    mapped_issuers = {str(e.get("cik")) for e, flag in zip(events, map_flags) if flag and e.get("cik")}
    filing_rows = list(filings or [])
    rates = [x["price_windows"] / x["events"] for x in by_family.values() if x["events"]]
    family_mapping_rates = [x["mapped"] / x["events"] for x in by_family.values() if x["events"]]
    delisting_records = sum(1 for m in mappings if m.get("status") == "DELISTED")
    report = {"events": len(events), "mapped_events": sum(map_flags), "price_window_events": sum(win_flags),
              "mapping_rate": sum(map_flags) / len(events) if events else 0.0,
              "price_window_rate": sum(win_flags) / len(events) if events else 0.0,
              "denominators": {"selected_filings": len(filing_rows) if filing_rows else None,
                               "event_tags": len(events), "event_accessions": len(event_accessions),
                               "event_issuers": len(event_issuers), "mapped_event_accessions": len(mapped_accessions),
                               "windowed_event_accessions": len(windowed_accessions), "mapped_event_issuers": len(mapped_issuers)},
              "accession_mapping_rate": len(mapped_accessions) / len(event_accessions) if event_accessions else 0.0,
              "accession_price_window_rate": len(windowed_accessions) / len(event_accessions) if event_accessions else 0.0,
              "filing_mapping_rate": (sum(1 for f in filing_rows if str(f.get("accession_number")) in mapped_accessions) / len(filing_rows)) if filing_rows else None,
              "delisting_coverage": delisting_records > 0, "delisting_records": delisting_records, "by_family": by_family, "by_year": by_year, "by_issuer": by_issuer,
              "missingness_comparison": {"mapped": sum(map_flags), "unmapped": len(events) - sum(map_flags), "windowed": sum(win_flags), "window_missing": len(events) - sum(win_flags)},
              "promotion_gate": bool(events and len(mapped_accessions) / len(event_accessions) >= min_rate and len(windowed_accessions) / len(event_accessions) >= min_rate and rates and min(rates) >= family_floor and family_mapping_rates and min(family_mapping_rates) >= family_floor and delisting_records > 0)}
    return report


def write_report(path: str | Path, report: dict) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def build_phase_a_pilot(root: str | Path | None = None) -> tuple[list[dict], list[dict]]:
    """Build the pilot mapping directly from cached SEC filing documents."""
    base = Path(root) if root else DATA_DIR / "atlas" / "phaseA_pilot"
    filings_path, raw_dir = base / "filings.jsonl", base / "raw"
    if not filings_path.exists():
        raise FileNotFoundError(f"missing frozen census: {filings_path}")
    rows = []
    for line in filings_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        source = raw_dir / f"{row['accession_number']}.txt"
        if source.exists():
            row["document"] = source.read_text(encoding="utf-8", errors="replace")
            rows.append(row)
    return build_intervals(rows)


def select_trade_class(mappings: Iterable[dict], dollar_adv: dict | None = None) -> dict[str, dict]:
    """Freeze one primary common class per issuer; exposure is issuer-capped.

    Among eligible COMMON intervals, select highest trailing dollar ADV when
    supplied. Without point-in-time ADV, issuers with multiple eligible common
    classes are rejected rather than selected by an arbitrary identifier.
    """
    dollar_adv = dollar_adv or {}
    chosen = {}
    candidates = {}
    for row in mappings:
        if str(row.get("instrument_class", "COMMON")).upper() != "COMMON" or row.get("status") == "CONFLICT":
            continue
        cik = str(row.get("cik", ""))
        candidates.setdefault(cik, []).append(row)
    for cik, issuer_rows in candidates.items():
        # Without point-in-time ADV, multiple eligible classes are
        # economically ambiguous; fail closed instead of picking by ID.
        if len(issuer_rows) > 1 and not any(float(dollar_adv.get(r.get("security_id"), 0.0)) > 0 for r in issuer_rows):
            continue
        for row in issuer_rows:
            cik = str(row.get("cik", ""))
            score = (float(dollar_adv.get(row.get("security_id"), 0.0)), row.get("security_id", ""))
            if cik not in chosen or score > chosen[cik]["_score"]:
                chosen[cik] = {"security_id": row.get("security_id"), "ticker": row.get("ticker"), "cik": cik,
                               "issuer_exposure_cap": True, "selection_rule": SHARE_CLASS_TRADING_RULE, "_score": score}
    return {cik: {k: v for k, v in row.items() if k != "_score"} for cik, row in chosen.items()}


def linkage_failure_decomposition(events: Iterable[dict], mappings: Iterable[dict], raw_root: str | Path | None = None) -> dict:
    """Explain unmapped event tags by accession, class ambiguity and interval."""
    events, mappings = list(events), list(mappings)
    raw_root = Path(raw_root) if raw_root else None
    by_cik = {}
    for m in mappings:
        by_cik.setdefault(str(m.get("cik", "")), []).append(m)
    rows = []
    for e in events:
        cik, when = str(e.get("cik", "")), _time(e["public_time"])
        valid = any(m.get("status") != "CONFLICT" and _time(m["valid_from"]) <= when and (not m.get("valid_to") or when < _time(m["valid_to"])) for m in by_cik.get(cik, []))
        if valid:
            continue
        category = "NO_PRIMARY_SYMBOL_EVIDENCE"
        if by_cik.get(cik):
            category = ("CONFLICTING_MAPPING" if all(m.get("status") == "CONFLICT" for m in by_cik[cik])
                        else "MISSING_EFFECTIVE_INTERVAL")
        elif raw_root and e.get("accession_number"):
            source = raw_root / f"{e['accession_number']}.txt"
            if not source.exists():
                category = "SOURCE_DOCUMENT_MISSING"
            else:
                mapping = extract_cover_mapping(source.read_text(encoding="utf-8", errors="replace"))
                if mapping and len(mapping.get("listed_tickers", [])) > 1:
                    category = "MULTI_CLASS_AMBIGUITY"
                elif re.search(r"ADR|PREFERRED|UNIT|WARRANT", f"{e.get('issuer_name','')} {e.get('ticker','')}".upper()):
                    category = "NON_COMMON_STOCK_INSTRUMENT"
        rows.append({"category": category, "atlas_event_id": e.get("atlas_event_id"), "accession_number": e.get("accession_number"), "cik": cik, "event_family": e.get("event_family") or EVENT_TYPES.get(e.get("event_type"), "unknown"), "year": str(when.year)})
    def counts(key):
        out = {}
        for row in rows:
            out.setdefault(row[key], {"tags": 0, "accessions": set()}); out[row[key]]["tags"] += 1; out[row[key]]["accessions"].add(row["accession_number"])
        return {k: {"tags": v["tags"], "accessions": len(v["accessions"])} for k, v in out.items()}
    return {"unmapped_tags": len(rows), "unmapped_accessions": len({r["accession_number"] for r in rows}), "by_category": counts("category"), "by_family": counts("event_family"), "by_year": counts("year"), "rows": rows}
