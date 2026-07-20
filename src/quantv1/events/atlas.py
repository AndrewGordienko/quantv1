"""SEC Event Atlas: source-anchored taxonomy and unsigned validation gate.

The Atlas is deliberately an extraction substrate, not a directional strategy.
Every event is tied to a permanent CIK/accession and exact public timestamp.
Stage 1 measures whether a structured event family is associated with unusual
absolute residual movement or volatility; it never assigns a trade direction.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_DIR
from ..db import connect


TAXONOMY_VERSION = "sec-event-atlas-v1"
MANIFEST_VERSION = "sec-event-atlas-manifest-v1"
DISCOVERY_START, DISCOVERY_END = "2022-01-01", "2024-12-31"
VALIDATION_START, VALIDATION_END = "2025-01-01", "2025-12-31"
PROSPECTIVE_START = "2026-01-01"

# The family/type vocabulary is intentionally explicit and small enough to audit.
# A future imported grounded 119-type taxonomy gets a new version and a mapping,
# never a silent replacement of this manifest.
EVENT_FAMILIES: dict[str, tuple[str, ...]] = {
    "guidance": ("guidance_raised", "guidance_lowered", "guidance_withdrawn", "guidance_maintained"),
    "leadership": ("ceo_departure", "cfo_departure", "executive_appointment", "board_change"),
    "auditor": ("auditor_change", "auditor_resignation", "going_concern_auditor"),
    "restructuring": ("restructuring", "layoffs", "facility_closure", "bankruptcy_risk"),
    "capital_return": ("buyback_authorization", "buyback_change", "dividend_change"),
    "financing_dilution": ("secondary_offering", "convertible_offering", "debt_refinancing", "covenant_change"),
    "merger_acquisition": ("merger_announced", "acquisition_announced", "deal_terminated", "deal_completed"),
    "commercial": ("major_customer_win", "major_customer_loss", "strategic_partnership", "contract_termination"),
    "government": ("government_contract", "government_contract_termination"),
    "litigation_regulatory": ("material_litigation", "regulatory_decision", "regulatory_investigation", "license_change"),
    "activist_ownership": ("activist_13d", "ownership_increase", "ownership_decrease"),
    "insider": ("insider_buy_cluster", "insider_sell_cluster", "executive_transaction"),
    "going_concern": ("going_concern_warning", "liquidity_warning", "default_notice"),
    "cybersecurity": ("cyber_incident", "data_breach", "systems_outage"),
    "restatement_controls": ("restatement", "internal_control_failure", "material_weakness"),
}
EVENT_TYPES = {event_type: family for family, values in EVENT_FAMILIES.items() for event_type in values}
FORMS = {"8-K", "8-K/A", "10-K", "10-Q", "SC 13D", "SC 13G", "3", "4", "5"}


def taxonomy_hash() -> str:
    return hashlib.sha256(json.dumps(EVENT_FAMILIES, sort_keys=True).encode()).hexdigest()


def ensure_table(con=None):
    own = con is None
    con = con or connect()
    con.execute("""
        CREATE TABLE IF NOT EXISTS atlas_events (
            atlas_event_id VARCHAR PRIMARY KEY,
            taxonomy_version VARCHAR,
            manifest_version VARCHAR,
            cik VARCHAR,
            issuer_name VARCHAR,
            ticker VARCHAR,
            accession_number VARCHAR,
            form VARCHAR,
            item_codes VARCHAR,
            event_family VARCHAR,
            event_type VARCHAR,
            public_time TIMESTAMP,
            known_at TIMESTAMP,
            source_url VARCHAR,
            source_sha256 VARCHAR,
            raw_path VARCHAR,
            extraction_version VARCHAR,
            split VARCHAR,
            status VARCHAR,
            metadata JSON
        )
    """)
    if own:
        con.close()


def _aware(value: str, field: str) -> datetime:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be ISO-8601 with timezone") from exc
    if dt.tzinfo is None:
        raise ValueError(f"{field} must include an explicit timezone")
    return dt.astimezone(timezone.utc)


def _split(public_time: datetime) -> str:
    d = public_time.date().isoformat()
    if DISCOVERY_START <= d <= DISCOVERY_END:
        return "discovery"
    if VALIDATION_START <= d <= VALIDATION_END:
        return "validation"
    if d >= PROSPECTIVE_START:
        return "prospective"
    return "out_of_sample"


def validate_record(record: dict, *, root: Path | None = None) -> dict:
    """Validate one source-anchored unsigned event; reject all future leakage fields."""
    required = ("atlas_event_id", "cik", "issuer_name", "ticker", "accession_number",
                "form", "event_type", "public_time", "known_at", "source_url",
                "source_sha256", "raw_path")
    missing = [key for key in required if not record.get(key)]
    if missing:
        raise ValueError(f"MISSING_FIELDS:{','.join(missing)}")
    if record.get("taxonomy_version") != TAXONOMY_VERSION or record.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("BAD_VERSION")
    form = str(record["form"]).upper()
    if form not in FORMS:
        raise ValueError("BAD_FORM")
    event_type = str(record["event_type"])
    if event_type not in EVENT_TYPES:
        raise ValueError("UNKNOWN_EVENT_TYPE")
    public_time = _aware(record["public_time"], "public_time")
    known_at = _aware(record["known_at"], "known_at")
    if known_at < public_time:
        raise ValueError("KNOWN_AT_BEFORE_PUBLIC_TIME")
    digest = str(record["source_sha256"]).lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("BAD_SOURCE_SHA256")
    if root is not None:
        raw = (root / str(record["raw_path"])).resolve()
        if not raw.is_file():
            raise ValueError("RAW_SOURCE_MISSING")
        if hashlib.sha256(raw.read_bytes()).hexdigest() != digest:
            raise ValueError("RAW_SOURCE_HASH_MISMATCH")
    # Unsigned Stage 1: direction, magnitude, realized outcome and labels are
    # intentionally not accepted in the canonical event record.
    forbidden = {"direction", "return", "residual_return", "label", "target", "trade_side"}
    if forbidden.intersection(record):
        raise ValueError("DIRECTIONAL_FIELD_IN_UNSIGNED_MANIFEST")
    return {
        "atlas_event_id": str(record["atlas_event_id"]),
        "taxonomy_version": TAXONOMY_VERSION, "manifest_version": MANIFEST_VERSION,
        "cik": str(record["cik"]).zfill(10), "issuer_name": str(record["issuer_name"]),
        "ticker": str(record["ticker"]).upper(), "accession_number": str(record["accession_number"]),
        "form": form, "item_codes": str(record.get("item_codes") or ""),
        "event_family": EVENT_TYPES[event_type], "event_type": event_type,
        "public_time": public_time, "known_at": known_at, "source_url": str(record["source_url"]),
        "source_sha256": digest, "raw_path": str(record["raw_path"]),
        "extraction_version": str(record.get("extraction_version") or "unclassified"),
        "split": _split(public_time), "status": "VERIFIED",
        "metadata": json.dumps({"novelty": record.get("novelty"), "contradiction": record.get("contradiction"),
                                "magnitude_proxy": record.get("magnitude_proxy")}, sort_keys=True),
    }


def ingest_manifest(path: str | Path, verbose: bool = True) -> dict:
    """Validate all JSONL records before inserting any; fail closed on one error."""
    manifest_path = Path(path)
    records, errors, seen = [], [], set()
    for line_no, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            raw = json.loads(line)
            if raw.get("atlas_event_id") in seen:
                raise ValueError("DUPLICATE_EVENT_ID")
            seen.add(raw.get("atlas_event_id"))
            records.append(validate_record(raw, root=manifest_path.parent))
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append({"line": line_no, "error": str(exc)})
    if errors:
        return {"status": "REJECTED", "records": 0, "errors": errors,
                "taxonomy_version": TAXONOMY_VERSION, "taxonomy_hash": taxonomy_hash()}
    con = connect()
    ensure_table(con)
    try:
        for r in records:
            con.execute("INSERT INTO atlas_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
                r["atlas_event_id"], r["taxonomy_version"], r["manifest_version"], r["cik"], r["issuer_name"],
                r["ticker"], r["accession_number"], r["form"], r["item_codes"], r["event_family"], r["event_type"],
                r["public_time"], r["known_at"], r["source_url"], r["source_sha256"], r["raw_path"],
                r["extraction_version"], r["split"], r["status"], r["metadata"]])
    except Exception:
        con.rollback()
        con.close()
        raise
    con.close()
    result = {"status": "INGESTED", "records": len(records), "errors": [],
              "by_split": pd.Series([r["split"] for r in records]).value_counts().to_dict() if records else {},
              "by_family": pd.Series([r["event_family"] for r in records]).value_counts().to_dict() if records else {},
              "taxonomy_version": TAXONOMY_VERSION, "taxonomy_hash": taxonomy_hash()}
    if verbose:
        print(f"SEC Event Atlas: {len(records)} verified events -> atlas_events")
    return result


def _unsigned_outcome(price: dict[str, pd.DataFrame], ticker: str, event_date: pd.Timestamp,
                      horizon: int, baseline_date: pd.Timestamp | None = None) -> dict | None:
    stock = price.get(ticker)
    market = price.get("SPY")
    if stock is None or market is None:
        return None
    dates = stock.index[stock.index > event_date]
    if len(dates) < horizon:
        return None
    entry_date, exit_date = dates[0], dates[horizon - 1]
    if entry_date not in market.index or exit_date not in market.index:
        return None
    raw = float(stock.loc[exit_date, "close"] / stock.loc[entry_date, "open"] - 1)
    mkt = float(market.loc[exit_date, "close"] / market.loc[entry_date, "open"] - 1)
    return {"abs_residual": abs(raw - mkt), "realized_abs": abs(raw),
            "residual": raw - mkt, "entry_date": entry_date, "exit_date": exit_date}


def unsigned_validation(verbose: bool = True) -> dict:
    """Stage 1: family support + unsigned absolute-move/volatility diagnostics."""
    con = connect(read_only=True)
    try:
        events = con.execute("""SELECT atlas_event_id, cik, ticker, accession_number, event_family, public_time, split
                                FROM atlas_events WHERE status='VERIFIED'""").fetchall()
    except Exception:
        con.close()
        report = {"status": "BLOCKED_NO_ATLAS_MANIFEST", "taxonomy_version": TAXONOMY_VERSION,
                  "taxonomy_hash": taxonomy_hash(), "n_verified_events": 0,
                  "directional_model": "NOT_RUN"}
        (DATA_DIR / "sec_event_atlas_unsigned.json").write_text(json.dumps(report, indent=2))
        return report
    tickers = sorted({r[2] for r in events} | {"SPY"})
    prices = con.execute("""SELECT ticker, date, open, close, volume FROM prices
                           WHERE ticker IN (SELECT UNNEST(?)) ORDER BY date""", [tickers]).df()
    market_caps = dict(con.execute("SELECT ticker, market_cap FROM ticker_sectors").fetchall())
    con.close()
    price = {ticker: group.set_index("date").sort_index()
             for ticker, group in prices.groupby("ticker")}
    event_dates = {(ticker, pd.Timestamp(public_time).date()) for _, _, ticker, _, _, public_time, _ in events}
    rows = []
    for event_id, cik, ticker, accession, family, public_time, split in events:
        d = pd.Timestamp(public_time)
        d = (d.tz_convert(None) if d.tzinfo is not None else d).normalize()
        one = _unsigned_outcome(price, ticker, d, 1)
        five = _unsigned_outcome(price, ticker, d, 5)
        if one is not None or five is not None:
            rows.append({"atlas_event_id": event_id, "cik": cik, "accession": accession,
                         "ticker": ticker, "event_family": family,
                         "split": split, "event_date": d, "one": one, "five": five})
    frame = pd.DataFrame(rows)
    summary = []
    # Tags from one accession/date share one market outcome. Keep tag-level
    # counts for extraction coverage, but use accession/date clusters for
    # unsigned effect sizes and controls.
    cluster_frame = (frame.drop_duplicates(["accession", "ticker", "event_date", "event_family"])
                     if not frame.empty else frame)
    if not cluster_frame.empty:
        for (family, split), group in cluster_frame.groupby(["event_family", "split"]):
            one = [r.one["abs_residual"] for r in group.itertuples() if r.one]
            five = [r.five["abs_residual"] for r in group.itertuples() if r.five]
            summary.append({"event_family": family, "split": split, "n": int(len(group)),
                            "n_ticker_date_clusters": int(group[["ticker", "event_date"]].drop_duplicates().shape[0]),
                            "mean_abs_residual_1d": float(np.mean(one)) if one else None,
                            "mean_abs_residual_5d": float(np.mean(five)) if five else None,
                            "median_abs_residual_5d": float(np.median(five)) if five else None})
    # Same-company normal controls are deterministic pre-event dates. They are
    # not used to choose families; they only answer whether the event-day label
    # separates from that company's ordinary movement.
    control_rows = []
    for r in cluster_frame.to_dict("records"):
        stock = price.get(r["ticker"])
        if stock is None:
            continue
        available = [d for d in stock.index if d < r["event_date"] and
                     (r["ticker"], d.date()) not in event_dates]
        if len(available) < 61:
            continue
        normal_date = available[-21]
        random_idx = int(hashlib.sha256(f"atlas-control|{r['atlas_event_id']}".encode()).hexdigest()[:8], 16) % len(available)
        for label, control_date in (("same_company_normal", normal_date),
                                    ("shuffled_timestamp", available[random_idx])):
            outcome = _unsigned_outcome(price, r["ticker"], pd.Timestamp(control_date), 5)
            if outcome:
                control_rows.append({"control": label, "event_family": r["event_family"],
                                     "abs_residual_5d": outcome["abs_residual"]})
    controls = []
    if control_rows:
        cf = pd.DataFrame(control_rows)
        for (control, family), group in cf.groupby(["control", "event_family"]):
            controls.append({"control": control, "event_family": family, "n": int(len(group)),
                             "mean_abs_residual_5d": float(group["abs_residual_5d"].mean()),
                             "median_abs_residual_5d": float(group["abs_residual_5d"].median())})
    family_table = []
    if not frame.empty:
        event_means = cluster_frame.groupby("event_family")["five"].apply(
            lambda s: float(np.mean([x["abs_residual"] for x in s if x])) if any(s) else None)
        normal = {r["event_family"]: r["mean_abs_residual_5d"] for r in controls
                  if r["control"] == "same_company_normal"}
        for family, group in cluster_frame.groupby("event_family"):
            mean = event_means.get(family)
            family_table.append({"family": family, "events": int(len(group)),
                                 "accessions": int(group["accession"].nunique()),
                                 "companies": int(group["cik"].nunique()),
                                 "extraction_precision": None,
                                 "extraction_recall": None,
                                 "abnormal_volatility_lift_5d": (mean - normal[family]) if mean is not None and family in normal else None,
                                 "stable_by_year": None,
                                 "goldset_status": "UNLABELED"})
    coverage = {"event_tags": len(events), "priced_event_tags": len(frame),
                "priced_tag_rate": float(len(frame) / len(events)) if events else 0.0,
                "priced_cluster_rate": float(len(cluster_frame) / len(events)) if events else 0.0,
                "priced_accessions": int(cluster_frame["accession"].nunique()) if not cluster_frame.empty else 0,
                "unique_event_ciks": len({r[1] for r in events}),
                "unique_event_tickers": len({r[2] for r in events}),
                "pit_ticker_mapping": "NOT_AVAILABLE_IN_CURRENT_SEC_CATALOG",
                "delisting_coverage": "NOT_AVAILABLE",
                "price_source": "local prices table; current-link diagnostic only",
                "market_cap_proxy_linked_tickers": int(sum(1 for t in set(r[2] for r in events) if market_caps.get(t))),
                "liquidity_proxy": "not point-in-time; volume data exists only for current price-linked names",
                "event_tags_by_year": pd.Series([str(pd.Timestamp(r[5]).year) for r in events]).value_counts().to_dict(),
                "priced_event_tags_by_year": frame.assign(year=frame["event_date"].dt.year).groupby("year").size().to_dict() if not frame.empty else {}}
    report = {"status": "UNSIGNED_DIAGNOSTIC_ONLY", "taxonomy_version": TAXONOMY_VERSION,
              "taxonomy_hash": taxonomy_hash(), "n_verified_events": len(events),
              "n_with_price_outcome": len(frame), "by_family_split": summary,
              "controls": controls, "coverage": coverage,
              "phaseA_table": family_table,
              "directional_model": "NOT_RUN",
              "warning": "This validates unsigned market impact only; current ticker links are not point-in-time, no delisting master is present, and this is not directional alpha or a trade selection."}
    (DATA_DIR / "sec_event_atlas_unsigned.json").write_text(json.dumps(report, indent=2, default=str))
    if verbose:
        print(f"SEC Event Atlas unsigned diagnostic: {len(events)} events, {len(frame)} priced")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["unsigned"])
    args = parser.parse_args()
    unsigned_validation()
