"""Deterministic SEC 8-K Phase-A census and source cache.

Selection is return-blind: choose a fixed CIK sample, enumerate qualifying
8-K/8-K/A submissions in 2022--2024, then select filings by a hash of accession
and census version.  Text is fetched only after the denominator is frozen.
The module records current SEC ticker links as *non-PIT* until a historical
security master proves otherwise; those links are never silently called
point-in-time mappings.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

from .. import net
from ..config import DATA_DIR, ROOT as REPO_ROOT
from ..events.atlas import EVENT_TYPES, EVENT_FAMILIES, MANIFEST_VERSION, TAXONOMY_VERSION
from ..ingest.earnings import _get_json, _get_text, _sec_documents, _plain_text
from ..ingest.sec_entities import fetch as fetch_entities


CENSUS_VERSION = "sec-atlas-phaseA-pilot-v1"
START, END = "2022-01-01", "2024-12-31"
ROOT = DATA_DIR / "atlas" / "phaseA_pilot"
RAW = ROOT / "raw"
MANIFEST = ROOT / "events.jsonl"
FILINGS = ROOT / "filings.jsonl"
REJECTIONS = ROOT / "rejection_ledger.jsonl"
GOLDSET = ROOT / "goldset_skeleton.jsonl"
PUBLISHED_EVENTS = REPO_ROOT / "goldset" / "sec_event_atlas_phaseA_pilot.jsonl"
PUBLISHED_GOLDSET = REPO_ROOT / "goldset" / "sec_event_atlas_goldset_skeleton.jsonl"
PUBLISHED_UNSIGNED = REPO_ROOT / "goldset" / "sec_event_atlas_unsigned_summary.json"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data"


def _key(value: str) -> str:
    return hashlib.sha256(f"{CENSUS_VERSION}|{value}".encode()).hexdigest()


def select_companies(target: int = 500) -> list[dict]:
    """Deterministic broad current SEC catalog sample; no prices or outcomes."""
    rows = []
    for row in fetch_entities():
        title = str(row.get("title", ""))
        if not row.get("cik") or not row.get("ticker") or any(x in title.upper() for x in ("ETF", "FUND", "TRUST")):
            continue
        rows.append({**row, "selection_key": _key(str(row["cik"]))})
    # Multiple share-class tickers can point to one CIK. Keep one deterministic
    # representative per issuer; the CIK remains the permanent identity anchor.
    deduped = {}
    for row in rows:
        current = deduped.get(row["cik"])
        if current is None or row["ticker"] < current["ticker"]:
            deduped[row["cik"]] = row
    rows = list(deduped.values())
    rows.sort(key=lambda r: (r["selection_key"], r["cik"]))
    return rows[:target]


def _submission_rows(company: dict) -> list[dict]:
    cik = str(company["cik"]).zfill(10)
    payload = _get_json(SUBMISSIONS.format(cik=cik))
    all_rows = []
    recent = payload.get("filings", {}).get("recent", {})
    all_rows.extend(_rows_from_block(company, recent))
    # SEC moves older submissions to separate JSON files. Fetch only files whose
    # declared date range overlaps the frozen period.
    for older in payload.get("filings", {}).get("files", []):
        if str(older.get("filingTo", "0000")) < START or str(older.get("filingFrom", "9999")) > END:
            continue
        url = "https://data.sec.gov/submissions/" + older["name"]
        all_rows.extend(_rows_from_block(company, _get_json(url)))
    return all_rows


def _rows_from_block(company: dict, block: dict) -> list[dict]:
    rows = []
    if not block:
        return rows
    n = len(block.get("accessionNumber", []))
    for i in range(n):
        if block.get("form", [""])[i] not in {"8-K", "8-K/A"}:
            continue
        filing_date = str(block.get("filingDate", [""])[i])
        if not START <= filing_date <= END:
            continue
        acceptance = str(block.get("acceptanceDateTime", [""])[i] or "")
        if not acceptance or "T" not in acceptance:
            continue
        rows.append({"cik": company["cik"], "issuer_name": company["title"],
                     "ticker": company["ticker"], "accession_number": block["accessionNumber"][i],
                     "form": block["form"][i], "filing_date": filing_date,
                     "acceptance_time": acceptance, "items": block.get("items", [""])[i] or "",
                     "primary_document": block.get("primaryDocument", [""])[i] or "",
                     "source": "SEC submissions API"})
    return rows


def discover(target_companies: int = 500, target_filings: int = 500,
             workers: int = 6, verbose: bool = True) -> dict:
    """Freeze company and filing denominator; this step never fetches outcomes."""
    ROOT.mkdir(parents=True, exist_ok=True)
    companies = select_companies(target_companies)
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(_submission_rows, company): company for company in companies}
        for future in as_completed(jobs):
            company = jobs[future]
            try:
                rows.extend(future.result())
            except Exception as exc:  # retain rejection denominator
                errors.append({"cik": company["cik"], "ticker": company["ticker"], "reason": f"SUBMISSIONS_ERROR:{exc}"})
    rows.sort(key=lambda r: (_key(r["accession_number"]), r["accession_number"]))
    selected = rows[:target_filings]
    # Freeze the denominator before fetching source documents.
    (ROOT / "selection_manifest.json").write_text(json.dumps({
        "census_version": CENSUS_VERSION, "period": [START, END],
        "company_selection": [{k: c[k] for k in ("cik", "ticker", "title", "selection_key")} for c in companies],
        "candidate_filings": len(rows), "selected_filings": len(selected),
        "selection_is_return_blind": True, "selection_hash": hashlib.sha256(
            json.dumps(selected, sort_keys=True).encode()).hexdigest()}, indent=2))
    with FILINGS.open("w") as handle:
        for row in selected:
            handle.write(json.dumps(row) + "\n")
    with REJECTIONS.open("w") as handle:
        for row in errors:
            handle.write(json.dumps(row) + "\n")
    result = {"status": "DENOMINATOR_FROZEN", "companies": len(companies),
              "candidate_filings": len(rows), "selected_filings": len(selected),
              "submission_errors": len(errors), "period": [START, END],
              "selection_manifest": str(ROOT / "selection_manifest.json")}
    if verbose:
        print(f"SEC census denominator: {len(selected)} filings from {len(companies)} CIKs; source fetch not started")
    return result


_RULES = [
    ("cybersecurity", "cyber_incident", r"cyber|ransomware|data breach|security incident"),
    ("restatement_controls", "restatement", r"restat(?:ement|ed)|material weakness|internal control"),
    ("going_concern", "going_concern_warning", r"going concern|liquidity warning|default notice"),
    ("auditor", "auditor_change", r"independent registered public accounting|auditor.*resign|auditor.*change"),
    ("leadership", "ceo_departure", r"chief executive officer|chief financial officer|ceo|cfo|executive officer"),
    ("restructuring", "restructuring", r"restructur|layoff|workforce reduction|facility closure"),
    ("financing_dilution", "secondary_offering", r"offering|registered direct|convertible|debt refinancing|covenant"),
    ("capital_return", "buyback_authorization", r"repurchase|buyback|share repurchase|dividend"),
    ("merger_acquisition", "merger_announced", r"merger|acquisition|business combination|terminate.*agreement"),
    ("government", "government_contract", r"department of defense|government contract|federal contract"),
    ("commercial", "major_customer_win", r"customer|contract award|strategic partnership|agreement"),
    ("litigation_regulatory", "regulatory_decision", r"regulatory|litigation|investigation|settlement|license"),
    ("activist_ownership", "activist_13d", r"schedule 13d|activist|beneficial ownership"),
    ("insider", "executive_transaction", r"insider|officer transaction|director transaction"),
    ("guidance", "guidance_raised", r"guidance|outlook|expects|forecast|preliminary results"),
]


def _classify(text: str, items: str) -> list[dict]:
    plain = _plain_text(text).lower()
    # Item code gives a deterministic first pass; evidence phrase is retained.
    out = []
    for family, event_type, pattern in _RULES:
        match = re.search(pattern, plain)
        if match:
            out.append({"event_family": family, "event_type": event_type,
                        "evidence": plain[max(0, match.start() - 120):match.end() + 240],
                        "classifier": "rules-phaseA-v1"})
    # Multiple event types in one filing are retained; duplicate family matches
    # collapse only exact event types.
    unique = {}
    for row in out:
        unique[row["event_type"]] = row
    return list(unique.values())


def _issuer_split(cik: str) -> str:
    """Issuer-disjoint development/sealed split, frozen by census hash."""
    # Do not split records: all filings for an issuer stay in one partition.
    return "sealed" if int(hashlib.sha256(f"{CENSUS_VERSION}|issuer|{cik}".encode()).hexdigest()[:8], 16) % 5 == 0 else "development"


def _write_goldset_skeleton(events: list[dict], controls: list[dict],
                            filings: list[dict] | None = None, min_controls: int = 20) -> int:
    """Freeze a stratified human-label queue without labels.

    Controls beyond the automatically unclassified filings are explicitly marked
    ``control_candidate``.  They are not treated as controls until a human
    adjudicator confirms ``routine/no-material``; this avoids manufacturing a
    negative label merely to hit a quota.
    """
    selected = []
    by_family: dict[str, list[dict]] = {}
    for event in events:
        # One event record includes its evidence anchor; family is recovered from
        # the taxonomy map rather than trusting classifier output.
        family = next((f for f, types in EVENT_FAMILIES.items() if event["event_type"] in types), "unknown")
        by_family.setdefault(family, []).append(event)
    # Four examples per event family, with explicit multi-event and exhibit
    # strata where the pilot contains them.
    for family, group in sorted(by_family.items()):
        ordered = sorted(group, key=lambda r: _key(r["atlas_event_id"]))
        for event in ordered[:4]:
            accession_count = sum(1 for x in events if x["accession_number"] == event["accession_number"])
            src = {**event, "document_surface": event.get("evidence_location", "exhibit" if event.get("exhibits") else "main_text"),
                   "multi_event_filing": accession_count > 1}
            selected.append({"record_type": "event", "label_status": "UNLABELED",
                             "stratum_family": family, "issuer_split": _issuer_split(str(event["cik"])),
                             "source": src})
    control_rows = sorted(controls, key=lambda r: _key(r["accession_number"]))
    for control in control_rows:
        selected.append({"record_type": "routine_control", "label_status": "UNLABELED",
                         "control_status": "observed_unclassified",
                         "stratum_family": "routine_control", "issuer_split": _issuer_split(str(control["cik"])),
                         "source": control})
    # The classifier may find fewer than 15 truly unclassified filings. Fill a
    # 20-row review stratum from the frozen filing denominator, but call these
    # candidates (never controls) until human review confirms them.
    if filings and len(control_rows) < min_controls:
        represented = {x["accession_number"] for x in events} | {x["accession_number"] for x in controls}
        candidates = [x for x in filings if x["accession_number"] not in represented]
        # If the heuristic tagged nearly every filing, sample deterministic
        # single-event accessions as difficult negative candidates for review.
        if len(candidates) < min_controls - len(control_rows):
            counts = {}
            for x in events:
                counts[x["accession_number"]] = counts.get(x["accession_number"], 0) + 1
            candidates = [x for x in filings if counts.get(x["accession_number"], 0) <= 1 and x["accession_number"] not in {c["accession_number"] for c in controls}]
        need = min_controls - len(control_rows)
        for row in sorted(candidates, key=lambda r: _key(r["accession_number"]))[:need]:
            selected.append({"record_type": "routine_control_candidate", "label_status": "UNLABELED",
                             "control_status": "candidate_needs_human_confirmation",
                             "stratum_family": "routine_control", "issuer_split": _issuer_split(str(row["cik"])),
                             "source": row})
    with GOLDSET.open("w") as handle:
        for row in selected:
            handle.write(json.dumps({**row, "human_label": None,
                                     "evidence_span": None, "magnitude_label": None,
                                     "document_detection": None, "event_type_label": None,
                                     "evidence_grounding": None}) + "\n")
    return len(selected)


def _extract_one(filing: dict) -> tuple[list[dict], dict | None, dict | None]:
    accession = filing["accession_number"]
    acc_compact = accession.replace("-", "")
    url = f"{ARCHIVES}/{int(filing['cik'])}/{acc_compact}/{accession}.txt"
    raw_path = RAW / f"{accession}.txt"
    try:
        if raw_path.exists():
            raw = raw_path.read_text(encoding="utf-8", errors="replace")
        else:
            raw = _get_text(url)
            raw_path.write_text(raw, encoding="utf-8")
        digest = hashlib.sha256(raw.encode()).hexdigest()
        documents = _sec_documents(raw)
        primary = next((d for d in documents if d["filename"].lower() == filing["primary_document"].lower()), None)
        exhibits = [{"type": d["type"], "filename": d["filename"],
                     "sha256": hashlib.sha256(d["raw_text"].encode()).hexdigest()}
                    for d in documents if d["type"].startswith("EX-")]
        classified = _classify(raw, filing["items"])
        if not classified:
            return [], {**filing, "classification": "NO_MATERIAL_EVENT",
                        "source_sha256": digest, "raw_path": str(raw_path.relative_to(ROOT)),
                        "primary_sha256": hashlib.sha256(primary["raw_text"].encode()).hexdigest() if primary else None,
                        "exhibits": exhibits}, None
        events = []
        for ordinal, event in enumerate(classified):
            evidence_location = "primary_document" if primary and event["evidence"].split()[:3] and all(
                token in _plain_text(primary["raw_text"]).lower() for token in event["evidence"].lower().split()[:3]
            ) else ("exhibit" if exhibits else "filing_text")
            events.append({
                "atlas_event_id": hashlib.sha256(f"{accession}|{event['event_type']}|{ordinal}".encode()).hexdigest()[:24],
                "taxonomy_version": TAXONOMY_VERSION, "manifest_version": MANIFEST_VERSION,
                "cik": filing["cik"], "issuer_name": filing["issuer_name"], "ticker": filing["ticker"],
                "accession_number": accession, "form": filing["form"], "item_codes": filing["items"],
                "event_type": event["event_type"], "public_time": filing["acceptance_time"],
                "known_at": datetime.now(timezone.utc).isoformat(), "source_url": url,
                "source_sha256": digest, "raw_path": str(raw_path.relative_to(ROOT)),
                "extraction_version": event["classifier"],
                "ticker_mapping_status": "CURRENT_SEC_CATALOG_NOT_PIT",
                "ticker_mapping_source": "sec_company_tickers_at_pull",
                "evidence": event["evidence"], "primary_sha256": hashlib.sha256(primary["raw_text"].encode()).hexdigest() if primary else None,
                "exhibits": exhibits, "evidence_location": evidence_location,
            })
        return events, None, None
    except Exception as exc:  # preserve source-specific rejection reason
        return [], None, {**filing, "reason": f"FETCH_OR_PARSE_ERROR:{exc}"}


def fetch_and_extract(verbose: bool = True, workers: int = 6) -> dict:
    """Fetch frozen filings, cache raw SEC sources, and emit an event manifest."""
    if not FILINGS.exists():
        raise RuntimeError("run discover first; denominator must be frozen before source fetch")
    RAW.mkdir(parents=True, exist_ok=True)
    events, rejection, controls = [], [], []
    filings = [json.loads(line) for line in FILINGS.read_text().splitlines() if line.strip()]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_extract_one, filings))
    for ev, control, rejected in results:
        events.extend(ev)
        if control:
            controls.append(control)
        if rejected:
            rejection.append(rejected)
    with MANIFEST.open("w") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    with (ROOT / "controls.jsonl").open("w") as handle:
        for row in controls:
            handle.write(json.dumps(row) + "\n")
    with (ROOT / "extraction_rejections.jsonl").open("w") as handle:
        for row in rejection:
            handle.write(json.dumps(row) + "\n")
    goldset = _write_goldset_skeleton(events, controls, filings)
    result = {"status": "EXTRACTED", "filings": sum(1 for _ in FILINGS.open()),
              "events": len(events), "no_material_controls": len(controls),
              "goldset_skeleton": goldset, "rejected": len(rejection), "manifest": str(MANIFEST),
              "ticker_mapping_status": "CURRENT_SEC_CATALOG_NOT_PIT"}
    if verbose:
        print(f"SEC census extraction: {len(events)} events, {len(controls)} controls, {len(rejection)} rejected")
    return result


def export_pilot() -> dict:
    """Publish slim real event/accession records and the unlabeled gold queue."""
    if not MANIFEST.exists():
        raise RuntimeError("run extract first")
    PUBLISHED_EVENTS.parent.mkdir(parents=True, exist_ok=True)
    keep = ("atlas_event_id", "cik", "issuer_name", "ticker", "accession_number", "form",
            "item_codes", "event_type", "public_time", "source_url", "source_sha256",
            "raw_path", "primary_sha256", "exhibits", "ticker_mapping_status")
    with PUBLISHED_EVENTS.open("w") as out:
        for line in MANIFEST.read_text().splitlines():
            row = json.loads(line)
            out.write(json.dumps({k: row.get(k) for k in keep}, sort_keys=True) + "\n")
    if GOLDSET.exists():
        PUBLISHED_GOLDSET.write_text(GOLDSET.read_text())
    unsigned = DATA_DIR / "sec_event_atlas_unsigned.json"
    if unsigned.exists():
        PUBLISHED_UNSIGNED.write_text(unsigned.read_text())
    return {"status": "PUBLISHED", "events": sum(1 for _ in PUBLISHED_EVENTS.open()),
            "event_catalog": str(PUBLISHED_EVENTS), "goldset": str(PUBLISHED_GOLDSET),
            "unsigned_summary": str(PUBLISHED_UNSIGNED)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["discover", "extract", "export"])
    parser.add_argument("--companies", type=int, default=500)
    parser.add_argument("--filings", type=int, default=500)
    args = parser.parse_args()
    if args.command == "discover":
        result = discover(args.companies, args.filings)
    elif args.command == "extract":
        result = fetch_and_extract()
    else:
        result = export_pilot()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
