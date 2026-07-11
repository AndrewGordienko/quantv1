"""MGRM real-filing gold-set corpus: deterministic selection, freeze, manifest.

Data-phase tooling only. It does NOT change the model, research harness,
promotion gates, or certification architecture -- it reads the eligible SEC
pool, applies a deterministic pre-output selection rule, and freezes a manifest
plus a human-labelling skeleton. The development and certification sets are kept
in SEPARATE files so the certification gold set (guidance_goldset.GOLD_PATH)
never contains development documents.

Integrity rules enforced here:
  * selection is deterministic and uses only filing metadata + a structural
    format hint from raw HTML -- never extractor output;
  * development and certification sets are company-disjoint;
  * the extractor prefill is written to a separate, clearly non-authoritative
    file and is produced for the DEVELOPMENT set only, so the certification set
    stays sealed and unbiased until the extractor is frozen;
  * human labels are authoritative and are filled into the skeletons by a human.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .. import net
from ..config import ROOT
from ..db import connect
from .earnings import (
    _SEC_ARCHIVES, _cached_sec_filing, _sec_documents, build_universe,
)
from .guidance import structured_extract


SELECTION_VERSION = "mgrm-corpus-select-v1"
CRAWL_VERSION = "mgrm-corpus-crawl-v1"
GOLDSET_DIR = ROOT / "goldset"
MANIFEST_PATH = GOLDSET_DIR / "mgrm_corpus_manifest.jsonl"
DEV_PREFILL_PATH = GOLDSET_DIR / "mgrm_dev_prefill.jsonl"
DEV_SKELETON_PATH = GOLDSET_DIR / "mgrm_dev_labels.skeleton.jsonl"
CERT_SKELETON_PATH = GOLDSET_DIR / "mgrm_cert_labels.skeleton.jsonl"

DEV_TARGET = 20
CERT_TARGET = 30
MIN_SECTORS = 6
MAX_DOCS_PER_COMPANY = 4
# Canonical labelling document per filing: prefer the earnings-release exhibit.
_DOC_PRIORITY = {"EX-99.1": 0, "EX-99.01": 0, "EX-99.2": 1, "EX-99.02": 1,
                 "8-K": 2, "8-K/A": 2}


def _key(value: str) -> str:
    return hashlib.sha256(f"{SELECTION_VERSION}|{value}".encode()).hexdigest()


def _rel_raw_path(accession: str, sha256: str) -> str:
    """Repository-relative, portable path (no absolute paths, no username)."""
    return f"data/raw/mgrm/{accession}/{sha256}.html"


def _primary_url(cik: str, accession: str, primary_document: str) -> str:
    directory = f"{_SEC_ARCHIVES}/{int(cik)}/{accession.replace('-', '')}"
    return f"{directory}/{primary_document}"


def crawl_targets(per_sector: int = 4,
                  universe: list[dict] | None = None) -> list[dict]:
    """Deterministic, sector-stratified subset of the frozen universe."""
    universe = universe or build_universe()
    con = connect(read_only=True)
    try:
        sectors = dict(con.execute(
            "SELECT ticker, sector FROM ticker_sectors"
        ).fetchall())
    finally:
        con.close()
    by_sector: dict[str, list[dict]] = defaultdict(list)
    for company in universe:
        sector = sectors.get(company["ticker"])
        if sector:
            by_sector[sector].append({**company, "sector": sector})
    targets = []
    for sector in sorted(by_sector):
        ranked = sorted(by_sector[sector], key=lambda c: _key(c["ticker"]))
        targets.extend(ranked[:per_sector])
    return targets


def _format_hint(raw_path: str) -> str:
    """Structural hint (table/prose/mixed) from raw HTML -- not extractor output."""
    try:
        html = Path(raw_path).read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return "unknown"
    has_table = "<table" in html
    has_prose = any(word in html for word in
                    ("guidance", "outlook", "expects", "anticipates"))
    return ("mixed" if has_table and has_prose else
            "table" if has_table else "prose" if has_prose else "prose")


def eligible_documents() -> list[dict]:
    """One canonical preserved document per eligible filing, with provenance."""
    con = connect(read_only=True)
    try:
        rows = con.execute("""
            SELECT d.document_id, d.accession_number, d.ticker, f.cik,
                   f.primary_document, d.earnings_event_id, d.document_type,
                   d.source_url, d.public_time, d.source_sha256, d.raw_path,
                   s.sector
            FROM mgrm_documents d
            JOIN mgrm_filings f USING (accession_number)
            LEFT JOIN ticker_sectors s ON s.ticker = d.ticker
            WHERE f.status='ELIGIBLE' AND d.status='PRESERVED'
        """).fetchall()
    finally:
        con.close()
    columns = ["document_id", "accession_number", "ticker", "cik",
               "primary_document", "earnings_event_id", "document_type",
               "source_url", "public_time", "source_sha256", "raw_path", "sector"]
    by_filing: dict[str, dict] = {}
    for row in rows:
        record = dict(zip(columns, row))
        accession = record["accession_number"]
        current = by_filing.get(accession)
        priority = _DOC_PRIORITY.get(record["document_type"], 9)
        if (current is None or priority < current["_priority"] or
                (priority == current["_priority"] and
                 _key(record["document_id"]) < _key(current["document_id"]))):
            by_filing[accession] = {**record, "_priority": priority}
    documents = []
    for record in by_filing.values():
        record.pop("_priority")
        record["sector"] = record["sector"] or "UNKNOWN"
        record["format_hint"] = _format_hint(record["raw_path"])
        record["selection_key"] = _key(record["accession_number"])
        record["relative_path"] = _rel_raw_path(record["accession_number"],
                                                 record["source_sha256"])
        record["primary_url"] = _primary_url(record["cik"],
                                             record["accession_number"],
                                             record["primary_document"])
        documents.append(record)
    return documents


def select_corpus(documents: list[dict] | None = None, *,
                  dev_n: int = DEV_TARGET, cert_n: int = CERT_TARGET) -> dict:
    """Deterministic, company-disjoint dev/cert split, sector-front-loaded."""
    documents = documents if documents is not None else eligible_documents()
    by_company: dict[str, list[dict]] = defaultdict(list)
    for document in documents:
        by_company[document["ticker"]].append(document)
    for ticker in by_company:
        by_company[ticker] = sorted(
            by_company[ticker], key=lambda d: d["selection_key"]
        )[:MAX_DOCS_PER_COMPANY]

    # Order companies: one lowest-key company per sector first (guarantees sector
    # spread), then the remaining companies by key. Deterministic throughout.
    company_sector = {t: docs[0]["sector"] for t, docs in by_company.items()}
    company_key = {t: min(d["selection_key"] for d in docs)
                   for t, docs in by_company.items()}
    first_in_sector, seen = [], set()
    for ticker in sorted(by_company, key=lambda t: company_key[t]):
        sector = company_sector[ticker]
        if sector not in seen:
            seen.add(sector)
            first_in_sector.append(ticker)
    ordered = first_in_sector + [t for t in sorted(by_company, key=lambda t: company_key[t])
                                 if t not in set(first_in_sector)]

    cert, dev = [], []
    cert_companies, dev_companies = set(), set()
    for ticker in ordered:
        docs = by_company[ticker]
        if len(cert) < cert_n:
            take = docs[:cert_n - len(cert)]
            cert.extend(take)
            cert_companies.add(ticker)
        elif len(dev) < dev_n:
            take = docs[:dev_n - len(dev)]
            dev.extend(take)
            dev_companies.add(ticker)
        if len(cert) >= cert_n and len(dev) >= dev_n:
            break
    return {"development": dev, "sealed_certification": cert,
            "dev_companies": sorted(dev_companies),
            "cert_companies": sorted(cert_companies),
            "requested": {"development": dev_n, "sealed_certification": cert_n}}


def _manifest_record(document: dict, split: str) -> dict:
    return {
        "document_id": document["document_id"], "ticker": document["ticker"],
        "company": document["ticker"], "sector": document["sector"],
        "accession_number": document["accession_number"],
        "earnings_event_id": document["earnings_event_id"],
        "document_url": document["source_url"],
        "primary_url": document["primary_url"],
        "document_type": document["document_type"],
        "public_time": str(document["public_time"]),
        "source_sha256": document["source_sha256"],
        "raw_path": document["relative_path"],
        "format_hint": document["format_hint"],
        "split": split, "selection_rule_version": SELECTION_VERSION,
    }


def freeze(selection: dict | None = None) -> dict:
    """Write the frozen manifest, dev prefill (suggestions only), and skeletons."""
    selection = selection if selection is not None else select_corpus()
    GOLDSET_DIR.mkdir(parents=True, exist_ok=True)
    dev, cert = selection["development"], selection["sealed_certification"]

    manifest = ([_manifest_record(d, "development") for d in dev] +
                [_manifest_record(d, "sealed_certification") for d in cert])
    _write_jsonl(MANIFEST_PATH, manifest, header=[
        "MGRM corpus selection manifest (frozen).",
        f"selection_rule_version={SELECTION_VERSION}; deterministic, pre-output.",
        "Human labels are authoritative and filled into the skeleton files.",
    ])

    # Extractor prefill: DEVELOPMENT ONLY, non-authoritative suggestions. The
    # certification set stays sealed -- no extractor is run over it here.
    prefill = []
    for document in dev:
        html = Path(document["raw_path"]).read_text(encoding="utf-8",
                                                    errors="replace")
        prefill.append({"document_id": document["document_id"],
                        "ticker": document["ticker"],
                        "suggested_records": structured_extract(html)})
    _write_jsonl(DEV_PREFILL_PATH, prefill, header=[
        "MGRM DEVELOPMENT extractor prefill -- SUGGESTIONS ONLY, NOT AUTHORITATIVE.",
        "These never populate expected labels; a human labels the source docs.",
    ])

    _write_jsonl(DEV_SKELETON_PATH, [_skeleton(d, "development") for d in dev],
                 header=_skeleton_header("development"))
    _write_jsonl(CERT_SKELETON_PATH, [_skeleton(d, "sealed_certification")
                                      for d in cert],
                 header=_skeleton_header("sealed_certification"))
    return {"manifest": len(manifest), "development": len(dev),
            "sealed_certification": len(cert),
            "dev_prefill": len(prefill),
            "distribution": distribution(selection)}


def _skeleton(document: dict, split: str) -> dict:
    return {
        "doc_id": document["document_id"], "company": document["ticker"],
        "sector": document["sector"], "format": document["format_hint"],
        "accession": document["accession_number"],
        "earnings_event_id": document["earnings_event_id"],
        "source_url": document["source_url"],
        "raw_path": document["relative_path"], "split": split,
        # Human fills these. no_guidance=null means "not yet labelled".
        "no_guidance": None, "expected": [],
    }


def _skeleton_header(split: str) -> list[str]:
    return [
        f"MGRM {split} labelling skeleton. HUMAN LABELS ARE AUTHORITATIVE.",
        "Read each source document and fill no_guidance (true/false) and, if",
        "guidance exists, one expected record per (metric, guided period) with:",
        "metric, period, units, currency, low, high, midpoint, status, action,",
        "and the exact supporting sentence or table cells (evidence).",
        "Do NOT copy the extractor prefill blindly; the prefill is only a hint.",
    ]


def distribution(selection: dict | None = None) -> dict:
    selection = selection if selection is not None else select_corpus()
    everything = selection["development"] + selection["sealed_certification"]
    return {
        "documents": len(everything),
        "by_split": {"development": len(selection["development"]),
                     "sealed_certification": len(selection["sealed_certification"])},
        "sectors": dict(Counter(d["sector"] for d in everything)),
        "formats": dict(Counter(d["format_hint"] for d in everything)),
        "distinct_companies": len({d["ticker"] for d in everything}),
        "company_disjoint": not (set(selection["dev_companies"]) &
                                 set(selection["cert_companies"])),
        "sectors_covered": len({d["sector"] for d in everything}),
        "meets_sector_minimum": len({d["sector"] for d in everything}) >= MIN_SECTORS,
        "note_actions": "raised/lowered/reaffirmed/initiated/withdrawn counts are "
                        "a labelling outcome and are unknown until humans label.",
    }


def load_manifest(path: Path | None = None) -> list[dict]:
    path = path or MANIFEST_PATH
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            records.append(json.loads(line))
    return records


def _reproduce(record: dict) -> str | None:
    """Reproduce the exact saved exhibit content from the frozen primary URL."""
    filing_text, _ = _cached_sec_filing(record["primary_url"],
                                        f"sec:{record['accession_number']}")
    for document in _sec_documents(filing_text):
        raw = document["raw_text"]
        if hashlib.sha256(raw.encode()).hexdigest() == record["source_sha256"]:
            return raw
    return None


def rehydrate(*, verbose: bool = True) -> dict:
    """Reconstruct missing corpus documents from the frozen manifest.

    For each manifest entry: if the local file exists, verify its SHA-256
    against the manifest; otherwise re-fetch the frozen primary URL, reproduce
    the exact discovery parse, and write it ONLY if the SHA-256 matches
    exactly. Never reruns selection or mutates the manifest.
    """
    records = load_manifest()
    present = downloaded = refused = 0
    problems = []
    for record in records:
        target = ROOT / record["raw_path"]
        want = record["source_sha256"]
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() == want:
                present += 1
            else:
                refused += 1
                problems.append({"document_id": record["document_id"],
                                 "reason": "LOCAL_SHA_MISMATCH"})
            continue
        content = _reproduce(record)
        if content is None:
            refused += 1
            problems.append({"document_id": record["document_id"],
                             "reason": "SHA_MISMATCH_ON_REFETCH"})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        downloaded += 1
        if verbose:
            print(f"  rehydrated {record['document_id']} ({record['ticker']})")
    return {"documents": len(records), "present": present,
            "downloaded": downloaded, "refused": refused,
            "reselected": False, "problems": problems}


def _write_jsonl(path: Path, records: list[dict], *, header: list[str]) -> None:
    lines = [f"# {line}" for line in header]
    lines.extend(json.dumps(record, default=str) for record in records)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
