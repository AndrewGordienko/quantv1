"""Point-in-time earnings-event and consensus ingestion.

SEC Item 2.02 filings provide a conservative, exact public timestamp.  They are
never mislabeled as the earliest release: a canonical event becomes
``VERIFIED_EARLIEST`` only after a reviewed company-IR or direct press-wire
record is inserted. Historical consensus arrives through a vendor manifest and
is rejected unless its archived snapshot predates the event.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
import urllib.request
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

from .. import net  # noqa: F401 - verified OS trust store
from ..config import DATA_DIR
from ..db import connect

SPRINT_VERSION = "earnings-alpha-v1"
UNIVERSE_VERSION = "earnings-alpha-v1-2021-06-30"
SAMPLE_START = date(2021, 7, 1)
SAMPLE_END = date(2026, 6, 30)
VALIDATION_START = date(2024, 7, 1)
FINAL_TEST_START = date(2025, 7, 1)
# Backwards-compatible name; final-test rows must not be used for selection.
HOLDOUT_START = FINAL_TEST_START
_SEC_SUBMISSIONS = "https://data.sec.gov/submissions"
_SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
_ET = ZoneInfo("America/New_York")
DIRECT_SOURCE_TYPES = {"company_ir", "press_release_wire"}
SEC_CLASSIFIER_VERSION = "sec-earnings-release-v1"
EARNINGS_FEATURE_VERSION = "structured-earnings-v1"
SEC_FILING_CACHE = DATA_DIR / "cache" / "sec_earnings_filings"
_DOCUMENT_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.IGNORECASE | re.DOTALL)
_MONTH_DATE_RE = re.compile(
    r"\bended\s+"
    r"(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(\d{1,2}),?\s+(20\d{2})",
    re.IGNORECASE,
)
_ANY_MONTH_DATE_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(\d{1,2}),?\s+(20\d{2})", re.IGNORECASE,
)


class EarningsDataError(ValueError):
    pass


class _TextExtractor(HTMLParser):
    """Small dependency-free HTML-to-text extractor for SEC exhibits."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style"}:
            self.ignored += 1

    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style"} and self.ignored:
            self.ignored -= 1

    def handle_data(self, data):
        if not self.ignored:
            self.parts.append(data)


def _plain_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(value)
    return " ".join(" ".join(parser.parts).replace("\xa0", " ").split())


def _utc_naive(value: str | datetime, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise EarningsDataError(f"{field} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise EarningsDataError(f"{field} must include an explicit UTC offset")
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _sec_acceptance_utc(value: str) -> datetime:
    """Parse SEC acceptance time; legacy timezone-less values are US/Eastern."""
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_ET)
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _release_session(public_time: datetime) -> str:
    local = public_time.replace(tzinfo=timezone.utc).astimezone(_ET)
    if local.weekday() >= 5:
        return "UNKNOWN"
    minute = local.hour * 60 + local.minute
    if minute < 9 * 60 + 30:
        return "BMO"
    if minute >= 16 * 60:
        return "AMC"
    return "DURING"


def _company_bucket(ticker: str) -> str:
    value = int(hashlib.sha256(f"{SPRINT_VERSION}|{ticker}".encode()).hexdigest()[:8], 16)
    return "UNSEEN_COMPANY" if value % 5 == 0 else "TRAIN_COMPANY"


def build_universe(as_of: date = date(2021, 6, 30), target: int = 500,
                   min_price: float = 5.0, min_adv: float = 10_000_000,
                   persist: bool = True) -> list[dict]:
    """Freeze a pre-sample liquid universe using only trailing data as of start."""
    con = connect()
    start = as_of - timedelta(days=90)
    rows = con.execute("""
        SELECT p.ticker, e.cik, AVG(p.close*p.volume) AS adv,
               arg_max(p.close, p.date) AS last_price
        FROM prices p JOIN sec_entities e USING (ticker)
        JOIN ticker_sectors s USING (ticker)
        WHERE p.date BETWEEN ? AND ? AND p.close IS NOT NULL AND p.volume IS NOT NULL
          AND p.ticker NOT LIKE '%.%' AND p.ticker NOT LIKE '%-%'
          AND s.sector <> 'Unknown'
          AND upper(e.title) NOT LIKE '%ETF%'
          AND upper(e.title) NOT LIKE '%FUND%'
          AND upper(e.title) NOT LIKE '%PORTFOLIO%'
          AND upper(e.title) NOT LIKE '%ISHARES%'
          AND upper(e.title) NOT LIKE '%SPDR%'
        GROUP BY p.ticker, e.cik
        HAVING COUNT(*) >= 40 AND AVG(p.close*p.volume) >= ?
               AND arg_max(p.close, p.date) >= ?
        ORDER BY adv DESC, p.ticker
        LIMIT ?
    """, [start, as_of, min_adv, min_price, target]).fetchall()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    result = [{"ticker": ticker, "cik": cik, "trailing_adv": float(adv),
               "last_price": float(price), "company_bucket": _company_bucket(ticker)}
              for ticker, cik, adv, price in rows]
    if persist:
        con.executemany("""
            INSERT INTO earnings_universe_snapshots
                (universe_version,ticker,eligibility_as_of,trailing_adv,last_price,
                 company_bucket,included,exclusion_reason,first_seen_at,metadata)
            VALUES (?,?,?,?,?,?,TRUE,NULL,?,?) ON CONFLICT DO NOTHING
        """, [(UNIVERSE_VERSION, row["ticker"], as_of, row["trailing_adv"],
                row["last_price"], row["company_bucket"], now,
                json.dumps({"min_price": min_price, "min_adv": min_adv,
                            "lookback_days": 90, "selection_is_point_in_time": True,
                            "security_filter": "resolved company sector; ETF/fund names excluded"}))
               for row in result])
    con.close()
    return result


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={
        "User-Agent": net.DEFAULT_UA,
    })
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _get_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": net.DEFAULT_UA})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - retry audited SEC transport failures
            last_error = exc
            time.sleep(1.0 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _sec_filing_text_url(source_url: str, source_id: str) -> str:
    accession = source_id.removeprefix("sec:")
    directory = source_url.rsplit("/", 1)[0]
    return f"{directory}/{accession}.txt"


def _cached_sec_filing(source_url: str, source_id: str, *, force: bool = False) -> tuple[str, str]:
    """Return primary 8-K plus linked 99.x exhibits, caching the compact bundle."""
    SEC_FILING_CACHE.mkdir(parents=True, exist_ok=True)
    accession = source_id.removeprefix("sec:")
    path = SEC_FILING_CACHE / f"{accession}.compact.txt"
    if path.exists() and not force:
        return path.read_text(encoding="utf-8", errors="replace"), source_url
    primary = _get_text(source_url)
    links: list[tuple[str, str]] = []
    for match in re.finditer(
            r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
            primary, re.IGNORECASE | re.DOTALL):
        href, label = match.group(1), _plain_text(match.group(2)).lower()
        filename = href.rsplit("/", 1)[-1].lower()
        if (re.search(r"\b99\s*\.\s*(?:0?1|0?2)\b", label) or
                re.search(r"(?:ex(?:hibit)?[-_]?99|ex99|991\.(?:htm|html|txt))", filename) or
                ("sec-extract:exhibit" in match.group(0).lower() and
                 re.search(r"press release|financial|earnings|quarterly results", label))):
            absolute = urljoin(source_url, href)
            if absolute not in {link for link, _ in links}:
                exhibit_type = "EX-99.2" if re.search(r"99\s*\.\s*0?2", label) else "EX-99.1"
                links.append((absolute, exhibit_type))
    parts = [f"<DOCUMENT><TYPE>8-K<FILENAME>{source_url.rsplit('/', 1)[-1]}"
             f"<TEXT>{primary}</TEXT></DOCUMENT>"]
    for exhibit_url, exhibit_type in links[:3]:
        exhibit = _get_text(exhibit_url)
        parts.append(f"<DOCUMENT><TYPE>{exhibit_type}"
                     f"<FILENAME>{exhibit_url.rsplit('/', 1)[-1]}"
                     f"<TEXT>{exhibit}</TEXT></DOCUMENT>")
    text = "\n".join(parts)
    path.write_text(text, encoding="utf-8")
    return text, source_url


def _sec_documents(filing_text: str) -> list[dict]:
    documents = []
    for block in _DOCUMENT_RE.findall(filing_text):
        def field(name: str) -> str:
            match = re.search(rf"<{name}>\s*([^\r\n<]+)", block, re.IGNORECASE)
            return match.group(1).strip() if match else ""
        text_match = re.search(r"<TEXT>(.*?)</TEXT>", block,
                               re.IGNORECASE | re.DOTALL)
        raw_text = text_match.group(1) if text_match else block
        documents.append({
            "type": field("TYPE").upper(),
            "filename": field("FILENAME"),
            "description": field("DESCRIPTION"),
            "raw_text": raw_text,
            "plain_text": _plain_text(raw_text),
        })
    return documents


def _extract_fiscal_period(text: str, public_time: datetime) -> str | None:
    def parsed(matches) -> list[date]:
        values = []
        for month, day, year in matches:
            try:
                value = datetime.strptime(f"{month} {day} {year}", "%B %d %Y").date()
            except ValueError:
                continue
            if (public_time.date() - timedelta(days=370) <= value <=
                    public_time.date() + timedelta(days=7)):
                values.append(value)
        return values

    dates = parsed(_MONTH_DATE_RE.findall(text))
    if dates:
        return str(max(dates))
    # Some bank releases say "second-quarter 2021" and label their current
    # balance-sheet date only as "as of June 30, 2021". Exclude dates near the
    # filing day so the press-release date itself cannot become the period end.
    for value in parsed(_ANY_MONTH_DATE_RE.findall(text)):
        try:
            if value <= public_time.date() - timedelta(days=5):
                dates.append(value)
        except TypeError:
            continue
    return str(max(dates)) if dates else None


def classify_sec_filing_text(filing_text: str, public_time: datetime) -> dict:
    """Conservatively identify an earnings release and its furnished exhibit.

    Item 2.02 is only candidate generation. Promotion requires explicit results
    language plus financial-statement evidence in the 8-K or Exhibit 99.x.
    """
    documents = _sec_documents(filing_text)
    primary = [doc for doc in documents if doc["type"] in {"8-K", "8-K/A"}]
    exhibits = [doc for doc in documents
                if re.fullmatch(r"EX-99(?:\.0?1|\.1|\.01|\.2|\.02)?", doc["type"])]
    selected = primary + exhibits
    combined = " ".join(doc["plain_text"] for doc in selected).lower()
    release_patterns = {
        "financial_results": r"(?:financial|quarterly|annual)\s+results",
        "results_for_period": r"results\s+for\s+the\s+(?:fiscal\s+)?(?:quarter|year)",
        "reports_period": r"reports?\s+(?:fiscal\s+)?(?:first|second|third|fourth|quarter|year).{0,80}results",
        "press_release_results": r"press\s+release.{0,160}(?:financial|operating)\s+results",
    }
    matched_phrases = [name for name, pattern in release_patterns.items()
                       if re.search(pattern, combined, re.DOTALL)]
    anchors = {
        "revenue": bool(re.search(r"\b(?:revenue|net sales)\b", combined)),
        "income": bool(re.search(r"\b(?:net income|net loss|operating income)\b", combined)),
        "eps": bool(re.search(r"\b(?:earnings|loss) per (?:diluted )?share\b|\bdiluted eps\b",
                              combined)),
        "statements": bool(re.search(r"(?:condensed|consolidated)\s+(?:financial\s+)?statements",
                                     combined)),
    }
    anchor_names = [name for name, present in anchors.items() if present]
    period_end = _extract_fiscal_period(combined, public_time)
    exhibit = next((doc for doc in exhibits
                    if doc["type"] in {"EX-99.1", "EX-99.01"}),
                   exhibits[0] if exhibits else None)
    is_earnings = bool(matched_phrases and len(anchor_names) >= 2 and period_end)
    return {
        "event_classification": ("VERIFIED_EARNINGS_RELEASE" if is_earnings
                                 else "NOT_EARNINGS"),
        "fiscal_period_end": period_end,
        "timestamp_quality": "TIER_2_SEC_ACCEPTANCE" if is_earnings else None,
        "matched_release_phrases": matched_phrases,
        "financial_anchors": anchor_names,
        "exhibit_filename": exhibit["filename"] if exhibit else None,
        "exhibit_type": exhibit["type"] if exhibit else None,
        "document_count": len(documents),
        "classifier_version": SEC_CLASSIFIER_VERSION,
    }


def _columnar_records(payload: dict) -> list[dict]:
    if "filings" in payload:
        payload = payload.get("filings", {}).get("recent", {})
    accessions = payload.get("accessionNumber") or []
    records = []
    for index in range(len(accessions)):
        records.append({key: values[index] if index < len(values) else None
                        for key, values in payload.items() if isinstance(values, list)})
    return records


def _event_id(ticker: str, identity: str) -> str:
    value = f"{SPRINT_VERSION}|{ticker}|{identity}"
    return hashlib.sha1(value.encode()).hexdigest()[:24]


def _quarter(period_end: str) -> str:
    month = int(period_end[5:7])
    return f"Q{((month - 1) // 3) + 1}"


def _sec_source(ticker: str, cik: str, record: dict) -> tuple | None:
    form = str(record.get("form") or "")
    items = str(record.get("items") or "")
    filing_date = str(record.get("filingDate") or "")
    if form not in {"8-K", "8-K/A"} or "2.02" not in items:
        return None
    if not (str(SAMPLE_START) <= filing_date <= str(SAMPLE_END)):
        return None
    acceptance = record.get("acceptanceDateTime")
    if not acceptance:
        return None
    public_time = _sec_acceptance_utc(acceptance)
    reported_date = str(record.get("reportDate") or filing_date)
    accession = str(record["accessionNumber"])
    accession_path = accession.replace("-", "")
    primary_document = str(record.get("primaryDocument") or "")
    url = f"{_SEC_ARCHIVES}/{int(cik)}/{accession_path}/{primary_document}"
    # Item 2.02 also contains operational updates (for example delivery counts).
    # Keep one candidate per accession until direct/reviewed evidence confirms it.
    earnings_event_id = _event_id(ticker, accession)
    source_id = f"sec:{accession}"
    metadata = json.dumps({"form": form, "items": items, "filing_date": filing_date,
                           "reported_date": reported_date,
                           "event_classification": "UNVERIFIED_EARNINGS_CANDIDATE",
                           "acceptance_timezone_rule": "source offset or America/New_York"})
    return (earnings_event_id, ticker, str(cik).zfill(10), None,
            "UNKNOWN", source_id, public_time, "sec_8k_item_202", url, False,
            None, metadata)


def _canonicalize(con, event_ids: list[str]) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for event_id in sorted(set(event_ids)):
        sources = con.execute("""
            SELECT source_id, public_time, source_type, source_url, is_direct_release,
                   json_extract_string(metadata, '$.ticker') ticker,
                   json_extract_string(metadata, '$.cik') cik,
                   json_extract_string(metadata, '$.fiscal_period_end') period_end,
                   json_extract_string(metadata, '$.fiscal_quarter') fiscal_quarter
            FROM earnings_event_sources WHERE earnings_event_id=?
            ORDER BY is_direct_release DESC, public_time, source_id
        """, [event_id]).fetchall()
        if not sources:
            continue
        direct = [source for source in sources if source[4]]
        reviewed_sec = [source for source in sources if source[2].startswith("sec_") and
                        con.execute("""
                            SELECT json_extract_string(metadata, '$.event_classification')
                            FROM earnings_event_sources
                            WHERE earnings_event_id=? AND source_id=?
                        """, [event_id, source[0]]).fetchone()[0] == "VERIFIED_EARNINGS_RELEASE"]
        eligible = direct or reviewed_sec or sources
        source = min(eligible, key=lambda row: row[1])
        timestamp_status = ("VERIFIED_EARLIEST" if direct else
                            "CONSERVATIVE_SEC_ONLY" if reviewed_sec else
                            "UNVERIFIED_2_02_CANDIDATE")
        session_status = "VERIFIED" if direct else "INFERRED" if reviewed_sec else "UNKNOWN"
        timestamp_quality = ("TIER_1_VERIFIED_EARLIEST" if direct else
                             "TIER_2_SEC_ACCEPTANCE" if reviewed_sec else
                             "UNVERIFIED")
        con.execute("""
            INSERT INTO earnings_events
                (earnings_event_id,ticker,cik,fiscal_period_end,fiscal_quarter,
                 earliest_public_time,release_session,timestamp_status,
                 release_session_status,primary_source_id,primary_source_url,
                 event_version,first_seen_at,metadata)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT (earnings_event_id) DO UPDATE SET
                earliest_public_time=excluded.earliest_public_time,
                release_session=excluded.release_session,
                timestamp_status=excluded.timestamp_status,
                release_session_status=excluded.release_session_status,
                primary_source_id=excluded.primary_source_id,
                primary_source_url=excluded.primary_source_url,
                event_version=excluded.event_version,
                ticker=excluded.ticker,cik=excluded.cik,
                fiscal_period_end=excluded.fiscal_period_end,
                fiscal_quarter=excluded.fiscal_quarter,
                metadata=excluded.metadata
        """, [event_id, source[5], source[6], source[7], source[8], source[1],
              _release_session(source[1]), timestamp_status, session_status,
              source[0], source[3], SPRINT_VERSION, now,
              json.dumps({"canonicalization":
                          "earliest direct release, reviewed SEC fallback, else unverified candidate",
                          "timestamp_quality": timestamp_quality})])


def _repair_unverified_statuses(con) -> None:
    """Neutralize legacy Item 2.02 rows that were once over-classified."""
    con.execute("""
        UPDATE earnings_events e
        SET timestamp_status='UNVERIFIED_2_02_CANDIDATE',
            release_session_status='UNKNOWN'
        WHERE EXISTS (
            SELECT 1 FROM earnings_event_sources s
            WHERE s.earnings_event_id=e.earnings_event_id AND s.source_type LIKE 'sec_%'
        )
          AND NOT EXISTS (
            SELECT 1 FROM earnings_event_sources s
            WHERE s.earnings_event_id=e.earnings_event_id
              AND (s.is_direct_release=TRUE OR
                   json_extract_string(s.metadata,'$.event_classification')=
                       'VERIFIED_EARNINGS_RELEASE')
        )
    """)


def _dedupe_earnings_revisions(con) -> int:
    """Keep one conservative event per ticker/fiscal period.

    Later Item 2.02 amendments or supplemental filings remain in provenance but
    cannot become duplicate model observations.
    """
    rows = con.execute("""
        SELECT earnings_event_id,ticker,fiscal_period_end,earliest_public_time,metadata
        FROM earnings_events
        WHERE timestamp_status='CONSERVATIVE_SEC_ONLY'
          AND fiscal_period_end IS NOT NULL
        ORDER BY ticker,fiscal_period_end,earliest_public_time,earnings_event_id
    """).fetchall()
    canonical: dict[tuple[str, date], str] = {}
    duplicates = 0
    for event_id, ticker, period_end, _, raw_metadata in rows:
        key = (ticker, period_end)
        if key not in canonical:
            canonical[key] = event_id
            continue
        metadata = json.loads(raw_metadata or "{}")
        metadata.update({"canonical_earnings_event_id": canonical[key],
                         "excluded_reason": "later filing for same ticker/fiscal period"})
        con.execute("""
            UPDATE earnings_events
            SET timestamp_status='DUPLICATE_EARNINGS_REVISION',metadata=?
            WHERE earnings_event_id=?
        """, [json.dumps(metadata), event_id])
        duplicates += 1
    return duplicates


def acquire_sec_events(universe: list[dict] | None = None,
                       verbose: bool = True) -> dict:
    """Acquire five years of Item 2.02 filings for the frozen universe."""
    universe = universe or build_universe()
    con = connect()
    inserted = failed = skipped = 0
    affected: list[str] = []
    for index, company in enumerate(universe, 1):
        ticker, cik = company["ticker"], str(company["cik"]).zfill(10)
        watermark = con.execute("""
            SELECT status FROM earnings_acquisition_watermarks
            WHERE ticker=? AND source='sec_item_202' AND sample_start=? AND sample_end=?
        """, [ticker, SAMPLE_START, SAMPLE_END]).fetchone()
        if watermark and watermark[0] == "COMPLETE":
            skipped += 1
            continue
        try:
            root = _get_json(f"{_SEC_SUBMISSIONS}/CIK{cik}.json")
            payloads = [root]
            for older in root.get("filings", {}).get("files", []):
                filing_from = str(older.get("filingFrom") or "0000-00-00")
                filing_to = str(older.get("filingTo") or "9999-99-99")
                if filing_to < str(SAMPLE_START) or filing_from > str(SAMPLE_END):
                    continue
                payloads.append(_get_json(f"{_SEC_SUBMISSIONS}/{older['name']}"))
                time.sleep(0.11)
            sources = [source for payload in payloads
                       for record in _columnar_records(payload)
                       if (source := _sec_source(ticker, cik, record))]
            rows = []
            for source in sources:
                event_id, tk, source_cik, period_end, quarter, source_id, public_time, \
                    source_type, url, direct, digest, raw_metadata = source
                metadata = json.loads(raw_metadata)
                metadata.update({"ticker": tk, "cik": source_cik,
                                 "fiscal_period_end": period_end,
                                 "fiscal_quarter": quarter})
                rows.append((event_id, source_id, public_time, source_type, url,
                             direct, datetime.now(timezone.utc).replace(tzinfo=None),
                             digest, json.dumps(metadata)))
                affected.append(event_id)
            if rows:
                con.executemany("""
                    INSERT INTO earnings_event_sources
                        (earnings_event_id,source_id,public_time,source_type,source_url,
                         is_direct_release,retrieved_at,source_sha256,metadata)
                    VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
                """, rows)
            con.execute("""
                INSERT INTO earnings_acquisition_watermarks
                    (ticker,source,sample_start,sample_end,candidate_count,status,
                     updated_at,metadata)
                VALUES (?,'sec_item_202',?,?,?,'COMPLETE',?,?)
                ON CONFLICT (ticker,source,sample_start,sample_end) DO UPDATE SET
                    candidate_count=excluded.candidate_count,status=excluded.status,
                    updated_at=excluded.updated_at,metadata=excluded.metadata
            """, [ticker, SAMPLE_START, SAMPLE_END, len(rows),
                  datetime.now(timezone.utc).replace(tzinfo=None),
                  json.dumps({"item_202_is_candidate_only": True})])
            inserted += len(rows)
        except Exception as exc:  # noqa: BLE001 - audit failures, continue universe
            failed += 1
            con.execute("""
                INSERT INTO earnings_acquisition_watermarks
                    (ticker,source,sample_start,sample_end,candidate_count,status,
                     updated_at,metadata)
                VALUES (?,'sec_item_202',?,?,0,'FAILED',?,?)
                ON CONFLICT (ticker,source,sample_start,sample_end) DO UPDATE SET
                    status=excluded.status,updated_at=excluded.updated_at,
                    metadata=excluded.metadata
            """, [ticker, SAMPLE_START, SAMPLE_END,
                  datetime.now(timezone.utc).replace(tzinfo=None),
                  json.dumps({"error": str(exc)[:500]})])
            if verbose:
                print(f"  {ticker}: SEC acquisition failed: {exc}")
        if verbose and index % 25 == 0:
            print(f"  SEC {index}/{len(universe)} companies, sources={inserted}, failed={failed}")
        time.sleep(0.11)
    _canonicalize(con, affected)
    _repair_unverified_statuses(con)
    con.close()
    return {"companies": len(universe), "sec_sources": inserted, "failed": failed,
            "skipped_complete": skipped,
            "timestamp_status": "UNVERIFIED_2_02_CANDIDATE until releases are reviewed"}


def classify_sec_candidates(*, max_events: int | None = None, force: bool = False,
                            verbose: bool = True) -> dict:
    """Inspect every candidate's full filing and furnished 99.x exhibits.

    This is a deterministic high-precision screen, not a claim that SEC
    acceptance was the earliest release. Passing rows are Tier 2 and enter only
    the conservative bar-cost screen.
    """
    con = connect()
    limit = f"LIMIT {int(max_events)}" if max_events else ""
    rows = con.execute(f"""
        SELECT e.earnings_event_id,e.earliest_public_time,s.source_id,s.source_url,
               s.metadata
        FROM earnings_events e
        JOIN earnings_event_sources s USING (earnings_event_id)
        WHERE s.source_type='sec_8k_item_202'
          AND e.timestamp_status IN ('UNVERIFIED_2_02_CANDIDATE',
                                     'CONSERVATIVE_SEC_ONLY','NOT_EARNINGS',
                                     'DUPLICATE_EARNINGS_REVISION')
        ORDER BY e.earliest_public_time,e.earnings_event_id
        {limit}
    """).fetchall()
    classified = rejected = failed = skipped = 0
    affected: list[str] = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    pending = []
    for row in rows:
        metadata = json.loads(row[4] or "{}")
        if (not force and metadata.get("classifier_version") == SEC_CLASSIFIER_VERSION and
                metadata.get("event_classification") in
                {"VERIFIED_EARNINGS_RELEASE", "NOT_EARNINGS"}):
            skipped += 1
        else:
            pending.append(row)

    def fetch(row):
        try:
            text, url = _cached_sec_filing(row[3], row[2], force=force)
            return row, text, url, None
        except Exception as exc:  # noqa: BLE001
            return row, None, None, exc

    processed = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
     fetched_rows = (
         fetched
         for offset in range(0, len(pending), 64)
         for fetched in pool.map(fetch, pending[offset:offset + 64])
     )
     for fetched in fetched_rows:
        (event_id, public_time, source_id, source_url, raw_metadata), \
            filing_text, filing_url, fetch_error = fetched
        metadata = json.loads(raw_metadata or "{}")
        try:
            if fetch_error is not None:
                raise fetch_error
            result = classify_sec_filing_text(filing_text, public_time)
            metadata.update(result)
            metadata.update({
                "classification_method": "deterministic_full_filing_and_exhibit_rules",
                "classification_reviewed_at": now.isoformat(),
                "full_submission_url": filing_url,
            })
            digest = hashlib.sha256(filing_text.encode()).hexdigest()
            con.execute("""
                UPDATE earnings_event_sources
                SET metadata=?,source_sha256=?,retrieved_at=?
                WHERE earnings_event_id=? AND source_id=?
            """, [json.dumps(metadata), digest, now, event_id, source_id])
            exhibit_filename = result.get("exhibit_filename")
            if exhibit_filename:
                exhibit_id = f"sec_exhibit:{source_id.removeprefix('sec:')}:{exhibit_filename}"
                exhibit_metadata = {
                    **metadata,
                    "ticker": metadata.get("ticker"), "cik": metadata.get("cik"),
                    "fiscal_period_end": result.get("fiscal_period_end"),
                    "fiscal_quarter": "UNKNOWN",
                }
                exhibit_url = f"{source_url.rsplit('/', 1)[0]}/{exhibit_filename}"
                con.execute("""
                    INSERT INTO earnings_event_sources
                        (earnings_event_id,source_id,public_time,source_type,source_url,
                         is_direct_release,retrieved_at,source_sha256,metadata)
                    VALUES (?,?,?,?,?,FALSE,?,?,?) ON CONFLICT DO NOTHING
                """, [event_id, exhibit_id, public_time, "sec_exhibit_99",
                      exhibit_url, now, digest, json.dumps(exhibit_metadata)])
            affected.append(event_id)
            if result["event_classification"] == "VERIFIED_EARNINGS_RELEASE":
                classified += 1
            else:
                rejected += 1
        except Exception as exc:  # noqa: BLE001 - preserve failure provenance and continue
            failed += 1
            metadata.update({"classifier_version": SEC_CLASSIFIER_VERSION,
                             "classification_error": str(exc)[:500],
                             "classification_reviewed_at": now.isoformat()})
            con.execute("""
                UPDATE earnings_event_sources SET metadata=?,retrieved_at=?
                WHERE earnings_event_id=? AND source_id=?
            """, [json.dumps(metadata), now, event_id, source_id])
        processed += 1
        if verbose and processed % 100 == 0:
            print(f"  SEC review {processed + skipped}/{len(rows)} earnings={classified} "
                  f"rejected={rejected} failed={failed} skipped={skipped}", flush=True)
    _canonicalize(con, affected)
    _repair_unverified_statuses(con)
    con.execute("""
        UPDATE earnings_events SET timestamp_status='NOT_EARNINGS'
        WHERE earnings_event_id IN (
            SELECT earnings_event_id FROM earnings_event_sources
            WHERE source_type='sec_8k_item_202'
              AND json_extract_string(metadata,'$.event_classification')='NOT_EARNINGS'
        )
    """)
    duplicates = _dedupe_earnings_revisions(con)
    con.close()
    return {"candidates": len(rows), "tier_2_earnings": classified,
            "not_earnings": rejected, "failed": failed, "skipped": skipped,
            "duplicate_revisions": duplicates,
            "classifier_version": SEC_CLASSIFIER_VERSION}


def _load_manifest(path: str | Path) -> list[dict]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    return payload if isinstance(payload, list) else payload.get("records", [])


def ingest_release_manifest(path: str | Path) -> dict:
    """Append reviewed company-IR/press-wire earliest-release records."""
    records = _load_manifest(path)
    con = connect()
    affected = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = []
    for record in records:
        source_type = record.get("source_type")
        if source_type not in DIRECT_SOURCE_TYPES:
            raise EarningsDataError(f"direct source_type must be one of {DIRECT_SOURCE_TYPES}")
        if record.get("reviewed_earliest") is not True:
            raise EarningsDataError("direct release must be explicitly reviewed_earliest=true")
        parsed_url = urlparse(record.get("source_url", ""))
        if parsed_url.scheme != "https" or not parsed_url.hostname:
            raise EarningsDataError("direct release source_url must be HTTPS")
        public_time = _utc_naive(record["public_time"], "public_time")
        period_end = str(record["fiscal_period_end"])
        ticker = str(record["ticker"]).upper()
        event_id = record.get("earnings_event_id")
        if not event_id:
            nearby = con.execute("""
                SELECT earnings_event_id FROM earnings_events
                WHERE ticker=? AND timestamp_status='UNVERIFIED_2_02_CANDIDATE'
                  AND abs(epoch(earliest_public_time)-epoch(?)) <= 3*86400
                ORDER BY abs(epoch(earliest_public_time)-epoch(?)) LIMIT 1
            """, [ticker, public_time, public_time]).fetchone()
            event_id = nearby[0] if nearby else _event_id(ticker, period_end)
        source_id = str(record.get("source_id") or
                        hashlib.sha1(record["source_url"].encode()).hexdigest()[:20])
        metadata = {"ticker": ticker, "cik": str(record.get("cik") or ""),
                    "fiscal_period_end": period_end,
                    "fiscal_quarter": record.get("fiscal_quarter") or _quarter(period_end),
                    "reviewed_earliest": True}
        rows.append((event_id, source_id, public_time, source_type,
                     record["source_url"], True, now, record.get("source_sha256"),
                     json.dumps(metadata)))
        affected.append(event_id)
    con.executemany("""
        INSERT INTO earnings_event_sources
            (earnings_event_id,source_id,public_time,source_type,source_url,
             is_direct_release,retrieved_at,source_sha256,metadata)
        VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
    """, rows)
    _canonicalize(con, affected)
    _repair_unverified_statuses(con)
    con.close()
    return {"direct_release_sources": len(rows), "events_affected": len(set(affected))}


def ingest_sec_classification_manifest(path: str | Path) -> dict:
    """Review Item 2.02 candidates; verified rows remain conservative SEC-time."""
    records = _load_manifest(path)
    con = connect()
    affected = []
    for record in records:
        event_id = record["earnings_event_id"]
        classification = record.get("event_classification")
        if classification not in {"VERIFIED_EARNINGS_RELEASE", "NOT_EARNINGS"}:
            raise EarningsDataError("event_classification must be verified earnings or not earnings")
        sources = con.execute("""
            SELECT source_id,metadata FROM earnings_event_sources
            WHERE earnings_event_id=? AND source_type LIKE 'sec_%'
        """, [event_id]).fetchall()
        if not sources:
            raise EarningsDataError(f"no SEC candidate sources for {event_id}")
        for source_id, raw_metadata in sources:
            metadata = json.loads(raw_metadata or "{}")
            metadata["event_classification"] = classification
            if record.get("fiscal_period_end"):
                metadata["fiscal_period_end"] = record["fiscal_period_end"]
                metadata["fiscal_quarter"] = (record.get("fiscal_quarter") or
                                               _quarter(record["fiscal_period_end"]))
            con.execute("""
                UPDATE earnings_event_sources SET metadata=?
                WHERE earnings_event_id=? AND source_id=?
            """, [json.dumps(metadata), event_id, source_id])
        affected.append(event_id)
    _canonicalize(con, affected)
    _repair_unverified_statuses(con)
    con.execute("""
        UPDATE earnings_events SET timestamp_status='NOT_EARNINGS'
        WHERE earnings_event_id IN (
            SELECT earnings_event_id FROM earnings_event_sources
            WHERE json_extract_string(metadata,'$.event_classification')='NOT_EARNINGS'
        )
    """)
    con.close()
    return {"classified_candidates": len(records)}


def ingest_consensus_manifest(path: str | Path) -> dict:
    """Insert only historical, point-in-time consensus snapshots."""
    records = _load_manifest(path)
    con = connect()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = []
    for record in records:
        event = con.execute("""
            SELECT ticker,fiscal_period_end,earliest_public_time
            FROM earnings_events WHERE earnings_event_id=?
        """, [record["earnings_event_id"]]).fetchone()
        if not event:
            raise EarningsDataError(f"unknown earnings event {record['earnings_event_id']}")
        as_of = _utc_naive(record["estimate_as_of"], "estimate_as_of")
        if as_of >= event[2]:
            raise EarningsDataError("consensus snapshot must predate public release")
        if record.get("is_point_in_time") is not True or record.get("is_final_revised") is True:
            raise EarningsDataError("final/revised or non-point-in-time consensus is forbidden")
        rows.append((record["earnings_event_id"], event[0], event[1], record["metric"],
                     float(record["estimate_value"]), record.get("currency"),
                     record.get("analyst_count"), as_of, record["vendor"],
                     str(record["vendor_record_id"]), True, False, now,
                     json.dumps(record.get("metadata", {})), as_of,
                     record.get("forecast_dispersion", record.get("estimate_std")),
                     record.get("revision_breadth"), EARNINGS_FEATURE_VERSION))
    con.executemany("""
        INSERT INTO earnings_consensus_snapshots
            (earnings_event_id,ticker,fiscal_period_end,metric,estimate_value,
             currency,analyst_count,estimate_as_of,vendor,vendor_record_id,
             is_point_in_time,is_final_revised,ingested_at,metadata,known_at,
             forecast_dispersion,revision_breadth,feature_version)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
    """, rows)
    con.close()
    return {"consensus_snapshots": len(rows)}


def ingest_actuals_manifest(path: str | Path) -> dict:
    """Insert structured actuals with their own public timestamp/source."""
    records = _load_manifest(path)
    con = connect()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = []
    for record in records:
        event = con.execute("""
            SELECT earliest_public_time FROM earnings_events WHERE earnings_event_id=?
        """, [record["earnings_event_id"]]).fetchone()
        if not event:
            raise EarningsDataError(f"unknown earnings event {record['earnings_event_id']}")
        public_time = _utc_naive(record["public_time"], "public_time")
        if public_time < event[0] - timedelta(minutes=1):
            raise EarningsDataError("actual cannot become public before the canonical release")
        source_record_id = record.get("source_record_id")
        if not source_record_id:
            raise EarningsDataError("actual requires source_record_id")
        rows.append((record["earnings_event_id"], record["metric"],
                     float(record["actual_value"]), record.get("currency"), public_time,
                     record["source"], record["source_url"], now,
                     json.dumps(record.get("metadata", {})), public_time,
                     str(source_record_id), EARNINGS_FEATURE_VERSION))
    con.executemany("""
        INSERT INTO earnings_actuals
            (earnings_event_id,metric,actual_value,currency,public_time,source,
             source_url,ingested_at,metadata,known_at,source_record_id,feature_version)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
    """, rows)
    con.close()
    return {"actuals": len(rows)}


def _event_public_time(con, event_id: str) -> datetime:
    event = con.execute("""
        SELECT earliest_public_time FROM earnings_events WHERE earnings_event_id=?
    """, [event_id]).fetchone()
    if not event:
        raise EarningsDataError(f"unknown earnings event {event_id}")
    return event[0]


def ingest_guidance_manifest(path: str | Path) -> dict:
    records = _load_manifest(path)
    con = connect()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = []
    for record in records:
        event_time = _event_public_time(con, record["earnings_event_id"])
        public_time = _utc_naive(record["public_time"], "public_time")
        role = str(record.get("guidance_role") or "new")
        if role not in {"previous", "new"}:
            raise EarningsDataError("guidance_role must be previous or new")
        if role == "previous" and public_time >= event_time:
            raise EarningsDataError("previous guidance must predate the canonical release")
        if role == "new" and public_time < event_time - timedelta(minutes=1):
            raise EarningsDataError("new guidance cannot predate the canonical release")
        source_record_id = record.get("source_record_id")
        if not source_record_id:
            raise EarningsDataError("guidance requires source_record_id")
        status = str(record.get("guidance_status") or "AVAILABLE")
        if status not in {"AVAILABLE", "NO_GUIDANCE", "MISSING_DATA"}:
            raise EarningsDataError("invalid guidance_status")
        rows.append((record["earnings_event_id"], record["metric"],
                     record["guidance_period"], record.get("lower_value"),
                     record.get("upper_value"), record.get("currency"), public_time,
                     record["source"], record["source_url"], now,
                     json.dumps(record.get("metadata", {})), public_time,
                     str(source_record_id), status, role, EARNINGS_FEATURE_VERSION))
    con.executemany("""
        INSERT INTO earnings_guidance_snapshots
            (earnings_event_id,metric,guidance_period,lower_value,upper_value,
             currency,public_time,source,source_url,ingested_at,metadata,
             known_at,source_record_id,guidance_status,guidance_role,feature_version)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
    """, rows)
    con.close()
    return {"guidance_records": len(rows)}


def ingest_options_manifest(path: str | Path) -> dict:
    records = _load_manifest(path)
    con = connect()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = []
    for record in records:
        event_time = _event_public_time(con, record["earnings_event_id"])
        observed_at = _utc_naive(record["observed_at"], "observed_at")
        if observed_at >= event_time:
            raise EarningsDataError("options expectation must be observed before release")
        straddle = float(record["straddle_mid"])
        underlying = float(record["underlying_mid"])
        implied_move = float(record.get("implied_move", straddle / underlying))
        if straddle <= 0 or underlying <= 0 or not 0 < implied_move < 1:
            raise EarningsDataError("invalid options-implied move")
        rows.append((record["earnings_event_id"], observed_at,
                     record["expiration_date"], straddle, underlying, implied_move,
                     record.get("implied_volatility"), record["source"],
                     str(record["source_record_id"]), now,
                     json.dumps(record.get("metadata", {}))))
    con.executemany("""
        INSERT INTO earnings_options_expectations
            (earnings_event_id,observed_at,expiration_date,straddle_mid,
             underlying_mid,implied_move,implied_volatility,source,
             source_record_id,ingested_at,metadata)
        VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
    """, rows)
    con.close()
    return {"options_expectations": len(rows)}


def ingest_positioning_manifest(path: str | Path) -> dict:
    records = _load_manifest(path)
    con = connect()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = []
    for record in records:
        event_time = _event_public_time(con, record["earnings_event_id"])
        observed_at = _utc_naive(record["observed_at"], "observed_at")
        if observed_at >= event_time:
            raise EarningsDataError("positioning snapshot must predate release")
        rows.append((record["earnings_event_id"], observed_at,
                     record.get("short_interest_shares"), record.get("days_to_cover"),
                     record.get("institutional_ownership"),
                     record.get("passive_ownership"), record["source"],
                     str(record["source_record_id"]), now,
                     json.dumps(record.get("metadata", {}))))
    con.executemany("""
        INSERT INTO earnings_positioning_snapshots
            (earnings_event_id,observed_at,short_interest_shares,days_to_cover,
             institutional_ownership,passive_ownership,source,source_record_id,
             ingested_at,metadata)
        VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
    """, rows)
    con.close()
    return {"positioning_snapshots": len(rows)}


def ingest_call_manifest(path: str | Path) -> dict:
    """Insert exact-time prepared/Q&A transcript segments without behavior labels."""
    records = _load_manifest(path)
    con = connect()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = []
    for record in records:
        event_id = record["earnings_event_id"]
        event_time = _event_public_time(con, event_id)
        call_time = _utc_naive(record["call_public_time"], "call_public_time")
        if call_time < event_time:
            raise EarningsDataError("conference call cannot predate earnings release")
        if record.get("timestamp_precision") != "exact":
            raise EarningsDataError("conference call timestamp_precision must be exact")
        for index, segment in enumerate(record.get("segments") or []):
            section = segment.get("section")
            if section not in {"prepared", "question", "answer"}:
                raise EarningsDataError("call section must be prepared, question or answer")
            segment_time = (_utc_naive(segment["segment_time"], "segment_time")
                            if segment.get("segment_time") else None)
            segment_id = hashlib.sha1(
                f"{event_id}|{index}|{segment.get('speaker_name')}|{segment.get('text')}".encode()
            ).hexdigest()[:24]
            rows.append((segment_id, event_id, call_time, index, segment_time,
                         segment.get("speaker_name"), segment.get("speaker_role"),
                         section, segment["text"], record["source"],
                         record["source_url"], now,
                         json.dumps({"behavior_labels_added": False,
                                     **record.get("metadata", {})})))
    con.executemany("""
        INSERT INTO earnings_call_segments
            (segment_id,earnings_event_id,call_public_time,segment_index,
             segment_time,speaker_name,speaker_role,section,text,source,
             source_url,ingested_at,metadata)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
    """, rows)
    con.close()
    return {"call_segments": len(rows), "calls": len(records)}


if __name__ == "__main__":
    universe = build_universe()
    print(f"Frozen universe: {len(universe)} companies; "
          f"unseen={sum(r['company_bucket']=='UNSEEN_COMPANY' for r in universe)}")
