"""Public SEC/IR acquisition and fail-closed MGRM guidance extraction."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.request

from .. import net
from ..config import DATA_DIR
from ..db import connect
from .earnings import (
    SAMPLE_END,
    SAMPLE_START,
    _SEC_ARCHIVES,
    _SEC_SUBMISSIONS,
    _cached_sec_filing,
    _columnar_records,
    _get_json,
    _sec_acceptance_utc,
    _sec_documents,
    build_universe,
    classify_sec_filing_text,
)


DISCOVERY_VERSION = "mgrm-sec-202-701-v1"
EXTRACTOR_VERSION = "mgrm-guidance-numeric-ai-agreement-v1"
LINKER_VERSION = "mgrm-previous-guidance-v1"
RAW_ROOT = DATA_DIR / "raw" / "mgrm"
ALLOWED_METRICS = {
    "revenue", "eps", "ebitda", "operating_income", "gross_margin",
    "bookings", "arr", "capex",
}
STATUSES = {"AVAILABLE", "REAFFIRMED", "WITHDRAWN"}
ACTIONS = {"INITIATED", "RAISED", "LOWERED", "REAFFIRMED", "WITHDRAWN", "UNSPECIFIED"}

_METRICS = (
    ("gross_margin", r"gross\s+margin"),
    ("operating_income", r"operating\s+(?:income|profit)"),
    ("ebitda", r"(?:adjusted\s+)?ebitda"),
    ("bookings", r"bookings?"),
    ("arr", r"annual\s+recurring\s+revenue|\barr\b"),
    ("capex", r"capital\s+expenditures?|\bcapex\b"),
    ("eps", r"(?:diluted\s+)?(?:earnings|loss)\s+per\s+share|\beps\b"),
    ("revenue", r"(?:net\s+sales|revenue)"),
)
_GUIDANCE = re.compile(
    r"guidance|outlook|expects?|forecasts?|projects?|anticipates?|"
    r"raises?|lowers?|reaffirms?|withdraws?", re.IGNORECASE,
)
_NUMBER = re.compile(
    r"(?P<currency>[$€£])?\s*(?P<number>-?\d+(?:\.\d+)?)\s*"
    r"(?P<scale>billion|million|thousand|bn|mm|m|k)?\s*(?P<percent>%)?",
    re.IGNORECASE,
)


def _hash(value: str, length: int = 24) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:length]


def _event_for_filing(con, ticker: str, accession: str,
                      acceptance: datetime) -> str | None:
    exact = con.execute("""
        SELECT earnings_event_id FROM earnings_event_sources
        WHERE source_id=? LIMIT 1
    """, [f"sec:{accession}"]).fetchone()
    if exact:
        return exact[0]
    nearby = con.execute("""
        SELECT earnings_event_id FROM earnings_events
        WHERE ticker=? AND abs(epoch(earliest_public_time)-epoch(?)) <= 86400
          AND timestamp_status IN ('VERIFIED_EARLIEST','CONSERVATIVE_SEC_ONLY')
        ORDER BY abs(epoch(earliest_public_time)-epoch(?)) LIMIT 1
    """, [ticker, acceptance, acceptance]).fetchone()
    return nearby[0] if nearby else None


def discover_filings(*, start: date = SAMPLE_START, end: date = SAMPLE_END,
                     universe: list[dict] | None = None,
                     force: bool = False, verbose: bool = True) -> dict:
    """Discover 8-K Item 2.02/7.01 filings and preserve immutable documents."""
    universe = universe or build_universe()
    con = connect()
    discovered = documents = eligible = failed = 0
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for index, company in enumerate(universe, 1):
        ticker, cik = company["ticker"], str(company["cik"]).zfill(10)
        try:
            root = _get_json(f"{_SEC_SUBMISSIONS}/CIK{cik}.json")
            payloads = [root]
            for older in root.get("filings", {}).get("files", []):
                if (str(older.get("filingTo") or "9999-99-99") < str(start) or
                        str(older.get("filingFrom") or "0000-00-00") > str(end)):
                    continue
                payloads.append(_get_json(f"{_SEC_SUBMISSIONS}/{older['name']}"))
                time.sleep(0.11)
            for record in (item for payload in payloads
                           for item in _columnar_records(payload)):
                form = str(record.get("form") or "")
                items = str(record.get("items") or "")
                filing_date = str(record.get("filingDate") or "")
                if (form not in {"8-K", "8-K/A"} or
                        not ({"2.02", "7.01"} & set(items.split(","))) or
                        not (str(start) <= filing_date <= str(end)) or
                        not record.get("acceptanceDateTime")):
                    continue
                accession = str(record["accessionNumber"])
                if not force and con.execute(
                    "SELECT 1 FROM mgrm_filings WHERE accession_number=?",
                    [accession],
                ).fetchone():
                    continue
                acceptance = _sec_acceptance_utc(record["acceptanceDateTime"])
                primary = str(record.get("primaryDocument") or "")
                directory = f"{_SEC_ARCHIVES}/{int(cik)}/{accession.replace('-', '')}"
                source_url = f"{directory}/{primary}"
                event_id = _event_for_filing(con, ticker, accession, acceptance)
                filing_text, _ = _cached_sec_filing(
                    source_url, f"sec:{accession}", force=force
                )
                classification = classify_sec_filing_text(filing_text, acceptance)
                parsed = _sec_documents(filing_text)
                has_guidance = any(_GUIDANCE.search(doc["plain_text"])
                                   for doc in parsed)
                status = ("ELIGIBLE" if event_id and
                          (classification["event_classification"] ==
                           "VERIFIED_EARNINGS_RELEASE" or has_guidance)
                          else "UNLINKED_OR_NO_GUIDANCE")
                con.execute("""
                    INSERT OR REPLACE INTO mgrm_filings
                        (accession_number,earnings_event_id,ticker,cik,form,items,
                         acceptance_time,filing_date,primary_document,source_url,
                         first_seen_at,discovery_version,status,metadata)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, [accession, event_id, ticker, cik, form, items, acceptance,
                      filing_date, primary, source_url, now, DISCOVERY_VERSION,
                      status, json.dumps(classification)])
                discovered += 1
                eligible += int(status == "ELIGIBLE")
                for position, doc in enumerate(parsed):
                    if doc["type"] not in {"8-K", "8-K/A", "EX-99.1", "EX-99.01",
                                           "EX-99.2", "EX-99.02"}:
                        continue
                    raw = doc["raw_text"]
                    digest = hashlib.sha256(raw.encode()).hexdigest()
                    raw_dir = RAW_ROOT / accession
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    raw_path = raw_dir / f"{digest}.html"
                    if not raw_path.exists():
                        raw_path.write_text(raw, encoding="utf-8")
                    document_id = _hash(f"{accession}|{position}|{digest}")
                    doc_url = (f"{directory}/{doc['filename']}"
                               if doc["filename"] else source_url)
                    con.execute("""
                        INSERT INTO mgrm_documents
                            (document_id,accession_number,earnings_event_id,ticker,
                             document_type,source_url,source_sha256,raw_path,
                             public_time,first_seen_at,status,metadata)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
                    """, [document_id, accession, event_id, ticker, doc["type"],
                          doc_url, digest, str(raw_path), acceptance, now,
                          "PRESERVED", json.dumps({"filename": doc["filename"]})])
                    documents += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if verbose:
                print(f"  {ticker}: MGRM SEC acquisition failed: {exc}")
        if verbose and index % 25 == 0:
            print(f"  MGRM SEC {index}/{len(universe)} filings={discovered} "
                  f"documents={documents} failed={failed}", flush=True)
        time.sleep(0.11)
    con.close()
    return {"companies": len(universe), "filings": discovered,
            "eligible_filings": eligible, "documents": documents,
            "failed_companies": failed, "discovery_version": DISCOVERY_VERSION}


def _period(sentence: str) -> str:
    year = re.search(r"\b(20\d{2})\b", sentence)
    quarter = re.search(r"\b(?:Q([1-4])|(?:first|second|third|fourth) quarter)\b",
                        sentence, re.IGNORECASE)
    if re.search(r"full[- ]year|fiscal year|\bFY\b", sentence, re.IGNORECASE):
        return f"FY{year.group(1) if year else 'UNKNOWN'}"
    if quarter:
        number = quarter.group(1) or {
            "first": "1", "second": "2", "third": "3", "fourth": "4",
        }.get(quarter.group(0).lower().split()[0], "UNKNOWN")
        return f"Q{number}-{year.group(1) if year else 'UNKNOWN'}"
    return f"PERIOD-{year.group(1) if year else 'UNKNOWN'}"


def _number(match, metric: str) -> tuple[float, str, str | None] | None:
    value = float(match.group("number"))
    if 1900 <= value <= 2100 and not match.group("currency"):
        return None
    scale = (match.group("scale") or "").lower()
    multiplier = {"billion": 1e9, "bn": 1e9, "million": 1e6, "mm": 1e6,
                  "m": 1e6, "thousand": 1e3, "k": 1e3}.get(scale, 1.0)
    percent = bool(match.group("percent")) or metric == "gross_margin"
    units = "percent" if percent else "per_share" if metric == "eps" else "absolute"
    currency = {"$": "USD", "€": "EUR", "£": "GBP"}.get(match.group("currency"))
    return (value if percent else value * multiplier), units, currency


def _action(sentence: str) -> tuple[str, str]:
    lower = sentence.lower()
    if re.search(r"withdraw|suspend|no longer provid", lower):
        return "WITHDRAWN", "WITHDRAWN"
    if re.search(r"reaffirm|maintain|unchanged", lower):
        return "REAFFIRMED", "REAFFIRMED"
    if re.search(r"rais|increas|upward", lower):
        return "AVAILABLE", "RAISED"
    if re.search(r"lower|reduc|downward", lower):
        return "AVAILABLE", "LOWERED"
    if re.search(r"initiat|introduc|first time", lower):
        return "AVAILABLE", "INITIATED"
    return "AVAILABLE", "UNSPECIFIED"


def deterministic_extract(text: str) -> list[dict]:
    """High-precision numerical extraction; ambiguous fields stay unavailable."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?;])\s+", text)
                 if part.strip()]
    records = []
    for sentence in sentences:
        if not _GUIDANCE.search(sentence):
            continue
        metric = next((name for name, pattern in _METRICS
                       if re.search(pattern, sentence, re.IGNORECASE)), None)
        if not metric:
            continue
        status, action = _action(sentence)
        values = [parsed for match in _NUMBER.finditer(sentence)
                  if (parsed := _number(match, metric)) is not None]
        if status == "AVAILABLE" and not values:
            continue
        low = high = units = currency = None
        if values:
            numeric = [value[0] for value in values]
            low, high = (min(numeric[:2]), max(numeric[:2])) if len(numeric) > 1 else (
                numeric[0], numeric[0]
            )
            units = values[0][1]
            currency = next((value[2] for value in values if value[2]), None)
        records.append({
            "metric": metric, "guidance_period": _period(sentence),
            "lower_value": low, "upper_value": high,
            "midpoint": ((low + high) / 2 if low is not None else None),
            "units": units, "currency": currency,
            "guidance_status": status, "stated_action": action,
            "supporting_sentence": sentence[:2000],
            "confidence": 0.95 if values or status != "AVAILABLE" else 0.0,
        })
    unique = {}
    for record in records:
        key = (record["metric"], record["guidance_period"],
               record["supporting_sentence"])
        unique[key] = record
    return list(unique.values())


def _record(metric: str, period: str, values: list, status: str, action: str,
            evidence: str, source_kind: str) -> dict:
    low = high = units = currency = None
    if values:
        numeric = [value[0] for value in values]
        low, high = ((min(numeric[:2]), max(numeric[:2])) if len(numeric) > 1
                     else (numeric[0], numeric[0]))
        units = values[0][1]
        currency = next((value[2] for value in values if value[2]), None)
    return {
        "metric": metric, "guidance_period": period,
        "lower_value": low, "upper_value": high,
        "midpoint": ((low + high) / 2 if low is not None else None),
        "units": units, "currency": currency,
        "guidance_status": status, "stated_action": action,
        "supporting_sentence": evidence[:2000], "source_kind": source_kind,
        "confidence": 0.95 if values or status != "AVAILABLE" else 0.0,
    }


def extract_tables(html: str) -> list[dict]:
    """Extract guidance rows from HTML tables before falling back to prose.

    Guidance is frequently published in an outlook table (metric rows, a
    low/high or single-value column) rather than a prose sentence. Only tables
    whose surrounding context signals guidance are read, and the exact row text
    is preserved as evidence; ambiguous rows are skipped, never guessed.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []
    soup = BeautifulSoup(html, "html.parser")
    records = []
    for table in soup.find_all("table"):
        context = " ".join(table.get_text(" ", strip=True).split())
        heading = table.find_previous(["p", "h1", "h2", "h3", "h4", "td", "span"])
        heading_text = (" ".join(heading.get_text(" ", strip=True).split())
                        if heading else "")
        guided = bool(_GUIDANCE.search(context) or _GUIDANCE.search(heading_text))
        if not guided:
            continue
        period_source = f"{heading_text} {context}"
        for row in table.find_all("tr"):
            cells = [" ".join(cell.get_text(" ", strip=True).split())
                     for cell in row.find_all(["td", "th"])]
            cells = [cell for cell in cells if cell]
            if len(cells) < 2:
                continue
            label = cells[0]
            metric = next((name for name, pattern in _METRICS
                           if re.search(pattern, label, re.IGNORECASE)), None)
            if not metric:
                continue
            joined = " ".join(cells[1:])
            values = [parsed for match in _NUMBER.finditer(joined)
                      if (parsed := _number(match, metric)) is not None]
            if not values:
                continue
            status, action = _action(f"{heading_text} {label} {joined}")
            evidence = " | ".join(cells)
            records.append(_record(metric, _period(f"{label} {period_source}"),
                                   values, status, action, evidence, "table"))
    unique = {}
    for record in records:
        unique[(record["metric"], record["guidance_period"])] = record
    return list(unique.values())


def structured_extract(raw_html: str) -> list[dict]:
    """Table-first, then prose. Table rows win ties on (metric, period)."""
    from .earnings import _plain_text
    table_records = extract_tables(raw_html)
    prose_records = deterministic_extract(_plain_text(raw_html))
    for record in prose_records:
        record.setdefault("source_kind", "prose")
    combined = {}
    for record in table_records + prose_records:
        combined.setdefault((record["metric"], record["guidance_period"]), record)
    return list(combined.values())


def _schema() -> dict:
    nullable_number = {"anyOf": [{"type": "number"}, {"type": "null"}]}
    nullable_string = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    properties = {
        "metric": {"type": "string", "enum": sorted(ALLOWED_METRICS)},
        "guidance_period": {"type": "string"},
        "lower_value": nullable_number, "upper_value": nullable_number,
        "midpoint": nullable_number, "units": nullable_string,
        "currency": nullable_string,
        "guidance_status": {"type": "string", "enum": sorted(STATUSES)},
        "stated_action": {"type": "string", "enum": sorted(ACTIONS)},
        "supporting_sentence": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    return {"type": "object", "additionalProperties": False,
            "properties": {"records": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": properties, "required": list(properties),
            }}}, "required": ["records"]}


_LLM_SYSTEM = (
    "Extract only explicit forward management guidance. Copy the exact "
    "supporting sentence. Never infer missing numbers or periods."
)


def llm_config() -> dict | None:
    """Resolve the active extraction backend; no backend -> fail closed (None).

    MGRM_LLM_PROVIDER selects the backend: 'openai' (or any OpenAI-compatible
    /responses endpoint via OPENAI_BASE_URL) or 'ollama' (local, no key). When
    unset it is inferred from OPENAI_API_KEY, otherwise disabled. The provider
    and model version are recorded on every extraction for reproducibility.
    """
    provider = os.getenv("MGRM_LLM_PROVIDER", "").strip().lower()
    if not provider:
        provider = "openai" if os.getenv("OPENAI_API_KEY") else "none"
    if provider in {"none", "off", "disabled", ""}:
        return None
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return None
        return {"provider": "openai",
                "model": os.getenv("MGRM_LLM_MODEL", "gpt-5.4-mini"),
                "base_url": os.getenv("OPENAI_BASE_URL",
                                      "https://api.openai.com/v1").rstrip("/"),
                "api_key": key}
    if provider == "ollama":
        return {"provider": "ollama",
                "model": os.getenv("MGRM_OLLAMA_MODEL", "llama3.1"),
                "base_url": os.getenv("OLLAMA_BASE_URL",
                                      "http://localhost:11434").rstrip("/"),
                "api_key": None}
    raise ValueError(f"unknown MGRM_LLM_PROVIDER: {provider!r}")


def provider_tag(config: dict | None = None) -> str:
    config = config if config is not None else llm_config()
    return (f"{config['provider']}:{config['model']}" if config else "none:none")


def _openai_extract(config: dict, text: str) -> list[dict]:
    payload = {
        "model": config["model"],
        "input": [{"role": "system", "content": _LLM_SYSTEM},
                  {"role": "user", "content": text[:100_000]}],
        "text": {"format": {"type": "json_schema", "name": "mgrm_guidance",
                             "strict": True, "schema": _schema()}},
    }
    request = urllib.request.Request(
        f"{config['base_url']}/responses", data=json.dumps(payload).encode(),
        method="POST", headers={"Authorization": f"Bearer {config['api_key']}",
                                "Content-Type": "application/json",
                                "User-Agent": net.DEFAULT_UA},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    output_text = "".join(
        part.get("text", "") for item in result.get("output", [])
        for part in item.get("content", []) if part.get("type") == "output_text"
    )
    return json.loads(output_text).get("records", [])


def _ollama_extract(config: dict, text: str) -> list[dict]:
    payload = {
        "model": config["model"], "stream": False,
        "messages": [{"role": "system", "content": _LLM_SYSTEM},
                     {"role": "user", "content": text[:100_000]}],
        "format": _schema(), "options": {"temperature": 0},
    }
    request = urllib.request.Request(
        f"{config['base_url']}/api/chat", data=json.dumps(payload).encode(),
        method="POST", headers={"Content-Type": "application/json",
                                "User-Agent": net.DEFAULT_UA},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    content = (result.get("message") or {}).get("content", "")
    return json.loads(content).get("records", []) if content else []


def ai_extract(text: str, config: dict | None = None) -> list[dict] | None:
    """Provider-agnostic schema-constrained extraction; no backend -> None."""
    config = config if config is not None else llm_config()
    if config is None:
        return None
    if config["provider"] == "ollama":
        return _ollama_extract(config, text)
    return _openai_extract(config, text)


def _same_number(left, right) -> bool:
    if left is None or right is None:
        return left is right
    scale = max(abs(float(left)), abs(float(right)), 1.0)
    return math.isclose(float(left), float(right), rel_tol=0.01,
                        abs_tol=0.001 * scale)


def reconcile(deterministic: list[dict], ai: list[dict] | None) -> list[dict]:
    output = []
    for record in deterministic:
        matches = [] if ai is None else [candidate for candidate in ai
                  if candidate.get("metric") == record["metric"] and
                  candidate.get("guidance_period") == record["guidance_period"]]
        agreed = next((candidate for candidate in matches
                       if _same_number(candidate.get("lower_value"),
                                       record.get("lower_value")) and
                       _same_number(candidate.get("upper_value"),
                                       record.get("upper_value")) and
                       candidate.get("guidance_status") ==
                       record.get("guidance_status")), None)
        output.append({**record, "ai_record": agreed or (matches[0] if matches else None),
                       "agreement_status": ("AGREED" if agreed else
                                            "PENDING_AI" if ai is None else
                                            "DISAGREED")})
    return output


def extract_documents(*, max_documents: int | None = None,
                      force: bool = False, use_ai: bool = True) -> dict:
    con = connect()
    limit = f"LIMIT {int(max_documents)}" if max_documents else ""
    rows = con.execute(f"""
        SELECT d.document_id,d.earnings_event_id,d.ticker,d.raw_path,d.public_time
        FROM mgrm_documents d JOIN mgrm_filings f USING (accession_number)
        WHERE f.status='ELIGIBLE' ORDER BY d.public_time,d.document_id {limit}
    """).fetchall()
    extracted = agreed_count = disagreed = pending = 0
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    config = llm_config() if use_ai else None
    version = f"{EXTRACTOR_VERSION}|{provider_tag(config)}"
    for document_id, event_id, ticker, raw_path, public_time in rows:
        if not force and con.execute("""
            SELECT 1 FROM mgrm_guidance_extractions
            WHERE document_id=? AND extractor_version=? LIMIT 1
        """, [document_id, version]).fetchone():
            continue
        raw = Path(raw_path).read_text(encoding="utf-8", errors="replace")
        from .earnings import _plain_text
        deterministic = structured_extract(raw)
        ai_records = (ai_extract(_plain_text(raw), config)
                      if config is not None and deterministic else None)
        for record in reconcile(deterministic, ai_records):
            extraction_id = _hash(
                f"{document_id}|{version}|{record['metric']}|"
                f"{record['guidance_period']}|{record['supporting_sentence']}"
            )
            ai_record = record.pop("ai_record")
            agreement = record.pop("agreement_status")
            con.execute("""
                INSERT OR REPLACE INTO mgrm_guidance_extractions
                    (extraction_id,document_id,earnings_event_id,ticker,metric,
                     guidance_period,lower_value,upper_value,midpoint,units,currency,
                     guidance_status,stated_action,supporting_sentence,
                     deterministic_confidence,ai_confidence,deterministic_payload,
                     ai_payload,agreement_status,extractor_version,public_time,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, [extraction_id, document_id, event_id, ticker, record["metric"],
                  record["guidance_period"], record["lower_value"],
                  record["upper_value"], record["midpoint"], record["units"],
                  record["currency"], record["guidance_status"],
                  record["stated_action"], record["supporting_sentence"],
                  record["confidence"], ai_record.get("confidence") if ai_record else None,
                  json.dumps(record), json.dumps(ai_record) if ai_record else None,
                  agreement, version, public_time, now])
            extracted += 1
            agreed_count += int(agreement == "AGREED")
            disagreed += int(agreement == "DISAGREED")
            pending += int(agreement == "PENDING_AI")
    con.close()
    return {"documents": len(rows), "extractions": extracted,
            "agreed": agreed_count, "disagreed": disagreed,
            "pending_ai": pending, "extractor_version": version,
            "provider": provider_tag(config)}


def link_previous_guidance() -> dict:
    con = connect()
    rows = con.execute("""
        SELECT extraction_id,ticker,metric,guidance_period,midpoint,
               lower_value,upper_value,guidance_status,stated_action,public_time
        FROM mgrm_guidance_extractions WHERE agreement_status='AGREED'
        ORDER BY ticker,metric,guidance_period,public_time,extraction_id
    """).fetchall()
    history: dict[tuple[str, str, str], tuple] = {}
    linked = unmatched = 0
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for row in rows:
        (extraction_id, ticker, metric, period, midpoint, low, high,
         status, action, _) = row
        key = (ticker, metric, period)
        previous = history.get(key)
        revision = width_change = None
        classification = action
        link_status = "NO_PREVIOUS_GUIDANCE"
        previous_id = None
        if previous:
            previous_id, previous_mid, previous_low, previous_high = previous
            if status == "REAFFIRMED":
                midpoint, low, high = previous_mid, previous_low, previous_high
                revision, width_change, classification = 0.0, 0.0, "REAFFIRMED"
            elif status == "WITHDRAWN":
                classification = "WITHDRAWN"
            elif midpoint is not None and previous_mid not in {None, 0}:
                revision = (midpoint - previous_mid) / abs(previous_mid)
                previous_width = ((previous_high - previous_low) /
                                  max(abs(previous_mid), 1e-12))
                current_width = ((high - low) / max(abs(midpoint), 1e-12))
                width_change = current_width - previous_width
                classification = ("RAISED" if revision > 1e-9 else
                                  "LOWERED" if revision < -1e-9 else "REAFFIRMED")
            link_status = "LINKED"
            linked += 1
        else:
            unmatched += 1
        con.execute("""
            INSERT OR REPLACE INTO mgrm_guidance_links
                (extraction_id,previous_extraction_id,midpoint_revision,
                 range_width_change,revision_classification,link_status,
                 linker_version,created_at,metadata)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, [extraction_id, previous_id, revision, width_change,
              classification, link_status, LINKER_VERSION, now,
              json.dumps({"join_key": [ticker, metric, period]})])
        if status != "WITHDRAWN" and midpoint is not None:
            history[key] = (extraction_id, midpoint, low, high)
    con.close()
    return {"accepted_extractions": len(rows), "linked": linked,
            "unmatched": unmatched, "linker_version": LINKER_VERSION}


def snapshot_public_record(record: dict, *, first_seen_at: datetime | None = None) -> str:
    """Append an externally configured free public observation unchanged."""
    now = first_seen_at or datetime.now(timezone.utc).replace(tzinfo=None)
    url = str(record["source_url"])
    request = urllib.request.Request(url, headers={"User-Agent": net.DEFAULT_UA})
    with urllib.request.urlopen(request, timeout=90) as response:
        raw = response.read()
    digest = hashlib.sha256(raw).hexdigest()
    snapshot_id = _hash(f"{record['source']}|{url}|{digest}|{now.isoformat()}")
    raw_dir = DATA_DIR / "raw" / "public_expectations" / str(now.date())
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{snapshot_id}.raw"
    raw_path.write_bytes(raw)
    con = connect()
    con.execute("""
        INSERT INTO public_expectation_snapshots
            (snapshot_id,ticker,expectation_type,metric,period,source,source_url,
             source_time,first_seen_at,raw_sha256,raw_path,payload)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
    """, [snapshot_id, record.get("ticker"), record["expectation_type"],
          record.get("metric"), record.get("period"), record["source"], url,
          record.get("source_time"), now, digest, str(raw_path), json.dumps(record)])
    con.close()
    return snapshot_id


def collect_forward(records: list[dict] | None = None) -> dict:
    today = datetime.now(timezone.utc).date()
    sec = discover_filings(start=today - timedelta(days=7), end=today,
                           force=False, verbose=False)
    snapshots = [snapshot_public_record(record) for record in (records or [])]
    return {"sec": sec, "configured_public_snapshots": len(snapshots),
            "first_seen_at_is_immutable": True}
