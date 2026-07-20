"""Independent AI-adjudicated reference labels for the MGRM gold set.

NOT a human gold set. An independent AI-adjudicated reference set: gpt-5.6-sol
reads each frozen source filing and produces labels through a blind pass, an
independent verification pass, and an adjudication pass, followed by
deterministic validation. This is weaker than human labels (Sol and the Terra
extractor may share model-family biases) but far better than letting the
extractor grade itself.

Strict model separation is enforced: the reference labeler is gpt-5.6-sol and
the extractor under evaluation is gpt-5.6-terra. The reference-label generation
never sees any extractor (Terra) output. This module is data tooling only and
does not touch the trading model, research harness, promotion gates, or the
extractor architecture.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import urllib.request

from .. import net
from ..config import ROOT
from .earnings import _plain_text
from .guidance import ACTIONS, ALLOWED_METRICS, STATUSES
from . import mgrm_corpus


REFERENCE_MODEL = os.getenv("MGRM_REFERENCE_MODEL", "gpt-5.6-sol")
EXTRACTOR_MODEL = os.getenv("MGRM_LLM_MODEL", "gpt-5.6-terra")
PROMPT_VERSION = "mgrm-sol-adjudicated-v1"
LABEL_PROVENANCE = "ai_adjudicated_independent"
UNITS = ["absolute", "per_share", "percent"]
MAX_SOURCE_CHARS = 120_000
_NUMERIC_REQUIRED_STATUS = {"AVAILABLE", "REAFFIRMED"}

DEV_REFERENCE_PATH = mgrm_corpus.GOLDSET_DIR / "mgrm_dev_reference.jsonl"
CERT_REFERENCE_PATH = mgrm_corpus.GOLDSET_DIR / "mgrm_cert_reference.jsonl"
EXCEPTION_PATH = mgrm_corpus.GOLDSET_DIR / "mgrm_reference_exceptions.jsonl"

_BLIND_SYSTEM = (
    "You extract forward-looking management guidance from an earnings filing. "
    "Read the ENTIRE document. A guidance record is management's own forward "
    "outlook for a future fiscal period (not a reported actual). Copy the exact "
    "supporting sentence or table cells verbatim as evidence. Never infer a "
    "number or period that is not stated. If there is no forward guidance, set "
    "no_guidance=true and return no records."
)
_VERIFY_SYSTEM = (
    "You are auditing a candidate guidance label against the source filing. "
    "Re-read the source and actively try to DISPROVE the candidate: find missed "
    "guidance, wrong periods, wrong units, wrong ranges, wrong raised/lowered/"
    "reaffirmed/initiated/withdrawn classification, or fabricated evidence. "
    "Return agrees=true only if the candidate is exactly correct and complete. "
    "Otherwise return agrees=false with issues and a fully corrected label."
)
_ADJUDICATE_SYSTEM = (
    "Two independent labels of the same filing disagree. Re-read the source and "
    "return the single correct final label with a confidence of HIGH, MEDIUM or "
    "LOW. Prefer the label whose evidence is verbatim in the source."
)


def _nullable(kind: str) -> dict:
    return {"anyOf": [{"type": kind}, {"type": "null"}]}


def _record_schema() -> dict:
    properties = {
        "metric": {"type": "string", "enum": sorted(ALLOWED_METRICS)},
        "period": {"type": "string"},
        "units": {"type": "string", "enum": UNITS},
        "currency": _nullable("string"),
        "low": _nullable("number"), "high": _nullable("number"),
        "midpoint": _nullable("number"),
        "status": {"type": "string", "enum": sorted(STATUSES)},
        "action": {"type": "string", "enum": sorted(ACTIONS)},
        "evidence": {"type": "string"},
    }
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


def _label_schema() -> dict:
    return {"type": "object", "additionalProperties": False, "properties": {
        "no_guidance": {"type": "boolean"},
        "records": {"type": "array", "items": _record_schema()},
    }, "required": ["no_guidance", "records"]}


def _verify_schema() -> dict:
    return {"type": "object", "additionalProperties": False, "properties": {
        "agrees": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "corrected": _label_schema(),
    }, "required": ["agrees", "issues", "corrected"]}


def _adjudicate_schema() -> dict:
    schema = _label_schema()
    schema["properties"]["confidence"] = {"type": "string",
                                          "enum": ["HIGH", "MEDIUM", "LOW"]}
    schema["required"].append("confidence")
    return schema


def _require_model_separation() -> None:
    if REFERENCE_MODEL == EXTRACTOR_MODEL:
        raise RuntimeError(
            f"model separation violated: reference labeler and extractor are "
            f"both {REFERENCE_MODEL!r}")


def _sol_request(system: str, user: str, schema: dict, name: str) -> dict:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not configured")
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    payload = {"model": REFERENCE_MODEL,
               "input": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
               "text": {"format": {"type": "json_schema", "name": name,
                                    "strict": True, "schema": schema}},
               "max_output_tokens": 8000}
    request = urllib.request.Request(
        f"{base}/responses", data=json.dumps(payload).encode(), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": net.DEFAULT_UA})
    with urllib.request.urlopen(request, timeout=240) as response:
        result = json.load(response)
    text = "".join(part.get("text", "") for item in result.get("output", [])
                   for part in item.get("content", [])
                   if part.get("type") == "output_text")
    return {"data": json.loads(text), "response_id": result.get("id"),
            "created_at": result.get("created_at")}


def _norm(text: str) -> str:
    return " ".join(str(text).split()).lower()


def deterministic_validate(source_text: str, label: dict) -> list[str]:
    issues = []
    normalized = _norm(source_text)
    records = label.get("records", []) or []
    if label.get("no_guidance") and records:
        issues.append("no_guidance=true but records present")
    if not label.get("no_guidance") and not records:
        issues.append("guidance-present but no records")
    seen = set()
    for index, record in enumerate(records):
        tag = f"record {index + 1}"
        key = (record.get("metric"), record.get("period"))
        if key in seen:
            issues.append(f"{tag}: duplicate metric-period {key}")
        seen.add(key)
        if record.get("metric") not in ALLOWED_METRICS:
            issues.append(f"{tag}: metric not allowed")
        if record.get("units") not in UNITS:
            issues.append(f"{tag}: units not allowed")
        if record.get("status") not in STATUSES:
            issues.append(f"{tag}: status not allowed")
        if record.get("action") not in ACTIONS:
            issues.append(f"{tag}: action not allowed")
        evidence = _norm(record.get("evidence", ""))
        if not evidence or evidence not in normalized:
            issues.append(f"{tag}: evidence not found verbatim in source")
        low, high, mid = record.get("low"), record.get("high"), record.get("midpoint")
        if record.get("status") in _NUMERIC_REQUIRED_STATUS and None in (low, high, mid):
            issues.append(f"{tag}: {record.get('status')} needs low/high/midpoint")
        if None not in (low, high, mid):
            if not (low <= mid <= high):
                issues.append(f"{tag}: require low <= midpoint <= high")
            scale = max(abs(low), abs(high), 1.0)
            if abs(mid - (low + high) / 2) > 0.01 * scale:
                issues.append(f"{tag}: midpoint != (low+high)/2")
    return issues


def _final_confidence(agrees: bool, adjudicated: str | None,
                      validation_issues: list[str]) -> str:
    if validation_issues:
        return "LOW"
    if agrees:
        return "HIGH"
    return adjudicated or "MEDIUM"


def adjudicate_document(source_text: str) -> dict:
    """Blind -> verify -> (adjudicate on disagreement) -> deterministic validate."""
    _require_model_separation()
    truncated = source_text[:MAX_SOURCE_CHARS]
    response_ids, timestamps = [], []

    blind = _sol_request(_BLIND_SYSTEM, truncated, _label_schema(), "mgrm_label")
    response_ids.append(blind["response_id"]); timestamps.append(blind["created_at"])
    first = blind["data"]

    verify = _sol_request(
        _VERIFY_SYSTEM,
        f"SOURCE:\n{truncated}\n\nCANDIDATE LABEL:\n{json.dumps(first)}",
        _verify_schema(), "mgrm_verify")
    response_ids.append(verify["response_id"]); timestamps.append(verify["created_at"])
    verification = verify["data"]

    disagreements = verification.get("issues", [])
    adjudicated_conf = None
    if verification.get("agrees"):
        final = {"no_guidance": first["no_guidance"], "records": first["records"]}
    else:
        adjudication = _sol_request(
            _ADJUDICATE_SYSTEM,
            f"SOURCE:\n{truncated}\n\nLABEL A (blind):\n{json.dumps(first)}\n\n"
            f"LABEL B (verification):\n{json.dumps(verification.get('corrected'))}",
            _adjudicate_schema(), "mgrm_adjudicate")
        response_ids.append(adjudication["response_id"])
        timestamps.append(adjudication["created_at"])
        final = {"no_guidance": adjudication["data"]["no_guidance"],
                 "records": adjudication["data"]["records"]}
        adjudicated_conf = adjudication["data"]["confidence"]

    validation_issues = deterministic_validate(source_text, final)
    confidence = _final_confidence(bool(verification.get("agrees")),
                                   adjudicated_conf, validation_issues)
    expected = [] if final["no_guidance"] else final["records"]
    return {
        "no_guidance": final["no_guidance"], "expected": expected,
        "confidence": confidence, "first_pass_label": first,
        "verification_result": verification, "disagreements": disagreements,
        "validation_issues": validation_issues,
        "reference_model": REFERENCE_MODEL, "extractor_model": EXTRACTOR_MODEL,
        "prompt_version": PROMPT_VERSION, "response_ids": response_ids,
        "timestamps": timestamps, "label_provenance": LABEL_PROVENANCE,
    }


def _reference_path(split: str) -> Path:
    return DEV_REFERENCE_PATH if split == "development" else CERT_REFERENCE_PATH


def _existing(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["doc_id"]
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")}


def generate_reference(split: str, *, limit: int | None = None,
                       verbose: bool = True) -> dict:
    """Generate AI-adjudicated reference labels for a split (resumable)."""
    _require_model_separation()
    manifest = [record for record in mgrm_corpus.load_manifest()
                if record["split"] == split]
    path = _reference_path(split)
    done = _existing(path)
    labelled = exceptions = 0
    processed = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in manifest:
            doc_id = record["document_id"]
            if doc_id in done:
                continue
            if limit is not None and processed >= limit:
                break
            processed += 1
            source_file = ROOT / record["raw_path"]
            source_text = _plain_text(
                source_file.read_text(encoding="utf-8", errors="replace"))
            result = adjudicate_document(source_text)
            row = {"doc_id": doc_id, "company": record["company"],
                   "sector": record["sector"], "format": record["format_hint"],
                   "split": split, "source_url": record["document_url"],
                   "source_sha256": record["source_sha256"], **result}
            handle.write(json.dumps(row, default=str) + "\n")
            handle.flush()
            labelled += 1
            if result["confidence"] == "LOW" or result["validation_issues"]:
                with EXCEPTION_PATH.open("a", encoding="utf-8") as exc:
                    exc.write(json.dumps({"doc_id": doc_id, "split": split,
                                          "confidence": result["confidence"],
                                          "validation_issues": result["validation_issues"]},
                                         default=str) + "\n")
                exceptions += 1
            if verbose:
                print(f"  {doc_id} {record['company']}: {result['confidence']} "
                      f"({len(result['expected'])} rec, {len(result['validation_issues'])} issues)",
                      flush=True)
    return {"split": split, "labelled_now": labelled, "already_done": len(done),
            "exceptions": exceptions, "reference_path": str(path.relative_to(ROOT)),
            "reference_model": REFERENCE_MODEL}


def load_reference(split: str) -> list[dict]:
    path = _reference_path(split)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
