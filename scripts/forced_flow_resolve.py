"""Forced-flow announcement-timestamp resolver (return-blind, Tier 1/2 compliant).

Backlog #6. Resolves S&P 500 addition batches to VERIFIED announcement times per
docs/forced_flow_announcement_spec.md. Blindness: this module NEVER imports
prices/returns or computes an outcome — it only fetches the public release,
parses the machine-readable publication time, and pins the source by sha256.

Pipeline (proven feasible): raw HTTP GET of the S&P DJI / PR Newswire release ->
JSON-LD "datePublished" (exact minute) -> hash raw bytes -> validate the added
company/ticker actually appears with "S&P 500" -> VERIFIED Tier 1/2 record.

Deterministic chronological order; unresolved batches are preserved. Idempotent:
re-running merges by event_batch_id and never relaxes the tiers.

Usage: PYTHONPATH=src python scripts/forced_flow_resolve.py
Reads candidate URLs from goldset/forced_flow/announcement_candidates.jsonl
(fields: event_batch_id, effective_date, affected_tickers, company_check, url).
Writes goldset/forced_flow/announcement_manifest_v1.jsonl and
goldset/forced_flow/announcement_coverage_v1.json.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from html import unescape
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass
import requests

from quantv1.config import ROOT

FF = ROOT / "goldset" / "forced_flow"
CANDIDATES = FF / "announcement_candidates.jsonl"
MANIFEST = FF / "announcement_manifest_v1.jsonl"
LEGACY_RESOLVED = FF / "announcement_resolved.jsonl"
WORKLIST = FF / "announcement_worklist_v1.jsonl"
COVERAGE = FF / "announcement_coverage_v1.json"
UA = {"User-Agent": "Mozilla/5.0 (quantv1 forced-flow census; research)"}
NY = ZoneInfo("America/New_York")


def _tier(url: str) -> int:
    host = url.lower()
    if "spglobal.com" in host or "spdji.com" in host:
        return 1
    if any(n in host for n in ("prnewswire.com", "businesswire.com",
                               "globenewswire.com")):
        return 2
    return 3


def _session(dt: datetime) -> str:
    ny = dt.astimezone(NY)
    if ny.weekday() >= 5:
        return "AFTER_HOURS"
    minutes = ny.hour * 60 + ny.minute
    return "INTRADAY" if 9 * 60 + 30 <= minutes < 16 * 60 else "AFTER_HOURS"


def resolve_one(cand: dict) -> dict:
    url = cand["url"]
    try:
        r = requests.get(url, headers=UA, timeout=30)
    except Exception as e:  # noqa: BLE001
        return {**_stub(cand), "verification_status": "UNRESOLVED",
                "reject_reason": f"FETCH_ERROR:{type(e).__name__}"}
    if r.status_code != 200:
        return {**_stub(cand), "verification_status": "UNRESOLVED",
                "reject_reason": f"HTTP_{r.status_code}"}
    body = r.content
    text = r.text
    m = re.search(r'"datePublished"\s*:\s*"([^"]+)"', text)
    if not m:
        return {**_stub(cand), "verification_status": "UNRESOLVED",
                "reject_reason": "NO_MACHINE_TIMESTAMP"}
    try:
        dt = datetime.fromisoformat(m.group(1))
    except ValueError:
        return {**_stub(cand), "verification_status": "UNRESOLVED",
                "reject_reason": "BAD_TIMESTAMP"}
    if dt.tzinfo is None:
        return {**_stub(cand), "verification_status": "UNRESOLVED",
                "reject_reason": "NAIVE_TIMESTAMP"}
    # linkage validation: an EXCHANGE-QUALIFIED added ticker must appear with
    # 'S&P 500' (guards against e.g. "DOW" matching "Dow Jones"). Normalize HTML
    # (strip tags, unescape entities, collapse whitespace) so markup like
    # "NYSE:&nbsp;DOW" does not defeat the match. Applied uniformly.
    up = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", text))).upper()
    ticker_ok = any(
        re.search(r"(?:NYSE|NASDAQ|NASD|CBOE|NYSE ARCA|NYSE MKT)[:\s]{1,4}"
                  + re.escape(t.upper()), up)
        for t in cand["affected_tickers"])
    if not (ticker_ok and "S&P 500" in up):
        return {**_stub(cand), "verification_status": "UNRESOLVED",
                "reject_reason": "LINKAGE_UNCONFIRMED"}
    correction = "CORRECTION" if re.search(r"correct", text, re.I) else "ORIGINAL"
    return {
        "event_batch_id": cand["event_batch_id"],
        "announcement_public_time": dt.isoformat(),
        "announcement_timezone": "America/New_York",
        "effective_date": cand["effective_date"],
        "source_tier": _tier(url),
        "source_url": url,
        "source_sha256": hashlib.sha256(body).hexdigest(),
        "original_or_correction": correction,
        "affected_tickers": cand["affected_tickers"],
        "timestamp_precision": "exact_minute",
        "announcement_session": _session(dt),
        "first_executable_time": "NEXT_SESSION_OPEN",
        "verification_status": "VERIFIED",
    }


def _stub(cand: dict) -> dict:
    return {"event_batch_id": cand["event_batch_id"],
            "effective_date": cand["effective_date"],
            "affected_tickers": cand["affected_tickers"],
            "source_url": cand.get("url")}


def load_existing() -> dict:
    out = {}
    for path in (LEGACY_RESOLVED, MANIFEST):
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    rec.setdefault("verification_status", "VERIFIED")
                    out[rec["event_batch_id"]] = rec
    return out


def main() -> None:
    resolved = load_existing()
    new = 0
    if CANDIDATES.exists():
        for line in CANDIDATES.read_text().splitlines():
            if not line.strip():
                continue
            cand = json.loads(line)
            bid = cand["event_batch_id"]
            if resolved.get(bid, {}).get("verification_status") == "VERIFIED":
                continue
            rec = resolve_one(cand)
            resolved[bid] = rec
            status = rec["verification_status"]
            if status == "VERIFIED":
                new += 1
            print(f"{bid:18s} {status:10s} "
                  f"{rec.get('announcement_public_time') or rec.get('reject_reason')}")

    verified = [r for r in resolved.values()
                if r.get("verification_status") == "VERIFIED"]
    verified.sort(key=lambda r: r["effective_date"])
    MANIFEST.write_text("\n".join(json.dumps(r, sort_keys=True) for r in verified) + "\n")

    total_batches = sum(1 for line in WORKLIST.read_text().splitlines() if line.strip()) \
        if WORKLIST.exists() else None
    by_year, by_tier = {}, {}
    for r in verified:
        y = r["effective_date"][:4]
        by_year[y] = by_year.get(y, 0) + 1
        by_tier[str(r["source_tier"])] = by_tier.get(str(r["source_tier"]), 0) + 1
    claim = ("descriptive pilot only" if len(verified) < 50 else
             "underpowered candidate test" if len(verified) < 75 else "full test")
    coverage = {
        "rules_version": "forced-flow-announcement-rules-v1",
        "total_addition_batches": total_batches,
        "verified": len(verified),
        "verified_pct": round(100 * len(verified) / total_batches, 1) if total_batches else None,
        "remaining": (total_batches - len(verified)) if total_batches else None,
        "by_year": by_year, "by_tier": by_tier,
        "predeclared_claim_if_frozen_now": claim,
        "note": ("return-blind; verified-and-covered subset is the tradable "
                 "continuation sample, NOT 113; tiers never relaxed"),
    }
    COVERAGE.write_text(json.dumps(coverage, indent=2, sort_keys=True))
    print(f"\nnew_verified={new}  total_verified={len(verified)}/{total_batches}  "
          f"claim_if_frozen_now={claim}")
    print(f"wrote {MANIFEST.name}, {COVERAGE.name}")


if __name__ == "__main__":
    main()
