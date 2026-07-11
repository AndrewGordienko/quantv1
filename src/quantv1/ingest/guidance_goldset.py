"""Frozen gold-set audit for the MGRM guidance extractor.

Requiring the deterministic parser and the LLM to agree is a *precision filter*,
not proof of correctness: both can agree on the same wrong number, and the
intersection is biased toward easy prose. The scientific gate is measured
field-level accuracy against a frozen, manually labelled set of real filings.

This module loads that gold set (``goldset/mgrm_guidance_gold.jsonl``), runs the
extractor over each document, and reports precision/recall for guidance
detection plus per-field accuracy for period, units, range, midpoint and the
raised/lowered/reaffirmed action. Certification is granted only when the set is
large and diverse enough AND every frozen threshold is met; otherwise it is
BLOCKED (e.g. ``GOLDSET_TOO_SMALL``) and no historical MGRM run is justified.

The committed seed is small and exists to validate the audit machinery. It must
be expanded with human labels across 30-50 stratified companies, multiple
sectors and multiple filing/release formats before the extractor can be
certified and a bounded pilot run.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from ..config import DATA_DIR, ROOT
from .guidance import EXTRACTOR_VERSION, _same_number, structured_extract


GOLD_PATH = ROOT / "goldset" / "mgrm_guidance_gold.jsonl"
CERTIFICATION_PATH = DATA_DIR / "mgrm_extractor_certification.json"
AUDIT_VERSION = "mgrm-goldset-audit-v1"

# Frozen acceptance thresholds. Freeze BEFORE running historical MGRM.
MIN_GOLD_DOCUMENTS = 30
MIN_SECTORS = 5
MIN_FORMATS = 2
MIN_DETECTION_PRECISION = 0.80
MIN_DETECTION_RECALL = 0.70
MIN_PERIOD_ACCURACY = 0.90
MIN_RANGE_ACCURACY = 0.90
MIN_UNITS_ACCURACY = 0.90
MIN_ACTION_ACCURACY = 0.90


def load_goldset(path: Path | None = None) -> list[dict]:
    path = path or GOLD_PATH
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            records.append(json.loads(line))
    return records


def _key(record: dict) -> tuple[str, str]:
    # Predicted records use ``guidance_period``; gold labels use ``period``.
    period = record.get("guidance_period", record.get("period"))
    return (str(record.get("metric")), str(period))


def _predict(document: dict) -> list[dict]:
    html = document.get("document_html")
    if html is None and document.get("raw_path"):
        html = Path(document["raw_path"]).read_text(encoding="utf-8", errors="replace")
    return structured_extract(html or "")


def audit(goldset: list[dict] | None = None) -> dict:
    goldset = goldset if goldset is not None else load_goldset()
    detection = Counter()  # tp / fp / fn
    field_totals = Counter()
    field_correct = Counter()
    per_document = []
    sectors, formats = set(), set()
    for document in goldset:
        sectors.add(str(document.get("sector", "UNKNOWN")))
        formats.add(str(document.get("format", "unknown")))
        expected = {} if document.get("no_guidance") else {
            _key(record): record for record in document.get("expected", [])
        }
        predicted = {_key(record): record for record in _predict(document)}
        tp_keys = set(expected) & set(predicted)
        detection["tp"] += len(tp_keys)
        detection["fp"] += len(set(predicted) - set(expected))
        detection["fn"] += len(set(expected) - set(predicted))
        for key in tp_keys:
            want, got = expected[key], predicted[key]
            checks = {
                "period": True,  # period is part of the matched key
                "units": str(want.get("units")) == str(got.get("units")),
                "range": _same_number(want.get("low"), got.get("lower_value")) and
                         _same_number(want.get("high"), got.get("upper_value")),
                "midpoint": _same_number(want.get("midpoint"), got.get("midpoint")),
                "action": str(want.get("action")) == str(got.get("stated_action")),
            }
            for field, correct in checks.items():
                field_totals[field] += 1
                field_correct[field] += int(correct)
        per_document.append({
            "doc_id": document.get("doc_id"), "company": document.get("company"),
            "format": document.get("format"), "expected": len(expected),
            "predicted": len(predicted), "matched": len(tp_keys),
        })

    tp, fp, fn = detection["tp"], detection["fp"], detection["fn"]
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not fn else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    field_accuracy = {field: (field_correct[field] / field_totals[field]
                              if field_totals[field] else None)
                      for field in ("period", "units", "range", "midpoint", "action")}

    size_gate = {
        "documents": len(goldset) >= MIN_GOLD_DOCUMENTS,
        "sectors": len(sectors) >= MIN_SECTORS,
        "formats": len(formats) >= MIN_FORMATS,
    }
    accuracy_gate = {
        "detection_precision": precision >= MIN_DETECTION_PRECISION,
        "detection_recall": recall >= MIN_DETECTION_RECALL,
        "period_accuracy": (field_accuracy["period"] or 0.0) >= MIN_PERIOD_ACCURACY,
        "range_accuracy": (field_accuracy["range"] or 0.0) >= MIN_RANGE_ACCURACY,
        "units_accuracy": (field_accuracy["units"] or 0.0) >= MIN_UNITS_ACCURACY,
        "action_accuracy": (field_accuracy["action"] or 0.0) >= MIN_ACTION_ACCURACY,
    }
    gates = {**{f"size:{k}": v for k, v in size_gate.items()},
             **accuracy_gate}
    if not goldset:
        status = "NO_GOLDSET"
    elif not all(size_gate.values()):
        status = "GOLDSET_TOO_SMALL"
    elif all(accuracy_gate.values()):
        status = "CERTIFIED"
    else:
        status = "ACCURACY_BELOW_THRESHOLD"
    return {
        "status": status, "audit_version": AUDIT_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "documents": len(goldset), "sectors": sorted(sectors),
        "formats": sorted(formats),
        "detection": {"tp": tp, "fp": fp, "fn": fn,
                      "precision": precision, "recall": recall},
        "field_accuracy": field_accuracy,
        "thresholds": {
            "min_documents": MIN_GOLD_DOCUMENTS, "min_sectors": MIN_SECTORS,
            "min_formats": MIN_FORMATS,
            "detection_precision": MIN_DETECTION_PRECISION,
            "detection_recall": MIN_DETECTION_RECALL,
            "period_accuracy": MIN_PERIOD_ACCURACY,
            "range_accuracy": MIN_RANGE_ACCURACY,
            "units_accuracy": MIN_UNITS_ACCURACY,
            "action_accuracy": MIN_ACTION_ACCURACY,
        },
        "gates": gates, "certified": status == "CERTIFIED",
        "pilot_justified": status == "CERTIFIED",
        "per_document": per_document,
    }


def certify(goldset: list[dict] | None = None) -> dict:
    """Run the audit and, only if CERTIFIED, write a frozen certification file."""
    result = audit(goldset)
    if result["certified"]:
        CERTIFICATION_PATH.write_text(json.dumps(result, indent=2, default=str))
    return result
