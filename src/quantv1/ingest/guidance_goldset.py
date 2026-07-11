"""Frozen gold-set audit and certification for the MGRM guidance extractor.

Requiring the deterministic parser and the LLM to agree is a *precision filter*,
not proof of correctness: both can agree on the same wrong number, and the
intersection is biased toward easy prose. The scientific gate is measured
field-level accuracy against a frozen, manually labelled set of real filings.

Three evaluations are reported for every audit:
  * deterministic  -- the table+prose parser output;
  * ai             -- the configured LLM structured output;
  * reconciled     -- the final AGREED records that actually enter MGRM.

Certification measures the *reconciled* output, because that is what feeds the
model. With no AI backend configured there are no AGREED records, so end-to-end
certification cannot pass. Synthetic EXAMPLE-* fixtures validate the machinery
but never count toward the real document/sector/format requirements.

Certification is granted only when the real set is large and diverse AND the
reconciled accuracy clears every frozen threshold; the artifact records the
gold-set SHA-256, code hash, extractor version, provider/model, thresholds and
result so downstream gates can detect a stale or wrong-provider certification.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

from ..config import DATA_DIR, ROOT
from .guidance import (
    EXTRACTOR_VERSION, _same_number, ai_extract, llm_config, provider_tag,
    reconcile, structured_extract,
)


GOLD_PATH = ROOT / "goldset" / "mgrm_guidance_gold.jsonl"
CERTIFICATION_PATH = DATA_DIR / "mgrm_extractor_certification.json"
AUDIT_VERSION = "mgrm-goldset-audit-v2"

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


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def goldset_sha256(path: Path | None = None) -> str:
    path = path or GOLD_PATH
    if not path.exists():
        return "absent"
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _is_synthetic(document: dict) -> bool:
    return bool(document.get("synthetic")) or \
        str(document.get("company", "")).upper().startswith("EXAMPLE")


def _key(record: dict) -> tuple[str, str]:
    # Predicted records use ``guidance_period``; gold labels use ``period``.
    period = record.get("guidance_period", record.get("period"))
    return (str(record.get("metric")), str(period))


def _document_html(document: dict) -> str:
    html = document.get("document_html")
    if html is None and document.get("raw_path"):
        html = Path(document["raw_path"]).read_text(encoding="utf-8",
                                                    errors="replace")
    return html or ""


def _predict_variants(document: dict, config: dict | None) -> tuple[list, list, list]:
    """Deterministic, AI, and reconciled-AGREED predictions for one document."""
    from .earnings import _plain_text
    html = _document_html(document)
    deterministic = structured_extract(html)
    ai = ai_extract(_plain_text(html), config) if config is not None else None
    reconciled = [record for record in reconcile(deterministic, ai)
                  if record["agreement_status"] == "AGREED"]
    return deterministic, (ai or []), reconciled


def _score(goldset: list[dict], predictions: list[list[dict]]) -> dict:
    detection = Counter()
    field_totals = Counter()
    field_correct = Counter()
    for document, predicted_records in zip(goldset, predictions):
        expected = {} if document.get("no_guidance") else {
            _key(record): record for record in document.get("expected", [])
        }
        predicted = {_key(record): record for record in predicted_records}
        tp_keys = set(expected) & set(predicted)
        detection["tp"] += len(tp_keys)
        detection["fp"] += len(set(predicted) - set(expected))
        detection["fn"] += len(set(expected) - set(predicted))
        for key in tp_keys:
            want, got = expected[key], predicted[key]
            checks = {
                "period": True,
                "units": str(want.get("units")) == str(got.get("units")),
                "range": _same_number(want.get("low"), got.get("lower_value")) and
                         _same_number(want.get("high"), got.get("upper_value")),
                "midpoint": _same_number(want.get("midpoint"), got.get("midpoint")),
                "action": str(want.get("action")) == str(got.get("stated_action")),
            }
            for field, correct in checks.items():
                field_totals[field] += 1
                field_correct[field] += int(correct)
    tp, fp, fn = detection["tp"], detection["fp"], detection["fn"]
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not fn else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    field_accuracy = {field: (field_correct[field] / field_totals[field]
                              if field_totals[field] else None)
                      for field in ("period", "units", "range", "midpoint", "action")}
    return {"detection": {"tp": tp, "fp": fp, "fn": fn,
                          "precision": precision, "recall": recall},
            "field_accuracy": field_accuracy}


def audit(goldset: list[dict] | None = None) -> dict:
    goldset = goldset if goldset is not None else load_goldset()
    config = llm_config()
    det_preds, ai_preds, rec_preds = [], [], []
    for document in goldset:
        deterministic, ai, reconciled = _predict_variants(document, config)
        det_preds.append(deterministic)
        ai_preds.append(ai)
        rec_preds.append(reconciled)
    evaluations = {
        "deterministic": _score(goldset, det_preds),
        "ai": _score(goldset, ai_preds),
        "reconciled": _score(goldset, rec_preds),
    }
    certified_eval = evaluations["reconciled"]
    detection = certified_eval["detection"]
    field_accuracy = certified_eval["field_accuracy"]

    real = [document for document in goldset if not _is_synthetic(document)]
    real_sectors = {str(d.get("sector", "UNKNOWN")) for d in real}
    real_formats = {str(d.get("format", "unknown")) for d in real}
    size_gate = {
        "real_documents": len(real) >= MIN_GOLD_DOCUMENTS,
        "real_sectors": len(real_sectors) >= MIN_SECTORS,
        "real_formats": len(real_formats) >= MIN_FORMATS,
    }
    accuracy_gate = {
        "detection_precision": detection["precision"] >= MIN_DETECTION_PRECISION,
        "detection_recall": detection["recall"] >= MIN_DETECTION_RECALL,
        "period_accuracy": (field_accuracy["period"] or 0.0) >= MIN_PERIOD_ACCURACY,
        "range_accuracy": (field_accuracy["range"] or 0.0) >= MIN_RANGE_ACCURACY,
        "units_accuracy": (field_accuracy["units"] or 0.0) >= MIN_UNITS_ACCURACY,
        "action_accuracy": (field_accuracy["action"] or 0.0) >= MIN_ACTION_ACCURACY,
    }
    gates = {**{f"size:{k}": v for k, v in size_gate.items()}, **accuracy_gate}
    if not goldset:
        status = "NO_GOLDSET"
    elif not all(size_gate.values()):
        status = "GOLDSET_TOO_SMALL"
    elif config is None:
        status = "NO_AI_BACKEND"
    elif all(accuracy_gate.values()):
        status = "CERTIFIED"
    else:
        status = "ACCURACY_BELOW_THRESHOLD"
    certified = status == "CERTIFIED"
    return {
        "status": status, "audit_version": AUDIT_VERSION,
        "extractor_version": EXTRACTOR_VERSION, "provider": provider_tag(config),
        "goldset_sha256": goldset_sha256(), "code_hash": _git_hash(),
        "documents": len(goldset), "real_documents": len(real),
        "real_sectors": sorted(real_sectors), "real_formats": sorted(real_formats),
        "certified_output": "reconciled", "evaluations": evaluations,
        "detection": detection, "field_accuracy": field_accuracy,
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
        "gates": gates, "certified": certified,
        "pilot_justified": certified,
    }


def certify(goldset: list[dict] | None = None) -> dict:
    """Audit and, only if CERTIFIED, write the frozen certification artifact."""
    result = audit(goldset)
    if result["certified"]:
        CERTIFICATION_PATH.write_text(json.dumps(result, indent=2, default=str))
    return result


def certification_status() -> dict:
    """Hard gate: is a valid certification present for the current extractor?"""
    if not CERTIFICATION_PATH.exists():
        return {"certified": False, "reason": "CERTIFICATION_ABSENT"}
    record = json.loads(CERTIFICATION_PATH.read_text())
    if not record.get("certified"):
        return {"certified": False, "reason": "CERTIFICATION_NOT_GRANTED"}
    if record.get("goldset_sha256") != goldset_sha256():
        return {"certified": False, "reason": "CERTIFICATION_STALE_GOLDSET"}
    if record.get("extractor_version") != EXTRACTOR_VERSION:
        return {"certified": False, "reason": "CERTIFICATION_WRONG_EXTRACTOR"}
    if record.get("provider") != provider_tag():
        return {"certified": False, "reason": "CERTIFICATION_WRONG_PROVIDER"}
    return {"certified": True, "reason": "CERTIFIED",
            "provider": record.get("provider"),
            "goldset_sha256": record.get("goldset_sha256")}
