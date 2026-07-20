"""Human gold-set preparation and scoring for SEC Event Atlas Phase A.

The sealed partition is issuer-disjoint and is never used by extraction or
family selection.  This module only scores labels supplied by a human reviewer;
unlabeled rows are reported as pending rather than imputed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def score_goldset(path: str | Path) -> dict:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    labeled = [r for r in rows if r.get("human_label") is not None]
    out = {"rows": len(rows), "labeled": len(labeled), "pending": len(rows) - len(labeled),
           "partitions": {}, "protocol": "issuer-disjoint; sealed is evaluation-only"}
    for partition in ("development", "sealed"):
        subset = [r for r in labeled if r.get("issuer_split") == partition]
        # Human reviewers may override the extractor's document-level decision
        # explicitly.  Until then, the catalog record is the frozen prediction.
        predicted_events = [r for r in subset if (r.get("document_detection") if r.get("document_detection") is not None else r.get("record_type") == "event")]
        actual_events = [r for r in subset if str(r.get("human_label")).lower() in {"event", "material_event", "yes", "1", "true"}]
        tp = sum(1 for r in predicted_events if r in actual_events)
        fp = len(predicted_events) - tp
        fn = sum(1 for r in actual_events if r not in predicted_events)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        typed = [r for r in predicted_events if r in actual_events and r.get("event_type_label")]
        type_acc = (sum(1 for r in typed if r["event_type_label"] == r.get("source", {}).get("event_type")) / len(typed)) if typed else None
        grounded = [r for r in labeled if r.get("issuer_split") == partition and r.get("evidence_grounding") is not None]
        grounding = (sum(1 for r in grounded if bool(r.get("evidence_grounding"))) / len(grounded)) if grounded else None
        magnitudes = [r for r in subset if r.get("magnitude_label") is not None]
        out["partitions"][partition] = {"labeled": len(subset), "event_detection_precision": precision,
                                         "event_detection_recall": recall, "event_type_accuracy": type_acc,
                                         "evidence_grounding_accuracy": grounding,
                                         "magnitude_extraction_coverage": len(magnitudes) / len(subset) if subset else None,
                                         "routine_controls_confirmed": sum(1 for r in subset if r.get("record_type") == "routine_control" and str(r.get("human_label")).lower() in {"routine", "no_event", "no-material"})}
    return out


def write_scored_report(path: str | Path, output: str | Path) -> dict:
    report = score_goldset(path)
    Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
